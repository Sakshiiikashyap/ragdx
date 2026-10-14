"""P5 tests: gold-reference degradation (docs/intent P5) — pure, deterministic,
offline. Includes one integration test proving an oracle_chunk override (via
dataclasses.replace) changes ONLY the gold condition of the three-condition run
(fake LLM client; never touches Groq)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ragdx.datasets.mirage import MirageItem
from ragdx.faults import FaultedTask, control
from ragdx.faults.degrade import (LEVELS, drop_content, near_miss_chunk,
                                  pseudo_gold_chunk, resolve_gold)

ANSWER = "Paris"

CHUNKS = [
    "The capital of France is Paris and it sits on the Seine river banks",   # 0 gold
    "Photosynthesis converts sunlight into chemical energy stored in glucose",  # 1
    "The Eiffel tower was completed in 1889 and attracts millions of visitors",  # 2
    "Paris forty two hosts the largest football stadium in the country",      # 3 answer-bearing
]


def make_item(**kw) -> MirageItem:
    defaults = dict(query_id="q1", query="What is the capital of France?",
                    answers=[ANSWER], oracle_chunk=CHUNKS[0], oracle_id="d1",
                    oracle_support=1, chunks=list(CHUNKS),
                    chunk_ids=["d1", "d2", "d3", "d4"], gold_index=0)
    defaults.update(kw)
    return MirageItem(**defaults)


# ------------------------------------------------------------ drop_content
def test_drop_zero_is_byte_identical():
    s = "  Weird   spacing\tand  newlines\nkept  "
    assert drop_content(s, 0.0) == s


def test_drop_full_is_empty():
    assert drop_content("one two three four", 1.0) == ""


def test_drop_counts_and_determinism():
    text = " ".join(str(i) for i in range(100))
    for frac, keep in [(0.25, 75), (0.5, 50), (0.75, 25)]:
        out1, out2 = drop_content(text, frac), drop_content(text, frac)
        assert out1 == out2
        assert len(out1.split()) == keep


def test_drop_is_position_uniform_not_a_truncation():
    text = " ".join(str(i) for i in range(100))
    kept = {int(w) for w in drop_content(text, 0.5).split()}
    dropped = sorted(set(range(100)) - kept)
    assert len(dropped) == 50
    assert dropped[0] < 50 < dropped[-1]          # spread across whole chunk
    gaps = [b - a for a, b in zip(dropped, dropped[1:])]
    assert max(gaps) <= 4                          # regular stride for m=50


def test_drop_bad_fraction_raises():
    with pytest.raises(ValueError):
        drop_content("a b", -0.1)
    with pytest.raises(ValueError):
        drop_content("a b", 1.1)


# ------------------------------------------------------------ chunk swaps
def test_near_miss_is_non_gold_and_answer_free():
    task = control(make_item())
    got = near_miss_chunk(task)
    assert got is not None
    assert got != CHUNKS[0]
    assert ANSWER.lower() not in got.lower()
    assert got in CHUNKS                           # comes from the pool


def test_near_miss_deterministic():
    task = control(make_item())
    assert near_miss_chunk(task) == near_miss_chunk(task)


def test_near_miss_none_when_no_candidate():
    # pool = gold + an answer-bearing distractor only -> no answer-free candidate
    item = make_item(chunks=[CHUNKS[0], CHUNKS[3]],
                     chunk_ids=["d1", "d4"], gold_index=0)
    assert near_miss_chunk(control(item)) is None


def test_pseudo_gold_returns_answer_bearing_non_gold():
    task = control(make_item())
    got = pseudo_gold_chunk(task)
    assert got == CHUNKS[3]
    assert ANSWER.lower() in got.lower()


def test_pseudo_gold_none_when_only_gold_has_answer():
    item = make_item(chunks=[CHUNKS[0], CHUNKS[1]],
                     chunk_ids=["d1", "d2"], gold_index=0)
    assert pseudo_gold_chunk(control(item)) is None


def test_pseudo_gold_works_when_gold_removed_from_pool():
    # SEARA-style: surrogate picked from a pool that no longer holds the gold row
    task = replace(control(make_item()), pool_chunks=[CHUNKS[1], CHUNKS[3]],
                   pool_ids=["d2", "d4"], pool_orig_index=[1, 3])
    assert pseudo_gold_chunk(task) == CHUNKS[3]


# ------------------------------------------------------------ resolve_gold
def test_resolve_levels():
    task = control(make_item())
    assert resolve_gold(task, "clean") == CHUNKS[0]
    assert resolve_gold(task, "drop25") == drop_content(CHUNKS[0], 0.25)
    assert resolve_gold(task, "drop50") == drop_content(CHUNKS[0], 0.50)
    assert resolve_gold(task, "drop75") == drop_content(CHUNKS[0], 0.75)
    assert resolve_gold(task, "near_miss") == near_miss_chunk(task)
    assert resolve_gold(task, "pseudo_gold") == CHUNKS[3]
    with pytest.raises(ValueError):
        resolve_gold(task, "nonsense")


def test_levels_order():
    assert LEVELS == ["clean", "drop25", "drop50", "drop75",
                      "near_miss", "pseudo_gold"]


# ------------------------------------------------------------ integration
def _fake_llm_factory(calls: list):
    from ragdx.llm.client import LLMResponse

    class FakeLLM:
        def __init__(self, model="fake", cache_dir=None, log_path=None,
                     temperature=0.0, **kw):
            self.model, self.temperature = model, temperature
            self.cache_dir, self.log_path = cache_dir, log_path

        def chat(self, messages, **kw):
            calls.append(messages)
            user = messages[-1]["content"]
            if "Context:" in user:
                ctx = user.split("Context:", 1)[1].split("Answer:", 1)[0]
                text = ANSWER if ANSWER.lower() in ctx.lower() else "unknown"
            else:
                text = "unknown"
            return LLMResponse(text=text, model="fake")

    return FakeLLM


def test_oracle_override_changes_only_gold_condition(monkeypatch, tmp_path):
    from ragdx.faults import runner as R

    calls: list = []
    monkeypatch.setattr(R, "LLMClient", _fake_llm_factory(calls))
    parent = R.LLMClient(model="fake", cache_dir=tmp_path, log_path=tmp_path / "l.jsonl")

    clean = control(make_item())
    degraded = replace(control(make_item()), oracle_chunk=CHUNKS[1])  # answer gone

    r_clean = R.run_task(clean, parent)
    r_deg = R.run_task(degraded, parent)

    # clean gold answers; degraded gold does not; base/retrieved unchanged
    assert r_clean["c1"] == 1 and r_deg["c1"] == 0
    assert r_clean["c0"] == r_deg["c0"]
    assert r_clean["c2"] == r_deg["c2"]
    assert r_clean["R"] == r_deg["R"]

    base_p, gold_p, ret_p = calls[:3]
    base2, gold2, ret2 = calls[3:]
    assert base_p == base2 and ret_p == ret2
    assert gold_p != gold2
    assert ANSWER in gold_p[-1]["content"] and ANSWER not in gold2[-1]["content"]
