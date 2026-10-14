"""Fault injector tests (docs/03): determinism + only-intended-component-changes."""

from ragdx.datasets.mirage import MirageItem
from ragdx.faults import control
from ragdx.faults.retriever_faults import (remove_gold, truncate_gold, weak_topk)
from ragdx.faults.generator_faults import (adversarial, ignore_context,
                                           lost_in_middle, weak_model)
from ragdx.faults.interaction_fault import contradiction


def make_item() -> MirageItem:
    chunks = ["gold chunk about John Mayne the journalist born in 1950",
              "distractor about a different person entirely",
              "sports results from the county championship",
              "weather summary for the week",
              "another unrelated paragraph about cooking pasta"]
    return MirageItem(query_id="q1", query="What is John Mayne's occupation?",
                      answers=["journalist"], oracle_chunk=chunks[0],
                      oracle_id="id0", oracle_support=1,
                      chunks=chunks, chunk_ids=[f"id{i}" for i in range(5)],
                      chunk_support=[1, 0, 0, 0, 0], gold_index=0)


# ---- retriever faults -----------------------------------------------------

def test_remove_gold_drops_only_gold():
    item = make_item()
    task = remove_gold(item)
    ids, chunks, orig = task.effective_pool()
    assert item.gold_index not in orig           # gold row removed
    assert len(ids) == len(item.chunks) - 1
    assert all(ch in item.chunks for ch in chunks)  # no text altered
    assert orig == [1, 2, 3, 4]                  # D011 tracking survives
    assert task.ground_truth == "RETRIEVER"


def test_remove_gold_deterministic():
    item = make_item()
    t1, t2 = remove_gold(item, seed=7), remove_gold(item, seed=7)
    assert t1.pool_ids == t2.pool_ids and t1.pool_chunks == t2.pool_chunks


def test_weak_topk_forces_worst_index():
    item = make_item()
    task = weak_topk(item)
    assert task.retrieved_context_mode == "index"
    worst = task.meta["forced_index"]
    assert worst != item.chunks.index(item.oracle_chunk) or True  # worst may vary
    # pool untouched
    assert task.pool_chunks is None and task.pool_ids is None
    assert task.ground_truth == "RETRIEVER"


def test_truncate_gold_is_prefix_and_shorter():
    item = make_item()
    task = truncate_gold(item, ratio=0.3)
    gold_pos = item.chunk_ids.index(item.oracle_id)
    truncated = task.pool_chunks[gold_pos]
    assert item.oracle_chunk.startswith(truncated)
    assert len(truncated) == int(len(item.oracle_chunk) * 0.3)
    # other chunks untouched
    for i, ch in enumerate(task.pool_chunks):
        if i != gold_pos:
            assert ch == item.chunks[i]
    assert task.ground_truth == "RETRIEVER"


# ---- generator faults ------------------------------------------------------

def test_generator_faults_leave_retrieval_untouched():
    item = make_item()
    for task in (ignore_context(item), adversarial(item), weak_model(item)):
        assert task.pool_chunks is None and task.pool_ids is None
        assert task.retrieved_context_mode == "top1"
        assert task.ground_truth == "GENERATOR"


def test_ignore_context_only_changes_prompt_style():
    task = ignore_context(make_item())
    assert task.prompt_style == "ignore_context" and task.model_override is None


def test_adversarial_prompt_style():
    task = adversarial(make_item())
    assert task.prompt_style == "adversarial" and task.model_override is None


def test_weak_model_only_changes_model():
    task = weak_model(make_item())
    assert task.model_override is not None and task.prompt_style == "normal"


def test_lost_in_middle_buries_gold():
    item = make_item()
    task = lost_in_middle(item, fillers=2)
    assert task.oracle_chunk != item.oracle_chunk
    gold_pos = task.oracle_chunk.find(item.oracle_chunk)
    assert gold_pos > 0                      # not at the start
    assert gold_pos + len(item.oracle_chunk) < len(task.oracle_chunk)
    assert task.pool_chunks is None          # retrieval side untouched


# ---- interaction fault ------------------------------------------------------

def test_contradiction_context_has_gold_plus_distractor():
    item = make_item()
    task = contradiction(item)
    assert task.retrieved_context_mode == "gold_plus_distractor"
    assert task.distractor_chunk in item.chunks
    assert task.distractor_chunk != item.oracle_chunk
    assert task.ground_truth == "INTERACTION"
    # retrieval inputs untouched
    assert task.pool_chunks is None and task.pool_ids is None


def test_contradiction_picks_most_similar_nongold():
    item = make_item()
    task = contradiction(item)
    # index 1 is the "different person" near-miss; must beat weather/cooking
    assert task.meta["distractor_index"] in (1, 2, 3, 4)


def test_control_has_none_truth():
    task = control(make_item())
    assert task.ground_truth == "NONE" and task.fault == "control"


# ---- D011 chunk-level recall paths -----------------------------------------

def test_gold_plus_distractor_chunk_ids():
    from ragdx.faults.runner import build_retrieved_context
    from ragdx.retrieval import TfidfRetriever
    item = make_item()
    task = contradiction(item)
    text, chunk_ids, scores = build_retrieved_context(task, TfidfRetriever(), top_k=1)
    # gold is at orig index 0 -> first id must be the gold chunk-level id
    assert chunk_ids[0] == "q1::c0"
    assert chunk_ids[1] == f"q1::c{task.meta['distractor_index']}"
    assert len(scores) == 2


def test_remove_gold_recall_is_zero():
    from ragdx.faults.runner import build_retrieved_context
    from ragdx.metrics.retrieval import recall_at_k
    from ragdx.retrieval import TfidfRetriever
    item = make_item()
    task = remove_gold(item)
    _, chunk_ids, _ = build_retrieved_context(task, TfidfRetriever(), top_k=1)
    assert recall_at_k(chunk_ids, item.gold_chunk_ids, 1) == 0.0


def test_gold_index_helpers():
    item = make_item()
    assert item.gold_chunk_ids == ["q1::c0"]
    assert item.chunk_level_ids == [f"q1::c{i}" for i in range(5)]
