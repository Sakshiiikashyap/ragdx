"""P7 tests: repair-policy selection + utility eval assembly (offline; schema
must be consumable by analysis/plots.fig_utility)."""

from __future__ import annotations

import json

import pytest

from ragdx.analysis.utility import (build_utility_eval, select_repair)


def test_select_repair_cases():
    assert select_repair("RETRIEVER") == "topk3"
    assert select_repair("GENERATOR") == "grounding"
    assert select_repair("INTERACTION") == "no_op"
    assert select_repair("NONE") == "no_op"
    assert select_repair("ABSTAIN") == "no_op"


def _rows():
    # 4 failures: our labels select different repairs; some repairs succeed
    return [
        {"query_id": "q1", "pred_label": "RETRIEVER",
         "fixed": {"topk3": 1, "grounding": 0, "gold_rerank": 1}},
        {"query_id": "q2", "pred_label": "GENERATOR",
         "fixed": {"topk3": 0, "grounding": 1, "gold_rerank": 0}},
        {"query_id": "q3", "pred_label": "GENERATOR",
         "fixed": {"topk3": 1, "grounding": 0}},           # gold_rerank n/a
        {"query_id": "q4", "pred_label": "INTERACTION",
         "fixed": {"topk3": 0, "grounding": 0, "gold_rerank": 0}},
    ]


def test_utility_gains_hand_computed():
    doc = build_utility_eval(_rows(), seed=0)
    gains = {f["name"]: f["gain"] for f in doc["fixes"]}
    assert gains["our_fix"] == pytest.approx(0.5)      # q1 + q2 fixed
    assert gains["always_topk3"] == pytest.approx(0.5)  # q1 + q3
    assert gains["always_grounding"] == pytest.approx(0.25)
    assert gains["gold_rerank"] == pytest.approx(0.25)
    assert 0.0 <= gains["random_fix"] <= 1.0
    assert doc["n_failure"] == 4
    assert doc["n_gold_in_pool"] == 3
    assert doc["repair_selected_by_ours"] == {
        "topk3": 1, "grounding": 2, "no_op": 1}


def test_utility_sign_tests():
    doc = build_utility_eval(_rows(), seed=0)
    mc = doc["mcnemar_ours_vs"]
    assert set(mc) == {"random_fix", "always_topk3", "always_grounding"}
    # ours vs always_topk3: discordant = q2 (ours only) and q3 (topk3 only)
    assert mc["always_topk3"]["b_ours_only"] == 1
    assert mc["always_topk3"]["c_baseline_only"] == 1
    assert 0.0 <= mc["always_topk3"]["p_holm"] <= 1.0


def test_random_fix_deterministic():
    a = build_utility_eval(_rows(), seed=0)
    b = build_utility_eval(_rows(), seed=0)
    c = build_utility_eval(_rows(), seed=1)
    assert a == b
    ga = {f["name"]: f["gain"] for f in a["fixes"]}
    gc = {f["name"]: f["gain"] for f in c["fixes"]}
    assert ga["always_topk3"] == gc["always_topk3"]      # seed-independent parts
    assert ga["our_fix"] == gc["our_fix"]


def test_empty_failure_set_raises():
    with pytest.raises(ValueError):
        build_utility_eval([])


def test_plots_reads_utility_schema(tmp_path):
    from ragdx.analysis.plots import fig_utility
    doc = build_utility_eval(_rows(), seed=0)
    p = tmp_path / "results" / "exp08" / "eval.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(doc), encoding="utf-8")
    png = tmp_path / "u.png"
    assert fig_utility(tmp_path, png) is True and png.exists()
