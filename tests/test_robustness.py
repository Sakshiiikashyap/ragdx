"""P5 tests: per-level reports + exp06 eval assembly (offline, schema must be
consumable by analysis/plots.fig_degradation)."""

from __future__ import annotations

from pathlib import Path

from ragdx.analysis.robustness import build_eval, level_report


def _rec(qid, level, truth, pred, c1=1):
    return {"query_id": qid, "level": level, "ground_truth": truth,
            "pred_label": pred, "c1": c1, "c0": 1, "c2": 1, "R": 1.0}


RECORDS = [
    _rec("q1", "clean", "RETRIEVER", "RETRIEVER"),
    _rec("q2", "clean", "GENERATOR", "RETRIEVER", c1=0),
    _rec("q1", "drop50", "RETRIEVER", "NONE", c1=0),
    _rec("q2", "drop50", "GENERATOR", "RETRIEVER", c1=0),
]
LEVELS = ["clean", "drop50"]


def test_level_report_basic():
    rep = level_report(RECORDS, "clean")
    # RETRIEVER f1=2*(.5*1)/1.5=.6667, GENERATOR f1=0 -> macro=.3333
    assert rep == {"n": 2, "accuracy": 0.5, "macro_f1": 0.3333}
    assert level_report(RECORDS, "drop50")["n"] == 2


def test_level_report_empty_returns_none():
    assert level_report(RECORDS, "pseudo_gold") is None


def test_build_eval_schema_matches_plots(tmp_path):
    eval_doc = build_eval(RECORDS, LEVELS, tau_r=0.5, n_tasks=2,
                          skips={"pseudo_gold": 2},
                          baseline_name="recall_threshold",
                          baseline_preds={"q1": "RETRIEVER",
                                          "q2": "RETRIEVER"},
                          flip_ps=[0.0, 0.1], n_rep=5, seed=0)
    # what fig_degradation consumes
    assert eval_doc["conditions"] == LEVELS
    for sysname, per_cond in eval_doc["by_system"].items():
        for cond in LEVELS:
            assert isinstance(per_cond[cond]["macro_f1"], float)
    # baseline is gold-independent: same predictions at both levels
    b = eval_doc["by_system"]["recall_threshold"]
    assert b["clean"] == {"n": 2, "accuracy": 0.5, "macro_f1": 0.3333}
    assert b["drop50"] == b["clean"]
    # extras for the paper paragraph
    assert eval_doc["c1_rate"]["clean"] == 0.5
    assert eval_doc["c1_rate"]["drop50"] == 0.0     # degradation took effect
    assert eval_doc["skips"] == {"pseudo_gold": 2}
    assert eval_doc["best_baseline"] == "recall_threshold"
    assert eval_doc["by_fault_class"]["RETRIEVER"]["clean"] == 1.0
    assert eval_doc["judge_noise"][0]["flip_p"] == 0.0
    assert eval_doc["judge_noise"][0]["std_accuracy"] == 0.0


def test_build_eval_without_baseline():
    eval_doc = build_eval(RECORDS, LEVELS, tau_r=0.5, n_tasks=2, skips={})
    assert eval_doc["best_baseline"] is None
    assert set(eval_doc["by_system"]) == {"rule_v1"}


def test_plots_reads_our_schema(tmp_path):
    from ragdx.analysis.plots import fig_degradation
    import json
    eval_doc = build_eval(RECORDS, LEVELS, tau_r=0.5, n_tasks=2, skips={},
                          flip_ps=[0.0], n_rep=3, seed=0)
    out = tmp_path / "results" / "exp06" / "eval.json"
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps(eval_doc), encoding="utf-8")
    png = tmp_path / "deg.png"
    assert fig_degradation(tmp_path, png) is True and png.exists()
