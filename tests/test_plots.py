"""Tests for analysis/plots.py — every figure function must (a) write a PNG
when its input artifact exists under a given root, (b) return False without
raising when the input is missing (graceful skip), and (c) never touch the
real results/ tree when given a temp root."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragdx.analysis import plots


def _w(tmp_path: Path, rel: str, payload) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ------------------------------------------------------------ fixtures
TUNING = {"tau_r": 0.5, "tuned_on": "faultbench dev", "criterion": "macro-F1",
          "dev_macro_f1": 0.5, "frozen_date": "2026-10-15",
          "sweep": [{"tau_r": t, "macro_f1": 0.3 + 0.1 * t, "accuracy": 0.4 + t / 20}
                    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]}

EVAL = {"n": 61, "tau_r_frozen": 0.5, "accuracy": 0.55, "macro_f1": 0.48,
        "bootstrap_95ci": [0.42, 0.68],
        "per_class": {"RETRIEVER": {"precision": 0.8, "recall": 0.7, "f1": 0.75,
                                    "support": 23}},
        "confusion_truth_rows_pred_cols": {
            "RETRIEVER": {"RETRIEVER": 16, "NONE": 7},
            "GENERATOR": {"RETRIEVER": 12, "GENERATOR": 10, "NONE": 8},
            "INTERACTION": {"NONE": 6, "GENERATOR": 2}},
        "per_fault_accuracy": {"remove_gold": 0.7, "ignore_context": 0.33}}

ITEMS = [{"query_id": "q1", "fault": "remove_gold", "ground_truth": "RETRIEVER",
          "correct_attribution": True},
         {"query_id": "q2", "fault": "remove_gold", "ground_truth": "RETRIEVER",
          "correct_attribution": False},
         {"query_id": "q3", "fault": "ignore_context", "ground_truth": "GENERATOR",
          "correct_attribution": False},
         {"query_id": "q4", "fault": "contradiction", "ground_truth": "INTERACTION",
          "correct_attribution": True}]

EVAL04 = {"n": 61, "tau_r": 0.5, "gamma": 0.4,
          "systems": {
              "rule_v1": {"accuracy": 0.56, "macro_f1": 0.48,
                          "bootstrap_95ci_acc": [0.43, 0.68]},
              "recall_threshold": {"accuracy": 0.44, "macro_f1": 0.38,
                                   "bootstrap_95ci_acc": [0.31, 0.57]},
              "metric_rule": {"accuracy": 0.49, "macro_f1": 0.42,
                              "bootstrap_95ci_acc": [0.36, 0.62]},
              "llm_diagnoser": {"accuracy": 0.6, "macro_f1": 0.51,
                                "bootstrap_95ci_acc": [0.47, 0.72]}},
          "mcnemar_vs_rule_v1": {}}


# ------------------------------------------------------------ happy paths
@pytest.mark.parametrize("name,fixture_rel,payload", [
    ("tau_sweep", "results/exp03_dev/tuning.json", TUNING),
    ("confusion", "results/exp03/eval.json", EVAL),
    ("per_fault", "results/exp03/items.json", ITEMS),
    ("baselines", "results/exp04/eval.json", EVAL04),
])
def test_figure_written_when_input_exists(tmp_path, name, fixture_rel, payload,
                                          monkeypatch):
    _w(tmp_path, fixture_rel, payload)
    out = tmp_path / "fig" / f"{name}.png"
    wrote = plots.FIGURES[name](tmp_path, out)
    assert wrote is True
    assert out.exists() and out.stat().st_size > 1000
    # image is a real PNG
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("name", list(plots.FIGURES))
def test_graceful_skip_when_missing(tmp_path, name):
    out = tmp_path / "fig" / f"{name}.png"
    assert plots.FIGURES[name](tmp_path, out) is False
    assert not out.exists()


def test_all_mode_writes_everything_available(tmp_path, capsys):
    _w(tmp_path, "results/exp03/eval.json", EVAL)
    _w(tmp_path, "results/exp04/eval.json", EVAL04)
    written = plots.run_all(tmp_path, tmp_path / "fig")
    assert "confusion" in written and "baselines" in written
    assert "tau_sweep" not in written
    assert (tmp_path / "fig" / "confusion.png").exists()


def test_figures_dir_created(tmp_path):
    _w(tmp_path, "results/exp03_dev/tuning.json", TUNING)
    out = tmp_path / "nested" / "deep" / "tau_sweep.png"
    assert plots.FIGURES["tau_sweep"](tmp_path, out) is True
    assert out.exists()
