"""exp03b: FaultBench TEST split, single use, with FROZEN tau_R from dev.

Rule v1 (and v2 where k-sample data exists) vs ground truth:
accuracy, macro-F1, per-class P/R/F1, confusion matrix, bootstrap 95% CI.

Usage: python experiments/exp03b_faultbench_test.py
Outputs: results/exp03/{items.json, eval.json, manifest.json}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdx.analysis.stats import (bootstrap_ci, confusion_matrix, macro_f1,
                                  per_class_prf)
from ragdx.attribution.rule import AttributionThresholds, attribute
from ragdx.faults.loader import load_rows, rows_to_tasks
from ragdx.faults.runner import run_tasks
from ragdx.llm.client import LLMClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    frozen = json.loads((ROOT / "configs" / "frozen_thresholds.json").read_text(encoding="utf-8"))
    tau = frozen["tau_r"]
    rows = load_rows(ROOT / "data" / "faultbench" / "test.jsonl")
    tasks = rows_to_tasks(rows)
    print(f"exp03b: test split, {len(tasks)} tasks, frozen tau_r={tau}")
    client = LLMClient(model="openai/gpt-oss-20b", cache_dir=ROOT / ".llm_cache",
                       log_path=ROOT / "results" / "_llm_calls.jsonl")
    run = run_tasks(tasks, client, ROOT / "results" / "exp03", "exp03",
                    tau_r=tau, top_k=1)
    records = run["records"]

    # re-attribute with frozen tau (records already carry it, but be explicit)
    preds = [attribute(r["c0"], r["c1"], r["c2"], r=r["R"],
                       thresholds=AttributionThresholds(tau_r=tau)).label.value
             for r in records]
    truth = [r["ground_truth"] for r in records]
    for r, p in zip(records, preds):
        r["pred_label_frozen"] = p
        r["correct_attribution"] = (p == r["ground_truth"])

    per_item_acc = [1.0 if p == t else 0.0 for p, t in zip(preds, truth)]
    mean, lo, hi = bootstrap_ci(per_item_acc, n_boot=10000, seed=0)
    eval_report = {
        "n": len(records), "tau_r_frozen": tau,
        "accuracy": round(sum(per_item_acc) / len(per_item_acc), 4),
        "bootstrap_95ci": [round(lo, 4), round(hi, 4)],
        "macro_f1": round(macro_f1(truth, preds), 4),
        "per_class": per_class_prf(truth, preds),
        "confusion_truth_rows_pred_cols": confusion_matrix(truth, preds),
        "per_fault_accuracy": {
            f: round(sum(1 for r in records if r["fault"] == f and r["correct_attribution"])
                     / max(1, sum(1 for r in records if r["fault"] == f)), 4)
            for f in sorted({r["fault"] for r in records})},
    }
    (ROOT / "results" / "exp03" / "eval.json").write_text(
        json.dumps(eval_report, indent=2), encoding="utf-8")
    (ROOT / "results" / "exp03" / "items.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = json.loads((ROOT / "results" / "exp03" / "manifest.json").read_text(encoding="utf-8"))
    manifest["eval_written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (ROOT / "results" / "exp03" / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                                              encoding="utf-8")
    print(json.dumps(eval_report, indent=2))


if __name__ == "__main__":
    main()
