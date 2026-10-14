"""exp03a: run rule v1 on FaultBench DEV split, tune tau_R on dev ONLY,
freeze thresholds to configs/frozen_thresholds.json (pre-registration step).

Usage: python experiments/exp03a_faultbench_dev.py
Outputs: results/exp03_dev/{items.json, tuning.json, manifest.json}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdx.analysis.stats import macro_f1
from ragdx.attribution.rule import AttributionThresholds, attribute
from ragdx.faults.loader import load_rows, rows_to_tasks
from ragdx.faults.runner import run_tasks
from ragdx.llm.client import LLMClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = load_rows(ROOT / "data" / "faultbench" / "dev.jsonl")
    tasks = rows_to_tasks(rows)
    print(f"exp03a: dev split, {len(tasks)} tasks")
    client = LLMClient(model="openai/gpt-oss-20b", cache_dir=ROOT / ".llm_cache",
                       log_path=ROOT / "results" / "_llm_calls.jsonl")
    run = run_tasks(tasks, client, ROOT / "results" / "exp03_dev", "exp03_dev",
                    tau_r=0.5, top_k=1)
    records = run["records"]

    # tau sweep on dev (post-hoc over stored c0,c1,c2,R — no API calls)
    sweep = []
    for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        preds = [attribute(r["c0"], r["c1"], r["c2"], r=r["R"],
                           thresholds=AttributionThresholds(tau_r=tau)).label.value
                 for r in records]
        truth = [r["ground_truth"] for r in records]
        sweep.append({"tau_r": tau, "macro_f1": round(macro_f1(truth, preds), 4),
                      "accuracy": round(sum(t == p for t, p in zip(truth, preds)) / len(truth), 4)})
        print(f"  tau={tau:.1f} macroF1={sweep[-1]['macro_f1']:.4f} acc={sweep[-1]['accuracy']:.4f}")
    best = max(sweep, key=lambda s: s["macro_f1"])
    frozen = {"tau_r": best["tau_r"], "tuned_on": "faultbench dev",
              "criterion": "macro-F1", "dev_macro_f1": best["macro_f1"],
              "sweep": sweep, "frozen_date": __import__("time").strftime("%Y-%m-%d")}
    (ROOT / "configs" / "frozen_thresholds.json").write_text(
        json.dumps(frozen, indent=2), encoding="utf-8")
    (ROOT / "results" / "exp03_dev" / "tuning.json").write_text(
        json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"FROZEN tau_r={best['tau_r']} (dev macro-F1 {best['macro_f1']}) -> configs/frozen_thresholds.json")


if __name__ == "__main__":
    main()
