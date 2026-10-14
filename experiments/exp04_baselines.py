"""exp04: attribution baselines on the SAME FaultBench test records as exp03.

(a) recall-threshold (tuned on dev via the same tau sweep, frozen)
(b) RAGChecker-inspired metric rule (tau_R + grounding gamma tuned on dev)
(c) LLM-as-diagnoser (live calls, one per test task)
Comparison: accuracy/macro-F1 + McNemar vs rule v1 with Holm correction.

Usage: python experiments/exp04_baselines.py  (requires results/exp03/items.json)
Outputs: results/exp04/{items.json, eval.json, manifest.json}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdx.analysis.stats import (bootstrap_ci, holm_correction, macro_f1,
                                  mcnemar_exact)
from ragdx.attribution.baselines import (baseline_llm_diagnoser,
                                         baseline_metric_rule,
                                         baseline_recall_threshold,
                                         grounding_proxy)
from ragdx.llm.client import LLMClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    frozen = json.loads((ROOT / "configs" / "frozen_thresholds.json").read_text(encoding="utf-8"))
    tau = frozen["tau_r"]
    records = json.loads((ROOT / "results" / "exp03" / "items.json").read_text(encoding="utf-8"))
    truth = [r["ground_truth"] for r in records]

    # gamma tuning for metric rule: sweep on the record set (dev-tuned would be
    # ideal; grounding proxy has one sensible operating point — sweep 0.3..0.7
    # on DEV would require dev records; reuse dev results if present)
    dev_path = ROOT / "results" / "exp03_dev" / "items.json"
    gamma = 0.5
    if dev_path.exists():
        dev_records = json.loads(dev_path.read_text(encoding="utf-8"))
        dev_truth = [r["ground_truth"] for r in dev_records]
        best_gamma, best_f1 = 0.5, -1.0
        for g in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            preds_g = [baseline_metric_rule(
                r["R"], r["retrieved_text"], r["retrieved_context"], tau_r=tau, gamma=g).value
                for r in dev_records]
            f1 = macro_f1(dev_truth, preds_g)
            if f1 > best_f1:
                best_gamma, best_f1 = g, f1
        gamma = best_gamma
        print(f"gamma tuned on dev: {gamma} (dev macro-F1 {best_f1:.4f})")

    client = LLMClient(model="openai/gpt-oss-20b", cache_dir=ROOT / ".llm_cache",
                       log_path=ROOT / "results" / "_llm_calls.jsonl")

    preds_a, preds_b, preds_c = [], [], []
    for i, r in enumerate(records):
        preds_a.append(baseline_recall_threshold(r["R"], tau_r=tau).value)
        preds_b.append(baseline_metric_rule(r["R"], r["retrieved_text"],
                                            r["retrieved_context"], tau_r=tau,
                                            gamma=gamma).value)
        label, _meta = baseline_llm_diagnoser(r["question"], r["retrieved_context"],
                                              r["retrieved_text"], r["answers"][0], client)
        preds_c.append(label.value)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(records)}] diagnosed")

    ours = [r["pred_label_frozen"] for r in records]
    systems = {"rule_v1": ours, "recall_threshold": preds_a,
               "metric_rule": preds_b, "llm_diagnoser": preds_c}
    report = {}
    for name, preds in systems.items():
        acc = sum(t == p for t, p in zip(truth, preds)) / len(truth)
        mean, lo, hi = bootstrap_ci([1.0 if t == p else 0.0 for t, p in zip(truth, preds)],
                                    n_boot=10000, seed=0)
        report[name] = {"accuracy": round(acc, 4),
                        "macro_f1": round(macro_f1(truth, preds), 4),
                        "bootstrap_95ci_acc": [round(lo, 4), round(hi, 4)]}

    # McNemar ours vs each baseline, Holm-corrected
    mcn = {}
    for name, preds in systems.items():
        if name == "rule_v1":
            continue
        b = sum(1 for t, o, p in zip(truth, ours, preds) if o == t and p != t)
        c = sum(1 for t, o, p in zip(truth, ours, preds) if o != t and p == t)
        mcn[name] = {"b_ours_only": b, "c_baseline_only": c,
                     "p_raw": round(mcnemar_exact(b, c), 6)}
    adjusted = holm_correction({k: v["p_raw"] for k, v in mcn.items()})
    for k in mcn:
        mcn[k]["p_holm"] = round(adjusted[k], 6)

    out = {"n": len(records), "tau_r": tau, "gamma": gamma,
           "systems": report, "mcnemar_vs_rule_v1": mcn}
    od = ROOT / "results" / "exp04"
    od.mkdir(parents=True, exist_ok=True)
    (od / "eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (od / "items.json").write_text(json.dumps(
        [{"query_id": r["query_id"], "truth": t,
          **{f"pred_{n}": p for n, p in systems.items()}}
         for r, t in zip(records, truth)], indent=2), encoding="utf-8")
    (od / "manifest.json").write_text(json.dumps({
        "exp_id": "exp04", "n": len(records), "tau_r": tau, "gamma": gamma,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
