"""exp03c: D020 secondary analysis - failure-subset attribution accuracy.

Pre-specified from DEV evidence (2026-10-16, committed BEFORE test eval
existed; see decisions.md D020). Reads stored item records only - no API
calls, no tuning, no re-generation.

Reports, for dev and test:
  activation_rate   faulted items whose retrieved-condition answer is wrong
                    (c2==0); benign injections (c2==1) did not produce a
                    failure to attribute
  all_items         primary view: acc/macro-F1 vs injected-component truth
                    (echoes exp03b eval.json; kept for one-file comparison)
  failure_subset    secondary view: acc/macro-F1/per-class/confusion on
                    c2==0 items only (attribution's actual operating set;
                    NONE is unreachable there by rule construction)
  benign_subset     c2==1 items: label distribution (rule is expected to
                    emit NONE - "no failure to attribute")

Outputs: results/exp03/eval_subset.json (and results/exp03_dev/ for dev).
Usage: python experiments/exp03c_failure_subset.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdx.analysis.stats import confusion_matrix, macro_f1

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["RETRIEVER", "GENERATOR", "INTERACTION", "NONE"]


def analyze(items: list[dict], split: str) -> dict:
    faulted = [i for i in items if i.get("fault") not in (None, "control")]
    fail = [i for i in faulted if i["c2"] == 0]
    benign = [i for i in faulted if i["c2"] == 1]

    def acc(grp: list[dict]) -> float:
        return round(sum(i["ground_truth"] == i["pred_label"] for i in grp)
                     / len(grp), 4) if grp else 0.0

    truth = [i["ground_truth"] for i in faulted]
    pred = [i["pred_label"] for i in faulted]
    f_truth = [i["ground_truth"] for i in fail]
    f_pred = [i["pred_label"] for i in fail]

    per_fault_act = {}
    for f in sorted({i["fault"] for i in faulted}):
        grp = [i for i in faulted if i["fault"] == f]
        bad = [i for i in grp if i["c2"] == 0]
        per_fault_act[f] = {"n": len(grp), "activated": len(bad),
                            "activation_rate": round(len(bad) / len(grp), 4),
                            "failure_subset_acc": acc(bad)}

    return {
        "split": split,
        "n_faulted": len(faulted),
        "activation_rate": round(len(fail) / len(faulted), 4) if faulted else 0.0,
        "n_failure_subset": len(fail),
        "n_benign_subset": len(benign),
        "all_items": {
            "accuracy": acc(faulted),
            "macro_f1": round(macro_f1(truth, pred), 4),
            "pred_distribution": dict(Counter(pred)),
        },
        "failure_subset": {
            "accuracy": acc(fail),
            "macro_f1": round(macro_f1(f_truth, f_pred), 4) if fail else 0.0,
            "n": len(fail),
            "confusion": confusion_matrix(f_truth, f_pred, CLASSES)
            if fail else None,
        },
        "benign_subset": {
            "n": len(benign),
            "accuracy_vs_injected_label": acc(benign),
            "pred_distribution": dict(Counter(i["pred_label"] for i in benign)),
        },
        "per_fault": per_fault_act,
        "note": "D020 secondary analysis: pre-specified from dev 2026-10-16 "
                "and committed before test eval existed; analysis-only.",
    }


def main() -> None:
    for rel, split in (("results/exp03_dev/items.json", "dev"),
                       ("results/exp03/items.json", "test")):
        p = ROOT / rel
        if not p.exists():
            print(f"SKIP {rel} (not yet written)")
            continue
        doc = analyze(json.loads(p.read_text(encoding="utf-8")), split)
        out = p.parent / "eval_subset.json"
        out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"[{split}] activation={doc['activation_rate']:.3f} "
              f"all_items acc={doc['all_items']['accuracy']:.4f} "
              f"failure_subset acc={doc['failure_subset']['accuracy']:.4f} "
              f"(n={doc['failure_subset']['n']})")
        print(f"[{split}] benign preds={doc['benign_subset']['pred_distribution']}")
        print(f"[{split}] per_fault activation: "
              + ", ".join(f"{f}:{v['activation_rate']:.2f}"
                          for f, v in doc["per_fault"].items()))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
