"""exp06: P5 robustness study on FaultBench TEST tasks (nothing tuned here —
frozen tau only; results are measurements, not thresholds).

(a) Gold-reference degradation: re-run the three conditions with the gold
    context swapped per level (clean / drop25 / drop50 / drop75 / near_miss /
    pseudo_gold — see ragdx.faults.degrade). Rule v1 @ frozen tau vs the best
    exp04 baseline, whose predictions are frozen and therefore gold-independent.
    Base/retrieved calls at every level are byte-identical to the exp03b run
    -> cache hits; only the swapped gold condition bills fresh API tokens.
(b) Judge-noise flip sweep over clean records (offline, zero API).

Requires: data/faultbench/test.jsonl, configs/frozen_thresholds.json,
results/exp04/{eval,items}.json (frozen baseline; degrades without it if missing).
Outputs: results/exp06/{items.json, eval.json, manifest.json}

Usage: python experiments/exp06_robustness.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdx.analysis.robustness import build_eval
from ragdx.attribution.rule import AttributionThresholds
from ragdx.faults.degrade import LEVELS, resolve_gold
from ragdx.faults.loader import load_rows, rows_to_tasks
from ragdx.faults.runner import run_task
from ragdx.llm.client import LLMClient

ROOT = Path(__file__).resolve().parents[1]


def _git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def load_frozen_baseline() -> tuple[str | None, dict[str, str] | None]:
    """Best non-rule system from exp04, frozen (predictions keyed by query_id)."""
    ev4, it4 = ROOT / "results" / "exp04" / "eval.json", \
        ROOT / "results" / "exp04" / "items.json"
    if not (ev4.exists() and it4.exists()):
        print("WARNING: exp04 not found - running without baseline comparison")
        return None, None
    systems = {k: v for k, v in json.loads(ev4.read_text(encoding="utf-8"))["systems"].items()
               if k != "rule_v1"}
    best = max(systems, key=lambda k: (systems[k]["macro_f1"], k))
    # exp04 items are one row per record in records order; accept both the
    # scalar format and the legacy bug where every pred_* held the full
    # per-system vector (fixed 2026-10-17, see D023 - vector[i] = row i's pred)
    preds: dict[str, str] = {}
    for i, it in enumerate(json.loads(it4.read_text(encoding="utf-8"))):
        p = it[f"pred_{best}"]
        preds[it["query_id"]] = p[i] if isinstance(p, list) else p
    print(f"best baseline from exp04: {best} "
          f"(macro-F1 {systems[best]['macro_f1']:.3f})")
    return best, preds


def main() -> None:
    frozen = json.loads((ROOT / "configs" / "frozen_thresholds.json").read_text(encoding="utf-8"))
    tau = frozen["tau_r"]
    tasks = rows_to_tasks(load_rows(ROOT / "data" / "faultbench" / "test.jsonl"))
    print(f"exp06: {len(tasks)} test tasks x {len(LEVELS)} levels, frozen tau_r={tau}")

    client = LLMClient(model="openai/gpt-oss-20b", cache_dir=ROOT / ".llm_cache",
                       log_path=ROOT / "results" / "_llm_calls.jsonl")
    th = AttributionThresholds(tau_r=tau)
    best_name, best_preds = load_frozen_baseline()

    records: list[dict] = []
    skips = {lv: 0 for lv in LEVELS}
    t0 = time.time()
    for i, task in enumerate(tasks):
        for lv in LEVELS:
            gold = resolve_gold(task, lv)
            if gold is None:
                skips[lv] += 1
                continue
            t2 = task if lv == "clean" else replace(task, oracle_chunk=gold)
            rec = run_task(t2, client, thresholds=th, top_k=1)
            rec["level"] = lv
            records.append(rec)
            ok = "OK  " if rec["pred_label"] == rec["ground_truth"] else "MISS"
            print(f"[{i+1}/{len(tasks)}] {lv:10s} {ok} {task.fault:16s} "
                  f"truth={rec['ground_truth']:11s} pred={rec['pred_label']:11s} "
                  f"(c={rec['c0']}{rec['c1']}{rec['c2']} R={rec['R']:.2f})")
    if any(skips.values()):
        print(f"skips (no valid substitute): {skips}")

    eval_doc = build_eval(records, LEVELS, tau_r=tau, n_tasks=len(tasks),
                          skips=skips, baseline_name=best_name,
                          baseline_preds=best_preds)
    od = ROOT / "results" / "exp06"
    od.mkdir(parents=True, exist_ok=True)
    (od / "items.json").write_text(json.dumps(records, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    (od / "eval.json").write_text(json.dumps(eval_doc, indent=2), encoding="utf-8")
    manifest = {"exp_id": "exp06", "n_tasks": len(tasks),
                "n_records": len(records), "levels": LEVELS, "skips": skips,
                "tau_r": tau, "model": client.model, "git_hash": _git_hash(),
                "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "duration_s": round(time.time() - t0, 1),
                "total_tokens": sum(r["tokens"] for r in records)}
    (od / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nrule v1 accuracy by level:")
    for lv, rep in eval_doc["by_system"]["rule_v1"].items():
        print(f"  {lv:10s} acc={rep['accuracy']:.3f} macroF1={rep['macro_f1']:.3f} "
              f"n={rep['n']} c1_rate={eval_doc['c1_rate'][lv]:.3f}")
    if best_name:
        print(f"vs {best_name}:",
              {lv: rep["macro_f1"]
               for lv, rep in eval_doc["by_system"][best_name].items()})
    noise = [(j["flip_p"], round(j["mean_accuracy"], 3))
             for j in eval_doc["judge_noise"]]
    print(f"judge noise: {noise}")
    print(f"wrote {od / 'eval.json'}")


if __name__ == "__main__":
    main()
