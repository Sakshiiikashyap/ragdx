"""exp07: P6 multi-model generalization — rerun the FaultBench TEST protocol
with two additional Groq model families under the SAME frozen tau
(pre-registered threshold transfer; no per-model tuning is performed).

Model availability is checked against the live Groq /models list at runtime;
an unavailable model is skipped and reported, never silently replaced.

Honest framing input: weak_model-fault tasks override the generator with
WEAK_MODEL by fault definition, so those tasks are identical across exp07
models (and cache-shared); per-model differences come from the remaining
fault classes.

Requires: data/faultbench/test.jsonl, configs/frozen_thresholds.json,
results/exp03/{items,eval}.json (main model).
Outputs: results/exp07/{<slug>/items.json..., eval.json, manifest.json}
Usage: python experiments/exp07_multimodel.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdx.analysis.stats import cohen_kappa, macro_f1
from ragdx.faults.loader import load_rows, rows_to_tasks
from ragdx.faults.runner import run_tasks
from ragdx.llm.client import LLMClient

ROOT = Path(__file__).resolve().parents[1]
MAIN_MODEL = "openai/gpt-oss-20b"
EXTRA_MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]


def _git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def _slug(model: str) -> str:
    return model.replace("/", "_").replace("-", "_")


def live_models(client: LLMClient) -> set[str]:
    try:
        return {m.id for m in client.client.models.list().data}
    except Exception as e:  # availability check must not kill the run
        print(f"WARNING: /models check failed ({e}) - skipping availability gate")
        return set()


def main() -> None:
    frozen = json.loads((ROOT / "configs" / "frozen_thresholds.json").read_text(encoding="utf-8"))
    tau = frozen["tau_r"]
    tasks = rows_to_tasks(load_rows(ROOT / "data" / "faultbench" / "test.jsonl"))
    print(f"exp07: {len(tasks)} test tasks, frozen tau_r={tau}, "
          f"extra models {EXTRA_MODELS}")

    main_items = json.loads((ROOT / "results" / "exp03" / "items.json").read_text(encoding="utf-8"))
    main_eval = json.loads((ROOT / "results" / "exp03" / "eval.json").read_text(encoding="utf-8"))
    main_preds = {r["query_id"]: r["pred_label_frozen"] for r in main_items}

    models: dict[str, dict] = {MAIN_MODEL: {
        "accuracy": main_eval["accuracy"], "macro_f1": main_eval["macro_f1"],
        "n": main_eval["n"], "role": "main"}}
    stability: dict[str, dict] = {}
    skipped: dict[str, str] = {}

    client0 = LLMClient(model=MAIN_MODEL, cache_dir=ROOT / ".llm_cache",
                        log_path=ROOT / "results" / "_llm_calls.jsonl")
    available = live_models(client0)
    print(f"live Groq models: {sorted(available) if available else 'unknown'}")

    t0 = time.time()
    for model in EXTRA_MODELS:
        if available and model not in available:
            skipped[model] = "not in live /models list"
            print(f"SKIP {model}: {skipped[model]}")
            continue
        slug = _slug(model)
        client = LLMClient(model=model, cache_dir=ROOT / ".llm_cache",
                           log_path=ROOT / "results" / "_llm_calls.jsonl")
        run = run_tasks(tasks, client, ROOT / "results" / "exp07" / slug,
                        f"exp07_{slug}", tau_r=tau, top_k=1)
        recs = run["records"]
        preds = {r["query_id"]: r["pred_label"] for r in recs}
        truth = {r["query_id"]: r["ground_truth"] for r in recs}
        qids = sorted(preds)
        models[model] = {
            "accuracy": round(sum(preds[q] == truth[q] for q in qids) / len(qids), 4),
            "macro_f1": round(macro_f1([truth[q] for q in qids],
                                       [preds[q] for q in qids]), 4),
            "n": len(qids), "role": "extra"}
        shared = [q for q in qids if q in main_preds]
        agree = sum(main_preds[q] == preds[q] for q in shared) / len(shared)
        stability[model] = {
            "agreement_vs_main": round(agree, 4),
            "kappa_vs_main": round(cohen_kappa([main_preds[q] for q in shared],
                                               [preds[q] for q in shared]), 4)}
        print(f"{model}: macroF1={models[model]['macro_f1']:.3f} "
              f"acc={models[model]['accuracy']:.3f} "
              f"agreement_vs_main={stability[model]['agreement_vs_main']:.3f} "
              f"kappa={stability[model]['kappa_vs_main']:.3f}")

    eval_doc = {"n": len(tasks), "tau_r": tau, "models": models,
                "stability": stability, "skipped": skipped,
                "note": "weak_model-fault tasks use the fixed WEAK_MODEL "
                        "override by definition and are identical across "
                        "exp07 models (cache-shared)"}
    od = ROOT / "results" / "exp07"
    od.mkdir(parents=True, exist_ok=True)
    (od / "eval.json").write_text(json.dumps(eval_doc, indent=2), encoding="utf-8")
    (od / "manifest.json").write_text(json.dumps({
        "exp_id": "exp07", "main_model": MAIN_MODEL,
        "extra_models": EXTRA_MODELS, "skipped": skipped,
        "tau_r": tau, "git_hash": _git_hash(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_s": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    print(json.dumps(models, indent=2))
    print(f"wrote {od / 'eval.json'}")


if __name__ == "__main__":
    main()
