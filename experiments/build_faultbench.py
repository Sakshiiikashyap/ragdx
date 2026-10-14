"""Build RAG-FaultBench: stratified dev/test tasks over a MIRAGE subset.

Design (docs/03 + D007): n_questions questions, each contributes ONE fault
variant, round-robin across the injector catalogue so classes are balanced.
Splits are stratified 50/50 dev/test by fault type with a fixed seed.
The dev split is for threshold tuning ONLY; test is touched once (P4).

Writes data/faultbench/{dev,test}.jsonl (task descriptors, no API calls)
+ data/faultbench/README.md (dataset card) + manifest with class balance.
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdx.datasets.mirage import load_mirage
from ragdx.faults import FaultedTask, control
from ragdx.faults.generator_faults import INJECTORS as GEN_INJECTORS
from ragdx.faults.interaction_fault import INJECTORS as INT_INJECTORS
from ragdx.faults.retriever_faults import INJECTORS as RET_INJECTORS

ROOT = Path(__file__).resolve().parents[1]

CATALOG = {**RET_INJECTORS, **GEN_INJECTORS, **INT_INJECTORS}


def task_to_dict(task: FaultedTask) -> dict:
    return {"fault": task.fault, "ground_truth": task.ground_truth,
            "query_id": task.item.query_id, "prompt_style": task.prompt_style,
            "model_override": task.model_override,
            "retrieved_context_mode": task.retrieved_context_mode,
            "oracle_chunk_override": task.oracle_chunk,
            "pool_ids_override": task.pool_ids,
            "pool_chunks_override": task.pool_chunks,
            "distractor_chunk": task.distractor_chunk, "meta": task.meta}


def build(n_questions: int = 120, seed: int = 42,
          dev_frac: float = 0.5) -> tuple[list[dict], list[dict]]:
    items = load_mirage(n=n_questions)
    rng = random.Random(seed)
    names = list(CATALOG.keys())
    tasks: list[FaultedTask] = []
    for i, item in enumerate(items):
        name = names[i % len(names)]
        task = CATALOG[name](item, seed=seed)
        if task.ground_truth == "NONE" and "fallback" in task.fault:
            task = control(item)  # keep class balance honest
        tasks.append(task)
    rng.shuffle(tasks)
    n_dev = int(len(tasks) * dev_frac)
    # stratify: alternate per fault class into dev/test
    by_class: dict[str, list[dict]] = {}
    for t in tasks:
        by_class.setdefault(t.ground_truth, []).append(task_to_dict(t))
    dev, test = [], []
    for cls, rows in by_class.items():
        rng.shuffle(rows)
        k = int(len(rows) * dev_frac)
        dev.extend(rows[:k])
        test.extend(rows[k:])
    rng.shuffle(dev)
    rng.shuffle(test)
    return dev, test


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    dev, test = build(n_questions=n)
    out = ROOT / "data" / "faultbench"
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in (("dev", dev), ("test", test)):
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    balance = {"dev_truth": dict(Counter(r["ground_truth"] for r in dev)),
               "test_truth": dict(Counter(r["ground_truth"] for r in test)),
               "dev_fault": dict(Counter(r["fault"] for r in dev)),
               "test_fault": dict(Counter(r["fault"] for r in test))}
    (out / "manifest.json").write_text(json.dumps({
        "n_dev": len(dev), "n_test": len(test), "seed": 42,
        "dataset": "nlpai-lab/mirage", "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "balance": balance}, indent=2), encoding="utf-8")
    card = f"""# RAG-FaultBench (v0, pilot)

Fault-injected RAG failure benchmark with KNOWN failure-source labels.

- Source questions: `nlpai-lab/mirage` (first {len(dev)+len(test)} train rows), HF datasets.
- Faults: retriever (remove_gold, weak_topk, truncate_gold), generator
  (ignore_context, adversarial_prompt, weak_model, lost_in_middle),
  interaction (contradiction), plus NONE controls.
- Ground truth = the injected fault's component; natural (uninjected) failures
  are labelled NONE-control and may contain unforced errors (documented caveat).
- Splits: dev (threshold tuning) / test (single use), stratified by truth label,
  fixed seed 42. Class balance in manifest.json.
- Known limitation: injected faults may be easier than natural faults
  (docs/03 caveat); validate on human-annotated natural failures later.
"""
    (out / "README.md").write_text(card, encoding="utf-8")
    print(json.dumps(balance, indent=2))
    print(f"dev={len(dev)} test={len(test)} -> {out}")


if __name__ == "__main__":
    main()
