"""exp08: P7 repair-utility on the FaultBench TEST failure set — records from
the frozen exp03b run where the retrieved condition failed (c2==0).

Repairs (one cached call each):
  topk3        deployable: TF-IDF top-3 over the effective pool (fixes weak
               top-1 retrieval)
  gold_rerank  oracle-assisted: same top-3 with the gold row forced first when
               the gold row is in the pool (labelled honestly in the paper)
  ground       deployable: stricter grounding system prompt over the CURRENT
               retrieved context (fixes context-ignoring generation)
Policies (utility.build_utility_eval, same failure set): our_fix (repair
selected by our frozen attribution label), random_fix (seeded), always_topk3,
always_grounding, gold_rerank-as-universal-policy.

Requires: results/exp03/items.json + data/faultbench/test.jsonl.
Outputs: results/exp08/{items.json, eval.json, manifest.json}
Usage: python experiments/exp08_utility.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdx.analysis.utility import build_utility_eval
from ragdx.faults.loader import load_rows, rows_to_tasks
from ragdx.faults.runner import ANSWER_PROMPT
from ragdx.llm.client import LLMClient
from ragdx.metrics import generation as gen
from ragdx.retrieval import TfidfRetriever

ROOT = Path(__file__).resolve().parents[1]

STRICT_GROUNDING = ("Answer ONLY from the provided context. Every claim must "
                    "be directly supported by the context text. If the "
                    "context does not contain the answer, reply exactly: "
                    "CANNOT ANSWER.")


def _git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def topk_context(task, retriever: TfidfRetriever, k: int = 3,
                 force_gold_first: bool = False) -> str:
    """Context for the repaired retriever: normal top-k over the effective
    pool; optionally with the gold row (D011 index) forced to position 1."""
    _, chunks, orig = task.effective_pool()
    ranked = retriever.retrieve(task.item.query, chunks,
                                top_k=min(k, len(chunks)))
    sel = [r.index for r in ranked]
    if force_gold_first and task.item.gold_index in orig:
        gpos = orig.index(task.item.gold_index)
        sel = [gpos] + [i for i in sel if i != gpos]
    return "\n".join(f"{j+1}. {chunks[i]}" for j, i in enumerate(sel))


def main() -> None:
    items = json.loads((ROOT / "results" / "exp03" / "items.json").read_text(encoding="utf-8"))
    failures = [r for r in items if r["c2"] == 0]
    tasks = {t.item.query_id: t
             for t in rows_to_tasks(load_rows(ROOT / "data" / "faultbench" / "test.jsonl"))}
    print(f"exp08: failure set = {len(failures)}/{len(items)} items (c2==0)")

    client = LLMClient(model="openai/gpt-oss-20b", cache_dir=ROOT / ".llm_cache",
                       log_path=ROOT / "results" / "_llm_calls.jsonl")
    retriever = TfidfRetriever()
    rows, tokens = [], 0
    t0 = time.time()
    for i, rec in enumerate(failures):
        task = tasks[rec["query_id"]]
        _, _, orig = task.effective_pool()
        gold_in_pool = task.item.gold_index in orig
        q = rec["question"]

        fixed: dict[str, int] = {}
        ctx3 = topk_context(task, retriever, k=3)
        r = client.chat([{"role": "user",
                          "content": ANSWER_PROMPT.format(question=q, context=ctx3)}])
        fixed["topk3"] = int(gen.em_loose(rec["answers"], r.text) >= 1.0)
        tokens += r.prompt_tokens + r.completion_tokens

        if gold_in_pool:
            ctxr = topk_context(task, retriever, k=3, force_gold_first=True)
            rr = client.chat([{"role": "user",
                               "content": ANSWER_PROMPT.format(question=q, context=ctxr)}])
            fixed["gold_rerank"] = int(gen.em_loose(rec["answers"], rr.text) >= 1.0)
            tokens += rr.prompt_tokens + rr.completion_tokens

        gr = client.chat([{"role": "system", "content": STRICT_GROUNDING},
                          {"role": "user",
                           "content": ANSWER_PROMPT.format(
                               question=q, context=rec["retrieved_context"])}])
        fixed["grounding"] = int(gen.em_loose(rec["answers"], gr.text) >= 1.0)
        tokens += gr.prompt_tokens + gr.completion_tokens

        rows.append({"query_id": rec["query_id"],
                     "pred_label": rec["pred_label_frozen"], "fixed": fixed})
        print(f"[{i+1}/{len(failures)}] {rec['query_id']:16s} "
              f"pred={rec['pred_label_frozen']:11s} fixed={fixed}")

    eval_doc = build_utility_eval(rows, seed=0)
    od = ROOT / "results" / "exp08"
    od.mkdir(parents=True, exist_ok=True)
    (od / "items.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (od / "eval.json").write_text(json.dumps(eval_doc, indent=2), encoding="utf-8")
    (od / "manifest.json").write_text(json.dumps({
        "exp_id": "exp08", "n_failure": len(rows),
        "model": client.model, "git_hash": _git_hash(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_s": round(time.time() - t0, 1),
        "total_tokens": tokens}, indent=2), encoding="utf-8")
    print(json.dumps(eval_doc["fixes"], indent=2))
    print(f"sign tests: {json.dumps(eval_doc['mcnemar_ours_vs'], indent=2)}")
    print(f"wrote {od / 'eval.json'}")


if __name__ == "__main__":
    main()
