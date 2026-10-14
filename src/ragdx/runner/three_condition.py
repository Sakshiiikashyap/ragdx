"""Three-condition test runner (Objective 1 — ported from MIRAGE-main/run_10.py).

For each question, generate answers under:
  base      (c0): question only
  gold      (c1): question + oracle chunk
  retrieved (c2): question + top-k retrieved chunks
Then score with MIRAGE EM-loose, compute retrieval metrics, run attribution
rule v1, and write per-item JSON + manifest.json into results/<exp_id>/.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ragdx.attribution.rule import Label, attribute, AttributionThresholds
from ragdx.datasets.mirage import MirageItem
from ragdx.llm.client import LLMClient
from ragdx.metrics import generation as gen
from ragdx.metrics import mirage as mg
from ragdx.metrics import retrieval as ret
from ragdx.retrieval import get_retriever

ANSWER_PROMPT = """Answer the question strictly using the provided context.

Do not use outside knowledge.
If the answer is stated in the context, use that information.
Give only the answer, without explanation.

Question:
{question}

Context:
{context}

Answer:"""

BASE_PROMPT = """Answer the question.

Question:
{question}

Give only the answer, without explanation."""


@dataclass
class RunConfig:
    exp_id: str
    model: str = "openai/gpt-oss-20b"
    retriever: str = "tfidf"
    top_k: int = 1
    n_questions: int = 10
    offset: int = 0
    tau_r: float = 0.5
    seed: int = 0
    extra: dict = field(default_factory=dict)


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "no-git"


def run_three_condition(cfg: RunConfig, items: list[MirageItem],
                        client: LLMClient, out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    retriever = get_retriever(cfg.retriever)
    thresholds = AttributionThresholds(tau_r=cfg.tau_r)
    t_start = time.time()

    records = []
    for i, item in enumerate(items):
        retrieved = retriever.retrieve(item.query, item.chunks, top_k=cfg.top_k)
        retrieved_ids = [item.chunk_ids[r.index] for r in retrieved]
        # D011: chunk-level ids for attribution recall (doc-level ids are
        # degenerate in MIRAGE pools — all chunks share one mapped_id)
        retrieved_chunk_ids = [f"{item.query_id}::c{r.index}" for r in retrieved]
        retrieved_context = "\n".join(f"{j+1}. {item.chunks[r.index]}"
                                      for j, r in enumerate(retrieved))

        base_resp = client.chat([{"role": "user",
                                  "content": BASE_PROMPT.format(question=item.query)}])
        gold_resp = client.chat([{"role": "user",
                                  "content": ANSWER_PROMPT.format(question=item.query,
                                                                  context=item.oracle_chunk)}])
        ret_resp = client.chat([{"role": "user",
                                 "content": ANSWER_PROMPT.format(question=item.query,
                                                                 context=retrieved_context)}])

        c0 = int(gen.em_loose(item.answers, base_resp.text) >= 1.0)
        c1 = int(gen.em_loose(item.answers, gold_resp.text) >= 1.0)
        c2 = int(gen.em_loose(item.answers, ret_resp.text) >= 1.0)

        r_at_k = ret.recall_at_k(retrieved_chunk_ids, item.gold_chunk_ids, cfg.top_k)
        attr = attribute(c0, c1, c2, r=r_at_k, thresholds=thresholds)

        records.append({
            "query_id": item.query_id, "question": item.query,
            "answers": item.answers,
            "base": {"text": base_resp.text, "em_loose": c0,
                     "em_strict": gen.em_strict(item.answers, base_resp.text),
                     "token_f1": gen.token_f1(item.answers, base_resp.text),
                     "tokens": base_resp.prompt_tokens + base_resp.completion_tokens,
                     "cached": base_resp.cached},
            "gold": {"text": gold_resp.text, "em_loose": c1,
                     "em_strict": gen.em_strict(item.answers, gold_resp.text),
                     "token_f1": gen.token_f1(item.answers, gold_resp.text),
                     "tokens": gold_resp.prompt_tokens + gold_resp.completion_tokens,
                     "cached": gold_resp.cached},
            "retrieved": {"text": ret_resp.text, "em_loose": c2,
                          "em_strict": gen.em_strict(item.answers, ret_resp.text),
                          "token_f1": gen.token_f1(item.answers, ret_resp.text),
                          "tokens": ret_resp.prompt_tokens + ret_resp.completion_tokens,
                          "cached": ret_resp.cached,
                          "ids": retrieved_ids, "scores": [r.score for r in retrieved],
                          "chunk_ids": retrieved_chunk_ids,
                          "gold_chunk_ids": item.gold_chunk_ids,
                          "gold_index": item.gold_index,
                          "recall_at_k": r_at_k},
            "label": attr.label.value,
            "evidence": attr.evidence,
        })
        print(f"[{i+1}/{len(items)}] c0={c0} c1={c1} c2={c2} "
              f"R={r_at_k:.2f} -> {attr.label.value}")

    base_labels = [r["base"]["em_loose"] for r in records]
    gold_labels = [r["gold"]["em_loose"] for r in records]
    ret_labels = [r["retrieved"]["em_loose"] for r in records]
    mm = mg.mirage_metrics(base_labels, ret_labels, gold_labels)

    from collections import Counter
    label_dist = Counter(r["label"] for r in records)
    summary = {
        "n": len(records),
        "accuracy": {"base": sum(base_labels) / len(records) if records else 0,
                     "gold": sum(gold_labels) / len(records) if records else 0,
                     "retrieved": sum(ret_labels) / len(records) if records else 0},
        "mirage_metrics": {"nv": mm.nv, "ca": mm.ca, "ci": mm.ci, "cm": mm.cm,
                           "ranking_score": mm.ranking_score,
                           "pattern_counts": mm.counts, "unmapped": mm.unmapped},
        "attribution_distribution": dict(label_dist),
        "mean_recall_at_k": (sum(r["retrieved"]["recall_at_k"] for r in records)
                             / len(records) if records else 0),
    }

    manifest = {
        "exp_id": cfg.exp_id,
        "git_hash": _git_hash(),
        "config": asdict(cfg),
        "config_hash": hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()[:16],
        "model": cfg.model,
        "dataset": "nlpai-lab/mirage",
        "n_items": len(records),
        "seed": cfg.seed,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_s": round(time.time() - t_start, 1),
        "total_tokens": sum(r[c]["tokens"] for r in records for c in ("base", "gold", "retrieved")),
        "cache_hits": sum(int(r[c]["cached"]) for r in records for c in ("base", "gold", "retrieved")),
    }

    (out / "items.json").write_text(json.dumps(records, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    return {"summary": summary, "manifest": manifest, "records": records}
