"""FaultBench split loading: JSONL rows -> FaultedTask objects.

The builder persisted task descriptors (no live dataset objects); here we
re-attach the MIRAGE items by query_id and recompute pool_orig_index (D011)
for modified pools by text-matching rows against the original chunks.
"""

from __future__ import annotations

import json
from pathlib import Path

from ragdx.datasets.mirage import load_mirage
from ragdx.faults import FaultedTask


def load_rows(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rows_to_tasks(rows: list[dict]) -> list[FaultedTask]:
    qids = sorted({r["query_id"] for r in rows})
    items = {it.query_id: it for it in load_mirage(n=None) if it.query_id in set(qids)}
    tasks = []
    for r in rows:
        item = items[r["query_id"]]
        orig = r.get("pool_orig_index")
        if orig is None and r.get("pool_chunks_override") is not None:
            # legacy rows without persisted indices: order-preserving text match
            pool = r["pool_chunks_override"]
            orig = []
            cursor = 0
            for ch in pool:
                while cursor < len(item.chunks) and item.chunks[cursor] != ch:
                    cursor += 1
                orig.append(cursor)
                cursor += 1
        tasks.append(FaultedTask(
            fault=r["fault"], ground_truth=r["ground_truth"], item=item,
            pool_ids=r.get("pool_ids_override"),
            pool_chunks=r.get("pool_chunks_override"),
            pool_orig_index=orig,
            oracle_chunk=r.get("oracle_chunk_override"),
            retrieved_context_mode=r.get("retrieved_context_mode", "top1"),
            distractor_chunk=r.get("distractor_chunk"),
            prompt_style=r.get("prompt_style", "normal"),
            model_override=r.get("model_override"),
            meta=r.get("meta", {})))
    return tasks
