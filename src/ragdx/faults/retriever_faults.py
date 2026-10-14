"""Retriever-side fault injectors (docs/03 catalogue).

- remove_gold:  gold chunk excluded from the candidate pool; retrieval must
  land on a distractor. Expected label: RETRIEVER.
- weak_topk:    pool intact, but the context passed to the generator is the
  worst TF-IDF-ranked chunk (simulates a weak retriever/reranker). Expected: RETRIEVER.
- truncate_gold: gold chunk in the pool is cut to its first `ratio` of
  characters, destroying the answer-bearing span while keeping the id
  retrievable. Expected: RETRIEVER (evidence no longer sufficient).

Deterministic given the item; `seed` reserved for future stochastic variants.
"""

from __future__ import annotations

from ragdx.attribution.rule import Label
from ragdx.datasets.mirage import MirageItem
from ragdx.faults import FaultedTask
from ragdx.metrics.retrieval import recall_at_k
from ragdx.retrieval import TfidfRetriever


def remove_gold(item: MirageItem, seed: int = 0) -> FaultedTask:
    # D011: gold identified by chunk TEXT position, not doc-level mapped_id
    if item.gold_index < 0 or len(item.chunks) < 2:
        return FaultedTask(fault="remove_gold_control_fallback", ground_truth="NONE",
                           item=item, meta={"reason": "no gold chunk in pool or pool too small"})
    ids = [cid for i, cid in enumerate(item.chunk_ids) if i != item.gold_index]
    chunks = [ch for i, ch in enumerate(item.chunks) if i != item.gold_index]
    orig = [i for i in range(len(item.chunks)) if i != item.gold_index]
    return FaultedTask(fault="remove_gold", ground_truth=Label.RETRIEVER.value,
                       item=item, pool_ids=list(ids), pool_chunks=list(chunks),
                       pool_orig_index=orig,
                       meta={"gold_in_pool": False, "removed_index": item.gold_index})


def weak_topk(item: MirageItem, seed: int = 0) -> FaultedTask:
    if len(item.chunks) < 2:
        return FaultedTask(fault="weak_topk_control_fallback", ground_truth="NONE",
                           item=item, meta={"reason": "pool too small"})
    ret = TfidfRetriever()
    ranked = ret.retrieve(item.query, item.chunks, top_k=len(item.chunks))
    worst = ranked[-1].index  # lowest similarity
    return FaultedTask(fault="weak_topk", ground_truth=Label.RETRIEVER.value,
                       item=item, retrieved_context_mode="index",
                       meta={"forced_index": worst, "gold_in_pool": True,
                             "gold_recall_still": recall_at_k([item.oracle_id], item.gold_ids, 1)})


def truncate_gold(item: MirageItem, seed: int = 0, ratio: float = 0.3) -> FaultedTask:
    if item.gold_index < 0:
        return FaultedTask(fault="truncate_gold_control_fallback", ground_truth="NONE",
                           item=item, meta={"reason": "no gold chunk in pool"})
    cut = max(1, int(len(item.oracle_chunk) * ratio))
    truncated = item.oracle_chunk[:cut]
    chunks = [truncated if i == item.gold_index else ch
              for i, ch in enumerate(item.chunks)]
    return FaultedTask(fault="truncate_gold", ground_truth=Label.RETRIEVER.value,
                       item=item, pool_chunks=chunks,
                       meta={"ratio": ratio, "truncated_len": cut,
                             "original_len": len(item.oracle_chunk)})


INJECTORS = {"remove_gold": remove_gold, "weak_topk": weak_topk,
             "truncate_gold": truncate_gold}
