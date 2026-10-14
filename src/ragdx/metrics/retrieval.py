"""Retrieval metrics: recall@k, precision@k, MRR, nDCG@k.

Standard definitions (Craswell et al. for MRR/nDCG; Manning et al. for
recall/precision). Gold set = MIRAGE oracle mapped_id (single gold chunk).
"""

from __future__ import annotations


def recall_at_k(retrieved_ids: list, gold_ids: list, k: int) -> float:
    if not gold_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    return len(top & set(gold_ids)) / len(set(gold_ids))


def precision_at_k(retrieved_ids: list, gold_ids: list, k: int) -> float:
    if k == 0:
        return 0.0
    top = retrieved_ids[:k]
    return len(set(top) & set(gold_ids)) / k


def mrr(retrieved_ids: list, gold_ids: list) -> float:
    gold = set(gold_ids)
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list, gold_ids: list, k: int) -> float:
    gold = set(gold_ids)
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in gold:
            dcg += 1.0 / (i.bit_length())  # log2(i+1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / ((i).bit_length()) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def gold_recall_flag(retrieved_ids: list, gold_ids: list, k: int, tau_r: float) -> bool:
    """R >= tau_R test used by the attribution rule (docs/02)."""
    return recall_at_k(retrieved_ids, gold_ids, k) >= tau_r
