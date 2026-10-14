"""MIRAGE dataset loader (nlpai-lab/mirage on HuggingFace).

Schema (verified against the live dataset 2026-10-14):
  query_id, query, answer (list[str]), doc_name, source,
  oracle: {mapped_id, doc_name, doc_chunk, support},
  doc_pool: {mapped_id: list, doc_name: list, doc_chunk: list, support: list}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MirageItem:
    query_id: str
    query: str
    answers: list[str]
    oracle_chunk: str
    oracle_id: Any
    oracle_support: int
    chunks: list[str]
    chunk_ids: list[Any]
    chunk_support: list[int] = field(default_factory=list)
    gold_index: int = -1   # position of the gold chunk TEXT within chunks

    @property
    def gold_ids(self) -> list[Any]:
        """Doc-level gold (MIRAGE upstream semantics)."""
        return [self.oracle_id]

    @property
    def chunk_level_ids(self) -> list[str]:
        """Synthetic per-chunk ids for chunk-level recall (see D011)."""
        return [f"{self.query_id}::c{i}" for i in range(len(self.chunks))]

    @property
    def gold_chunk_ids(self) -> list[str]:
        if 0 <= self.gold_index < len(self.chunks):
            return [f"{self.query_id}::c{self.gold_index}"]
        return []


def load_mirage(split: str = "train", n: int | None = None, offset: int = 0) -> list[MirageItem]:
    from datasets import load_dataset
    ds = load_dataset("nlpai-lab/mirage", split=split)
    if n is not None:
        ds = ds.select(range(offset, min(offset + n, len(ds))))
    items = []
    for row in ds:
        oracle = row["oracle"]
        pool = row["doc_pool"]
        answers = row["answer"]
        if isinstance(answers, str):
            import ast
            answers = ast.literal_eval(answers)
        chunks = list(pool["doc_chunk"])
        # D011: gold position = text match; fallback = MIRAGE support flag
        gold_index = -1
        if oracle["doc_chunk"] in chunks:
            gold_index = chunks.index(oracle["doc_chunk"])
        elif pool.get("support"):
            support = list(pool["support"])
            if 1 in support:
                gold_index = support.index(1)
        items.append(MirageItem(
            query_id=str(row["query_id"]),
            query=row["query"],
            answers=list(answers),
            oracle_chunk=oracle["doc_chunk"],
            oracle_id=oracle["mapped_id"],
            oracle_support=oracle["support"],
            chunks=chunks,
            chunk_ids=list(pool["mapped_id"]),
            chunk_support=list(pool["support"]) if pool.get("support") else [],
            gold_index=gold_index,
        ))
    return items
