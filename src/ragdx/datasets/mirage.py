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

    @property
    def gold_ids(self) -> list[Any]:
        return [self.oracle_id]


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
        items.append(MirageItem(
            query_id=str(row["query_id"]),
            query=row["query"],
            answers=list(answers),
            oracle_chunk=oracle["doc_chunk"],
            oracle_id=oracle["mapped_id"],
            oracle_support=oracle["support"],
            chunks=list(pool["doc_chunk"]),
            chunk_ids=list(pool["mapped_id"]),
            chunk_support=list(pool["support"]) if pool.get("support") else [],
        ))
    return items
