"""Retrieval over a question's doc_pool (MIRAGE-style local pool).

v1 ships TF-IDF + cosine (ported from the prototype). The interface is
retriever-agnostic so dense/hybrid variants (exp07) can slot in later.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    index: int
    score: float


class TfidfRetriever:
    name = "tfidf"

    def retrieve(self, query: str, chunks: list[str], top_k: int = 1) -> list[RetrievedChunk]:
        if not chunks:
            return []
        vectorizer = TfidfVectorizer()
        chunk_vectors = vectorizer.fit_transform(chunks)
        query_vector = vectorizer.transform([query])
        sims = cosine_similarity(query_vector, chunk_vectors)[0]
        order = sims.argsort()[::-1][:top_k]
        return [RetrievedChunk(index=int(i), score=float(sims[i])) for i in order]


def get_retriever(name: str):
    if name == "tfidf":
        return TfidfRetriever()
    raise ValueError(f"unknown retriever: {name} (dense/hybrid planned for exp07)")
