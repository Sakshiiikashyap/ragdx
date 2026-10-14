"""Gold-reference degradation for the P5 robustness study (intent P5).

Levels (LEVELS order = figure x-axis):
  clean        original gold chunk (control)
  drop25/50/75 remove floor(frac*n) words at an even stride — position-uniform
               and deterministic, so answer position alone does not decide
               survival (a plain truncation would)
  near_miss    TF-IDF-closest pool chunk to the gold that contains NO answer:
               a plausible-looking but uninformative substitute (wrong-but-
               relevant reference). None if no such chunk exists -> item is
               skipped for that level and the skip is reported.
  pseudo_gold  first pool chunk other than the gold that DOES contain an
               answer — the best-case answer-aware surrogate (SEARA-style
               pseudo-gold; ours is oracle-aware, stated honestly in the
               paper). None if none exists -> skip.

resolve_gold(task, level) -> str | None. All pure functions: no LLM calls.
"""

from __future__ import annotations

from ragdx.faults import FaultedTask
from ragdx.retrieval import TfidfRetriever

LEVELS = ["clean", "drop25", "drop50", "drop75", "near_miss", "pseudo_gold"]
DROP_FRACS = {"drop25": 0.25, "drop50": 0.50, "drop75": 0.75}


def drop_content(text: str, frac: float) -> str:
    """Remove floor(frac*len(words)) words at evenly spaced positions.

    frac=0 returns the input byte-identical; frac=1 returns "".
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"frac must be in [0,1], got {frac}")
    if frac == 0.0:
        return text
    words = text.split()
    n, m = len(words), int(len(words) * frac)
    if m <= 0:
        return text
    if m >= n:
        return ""
    # mid-point sampling of m positions over n words -> regular stride
    drop = {(2 * j + 1) * n // (2 * m) for j in range(m)}
    assert len(drop) == m  # proven: stride n/m >= 1 when m <= n
    return " ".join(w for i, w in enumerate(words) if i not in drop)


def _contains_answer(text: str, answers: list[str]) -> bool:
    t = text.lower()
    return any(a.strip() and a.strip().lower() in t for a in answers)


def _pool_rows(task: FaultedTask) -> list[tuple[int, str]]:
    """(pool row index, chunk text) excluding the gold row (D011: identified
    by original item index through pool_orig_index, not by text equality)."""
    _, chunks, orig = task.effective_pool()
    gold_orig = task.item.gold_index
    return [(i, ch) for i, ch in enumerate(chunks) if orig[i] != gold_orig]


def near_miss_chunk(task: FaultedTask) -> str | None:
    """Closest non-gold pool chunk to the gold by TF-IDF cosine that carries
    no answer; None when no answer-free candidate exists."""
    gold = task.effective_oracle()
    cand = [(i, ch) for i, ch in _pool_rows(task)
            if not _contains_answer(ch, task.item.answers)
            and ch != gold]
    if not cand:
        return None
    chunks = [ch for _, ch in cand]
    ranked = TfidfRetriever().retrieve(gold, chunks, top_k=len(chunks))
    return chunks[ranked[0].index] if ranked else None


def pseudo_gold_chunk(task: FaultedTask) -> str | None:
    """First non-gold pool chunk containing an answer (pool order, the gold
    excluded by original index); None when none exists."""
    for _, ch in _pool_rows(task):
        if _contains_answer(ch, task.item.answers):
            return ch
    return None


def resolve_gold(task: FaultedTask, level: str) -> str | None:
    """Gold-context content for a degradation level; None = skip (no valid
    substitute exists for this item)."""
    if level == "clean":
        return task.effective_oracle()
    if level in DROP_FRACS:
        return drop_content(task.effective_oracle(), DROP_FRACS[level])
    if level == "near_miss":
        return near_miss_chunk(task)
    if level == "pseudo_gold":
        return pseudo_gold_chunk(task)
    raise ValueError(f"unknown degradation level {level!r} (choose from {LEVELS})")
