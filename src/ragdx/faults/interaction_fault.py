"""Interaction fault: gold passage plus a high-similarity contradictory
distractor in the supplied context (docs/03). Expected: INTERACTION."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ragdx.attribution.rule import Label
from ragdx.datasets.mirage import MirageItem
from ragdx.faults import FaultedTask


def contradiction(item: MirageItem, seed: int = 0) -> FaultedTask:
    # D011: candidates = all chunk positions except the gold TEXT position
    candidates = [i for i in range(len(item.chunks)) if i != item.gold_index]
    if not candidates:
        return FaultedTask(fault="contradiction_control_fallback", ground_truth="NONE",
                           item=item, meta={"reason": "no non-gold candidate in pool"})
    vec = TfidfVectorizer()
    mats = vec.fit_transform([item.oracle_chunk] + item.chunks)
    sims = cosine_similarity(mats[0:1], mats[1:])[0]
    best = max(candidates, key=lambda i: sims[i])
    distractor = item.chunks[best]
    return FaultedTask(fault="contradiction", ground_truth=Label.INTERACTION.value,
                       item=item, retrieved_context_mode="gold_plus_distractor",
                       distractor_chunk=distractor,
                       meta={"distractor_index": best,
                             "distractor_similarity": round(float(sims[best]), 4)})


INJECTORS = {"contradiction": contradiction}
