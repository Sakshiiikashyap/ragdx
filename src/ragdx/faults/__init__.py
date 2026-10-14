"""Fault injection framework (O7, docs/03 catalogue).

Each injector is a pure, deterministic function of (item, seed) that returns
a FaultedTask describing how the three-condition run must be modified. The
ground-truth failure source label travels with the task; the injector never
calls an LLM (the faultbench runner executes the calls).

Ground-truth semantics follow docs/03: retriever faults corrupt the evidence
supply; generator faults corrupt how context is used; the interaction fault
plants a contradictory distractor alongside gold.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ragdx.datasets.mirage import MirageItem


@dataclass
class FaultedTask:
    fault: str
    ground_truth: str            # RETRIEVER | GENERATOR | INTERACTION | NONE(control)
    item: MirageItem
    pool_ids: list | None = None       # override doc_pool ids (default: item)
    pool_chunks: list[str] | None = None
    pool_orig_index: list[int] | None = None  # original item.chunks index per pool row (D011)
    oracle_chunk: str | None = None    # override gold context for condition c1
    retrieved_context_mode: str = "top1"   # top1 | worst1 | gold_buried | gold_plus_distractor
    distractor_chunk: str | None = None    # used by gold_plus_distractor
    prompt_style: str = "normal"           # normal | ignore_context | adversarial
    model_override: str | None = None
    meta: dict = field(default_factory=dict)

    def effective_pool(self) -> tuple[list, list[str], list[int]]:
        """Returns (doc_ids, chunks, orig_index) with D011 chunk tracking."""
        ids = self.pool_ids if self.pool_ids is not None else self.item.chunk_ids
        chunks = self.pool_chunks if self.pool_chunks is not None else self.item.chunks
        orig = self.pool_orig_index if self.pool_orig_index is not None \
            else list(range(len(chunks)))
        return ids, chunks, orig

    def effective_oracle(self) -> str:
        return self.oracle_chunk if self.oracle_chunk is not None else self.item.oracle_chunk


def control(item: MirageItem) -> FaultedTask:
    """No fault: expected label NONE or genuine unforced failure."""
    return FaultedTask(fault="control", ground_truth="NONE", item=item)
