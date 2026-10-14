"""Generator-side fault injectors (docs/03 catalogue).

- ignore_context: system prompt instructs the model to answer from memory
  even though gold context is supplied. Expected: GENERATOR.
- adversarial:    user-level injection claiming the context is fabricated;
  model should resist but often fails. Expected: GENERATOR.
- weak_model:     same prompts, smaller model substituted. Expected: GENERATOR.
- lost_in_middle: gold buried between filler chunks (context dilution).
  Expected: GENERATOR per docs/03 (context-handling, not evidence supply).

Retrieval inputs are untouched by all four.
"""

from __future__ import annotations

from ragdx.attribution.rule import Label
from ragdx.datasets.mirage import MirageItem
from ragdx.faults import FaultedTask

FILLER = ("The committee met on Tuesday to discuss procedural matters unrelated "
          "to the query at hand, and adjourned after approving the minutes. ") * 3

WEAK_MODEL = "allam-2-7b"  # smallest general model on Groq (verified 2026-10-14; llama-3.1-8b-instant retired)


def ignore_context(item: MirageItem, seed: int = 0) -> FaultedTask:
    return FaultedTask(fault="ignore_context", ground_truth=Label.GENERATOR.value,
                       item=item, prompt_style="ignore_context",
                       meta={"gold_supplied": True})


def adversarial(item: MirageItem, seed: int = 0) -> FaultedTask:
    return FaultedTask(fault="adversarial_prompt", ground_truth=Label.GENERATOR.value,
                       item=item, prompt_style="adversarial",
                       meta={"gold_supplied": True})


def weak_model(item: MirageItem, seed: int = 0) -> FaultedTask:
    return FaultedTask(fault="weak_model", ground_truth=Label.GENERATOR.value,
                       item=item, model_override=WEAK_MODEL,
                       meta={"model": WEAK_MODEL, "gold_supplied": True})


def lost_in_middle(item: MirageItem, seed: int = 0, fillers: int = 3) -> FaultedTask:
    buried = (FILLER * fillers) + "\n" + item.oracle_chunk + "\n" + (FILLER * fillers)
    return FaultedTask(fault="lost_in_middle", ground_truth=Label.GENERATOR.value,
                       item=item, oracle_chunk=buried,
                       meta={"filler_blocks": fillers, "gold_supplied": True,
                             "context_chars": len(buried)})


INJECTORS = {"ignore_context": ignore_context, "adversarial_prompt": adversarial,
             "weak_model": weak_model, "lost_in_middle": lost_in_middle}
