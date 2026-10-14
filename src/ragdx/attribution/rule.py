"""Attribution rule v1 — deterministic, per docs/02_ATTRIBUTION_SPEC.md.

Notation: c0/c1/c2 = correctness under No-Context / Gold-Context /
Real-Retrieval; R = gold recall@k of the retrieved set; gold_audit =
whether the gold chunk is judged sufficient for answering the question.

Decision table (docs/02, Rule v1), implemented in priority order:
  1. c2 = 1                       -> NONE   (flag PARAMETRIC if c0 = 1)
  2. c1 = 1 and c2 = 0, R < tau_R -> RETRIEVER
  3. c1 = 1 and c2 = 0, R >= tau_R-> INTERACTION (gold present but distracted)
  4. c0 = 1 and c1 = 0            -> GENERATOR (context misinterpretation)
  5. c0 = 0, c1 = 0, c2 = 0       -> GENERATOR; QUESTION_ISSUE if gold-audit
                                    flags insufficiency
  6. c0 = 0, c1 = 0, c2 = 1       -> ANOMALY (flag judge noise / lucky guess)
  7. anything else                -> ABSTAIN
GROUNDING is an orthogonal tag (over-summarisation / ungrounded claims),
not a primary label; callers may attach it to the evidence dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Label(str, Enum):
    RETRIEVER = "RETRIEVER"
    GENERATOR = "GENERATOR"
    INTERACTION = "INTERACTION"
    NONE = "NONE"
    QUESTION_ISSUE = "QUESTION_ISSUE"
    ANOMALY = "ANOMALY"
    ABSTAIN = "ABSTAIN"


@dataclass
class AttributionThresholds:
    tau_r: float = 0.5      # recall threshold (tuned on dev ONLY per docs/02)
    delta: float = 0.0      # used by rule v2 only
    theta: float = 0.5      # used by rule v2 only


@dataclass
class AttributionResult:
    label: Label
    evidence: dict = field(default_factory=dict)


def attribute(c0: int, c1: int, c2: int, r: float | None = None,
              gold_insufficient: bool = False,
              grounding_flag: bool = False,
              thresholds: AttributionThresholds | None = None) -> AttributionResult:
    th = thresholds or AttributionThresholds()
    ev: dict[str, Any] = {"c0": c0, "c1": c1, "c2": c2, "R": r,
                          "tau_r": th.tau_r, "gold_insufficient": gold_insufficient}
    if grounding_flag:
        ev["GROUNDING"] = True

    if c0 == 0 and c1 == 0 and c2 == 1:
        # SPEC NOTE (docs/02): the table has both "any|any|1 -> NONE" and the
        # more specific "0|0|1 -> ANOMALY (impossible-ish)". We resolve in
        # favor of the specific row; flagging the general row instead would
        # make ANOMALY unreachable. Pending user confirmation.
        label = Label.ANOMALY
        ev["reason"] = "correct only under retrieval, failed even with gold; check judge noise or lucky guess"
    elif c2 == 1:
        label = Label.NONE
        if c0 == 1:
            ev["parametric"] = True  # knew without context: context unnecessary
    elif c1 == 1 and c2 == 0:
        if r is None:
            return AttributionResult(Label.ABSTAIN, {**ev, "reason": "R required for RETRIEVER vs INTERACTION split"})
        if r < th.tau_r:
            label = Label.RETRIEVER
            ev["reason"] = "gold works, retrieved fails, gold not in top-k (R < tau_R)"
        else:
            label = Label.INTERACTION
            ev["reason"] = "gold works, retrieved fails despite R >= tau_R (distractor/noise)"
    elif c0 == 1 and c1 == 0:
        label = Label.GENERATOR
        ev["reason"] = "knew it with no context; gold context broke the answer"
    elif c0 == 0 and c1 == 0 and c2 == 0:
        if gold_insufficient:
            label = Label.QUESTION_ISSUE
            ev["reason"] = "fails even with gold, but gold-audit flags insufficient gold"
        else:
            label = Label.GENERATOR
            ev["reason"] = "fails even with sufficient gold context"
    else:
        label = Label.ABSTAIN
        ev["reason"] = "no rule matched"

    return AttributionResult(label, ev)
