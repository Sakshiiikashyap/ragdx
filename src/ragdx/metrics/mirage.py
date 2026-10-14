"""MIRAGE system-level metrics: NV, CA, CI, CM.

Definitions COPIED EXACTLY from MIRAGE upstream (evaluation.py,
Metrics.get_analysis_and_report; verified 2026-10-14 against the local
upstream copy MIRAGE-main/evaluation.py).

Per question with base/mix/oracle EM labels b, m, o (in {0,1}), evaluated
in elif order (first match wins):
  NV (noise vulnerability):   m=0 and o=1   -> retrieved context failed while gold worked
  CA (context acceptability): m=1 and o=1   -> retrieved context sufficed
  CI (context insensitivity): b=0 and o=0   -> context never helped (counts as failure both ways)
  CM (context misinterpretation): b=1 and o=0 -> knew without context, gold context broke it
Each score = count / total questions. Ranking score = -NV + CA - CI - CM.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class MirageMetricResult:
    total: int
    counts: dict  # per-pattern counts keyed "b_m_o"
    nv: float
    ca: float
    ci: float
    cm: float
    unmapped: int  # questions matching none of the four patterns

    @property
    def ranking_score(self) -> float:
        return -self.nv + self.ca - self.ci - self.cm


def pattern_key(base: int, mix: int, oracle: int) -> str:
    return f"{base}_{mix}_{oracle}"


def mirage_metrics(base_labels: list[int], mix_labels: list[int],
                   oracle_labels: list[int]) -> MirageMetricResult:
    assert len(base_labels) == len(mix_labels) == len(oracle_labels), "label lists must align"
    total = len(base_labels)
    if total == 0:
        return MirageMetricResult(0, {}, 0.0, 0.0, 0.0, 0.0, 0)
    counts: Counter = Counter()
    nv = ca = ci = cm = unmapped = 0
    for b, m, o in zip(base_labels, mix_labels, oracle_labels):
        counts[pattern_key(b, m, o)] += 1
        if m == 0 and o == 1:
            nv += 1
        elif m == 1 and o == 1:
            ca += 1
        elif b == 0 and o == 0:
            ci += 1
        elif b == 1 and o == 0:
            cm += 1
        else:
            unmapped += 1  # e.g. 0_0_1 anomaly, 1_1_1 parametric
    return MirageMetricResult(total=total, counts=dict(counts),
                              nv=nv / total, ca=ca / total,
                              ci=ci / total, cm=cm / total, unmapped=unmapped)
