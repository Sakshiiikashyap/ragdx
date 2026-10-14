"""Probabilistic attribution rule v2 (docs/02) with Wilson CIs and ABSTAIN.

Given k-sample per-condition correctness rates p0, p1, p2 (No/Gold/Retrieved)
and chunk-level gold recall R, decide:
  RETRIEVER   if (p1 - p2) > delta with CI excluding 0 and R < tau_R
  INTERACTION if (p1 - p2) > delta with CI excluding 0 and R >= tau_R
  GENERATOR   if p1 < theta
  NONE        if p2 >= theta
  else ABSTAIN
Risk-coverage: ABSTAIN items are uncovered; coverage = decided/n; selective
accuracy = correct decisions / decided (evaluated against ground truth by
the caller).

CIs: Wilson score interval for each p; the difference CI is the normal-approx
Wald interval on p1 - p2 (documented approximation; bootstrap CI planned for
the paper's stats module).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ragdx.attribution.rule import AttributionThresholds, Label


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def diff_ci(p1: float, p2: float, n1: int, n2: int,
            z: float = 1.96) -> tuple[float, float, float]:
    """Wald CI for p1 - p2 (independent samples of size n1, n2)."""
    diff = p1 - p2
    se = math.sqrt(max(p1 * (1 - p1) / max(n1, 1), 0) +
                   max(p2 * (1 - p2) / max(n2, 1), 0))
    return diff, diff - z * se, diff + z * se


@dataclass
class V2Result:
    label: Label
    coverage: float          # 1.0 if decided, 0.0 if abstain
    evidence: dict


def rule_v2(counts: tuple[int, int, int], ns: tuple[int, int, int],
            r: float | None,
            thresholds: AttributionThresholds | None = None) -> V2Result:
    """counts = (k0, k1, k2) successes; ns = sample sizes per condition."""
    th = thresholds or AttributionThresholds()
    k0, k1, k2 = counts
    n0, n1, n2 = ns
    p0, p1, p2 = k0 / max(n0, 1), k1 / max(n1, 1), k2 / max(n2, 1)
    diff, lo, hi = diff_ci(p1, p2, n1, n2)
    ev = {"p0": p0, "p1": p1, "p2": p2, "diff_p1_p2": diff,
          "diff_ci": [lo, hi], "R": r, "tau_r": th.tau_r,
          "delta": th.delta, "theta": th.theta}

    if diff > th.delta and lo > 0:
        if r is None:
            return V2Result(Label.ABSTAIN, 0.0, {**ev, "reason": "R missing"})
        label = Label.RETRIEVER if r < th.tau_r else Label.INTERACTION
    elif p1 < th.theta:
        label = Label.GENERATOR
    elif p2 >= th.theta:
        label = Label.NONE
    else:
        return V2Result(Label.ABSTAIN, 0.0, {**ev, "reason": "no CI-separated effect"})
    return V2Result(label, 1.0, ev)


def risk_coverage(confidences: list[float], correct: list[bool]) -> list[tuple[float, float]]:
    """Selective-risk curve: sort by confidence desc, ABSTAIN lowest first.
    Returns [(coverage, selective_accuracy)] points."""
    pairs = sorted(zip(confidences, correct), key=lambda x: -x[0])
    n = len(pairs)
    points = []
    seen_correct = 0
    for i, (conf, ok) in enumerate(pairs, start=1):
        seen_correct += int(ok)
        points.append((i / n, seen_correct / i))
    return points
