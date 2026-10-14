"""Judge-noise injection for P5: flip each recorded condition bit (c0,c1,c2)
independently with probability p, re-run attribution rule v1, and measure
accuracy against ground truth. Purely offline over recorded items (no API),
deterministic given (seed, p, rep)."""

from __future__ import annotations

import random
import statistics

from ragdx.attribution.rule import AttributionThresholds, attribute


def judge_noise_sweep(records: list[dict], tau_r: float, flip_ps: list[float],
                      n_rep: int = 200, seed: int = 0) -> list[dict]:
    """Per flip probability: mean/std attribution accuracy over n_rep trials.

    Returns [{"flip_p", "mean_accuracy", "std_accuracy", "n_rep", "n"}, ...]
    in the order of flip_ps. Each trial independently flips every recorded
    c-bit with probability p (simulating judge errors) before re-attributing.
    """
    if any(not (0.0 <= p <= 1.0) for p in flip_ps):
        raise ValueError(f"flip probabilities must be in [0,1]: {flip_ps}")
    if not records:
        raise ValueError("records must be non-empty")
    th = AttributionThresholds(tau_r=tau_r)
    out: list[dict] = []
    for p in flip_ps:
        accs: list[float] = []
        for rep in range(n_rep):
            rng = random.Random(seed * 1_000_003 + rep + round(p * 1000) * 7919)
            hits = 0
            for r in records:
                c0, c1, c2 = r["c0"], r["c1"], r["c2"]
                if p:  # each condition verdict errs independently w.p. p
                    if rng.random() < p:
                        c0 = 1 - c0
                    if rng.random() < p:
                        c1 = 1 - c1
                    if rng.random() < p:
                        c2 = 1 - c2
                pred = attribute(c0, c1, c2, r=r["R"], thresholds=th).label.value
                hits += int(pred == r["ground_truth"])
            accs.append(hits / len(records))
        mean = statistics.fmean(accs)
        sd = statistics.pstdev(accs) if n_rep > 1 else 0.0
        out.append({"flip_p": p, "mean_accuracy": mean, "std_accuracy": sd,
                    "n_rep": n_rep, "n": len(records)})
    return out
