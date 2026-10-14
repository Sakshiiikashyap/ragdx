"""Statistics helpers (docs/03): bootstrap 95% CI, McNemar exact test,
Holm-Bonferroni correction. Pure numpy/python; no API calls."""

from __future__ import annotations

import math
import random
from collections import Counter


def bootstrap_ci(values: list[float], n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI of the mean. Returns (mean, lo, hi)."""
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        s = sum(values[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (sum(values) / n, lo, hi)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar test on discordant pairs b, c
    (b = A correct B wrong; c = A wrong B correct). Binomial p, p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def holm_correction(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, keyed by input name."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)
        adjusted[name] = running_max
    return adjusted


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa between two labelings (P6: attribution stability)."""
    if len(a) != len(b) or not a:
        raise ValueError("label lists must be non-empty and equal length")
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1.0 - pe) if pe < 1.0 else 1.0


def macro_f1(truth: list[str], pred: list[str]) -> float:
    labels = sorted(set(truth) | set(pred))
    f1s = []
    tc, pc = Counter(truth), Counter(pred)
    for lab in labels:
        tp = sum(1 for t, p in zip(truth, pred) if t == lab and p == lab)
        fp = pc[lab] - tp
        fn = tc[lab] - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def per_class_prf(truth: list[str], pred: list[str]) -> dict[str, dict]:
    labels = sorted(set(truth) | set(pred))
    tc, pc = Counter(truth), Counter(pred)
    out = {}
    for lab in labels:
        tp = sum(1 for t, p in zip(truth, pred) if t == lab and p == lab)
        fp = pc[lab] - tp
        fn = tc[lab] - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[lab] = {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4), "support": tc[lab]}
    return out


def confusion_matrix(truth: list[str], pred: list[str]) -> dict[str, dict]:
    labels = sorted(set(truth) | set(pred))
    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(truth, pred):
        cm[t][p] += 1
    return cm
