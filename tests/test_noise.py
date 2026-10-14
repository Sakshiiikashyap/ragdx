"""P5 tests: judge-noise injection (flip probability sweep over the recorded
c-bits, no API cost) — deterministic, seeded, offline."""

from __future__ import annotations

import pytest

from ragdx.analysis.noise import judge_noise_sweep

RECORDS = [{"c0": 1, "c1": 1, "c2": 0, "R": 1.0, "ground_truth": "RETRIEVER"},
           {"c0": 0, "c1": 1, "c2": 0, "R": 0.0, "ground_truth": "RETRIEVER"},
           {"c0": 0, "c1": 0, "c2": 1, "R": 0.2, "ground_truth": "GENERATOR"},
           {"c0": 1, "c1": 1, "c2": 1, "R": 1.0, "ground_truth": "NONE"},
           {"c0": 0, "c1": 1, "c2": 1, "R": 0.8, "ground_truth": "INTERACTION"},
           {"c0": 0, "c1": 0, "c2": 0, "R": 0.0, "ground_truth": "GENERATOR"}]


def test_zero_flip_equals_plain_accuracy():
    out = judge_noise_sweep(RECORDS, tau_r=0.5, flip_ps=[0.0], n_rep=20, seed=0)
    (row,) = out
    assert row["flip_p"] == 0.0
    assert row["std_accuracy"] == 0.0            # no randomness at p=0
    # plain attribution accuracy of the records
    from ragdx.attribution.rule import AttributionThresholds, attribute
    th = AttributionThresholds(tau_r=0.5)
    acc = sum(attribute(r["c0"], r["c1"], r["c2"], r=r["R"], thresholds=th).label.value
              == r["ground_truth"] for r in RECORDS) / len(RECORDS)
    assert row["mean_accuracy"] == pytest.approx(acc)


def test_sweep_deterministic_given_seed():
    a = judge_noise_sweep(RECORDS, 0.5, [0.1, 0.3], n_rep=50, seed=7)
    b = judge_noise_sweep(RECORDS, 0.5, [0.1, 0.3], n_rep=50, seed=7)
    assert a == b
    c = judge_noise_sweep(RECORDS, 0.5, [0.1, 0.3], n_rep=50, seed=8)
    assert c != a


def test_full_flip_is_valid_and_bounded():
    out = judge_noise_sweep(RECORDS, 0.5, [1.0], n_rep=10, seed=0)
    (row,) = out
    assert 0.0 <= row["mean_accuracy"] <= 1.0
    assert row["std_accuracy"] >= 0.0
    assert row["n_rep"] == 10 and row["n"] == len(RECORDS)


def test_moderate_noise_differs_from_zero():
    out = judge_noise_sweep(RECORDS, 0.5, [0.0, 0.5], n_rep=200, seed=3)
    zero, half = out
    assert half["std_accuracy"] > 0.0            # randomness now active
    assert half["mean_accuracy"] < zero["mean_accuracy"]  # noise hurts on average


def test_bad_p_raises():
    with pytest.raises(ValueError):
        judge_noise_sweep(RECORDS, 0.5, [1.5], n_rep=5, seed=0)
