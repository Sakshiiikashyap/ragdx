"""Tests for cohen_kappa (P6 attribution stability across models)."""

from __future__ import annotations

import pytest

from ragdx.analysis.stats import cohen_kappa


def test_identical_labels_kappa_one():
    a = ["RETRIEVER", "GENERATOR", "NONE", "INTERACTION", "RETRIEVER"]
    assert cohen_kappa(a, list(a)) == pytest.approx(1.0)


def test_hand_computed_example():
    # 8/10 agree (po=.8); marginals both 60/40 -> pe=.52; k=(.8-.52)/.48
    a = ["X"] * 6 + ["Y"] * 4
    b = ["X"] * 5 + ["Y"] * 3 + ["X"] + ["Y"]
    assert cohen_kappa(a, b) == pytest.approx((0.8 - 0.52) / 0.48)


def test_disjoint_is_negative():
    a = ["X", "X", "Y", "Y"]
    b = ["Y", "Y", "X", "X"]
    assert cohen_kappa(a, b) < 0


def test_perfect_marginals_but_zero_agreement():
    # po=0, pe=0.5 -> k=-1
    assert cohen_kappa(["A", "B"], ["B", "A"]) == pytest.approx(-1.0)
