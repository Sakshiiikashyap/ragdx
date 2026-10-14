"""Tests for stats helpers with hand-computed values."""

from ragdx.analysis.stats import (bootstrap_ci, confusion_matrix, holm_correction,
                                  macro_f1, mcnemar_exact, per_class_prf)


def test_bootstrap_ci_brackets_mean():
    vals = [1.0, 0.0, 1.0, 1.0, 0.0] * 20
    mean, lo, hi = bootstrap_ci(vals, n_boot=2000, seed=1)
    assert lo <= mean <= hi
    assert 0.4 < lo < mean < hi < 0.8


def test_mcnemar_symmetric_extreme():
    # heavily discordant one way -> small p
    assert mcnemar_exact(20, 0) < 0.001
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) > 0.9


def test_holm_ordering():
    adj = holm_correction({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] <= adj["c"] <= adj["b"]        # monotone in raw p order
    assert adj["a"] == 0.03                        # 3 * 0.01
    assert adj["b"] <= 1.0


def test_macro_f1_perfect():
    assert macro_f1(["A", "B", "A"], ["A", "B", "A"]) == 1.0


def test_macro_f1_hand_computed():
    # truth [A,A,B], pred [A,B,B]: A: tp1 fp0 fn1 -> p1 r0.5 f1 2/3; B: tp1 fp1 fn0 -> p0.5 r1 f1 2/3
    assert abs(macro_f1(["A", "A", "B"], ["A", "B", "B"]) - 2 / 3) < 1e-9


def test_per_class_prf_counts():
    out = per_class_prf(["R", "R", "G"], ["R", "G", "G"])
    assert out["R"]["support"] == 2
    assert out["R"]["precision"] == 1.0 and out["R"]["recall"] == 0.5


def test_confusion_matrix_shape():
    cm = confusion_matrix(["A", "B"], ["A", "A"])
    assert cm["A"]["A"] == 1 and cm["B"]["A"] == 1 and cm["B"]["B"] == 0
