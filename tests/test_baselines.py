"""Tests for probabilistic v2 and baselines (no API calls)."""

import math

from ragdx.attribution.baselines import (baseline_metric_rule,
                                         baseline_recall_threshold,
                                         grounding_proxy)
from ragdx.attribution.probabilistic import (diff_ci, risk_coverage, rule_v2,
                                             wilson_ci)
from ragdx.attribution.rule import Label


# ---- Wilson CI -------------------------------------------------------------

def test_wilson_ci_center_reasonable():
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi
    assert 0.0 <= lo < hi <= 1.0


def test_wilson_ci_extremes():
    lo, hi = wilson_ci(100, 100)
    assert hi > 0.99 and lo > 0.9
    lo, hi = wilson_ci(0, 100)
    assert lo == 0.0 and hi < 0.1


def test_wilson_ci_empty():
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_diff_ci_excludes_zero_when_separated():
    diff, lo, hi = diff_ci(0.9, 0.2, 30, 30)
    assert diff > 0 and lo > 0


def test_diff_ci_includes_zero_when_close():
    diff, lo, hi = diff_ci(0.5, 0.5, 30, 30)
    assert lo <= 0 <= hi


# ---- rule v2 ----------------------------------------------------------------

def test_v2_retriever_when_gold_beats_retrieved_and_r_low():
    res = rule_v2((5, 27, 8), (30, 30, 30), r=0.1)
    assert res.label == Label.RETRIEVER and res.coverage == 1.0


def test_v2_interaction_same_effect_but_r_high():
    res = rule_v2((5, 27, 8), (30, 30, 30), r=0.9)
    assert res.label == Label.INTERACTION


def test_v2_generator_when_gold_fails():
    res = rule_v2((2, 3, 8), (30, 30, 30), r=0.9)
    assert res.label == Label.GENERATOR


def test_v2_none_when_retrieved_succeeds():
    res = rule_v2((2, 25, 28), (30, 30, 30), r=0.9)
    assert res.label == Label.NONE


def test_v2_abstain_when_effect_not_separated():
    # p1=0.667 >= theta, p2=0.467 < theta, but diff CI includes 0 -> ABSTAIN
    res = rule_v2((10, 20, 14), (30, 30, 30), r=0.5)
    assert res.label == Label.ABSTAIN and res.coverage == 0.0


def test_v2_none_when_no_effect_but_p2_high():
    # p1 == p2 == 0.667 >= theta -> NONE per spec priority
    res = rule_v2((10, 20, 20), (30, 30, 30), r=0.5)
    assert res.label == Label.NONE


def test_v2_abstain_when_r_missing_and_effect_present():
    res = rule_v2((5, 27, 8), (30, 30, 30), r=None)
    assert res.label == Label.ABSTAIN


def test_v2_evidence_fields():
    res = rule_v2((5, 27, 8), (30, 30, 30), r=0.1)
    assert res.evidence["p1"] > res.evidence["p2"]
    assert "diff_ci" in res.evidence


# ---- risk-coverage -----------------------------------------------------------

def test_risk_coverage_monotone_coverage_and_perfect_ranking():
    conf = [0.9, 0.8, 0.3, 0.1]
    correct = [True, True, False, False]
    pts = risk_coverage(conf, correct)
    assert pts[0][0] == 0.25 and pts[-1][0] == 1.0
    assert pts[-1][1] == 0.5           # 2/4 overall
    assert pts[0][1] == 1.0 and pts[1][1] == 1.0  # top-confidence correct


# ---- baselines ---------------------------------------------------------------

def test_recall_threshold_baseline():
    assert baseline_recall_threshold(0.2, 0.5) == Label.RETRIEVER
    assert baseline_recall_threshold(0.8, 0.5) == Label.NONE


def test_grounding_proxy_high_when_answer_in_context():
    assert grounding_proxy("journalist", "John Mayne was a journalist.") > 0.9
    assert grounding_proxy("astronaut", "John Mayne was a journalist.") < 0.1


def test_metric_rule_paths():
    assert baseline_metric_rule(0.0, "x", "y") == Label.RETRIEVER
    assert baseline_metric_rule(1.0, "astronaut", "John was a journalist") == Label.GENERATOR
    assert baseline_metric_rule(1.0, "journalist", "John was a journalist") == Label.NONE
