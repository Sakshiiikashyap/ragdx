"""Exhaustive tests for attribution rule v1 over all 8 correctness combos
x R above/below tau, plus gold-audit and parametric-flag behavior."""

import pytest

from ragdx.attribution.rule import Label, attribute, AttributionThresholds

TH = AttributionThresholds(tau_r=0.5)


@pytest.mark.parametrize("r,expected", [
    (0.0, Label.RETRIEVER), (0.25, Label.RETRIEVER), (0.49, Label.RETRIEVER),
    (0.5, Label.INTERACTION), (1.0, Label.INTERACTION),
])
def test_c1_only_success_split_by_recall(r, expected):
    # c0=0, c1=1, c2=0: gold works, retrieved fails
    res = attribute(0, 1, 0, r=r, thresholds=TH)
    assert res.label == expected


def test_missing_r_abstains_when_needed():
    res = attribute(0, 1, 0, r=None, thresholds=TH)
    assert res.label == Label.ABSTAIN


def test_c2_success_is_none_with_parametric_flag():
    res = attribute(1, 1, 1, r=1.0, thresholds=TH)
    assert res.label == Label.NONE
    assert res.evidence["parametric"] is True


def test_c2_success_without_base_is_plain_none():
    res = attribute(0, 1, 1, r=1.0, thresholds=TH)
    assert res.label == Label.NONE
    assert "parametric" not in res.evidence


def test_context_misinterpretation_generator():
    # c0=1, c1=0: knew it, gold context hurt
    res = attribute(1, 0, 0, r=1.0, thresholds=TH)
    assert res.label == Label.GENERATOR


def test_all_fail_with_sufficient_gold_is_generator():
    res = attribute(0, 0, 0, r=0.0, thresholds=TH)
    assert res.label == Label.GENERATOR


def test_all_fail_with_insufficient_gold_is_question_issue():
    res = attribute(0, 0, 0, r=0.0, gold_insufficient=True, thresholds=TH)
    assert res.label == Label.QUESTION_ISSUE


def test_lucky_retrieval_is_anomaly():
    res = attribute(0, 0, 1, r=0.0, thresholds=TH)
    assert res.label == Label.ANOMALY


def test_grounding_flag_is_orthogonal():
    res = attribute(0, 1, 0, r=0.0, grounding_flag=True, thresholds=TH)
    assert res.label == Label.RETRIEVER
    assert res.evidence["GROUNDING"] is True


def test_all_eight_patterns_have_defined_labels():
    """Every (c0,c1,c2) combo maps to a non-ABSTAIN label (with R supplied)."""
    for c0 in (0, 1):
        for c1 in (0, 1):
            for c2 in (0, 1):
                res = attribute(c0, c1, c2, r=1.0, thresholds=TH)
                assert res.label != Label.ABSTAIN, (c0, c1, c2)


def test_priority_c2_success_wins_over_generator_rule():
    # c0=1, c1=0, c2=1: rule 1 fires first (retrieved succeeded)
    res = attribute(1, 0, 1, r=1.0, thresholds=TH)
    assert res.label == Label.NONE


def test_evidence_contains_inputs():
    res = attribute(0, 1, 0, r=0.3, thresholds=TH)
    assert res.evidence["c0"] == 0 and res.evidence["c1"] == 1 and res.evidence["c2"] == 0
    assert res.evidence["R"] == 0.3
