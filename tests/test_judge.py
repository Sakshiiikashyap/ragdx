"""Tests for judge JSON parsing and agreement analysis (no API calls)."""

from ragdx.judge.judge import JudgeResult, _extract_json, agreement_report


def test_extract_json_simple():
    assert _extract_json('{"verdict": "correct", "confidence": 0.9}') == {
        "verdict": "correct", "confidence": 0.9}


def test_extract_json_with_surrounding_text():
    text = 'Sure! Here is my evaluation:\n{"verdict": "incorrect", "confidence": 0.7, "rationale": "wrong fact"}\nThanks!'
    assert _extract_json(text)["verdict"] == "incorrect"


def test_extract_json_invalid_returns_none():
    assert _extract_json("no json here") is None
    assert _extract_json("{broken json") is None


def test_judge_result_correct_property():
    assert JudgeResult("correct", 0.9, "", "").correct is True
    assert JudgeResult("incorrect", 0.9, "", "").correct is False
    assert JudgeResult(None, 0.0, "unparseable", "").correct is None


def test_agreement_report_all_agree():
    pairs = [(True, JudgeResult("correct", 0.9, "", "")),
             (False, JudgeResult("incorrect", 0.8, "", ""))]
    rep = agreement_report(pairs)
    assert rep["agreement"] == 1.0
    assert rep["confusion_em_rows_judge_cols"]["both_correct"] == 1
    assert rep["confusion_em_rows_judge_cols"]["both_incorrect"] == 1
    assert rep["n_unparseable"] == 0


def test_agreement_report_counts_unparseable():
    pairs = [(True, JudgeResult(None, 0.0, "unparseable", ""))]
    rep = agreement_report(pairs)
    assert rep["n"] == 0 and rep["n_unparseable"] == 1


def test_agreement_report_disagreements_and_ece_bounds():
    pairs = [(True, JudgeResult("incorrect", 0.6, "", "")),
             (False, JudgeResult("correct", 0.4, "", "")),
             (True, JudgeResult("correct", 0.95, "", "")),
             (False, JudgeResult("incorrect", 0.9, "", ""))]
    rep = agreement_report(pairs)
    assert rep["agreement"] == 0.5
    assert rep["confusion_em_rows_judge_cols"]["judge_only_correct"] == 1
    assert rep["confusion_em_rows_judge_cols"]["em_only_correct"] == 1
    assert 0.0 <= rep["ece"] <= 1.0
