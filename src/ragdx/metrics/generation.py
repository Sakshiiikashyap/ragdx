"""Generation metrics: EM (loose/strict) and token-F1.

Definitions COPIED EXACTLY from the MIRAGE upstream repository
(github.com/nlpai-lab/MIRAGE, file LLM.py, class LLM_Evaluator,
methods calculate_EM_loose / calculate_EM_strict / calculate_f1;
verified 2026-10-14 against the local upstream copy in MIRAGE-main/LLM.py).

- EM_loose: 1.0 if any normalized reference answer is a substring of the
  lowercased prediction, else 0.0.
- EM_strict: 1.0 if the lowercased prediction exactly equals any reference.
- F1: token-set (not multiset) F1 over the union of all reference answers.
"""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Light normalization used only for our substring checks in attribution
    helpers; MIRAGE EM functions below use .lower() only, as upstream."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def em_loose(answer_array: list[str], prediction: str) -> float:
    answer_set = {ans.lower() for ans in answer_array}
    pred = prediction.lower()
    return 1.0 if any(ans in pred for ans in answer_set) else 0.0


def em_strict(answer_array: list[str], prediction: str) -> float:
    answer_set = {ans.lower() for ans in answer_array}
    pred = prediction.lower()
    return 1.0 if any(ans == pred for ans in answer_set) else 0.0


def token_f1(answer_array: list[str], prediction: str) -> float:
    pred = prediction.lower()
    predicted_tokens = set(pred.split())
    answer_tokens = {token for ans in answer_array for token in ans.lower().split()}
    true_positives = len(predicted_tokens & answer_tokens)
    precision = true_positives / len(predicted_tokens) if predicted_tokens else 0.0
    recall = true_positives / len(answer_tokens) if answer_tokens else 0.0
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def correctness(answer_array: list[str], prediction: str) -> bool:
    """Primary binary judge used by the runner: MIRAGE EM_loose >= 0.5 threshold
    is just em_loose itself (already binary)."""
    return em_loose(answer_array, prediction) >= 1.0
