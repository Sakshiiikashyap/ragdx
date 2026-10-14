"""Attribution baselines (O8, docs/06 P5). All receive the SAME dev/test
splits; thresholds tuned on dev ONLY.

(a) recall_threshold: single-signal rule — RETRIEVER if R < tau else NONE.
    (The trivial baseline H1 beats.)
(b) metric_rule ("RAGChecker-inspired", D008): RETRIEVER if R < tau;
    else GENERATOR if the answer's grounding proxy (token overlap between
    answer and retrieved context) < gamma; else NONE. Documented as
    inspired-by, not the official package.
(c) llm_diagnoser: LLM-as-diagnoser given q, retrieved context, answer,
    reference (no access to the other conditions). Prompt:
    llm/prompts/diagnoser.md. Parses {label, confidence, rationale}.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ragdx.attribution.rule import Label
from ragdx.llm.client import LLMClient

DIAG_PROMPT = (Path(__file__).resolve().parents[1] / "llm" / "prompts" /
               "diagnoser.md").read_text(encoding="utf-8")


def baseline_recall_threshold(r: float, tau_r: float = 0.5) -> Label:
    return Label.RETRIEVER if r < tau_r else Label.NONE


def grounding_proxy(answer: str, context: str) -> float:
    """Fraction of answer content-tokens present in the context."""
    import re
    stop = {"the", "a", "an", "is", "was", "of", "in", "to", "and", "he", "she"}
    clean = lambda s: {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in stop}
    ans_tokens = clean(answer)
    if not ans_tokens:
        return 0.0
    ctx = clean(context)
    return len(ans_tokens & ctx) / len(ans_tokens)


def baseline_metric_rule(r: float, answer: str, context: str,
                         tau_r: float = 0.5, gamma: float = 0.5) -> Label:
    if r < tau_r:
        return Label.RETRIEVER
    if grounding_proxy(answer, context) < gamma:
        return Label.GENERATOR
    return Label.NONE


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def baseline_llm_diagnoser(question: str, context: str, answer: str,
                           reference: str, client: LLMClient) -> tuple[Label, dict]:
    user = (f"QUESTION:\n{question}\n\nRETRIEVED CONTEXT:\n{context}\n\n"
            f"SYSTEM ANSWER:\n{answer}\n\nREFERENCE ANSWER:\n{reference}\n\n"
            f"JSON label:")
    resp = client.chat([{"role": "system", "content": DIAG_PROMPT},
                        {"role": "user", "content": user}])
    data = _extract_json(resp.text)
    if not data or data.get("label") not in {l.value for l in Label} - {"ABSTAIN"}:
        return Label.ABSTAIN, {"raw": resp.text, "reason": "unparseable"}
    return Label(data["label"]), {"confidence": data.get("confidence"),
                                  "rationale": data.get("rationale")}
