"""LLM-as-judge correctness (O9-lite), plus agreement analysis vs EM metrics.

Judge output contract: JSON {"verdict": "correct"|"incorrect",
"confidence": 0..1, "rationale": str} — prompt in llm/prompts/judge.md.
Parsing is defensive: extracts the first balanced JSON object from the
reply; unparseable replies are counted, never silently coerced.

CLAUDE.md rule 4: judge runs use temperature 0.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ragdx.llm.client import LLMClient

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "llm" / "prompts" / "judge.md"

JUDGE_SYSTEM = _PROMPT_PATH.read_text(encoding="utf-8")

_TEMPLATE = """QUESTION:
{question}

REFERENCE ANSWERS:
{references}

CANDIDATE ANSWER:
{candidate}

JSON verdict:"""


@dataclass
class JudgeResult:
    verdict: str | None  # "correct" | "incorrect" | None if unparseable
    confidence: float
    rationale: str
    raw: str

    @property
    def correct(self) -> bool | None:
        if self.verdict == "correct":
            return True
        if self.verdict == "incorrect":
            return False
        return None


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def judge_answer(question: str, references: list[str], candidate: str,
                 client: LLMClient) -> JudgeResult:
    refs = "\n".join(f"- {r}" for r in references)
    user = _TEMPLATE.format(question=question, references=refs, candidate=candidate)
    resp = client.chat([{"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": user}])
    data = _extract_json(resp.text)
    if not data or data.get("verdict") not in ("correct", "incorrect"):
        return JudgeResult(None, 0.0, "unparseable", resp.text)
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return JudgeResult(data["verdict"], max(0.0, min(1.0, conf)),
                       str(data.get("rationale", "")), resp.text)


# ---- calibration analysis (no API calls; operates on stored results) ------

def agreement_report(pairs: list[tuple[bool, JudgeResult]]) -> dict:
    """pairs = (em_label, judge) over items where judge parsed OK."""
    valid = [(em, j) for em, j in pairs if j.correct is not None]
    n_unparseable = sum(1 for _, j in pairs if j.correct is None)
    if not valid:
        return {"n": 0, "n_unparseable": n_unparseable}
    agree = sum(1 for em, j in valid if em == j.correct)
    # 2x2 confusion: rows = EM, cols = judge
    tp = sum(1 for em, j in valid if em and j.correct)
    tn = sum(1 for em, j in valid if not em and not j.correct)
    fp = sum(1 for em, j in valid if not em and j.correct)   # judge says correct, EM says not
    fn = sum(1 for em, j in valid if em and not j.correct)   # judge says incorrect, EM says correct
    # ECE over 5 equal-width confidence bins
    bins = [(i / 5, (i + 1) / 5) for i in range(5)]
    ece = 0.0
    bin_stats = []
    for lo, hi in bins:
        sel = [(em, j) for em, j in valid
               if lo <= j.confidence < hi or (hi == 1.0 and j.confidence == 1.0)]
        if not sel:
            bin_stats.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0})
            continue
        # accuracy per bin = agreement of the judge's verdict with the EM
        # reference (D019: an earlier version used the rate of "correct"
        # verdicts as accuracy, which is a base rate, not calibration).
        acc_bin = sum(1 for em, j in sel if em == j.correct) / len(sel)
        conf_bin = sum(j.confidence for _, j in sel) / len(sel)
        ece += len(sel) / len(valid) * abs(acc_bin - conf_bin)
        bin_stats.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(sel),
                          "agreement_vs_em": round(acc_bin, 3),
                          "mean_confidence": round(conf_bin, 3)})
    return {"n": len(valid), "n_unparseable": n_unparseable,
            "agreement": round(agree / len(valid), 4),
            "confusion_em_rows_judge_cols": {"both_correct": tp, "both_incorrect": tn,
                                             "judge_only_correct": fp, "em_only_correct": fn},
            "ece": round(ece, 4), "bins": bin_stats}
