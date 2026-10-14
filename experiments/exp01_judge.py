"""exp01_judge: LLM-judge vs EM-loose agreement on exp01 items (O9-lite, P1).

Judges all 3 conditions of results/exp01/items.json (n x 3 calls, cached),
writes results/exp01_judge/{judge_items.json, agreement.json, manifest.json}.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdx.judge.judge import agreement_report, judge_answer
from ragdx.llm.client import LLMClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    src = ROOT / "results" / "exp01" / "items.json"
    items = json.loads(src.read_text(encoding="utf-8"))
    client = LLMClient(model="openai/gpt-oss-20b", cache_dir=ROOT / ".llm_cache",
                       log_path=ROOT / "results" / "_llm_calls.jsonl")

    out_records = []
    pairs: list[tuple[bool, "object"]] = []
    for i, it in enumerate(items):
        rec = {"query_id": it["query_id"]}
        for cond in ("base", "gold", "retrieved"):
            j = judge_answer(it["question"], it["answers"], it[cond]["text"], client)
            rec[cond] = {"verdict": j.verdict, "confidence": j.confidence,
                         "rationale": j.rationale,
                         "em": bool(it[cond]["em_loose"])}
            pairs.append((bool(it[cond]["em_loose"]), j))
        out_records.append(rec)
        print(f"[{i+1}/{len(items)}] judged")

    rep = agreement_report(pairs)
    out = ROOT / "results" / "exp01_judge"
    out.mkdir(parents=True, exist_ok=True)
    (out / "judge_items.json").write_text(json.dumps(out_records, indent=2), encoding="utf-8")
    (out / "agreement.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "exp_id": "exp01_judge", "source": "results/exp01/items.json",
        "model": "openai/gpt-oss-20b", "n_items": len(items),
        "n_judgments": len(pairs), "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
