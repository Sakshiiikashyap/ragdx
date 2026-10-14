"""exp01: three-condition run on a MIRAGE subset + attribution (Phase 1, O1+O3).

Usage: python experiments/exp01_three_condition.py [n_questions]
All numbers land in results/exp01/ (items.json, summary.json, manifest.json).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdx.datasets.mirage import load_mirage
from ragdx.llm.client import LLMClient
from ragdx.runner.three_condition import RunConfig, run_three_condition

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    cfg = RunConfig(exp_id="exp01", model="openai/gpt-oss-20b", retriever="tfidf",
                    top_k=1, n_questions=n, tau_r=0.5)
    print(f"exp01: n={n} model={cfg.model} retriever={cfg.retriever} top_k={cfg.top_k}")
    items = load_mirage(n=n, offset=0)
    client = LLMClient(model=cfg.model, cache_dir=ROOT / ".llm_cache",
                       log_path=ROOT / "results" / "_llm_calls.jsonl")
    result = run_three_condition(cfg, items, client, ROOT / "results" / cfg.exp_id)
    print("\n===== SUMMARY =====")
    import json
    print(json.dumps(result["summary"], indent=2))
    print(json.dumps(result["manifest"], indent=2))


if __name__ == "__main__":
    main()
