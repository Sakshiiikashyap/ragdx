# CLAUDE.md — RAG Failure Attribution (ragdx)

## Mission
Build and validate an automated **failure-attribution method** for RAG systems (Retriever / Generator / Both-Interaction / None), then wrap it in a tool. The research contribution is the *validated attribution method*, NOT the dashboard.

## Non-negotiable rules
1. Never invent citations, numbers, or results. Every number in the paper must come from a file in `results/` produced by a script in `experiments/`.
2. Every experiment: fixed seeds, config in YAML under `configs/`, outputs in `results/<exp_id>/`, a `manifest.json` (git hash, config hash, model versions, date).
3. All LLM calls go through `src/ragdx/llm/client.py` with on-disk caching (key = model+params+prompt hash) and logging of cost/tokens.
4. Use temperature 0 for judged/diagnostic runs unless the experiment is about sampling; report k-sample variance where specified.
5. Do not edit `docs/` spec files without telling me; propose diffs instead.
6. Write tests first for the attribution rule (`tests/test_attribution.py`), the metric functions, and the fault injectors.
7. Small commits, one task per commit, message format `phase-X: task`.
8. If a spec is ambiguous, ask me; do not guess metric definitions. MIRAGE metric definitions must be copied from the paper/repo (https://arxiv.org/abs/2504.17137, https://github.com/nlpai-lab/MIRAGE) and noted in `docs/07_CITATION_LEDGER.md`.

## Stack
Python 3.10+, FAISS/Chroma, sentence-transformers, FastAPI, Streamlit (dashboard), pandas/numpy/scipy/statsmodels, pytest, pydantic, hydra-or-plain-YAML.

## Repo layout
```
ragdx/
  CLAUDE.md
  configs/            # YAML per experiment
  data/               # raw/ processed/ (gitignored large files)
  docs/               # specs (this folder)
  src/ragdx/
    datasets/         # mirage.py garage.py hotpot.py custom.py
    retrieval/        # index.py retrievers.py rerank.py
    llm/              # client.py cache.py prompts/
    runner/           # three_condition.py (Objective 1 - DONE)
    metrics/          # retrieval.py generation.py mirage_metrics.py grounding.py
    attribution/      # rule.py probabilistic.py llm_diagnoser.py baselines.py
    faults/           # retriever_faults.py generator_faults.py
    judge/            # judge.py calibration.py
    analysis/         # stats.py plots.py tables.py
    app/              # api.py dashboard.py
  experiments/        # exp01_... exp09_ scripts
  results/            # generated, one folder per exp
  paper/              # main.tex, sections/, figures/, tables/, refs.bib
  tests/
```
