# ragdx — Attributing RAG Failures to Retriever or Generator

Controlled-condition (No-Context / Gold-Context / Real-Retrieval) diagnosis of RAG
failures, with an explicit abstention and per-question evidence. Research contribution
is the **validated attribution method**, not the dashboard (per `CLAUDE.md`).

Built on the B.Tech prototype in `Desktop\MIRAGE-main` (`run_10.py`, kept as
`experiments/_reference_run_10.py`), upgraded to the layout in `CLAUDE.md` with
caching, manifests, and tests. Methodologically anchored to MIRAGE (Park et al.,
NAACL 2025 Findings, arXiv:2504.17137).

## Status (2026-10-14, night 1)

| Objective | State |
|---|---|
| O1 three-condition runner | **DONE** — `src/ragdx/runner/three_condition.py`, per-item cache, manifest w/ git+config hash, token/cost log |
| O2 metrics | **DONE (pilot)** — `metrics/generation.py` (EM-loose/strict, token-F1, exact MIRAGE upstream defs), `metrics/mirage.py` (NV/CA/CI/CM w/ upstream elif-priority), `metrics/retrieval.py` (recall@k, P@k, MRR, nDCG) |
| O3 attribution rule v1 | **DONE** — `attribution/rule.py`, 35 tests incl. all 8 correctness combos × R above/below tau |
| O5 dashboard | **PILOT** — `app/dashboard.py` read-only static HTML over `results/` (Streamlit in Phase 5) |
| O4, O6–O12 | TODO — see `docs/05_PHASES_AND_TASKS.md` |

## Quickstart

```sh
export GROQ_API_KEY=...        # or put it in .env (gitignored)
make test                      # 35 unit tests, no API calls
python experiments/exp01_three_condition.py 50   # three-condition run on 50 MIRAGE questions
python -m ragdx.app.dashboard exp01             # writes results/exp01/dashboard.html
python -m ragdx.analysis.tables exp01           # Markdown summary + paper/tables/exp01_summary.tex
```

Runtime note: currently run with the `MIRAGE-main` venv (`C:\Users\pc\Desktop\MIRAGE-main\.venv\Scripts\python.exe`),
which already has `groq`, `datasets`, `scikit-learn`, `pytest`. A dedicated venv from
`requirements.txt` is the Phase-1 cleanup item.

## Rules that govern this repo

See `CLAUDE.md`: no invented numbers (everything in `results/` from a script), configs +
manifests per experiment, all LLM calls through the cached client, tests-first for
attribution/metrics, and never edit `docs/` specs without proposing a diff first.

## Paper

`paper/main.tex` (IEEEtran). Numbers are pulled only from `results/` artifacts via
`analysis/tables.py`; unfinished sections carry `\todo{}` markers by design.
