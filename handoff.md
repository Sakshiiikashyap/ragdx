# HANDOFF — ragdx (RAG failure attribution)

Last updated: 2026-10-15 (night of 14→15 Oct). Written for whoever picks this
up next (including "future me" after sleep). Read `intent.md` (the plan) and
`decisions.md` (append-only decision log, D001–D011) alongside this file.

## What this is

`ragdx` — a research project + tool that attributes RAG answer failures to
**Retriever / Generator / Interaction / None** by comparing answer correctness
across three controlled conditions (No-Context / Gold-Context / Retrieved),
plus a fault-injected benchmark (RAG-FaultBench) with known failure-source
labels to *validate* the attribution. Upstream docs live in `docs/` (specs
from the original project pack; do not edit without proposing diffs —
CLAUDE.md rule 5). Prototype ancestry: `C:\Users\pc\Desktop\MIRAGE-main`
(`run_10.py` kept as `experiments/_reference_run_10.py`).

## Environment (one-time setup facts)

- Runtime: `C:\Users\pc\Desktop\MIRAGE-main\.venv\Scripts\python.exe`
  (D001 — has groq, datasets, sklearn, pytest 9.1.1, rich; NOT the same as
  that folder's requirements.txt, which is aspirational).
- Repo root: `C:\Users\pc\OneDrive\Documents\Default Project` (git repo,
  branch `main`, OneDrive-synced — beware file locks during sync).
- Groq API key: repo-root `.env` (**gitignored, never committed**).
- LaTeX: `tools\tectonic.exe` (MSVC build, gitignored binary). Compile with
  `cd paper && ..\tools\tectonic.exe main.tex`.
- `PYTHONPATH=src` needed for every experiment/test invocation (no editable
  install yet — a `pip install -e .` cleanup item).

## State by part (see intent.md for full checklist)

| Part | State | Evidence |
|---|---|---|
| P1 judge + calibration | DONE | `results/exp01_judge/agreement.json`: judge↔EM agreement **97.3%** (150 judgments, 0 unparseable) |
| P2 fault injectors + FaultBench | DONE | 8 injector types + controls, 17 injector tests; `data/faultbench/` dev 59 / test 61 stratified |
| P3 attribution v2 + baselines + stats | CODE DONE (v2 run pending) | 81 tests green; `attribution/{rule,probabilistic,baselines}.py`, `analysis/stats.py` |
| P4 exp03a dev run + tau sweep | **IN FLIGHT — paused at 36/59 by Groq daily rate limit** (see Resume) | `results/exp03_dev/checkpoint.jsonl` (36 records, crash-safe) |
| P4 exp03b test, exp04 baselines | WRITTEN, NOT RUN | `experiments/exp03b_faultbench_test.py`, `exp04_baselines.py` |
| P5 robustness (exp06) | NOT STARTED | plan in intent.md |
| P6 multi-model (exp07-lite) | NOT STARTED — models chosen: `openai/gpt-oss-120b`, `qwen/qwen3.8-27b` (verified live via Groq /models 2026-10-14) | |
| P7 utility (exp08-lite) | NOT STARTED | |
| P8 paper | DRAFT compiles (2pp IEEEtran, real exp01 numbers, red \todo{} markers) | `paper/main.pdf` |
| P9 viva TUI | DONE (smoke-tested) | `viva_demo/run.bat` / `run.ps1` / `make -C viva_demo tui` |

## Key numbers so far (all from results/, regenerable)

- **exp01 pilot** (50 MIRAGE q, gpt-oss-20b, TF-IDF top-1 over 5-chunk pool):
  acc base/gold/retrieved = 0.30/0.88/0.46; NV 0.42, CA 0.46, CI 0.08, CM 0.04;
  attribution NONE 23 / RETRIEVER 21 / GENERATOR 6; mean chunk-level R@1 = 0.46.
- **Judge**: 97.3% agreement with EM-loose; ECE 0.43 (judge overconfident —
  all confidences ~0.99; honest framing: rescue mechanism, not calibrated prob).
- **FaultBench dev (36/59 done, provisional)**: rule v1 gets clean RETRIEVER
  faults (remove_gold, weak_topk) right ~always, but MISSES most GENERATOR
  faults (ignore/adversarial/weak_model make c1 fail → rule sees R=0 → says
  RETRIEVER) and truncate_gold (evidence destroyed but id retrievable → NONE).
  This is the *research finding*, not a bug: rule v1's blind spot is faults
  that corrupt the gold condition. Expect an honest, weaker-than-hoped main
  result — the paper must report per-fault accuracy, not just macro-F1.

## RESUME (rate limit has reset — likely by morning)

```powershell
cd "C:\Users\pc\OneDrive\Documents\Default Project"
$env:PYTHONPATH = "$PWD\src"
$PY = "C:\Users\pc\Desktop\MIRAGE-main\.venv\Scripts\python.exe"
& $PY experiments\exp03a_faultbench_dev.py    # resumes via cache; finishes 59/59 + tau sweep + freezes configs/frozen_thresholds.json
& $PY experiments\exp03b_faultbench_test.py   # SINGLE-USE test split; writes results/exp03/eval.json
& $PY experiments\exp04_baselines.py          # 3 baselines + McNemar/Holm; ~61 diagnoser calls
```

Cost note: exp03a/03b/04 consumed ~198k of the 200k/day Groq free tokens
today. Each resume replays cached answers at zero cost; only unfinished tasks
bill. If the daily limit is still hit: wait (Groq TPD resets ~24h) or reduce
scope in `data/faultbench/*.jsonl`.

Then per intent.md: P5 (exp06 robustness), P6 (exp07-lite multi-model),
P7 (exp08-lite utility), P8 (paper completion), and the v2 k-sample run
(deferred, needs fresh tokens).

## Known issues / risks

1. **Rate limit**: Groq free tier = 200k tokens/day for gpt-oss-20b. All
   planned runs share it. Cache aggressively (already the default) and prefer
   replay over re-run.
2. **v2 (probabilistic) never executed live** — module + tests exist, but
   k-sample data was deferred for cost; risk-coverage figure is `\todo` in
   the paper until then.
3. **Baseline (b) is "RAGChecker-inspired"**, not their official package
   (D008) — wording in the paper must stay honest about this.
4. **Human annotation (O9 full)** never happened (no second annotator);
   judge calibration is the substitute; paper says so.
5. **OneDrive**: repo lives inside OneDrive; large results churn can cause
   sync conflicts. `results/` and `.llm_cache/` are gitignored but not
   OneDrive-ignored.
6. **MIRAGE doc_pool ids are doc-level** (all chunks share one id) —
   chunk-level gold tracking via text match (D011) is load-bearing; do not
   "simplify" it back.
7. `weak_model` fault = `allam-2-7b` (llama-3.1-8b-instant was retired from
   Groq; verified 2026-10-14).

## Paper status

`paper/main.tex` (IEEEtran, compiles clean with tectonic): abstract, intro,
formulation, method (rule v1 + ANOMALY precedence note), experimental setup
+ pilot results with real numbers, limitations, 1 verified citation +
`\todo` markers for P4–P7 sections. Next: fill Sections 5–7 from
`results/exp03*`, `exp04`, `exp06*`, `exp07*`; verify remaining citations
(RAGChecker, GaRAGe, RAGAS, lost-in-the-middle, Zheng et al.) into
`docs/07_CITATION_LEDGER.md` before citing; run the red-team review
(P11 prompt in `docs/06`).

## Repo hygiene

- Branch `main`, 5 commits (`phase-0` … `phase-3`). Small commits, message
  format per CLAUDE.md rule 7.
- 81 unit tests, all offline: `python -m pytest -q`.
- Never commit: `.env` (key), `results/`, `data/raw|processed`, `tools/`,
  `paper/*.pdf`. See `.gitignore`.
- GitHub push: pending auth (no gh CLI on this machine; see last
  decisions.md entry / ask user for remote + credentials).
