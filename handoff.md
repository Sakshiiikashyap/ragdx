# HANDOFF — ragdx (RAG failure attribution)

Last updated: 2026-10-16 (server restarted overnight; P4 chain relaunched).
Read `intent.md` (the plan) and `decisions.md` (append-only decision log,
**D001–D020**) alongside this file.

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
| P1 judge + calibration | DONE (+ D019 fix) | `results/exp01_judge/agreement.json`: judge↔EM agreement **97.3%** (150 judgments, 0 unparseable), **ECE 0.0131** vs EM (D019: an earlier 0.43 "overconfident" figure was a miscalculation, fixed + regression test) |
| P2 fault injectors + FaultBench | DONE | 8 injector types + controls, 17 injector tests; `data/faultbench/` dev 59 / test 61 stratified |
| P3 attribution v2 + baselines + stats | CODE DONE (v2 k-sample run deferred) | **130 tests green**; `attribution/{rule,probabilistic,baselines}.py`, `analysis/stats.py` (incl. `cohen_kappa`) |
| P4 exp03a dev run + tau sweep | **DONE** — freeze written: **τ_r=0.1** (flat sweep, R binary at k=1; dev macro-F1 0.2163) | `configs/frozen_thresholds.json`, `results/exp03_dev/{items,eval_subset}.json` |
| P4 exp03b test, exp04 baselines | **IN FLIGHT** — `run_phase4_chain.py` relaunched 2026-10-16 after server restart; exp03b resumes from `results/exp03/checkpoint.jsonl` | chain shell output |
| P5 robustness (exp06) | **DONE 2026-10-17** — non-monotone aggregate (label-shift artifact), c1 monotone, judge noise −4.1pt@p=.30; D023 crash fixed via cache replay | `experiments/exp06_robustness.py`, `results/exp06/` |
| P6 multi-model (exp07) | **DONE 2026-10-22** — 120b 0.279/0.168, qwen3.8-27b 0.246/0.119, agreement 0.902, κ≈0.83, none skipped | `experiments/exp07_multimodel.py`, `results/exp07/` |
| P7 utility (exp08) | **DONE 2026-10-22** — our_fix 0.556; beats grounding (p=1.1e-5), n.s. vs random and vs always-topk3 (0.667); oracle 0.722 | `experiments/exp08_utility.py`, `results/exp08/` |
| P8 paper | **COMPLETE 2026-10-22** — 7pp, 0 errors/overfulls, all sections filled, 1 intentional todo (deferred k-sample); P11 red-team doc current | `paper/main.pdf` |
| P9 viva TUI | DONE (7 screens smoke-tested) | `viva_demo/run.bat` / `run.ps1` / `make -C viva_demo tui` |

## Key numbers so far (all from results/, regenerable)

- **exp01 pilot** (50 MIRAGE q, gpt-oss-20b, TF-IDF top-1 over 5-chunk pool):
  acc base/gold/retrieved = 0.30/0.88/0.46; NV 0.42, CA 0.46, CI 0.08, CM 0.04;
  attribution NONE 23 / RETRIEVER 21 / GENERATOR 6; mean chunk-level R@1 = 0.46.
- **Judge**: 97.3% agreement with EM-loose; **ECE 0.0131 vs EM reference**
  (D019 corrected; judge is well-calibrated against EM, still used only as a
  rescue because EM itself is imperfect).
- **FaultBench dev COMPLETE (59/59)**: overall rule v1 acc **0.2542**,
  macro-F1 **0.2163**. The headline structure (D020, pre-registered before
  test): **fault activation 29/59** (30 injections benign — answer still
  correct); **failure-subset (c2==0) acc 0.5172** (n=29); benign-subset:
  all 30 → NONE (rule perfectly consistent — no failure, no blame). Per-fault
  activation: remove_gold 1.00 (acc 1.00), weak_topk 0.80 (acc 0.80),
  lost_in_middle 0.71, adversarial/ignore_context 0.50, weak_model 0.29,
  contradiction 0.29, **truncate_gold 0.00** (never activates at k=1).
  Dominant activated-miss: generator-fault items with (c0,c1,c2)=(0,1,0),
  R=0 → rule says RETRIEVER (truth GENERATOR) — information-limited for rule
  v1; this is the paper's blind-spot finding (Sec. VIII), not a code bug.

## RESUME (chains are automated; restart = one command each)

```powershell
cd "C:\Users\pc\OneDrive\Documents\Default Project"
$env:PYTHONPATH = "$PWD\src"
$PY = "C:\Users\pc\Desktop\MIRAGE-main\.venv\Scripts\python.exe"
& $PY experiments\run_phase4_chain.py    # exp03a(replay) -> exp03b(checkpoint resume) -> exp04; sleeps out 429s, aborts loudly after 20 attempts
& $PY experiments\run_phase567_chain.py  # AFTER phase-4: exp06 -> exp07 -> exp08
& $PY experiments\exp03c_failure_subset.py  # D020 secondary analysis (offline) once results/exp03/items.json exists
& $PY -m ragdx.analysis.plots            # regenerate all paper figures from results/
```

Rate-limit notes: Groq free tier = 200k TPD on gpt-oss-20b; observed
resets come in irregular windows (~200k each; two resets happened within
4.4h on 10-15). Empirical cost ≈ **4.3k tokens per faultbench task**
(3 condition chats + judge rescues). Remaining budget pressure is on exp07
(≈525k full / 2 models) — if windows are scarce before the viva, run exp07
on a stratified subset or one model and label it honestly in the paper
(decision to be logged as D021 when taken). Cache makes every resume free.

Then: P8 fill-in — replace the 9 `\todo{}` markers in `paper/main.tex`
(each names its source artifact), recompile, final red-team skim, push.

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

`paper/main.tex` (IEEEtran, tectonic, **7 pp**, 0 errors/overfulls/citation
warnings; target ~8): COMPLETE — all experiment sections filled from
`results/` artifacts (2026-10-22). Tables T1–T6, figures F1–F3 + F5–F7
embedded; F4 (risk–coverage) deferred with rule v2 (`results/exp03_v2/`
pending, the single remaining intentional `\todo`). Abstract merged;
analysis findings (a)–(f); conclusion updated with multi-model + repair
results. Layout fixes: T1–T3 table sizing, 130pt math-set reflow,
consistent T-numbering. Red-team review: `docs/08_RED_TEAM_REVIEW.md`
(10 ranked rejection reasons; fixes applied).

## Repo hygiene

- Branch `main`, pushed to `https://github.com/Sakshiiikashyap/ragdx`
  (private) @ 42258ec+ (2026-10-16). Decisions D001–D020 committed
  (incl. D019 ECE fix, D020 pre-registration + its pre-test correction).
- **130 unit tests**, all offline: `python -m pytest -q`.
- Never commit: `.env` (key), `results/`, `data/raw|processed`, `tools/`,
  `paper/*.pdf`. See `.gitignore`.
- Auth: machine's stored GitHub credential via per-command
  `http.extraheader` only; token never written to config. `.env` verified
  absent from remote tree (2026-10-15). Flip to public via Settings if
  desired.
