# Intent — ragdx: from pilot to complete paper

**Mission (from CLAUDE.md):** build and validate an automated failure-attribution method
for RAG systems (Retriever / Generator / Both-Interaction / None), wrap it in a tool, and
produce a complete paper whose every number comes from `results/` produced by a script.

**Non-negotiables:** no invented numbers/citations; configs+manifests per experiment;
all LLM calls via the cached client; tests first; thresholds tuned on dev only, test
touched once; `\todo{}` where evidence is missing.

**Constraints (confirmed by user):** Groq-only API access (free tier — treat cost/rate
limits as scarce; cache everything); Windows; morning viva demo; full O1–O12 ambition
compressed honestly: validate RQ1/RQ2 with fault injection, robustness RQ3-lite, skip
multi-dataset corpus-scale generalization (RQ4) beyond multi-model, utility RQ5-lite.

---

## Parts (execute in order; each part = tests green + commit + NOTES entry)

### P1 Judge + reliability (O9-lite) — status: [x] DONE (2026-10-14)
- [x] `judge/judge.py`: LLM-as-judge correctness (T=0, JSON schema), prompt in `llm/prompts/judge.md`
- [x] Judge calibration vs EM-loose/strict on exp01's 50 items: **agreement 0.9733** (146/150), 0 unparseable; 3 judge-correct/EM-wrong (EM substring false negatives), 1 reverse; **ECE 0.0131** vs EM reference (**D019 bugfix 2026-10-15:** an earlier "ECE 0.43 = overconfident" figure compared confidence to the rate of "correct" verdicts (a base rate), not accuracy vs EM — fixed in `judge.py` + regression test; judge is *well calibrated against EM*, still used only as a rescue because the EM reference is imperfect, not a calibrated probability source)
- [x] Human-annotation CSV: DEFERRED (no second annotator available tonight; marked future work in paper)
- [x] Tests: judge parsing (valid/invalid JSON), agreement metrics — 42→57 tests total

### P2 Fault injectors + RAG-FaultBench (O7) — status: [x] DONE (2026-10-14)
- [x] Retriever: remove_gold, weak_topk, truncate_gold; Generator: ignore_context, adversarial_prompt, weak_model, lost_in_middle; Interaction: contradiction; controls via fallbacks
- [x] Each injector deterministic; tests verify only-intended-component changes (17 tests)
- [x] **D011 bugfix:** chunk-level gold identification (text match) — doc-level ids are degenerate in MIRAGE pools
- [x] FaultBench: 120 questions → dev 59 / test 61, stratified (dev: RET 22/GEN 30/INT 7; test: RET 23/GEN 30/INT 8), card + manifest
- **Cost actual:** ~0 API calls (injectors pure); bench run cost counted in P4

### P3 Attribution v2 + baselines (O6, O8) — status: [x] CODE DONE (2026-10-14)
- [x] `attribution/probabilistic.py`: Wilson CI, Wald diff CI, rule_v2 w/ ABSTAIN, risk_coverage
- [x] `attribution/baselines.py`: recall_threshold, metric_rule (RAGChecker-inspired, D008), llm_diagnoser
- [x] `analysis/stats.py`: bootstrap CI, McNemar exact, Holm, macro-F1, per-class, confusion
- [x] tau_R tuned on dev (exp03a sweep), frozen in `configs/frozen_thresholds.json` before test
- [ ] v2 k-sampling run (deferred: cost; attempt after P4 if rate limits allow)

### P4 Main study exp03+exp04 (RQ1, RQ2) — status: [~] RUNNING (2026-10-15)
- [x] `experiments/exp03a_faultbench_dev.py` (dev run + tau sweep + freeze) — RUNNING via `experiments/run_phase4_chain.py` (rate-limit aware, 52/59 dev as of 01:15 IST)
- [x] `experiments/exp03b_faultbench_test.py` (single-use test + eval report)
- [x] `experiments/exp04_baselines.py` (3 baselines + McNemar/Holm)
- [ ] Run exp03b + exp04 when dev run finishes; confusion/risk-coverage figures in `analysis/plots.py`

### P5 Robustness exp06 (RQ3-lite) — status: [~] CODE DONE, run queued (2026-10-15)
- [x] Gold degradation: drop 25/50/75% of gold chunk content (even-stride word deletion, D013); near-miss replacement; SEARA-style pseudo-gold (answer-containing chunk from pool as surrogate gold; skips reported) — `faults/degrade.py`, 20 tests
- [x] Judge-noise injection (flip probability sweep on labels, no API cost) — `analysis/noise.py`, 5 tests
- [x] Accuracy-vs-degradation curves for ours + frozen baseline — `analysis/plots.py::fig_degradation` + `analysis/robustness.py::level_report` (5 tests)
- [ ] Run `experiments/exp06_robustness.py` (queued behind P4; needs exp04 baseline items) — `run_phase567_chain.py` ready
- **Accept:** degradation table + figure; robustness paragraph in paper
- **Cost:** 60 test q × 3 levels ≈ 180 calls

### P6 Multi-model exp07-lite (RQ4-lite) — status: [~] CODE DONE, run queued (2026-10-15)
- [x] Query Groq /models at runtime; availability gate skips-and-reports (D016). Chosen extras: `openai/gpt-oss-120b` + `qwen/qwen3.8-27b` (3rd family is the main model itself); live check happens at exp07 start
- [x] Rerun exp03 pipeline per model on test split; per-model macro-F1; attribution stability (`cohen_kappa` + agreement vs main) — `experiments/exp07_multimodel.py` written; weak_model tasks cache-shared by construction (noted in eval)
- [ ] Run exp07 (queued behind P4 chain)
- **Accept:** generalization table (3 models); honest "single dataset, multi-model" framing in paper
- **Cost:** 2 extra models × 60 q × 3 cond ≈ 360 calls

### P7 Utility exp08-lite (RQ5-lite) — status: [~] CODE DONE, run queued (2026-10-15)
- [x] For RETRIEVER-labelled failures: top-k 1→3; for GENERATOR-labelled: stricter grounding prompt; plus gold-in-pool rerank as LABELED oracle-assisted ceiling; compare vs random-fix and always-one-component baselines on the same failure set — `analysis/utility.py` (7 tests) + `experiments/exp08_utility.py`
- [x] Paired gain + sign test (McNemar exact + Holm over discordant pairs, D017)
- [ ] Run exp08 (queued behind P4 chain; failure set = exp03 test items c2==0)
- **Accept:** utility bar chart + paragraph; honest "heuristic fixes" framing
- **Cost:** ≤ 150 calls

### P8 Paper completion — status: [~] DRAFT COMPILED (2026-10-15)
- [x] Verify remaining citations (RAGChecker NeurIPS 2024 D&B, GaRAGe ACL 2025 Findings, RAGAS EACL 2024, lost-in-the-middle, Zheng et al. MT-Bench, MIRAGE exact title, RAGVUE, SEARA) via web → ledger `docs/07_CITATION_LEDGER.md` (2026-10-15)
- [x] Sections 1–10 per `docs/04_PAPER_OUTLINE.md` + IEEE positioning table + injector table + rule table; exp01/judge/bench numbers embedded; **10 `\todo{}` markers** carry exp03–exp08 numbers (each names its source artifact)
- [x] Compile with `tools/tectonic.exe` — clean (0 undefined citations/refs, 0 overfull >10pt, ~5 pp draft)
- [ ] Fill `\todo`s from results/ + embed figures F1–F6 + final red-team self-review (P11 prompt) → fix pass (after P4 + P567 chains)
- **Accept:** complete `paper/main.pdf` ≤ venue page limit + appendix; zero unverified citations

### P9 Viva TUI demo (`viva_demo/`) — status: [x] DONE (2026-10-14)
- [x] Separate folder, own launcher (`run.ps1`, `run.bat`, `Makefile` target `tui`), zero changes to core repo (D010)
- [x] Rich-based: KPI dashboard, question browser (3 conditions + evidence), confusion matrix, risk-coverage, live "diagnose a question" mode (uses cache), paper/results explorer — 7 screens smoke-tested
- [x] Works offline from `results/` + `.llm_cache/`
- **Accept:** `viva_demo/run.bat` launches; graceful when results missing ✓

### P6b/P7b contingency
If rate limits or time block P5–P7: mark affected paper sections `\todo` with explicit
"future work" framing — never drop planned experiments silently.

---

## Execution discipline
- One part per commit batch (`phase-N: ...`); append every decision to `decisions.md`
- After each part: `pytest` green, commit, update this file's checkboxes
- Paper regenerated from scripts after every results-producing part
