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

### P4 Main study exp03+exp04 (RQ1, RQ2) — status: [x] DONE (2026-10-16)
- [x] `experiments/exp03a_faultbench_dev.py` (dev run + tau sweep + freeze) — **59/59, frozen τ_r=0.1** (flat sweep, R binary at k=1; dev macro-F1 0.2163)
- [x] `experiments/exp03b_faultbench_test.py` (single-use test + eval report) — **test acc 0.3115 [0.197, 0.426], macro-F1 0.1612**; per-fault + confusion written
- [x] `experiments/exp04_baselines.py` (3 baselines + McNemar/Holm) — recall 0.3443 / metric-rule 0.3443 / LLM-diagnoser 0.3934, **no significant diffs** (p_holm ≥ 0.68)
- [x] D020 activation/failure-subset analysis (`exp03c`): activation 36/61, **failure-subset acc 0.5278** (dev 0.5172 replication); figures: tau_sweep, confusion, per_fault, baselines (risk-coverage waits on deferred v2)

### P5 Robustness exp06 (RQ3-lite) — status: [x] DONE (2026-10-17; paper filled 2026-10-22)
- [x] Gold degradation: drop 25/50/75% of gold chunk content (even-stride word deletion, D013); near-miss replacement; SEARA-style pseudo-gold (answer-containing chunk from pool as surrogate gold; skips reported) — `faults/degrade.py`, 20 tests
- [x] Judge-noise injection (flip probability sweep on labels, no API cost) — `analysis/noise.py`, 5 tests
- [x] Accuracy-vs-degradation curves for ours + frozen baseline — `analysis/plots.py::fig_degradation` + `analysis/robustness.py::level_report` (5 tests)
- [x] Ran `exp06_robustness.py` (333 records; first attempt's LLM work fully cached — D023 crash on exp04 items bug, fixed, 51s cache replay): acc 0.3115/0.2787/0.2295/0.3279/0.3279 (pseudo 0.3214, n=28, 33 skips); c1_rate monotone 0.885→0.148; class-conditional collapse (RET 0.609→0.043, GEN 0.167→0.633) — non-monotone aggregate = label-shift artifact; judge noise −4.1pt at p=0.30 (200 reps)
- **Accept:** ✓ paper §VII-C + F5 (degradation.png), 0 overfull

### P6 Multi-model exp07-lite (RQ4-lite) — status: [x] DONE (2026-10-22)
- [x] Query Groq /models at runtime; availability gate skips-and-reports (D016). Chosen extras: `openai/gpt-oss-120b` + `qwen/qwen3.8-27b`; live check passed, none skipped
- [x] Reran exp03 pipeline per model on test split (851s): 120b 0.2787/0.1676, qwen3.8-27b 0.2459/0.1188; agreement 0.902, κ=0.826/0.829 vs main — labels stable across families; weak_model tasks cache-shared (noted)
- **Accept:** ✓ paper §VII-D + T6 + F6 (multimodel.png), honest "single dataset, multi-model" framing

### P7 Utility exp08-lite (RQ5-lite) — status: [x] DONE (2026-10-22)
- [x] Repairs on 36 activated failures (c2==0; 78,447 tokens): our_fix gain 0.556 vs random 0.389 (p_holm 0.14 n.s.), always_topk3 0.667 (n.s. vs ours), always_grounding 0.028 (p_holm 1.1e-5 ***); oracle gold_rerank 0.722 (28/36 in pool); rule selected topk3 30 / grounding 6
- **Accept:** ✓ paper §VII-E + F7 (utility.png); honest framing: attribution avoids misapplied repairs + reviewable evidence; superiority over the dominant single repair NOT established at n=36

### P8 Paper completion — status: [x] DONE (2026-10-22): 7pp, 0 errors/overfulls, 1 intentional `\todo`
- [x] Verify remaining citations via web → ledger `docs/07_CITATION_LEDGER.md` (2026-10-15)
- [x] Sections 1–10 + appendix; all experiment results filled from results/ artifacts (main study, D020, per-fault, baselines+McNemar, exp06, exp07, exp08); abstract merged; analysis (a)–(f); conclusion updated
- [x] Compile with `tools/tectonic.exe` — clean (0 errors/overfulls/citation warnings; **7 pp**, target ~8); fixed 4 hidden overfulls (130pt math set, T1–T3 tables) + T-numbering consistency (T2 rule, T3 injectors, T4 main, T5 per-fault, T6 multimodel); figures F1–F3, F5–F7 embedded (F4 risk-coverage deferred with v2, k-sample todo remains by design)
- **Accept:** ✓ complete `paper/main.pdf` ≤ venue page limit + appendix; zero unverified citations

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
