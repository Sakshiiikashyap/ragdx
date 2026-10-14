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
- [x] Judge calibration vs EM-loose/strict on exp01's 50 items: **agreement 0.9733** (146/150), 0 unparseable; 3 judge-correct/EM-wrong (EM substring false negatives), 1 reverse; ECE proxy 0.43 — judge overconfident (all confidences ~0.99); honest reporting: judge is a rescue mechanism for EM false negatives, not a calibrated probability source
- [x] Human-annotation CSV: DEFERRED (no second annotator available tonight; marked future work in paper)
- [x] Tests: judge parsing (valid/invalid JSON), agreement metrics — 42→57 tests total

### P2 Fault injectors + RAG-FaultBench (O7) — status: [x] DONE (2026-10-14)
- [x] Retriever: remove_gold, weak_topk, truncate_gold; Generator: ignore_context, adversarial_prompt, weak_model, lost_in_middle; Interaction: contradiction; controls via fallbacks
- [x] Each injector deterministic; tests verify only-intended-component changes (17 tests)
- [x] **D011 bugfix:** chunk-level gold identification (text match) — doc-level ids are degenerate in MIRAGE pools
- [x] FaultBench: 120 questions → dev 59 / test 61, stratified (dev: RET 22/GEN 30/INT 7; test: RET 23/GEN 30/INT 8), card + manifest
- **Cost actual:** ~0 API calls (injectors pure); bench run cost counted in P4

### P3 Attribution v2 + baselines (O6, O8) — status: [ ] TODO
- [ ] `attribution/probabilistic.py`: k-sample Wilson-CI rule with ABSTAIN (k=5, T>0) on dev only for pilot cost; risk-coverage curve
- [ ] `attribution/baselines.py`: (a) recall-threshold rule; (b) RAGChecker-style metric rule (claim recall/faithfulness proxies computed from existing metrics — documented as "RAGChecker-inspired", not official package, pending install check); (c) LLM-as-diagnoser (prompt `llm/prompts/diagnoser.md`, same info as our method minus conditions)
- [ ] Tune rule-v1 τ_R and baseline thresholds on **dev only**; freeze; test once
- **Accept:** baseline predictions on test CSV; thresholds recorded in configs
- **Cost:** diagnoser ~120 calls; v2 k-sampling deferred to dev subset (30q×5×3=450 calls) — run if rate limit allows, else mark `\todo`

### P4 Main study exp03+exp04 (RQ1, RQ2) — status: [ ] TODO
- [ ] `experiments/exp03_faultbench.py`: run rule v1 (+v2 if done) on test split → accuracy, macro-F1, per-class P/R, confusion matrix, bootstrap 95% CI
- [ ] `experiments/exp04_baselines.py`: same metrics for all baselines + McNemar vs ours (Holm-corrected)
- [ ] `analysis/stats.py` (bootstrap, McNemar), `analysis/plots.py` (matplotlib → `paper/figures/`)
- **Accept:** `results/exp03`, `results/exp04` with per-item CSVs + tables/figures; NOTES.md 5-line interpretation each
- **Cost:** runs reuse P2 cache (attribution is post-hoc on cached generations)

### P5 Robustness exp06 (RQ3-lite) — status: [ ] TODO
- [ ] Gold degradation: drop 25/50/75% of gold chunk content; near-miss replacement; SEARA-style pseudo-gold (answer-containing chunk from pool as surrogate gold)
- [ ] Judge-noise injection (flip probability sweep on labels, no API cost)
- [ ] Accuracy-vs-degradation curves for ours + best baseline
- **Accept:** degradation table + figure; robustness paragraph in paper
- **Cost:** 60 test q × 3 levels ≈ 180 calls

### P6 Multi-model exp07-lite (RQ4-lite) — status: [ ] TODO
- [ ] Query Groq /models at runtime; pick 3 distinct families available on free tier (current: openai/gpt-oss-20b, llama-3.3-70b-versatile, qwen/qwen3-32b — verify live)
- [ ] Rerun exp03 pipeline per model on test split; per-model macro-F1; attribution stability (label agreement across models)
- **Accept:** generalization table (3 models); honest "single dataset, multi-model" framing in paper
- **Cost:** 2 extra models × 60 q × 3 cond ≈ 360 calls

### P7 Utility exp08-lite (RQ5-lite) — status: [ ] TODO
- [ ] For RETRIEVER-labelled failures: apply top-k 1→3 + gold-in-pool rerank; for GENERATOR-labelled: stricter grounding prompt; compare vs random-fix and always-one-component baselines on the same failure set
- [ ] Paired gain + sign test
- **Accept:** utility bar chart + paragraph; honest "heuristic fixes" framing
- **Cost:** ≤ 150 calls

### P8 Paper completion — status: [ ] TODO
- [ ] Verify remaining citations (RAGChecker NeurIPS 2024 D&B, GaRAGe ACL 2025 Findings, RAGAS EACL 2024, lost-in-the-middle, Zheng et al. MT-Bench) via web → ledger
- [ ] Sections 1–10 per `docs/04_PAPER_OUTLINE.md` with ONLY results/ numbers; positioning table; figures F1–F6; tables T1–T6 (as evidence permits)
- [ ] Compile with `tools/tectonic.exe`; red-team self-review (P11 prompt) → fix pass
- **Accept:** complete `paper/main.pdf` ≤ venue page limit + appendix; zero unverified citations

### P9 Viva TUI demo (`viva_demo/`) — status: [ ] TODO
- [ ] Separate folder, own launcher (`run.ps1`, `run.bat`, `Makefile` target `tui`), zero changes to core repo
- [ ] Rich-based: KPI dashboard, question browser (3 conditions + evidence), confusion matrix, risk-coverage, live "diagnose a question" mode (uses cache), paper/results explorer
- [ ] Works offline from `results/` + `.llm_cache/`
- **Accept:** `viva_demo/run.bat` launches; graceful when results missing

### P6b/P7b contingency
If rate limits or time block P5–P7: mark affected paper sections `\todo` with explicit
"future work" framing — never drop planned experiments silently.

---

## Execution discipline
- One part per commit batch (`phase-N: ...`); append every decision to `decisions.md`
- After each part: `pytest` green, commit, update this file's checkboxes
- Paper regenerated from scripts after every results-producing part
