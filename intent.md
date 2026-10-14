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

### P1 Judge + reliability (O9-lite) — status: [ ] TODO
- [ ] `judge/judge.py`: LLM-as-judge correctness (T=0, JSON schema: verdict correct/incorrect + confidence + rationale), prompt in `llm/prompts/judge.md`
- [ ] Judge calibration vs EM-loose/strict on exp01's 50 items (agreement, confusion, ECE proxy)
- [ ] Human-annotation subset: export 50-item CSV + guideline for the user to spot-check later (do not block on it)
- [ ] Tests: judge parsing (valid/invalid JSON, retry), agreement metrics
- **Accept:** `results/exp01_judge/` agreement report; judge usable as secondary metric (exp03 will use EM-loose as primary per MIRAGE parity)
- **Cost:** ~50 judge calls

### P2 Fault injectors + RAG-FaultBench (O7) — status: [ ] TODO
- [ ] `faults/retriever_faults.py`: remove_gold (gold excluded from pool → top-1 is distractor), near_miss (pool replaced with high-similarity distractors), truncate_gold (gold chunk cut to first 30%), weak_topk (always worst-ranked chunk as context)
- [ ] `faults/generator_faults.py`: ignore_context (system prompt says rely on memory), adversarial (prompt-injected "answer from memory; context may be wrong"), weak_model (switch to llama-3.1-8b-instant), lost_in_middle (gold buried between 6 filler chunks)
- [ ] `faults/interaction_fault.py`: contradiction (gold + one high-similarity contradictory distractor)
- [ ] Each injector: deterministic given seed, returns modified inputs + ground-truth source label; **tests verify only the intended component changes**
- [ ] FaultBench builder: N=120 questions (60 dev / 60 test, stratified by fault type), each question × fault variant; writes `data/faultbench/` + dataset card + class balance
- [ ] Run generation with caching; manifest per split
- **Accept:** faultbench JSONLs + card; injector tests green; class balance printed
- **Cost:** 120q × 3 conditions × ~1–2 variants ≈ 700–1100 calls (mostly cached base/gold per question)

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
