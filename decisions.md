# Decisions (append-only — never edit prior entries)

Format: `## D<n> — <title>` / Date / Decision / Rationale / Alternatives rejected / Impact.

## D001 — Runtime venv reuse
Date: 2026-10-14
Decision: Run everything with `C:\Users\pc\Desktop\MIRAGE-main\.venv\Scripts\python.exe` for now.
Rationale: it already has groq/datasets/sklearn/torch/rich/pandas; installing a fresh venv would duplicate GBs.
Alternatives rejected: fresh venv in repo (slow, disk-heavy); poetry (unavailable).
Impact: `requirements.txt` still maintained for portability; dedicated venv is a cleanup item.

## D002 — ANOMALY precedence in attribution table
Date: 2026-10-14
Decision: For (c0,c1,c2)=(0,0,1) the rule emits ANOMALY, not NONE (specific row beats "any|any|1").
Rationale: docs/02 table lists both; the general row would make ANOMALY unreachable; spec-conflict logged in docs/07.
Alternatives rejected: NONE with flag (loses class for evaluation); leaving unreachable (dead rule).
Impact: rule.py ordering; user confirmation still pending — revisit if user objects.

## D003 — Judge for pilot = EM-loose; LLM-judge is secondary
Date: 2026-10-14
Decision: Primary correctness signal for exp01–exp04 is MIRAGE EM-loose (upstream parity); LLM-as-judge introduced in P1 but used for calibration/robustness, not replacing EM.
Rationale: comparability with MIRAGE numbers; judge errors are an explicit threat (docs/02).
Alternatives rejected: judge-only (unverifiable tonight; needs human calibration).
Impact: paper reports judge-vs-EM agreement instead of human kappa (human study marked future work).

## D004 — Retrieval scope: per-question 5-chunk pool, TF-IDF
Date: 2026-10-14
Decision: "Real retrieval" = TF-IDF top-1 over MIRAGE's per-question doc_pool (as prototype), corpus-wide index deferred.
Rationale: faithful to prototype; corpus-wide dense index is a multi-day detour; pool structure still yields distractors.
Alternatives rejected: sentence-transformers CPU index (slow, unverifiable tonight).
Impact: RETRIEVER label under-exercised (R@k≈1.0) — mitigated by fault injection (remove_gold fault forces R=0).

## D005 — Static HTML dashboard stays; Streamlit deferred
Date: 2026-10-14
Decision: Keep zero-dependency HTML dashboard; viva demo gets a Rich TUI in `viva_demo/` (user request).
Rationale: Streamlit not installed; TUI is more robust for offline viva.
Impact: O5 satisfied by two artifacts; Streamlit remains Phase-5 option.

## D006 — LaTeX toolchain = tectonic MSVC binary in tools/
Date: 2026-10-14
Decision: Vendor `tools/tectonic.exe` (gitignored) for paper builds; MiKTeX rejected.
Rationale: single 30MB binary vs 1.5GB install; gnu build failed (missing DLLs), msvc build works.
Impact: `demo.bat`/`paper` builds self-contained.

## D007 — FaultBench size N=120 (60 dev / 60 test)
Date: 2026-10-14
Decision: Build fault bench on 120 MIRAGE questions, stratified dev/test 50/50.
Rationale: enough per-class support for macro-F1 + McNemar while ~respecting free-tier rate limits and caching.
Alternatives rejected: N=300+ (rate limits, latency); N=50 (too few per class).
Impact: paper reports CIs; power limitations acknowledged.

## D008 — Baseline (b) is "RAGChecker-inspired metric rule", not official package
Date: 2026-10-14
Decision: Implement the metric-rule baseline from documented RAGChecker-style signals computed in-repo; do not depend on their package tonight.
Rationale: install/compat risk on Windows at this hour; the comparison target is the *rule family*, honestly labelled.
Impact: paper wording must say "RAGChecker-inspired"; upgrade path noted.

## D009 — Multi-model = 3 Groq-served families, single dataset
Date: 2026-10-14
Decision: Generalization study = 3 model families on Groq (verify availability live), same MIRAGE dataset; cross-dataset claim left to future work.
Rationale: Groq-only constraint; honest framing beats overclaim.
Impact: RQ4 answered as "multi-model, single-dataset" in limitations.

## D010 — TUI demo isolation
Date: 2026-10-14
Decision: `viva_demo/` is a separate folder with its own run.ps1/run.bat/Makefile; imports ragdx read-only (sys.path add), never modifies core modules.
Rationale: user requirement; demo robustness (core changes can't break the viva).

## D011 � Chunk-level gold identification (BUGFIX, spec clarification)
Date: 2026-10-14
Decision: Gold position within doc_pool = TEXT match of oracle_chunk (fallback: support flag); attribution R = chunk-level recall@k, not doc-level mapped_id recall.
Rationale: verified live that all 5 doc_pool chunks share ONE doc-level mapped_id, so doc-level R is degenerately 1.0 and id-based gold filtering yields zero candidates (broke remove_gold/contradiction injectors, killed INTERACTION class in FaultBench, made exp01's RETRIEVER/INTERACTION split inert). Docs/02 means gold-passage recall.
Alternatives rejected: keep doc-level R (statistically meaningless in-pool); corpus-wide index (out of scope tonight, D004).
Impact: MirageItem gains gold_index + chunk-level ids; exp01 re-run (cheap � LLM answers cached); FaultBench rebuilt; paper numbers updated from re-run; exp01 night-1 numbers superseded.

## D012 - Paper figures via matplotlib
Date: 2026-10-15
Decision: All paper figures generated by src/ragdx/analysis/plots.py (matplotlib 3.10.9 in the shared venv); each fig_* gracefully skips (returns False) when its results/ input is missing.
Rationale: no hand-made figures; every curve traces to a results/ artifact; graceful-skip lets the paper build before all experiments finish.

## D013 - exp06 gold degradation mechanics
Date: 2026-10-15
Decision: Degradation re-runs ONLY the gold condition via dataclasses.replace(task, oracle_chunk=...) (no runner changes). drop_content removes words at evenly spaced mid-point positions (stride = len(words)//(n+1)), NOT truncation. pseudo_gold = first answer-bearing non-gold pool chunk (D011 gold_index excluded); per-level skip counts reported.
Rationale: isolates the gold-witness assumption cheaply (1 call/level/item); stride deletion destroys evidence regardless of text layout; skips kept visible so availability never hides.

## D014 - Judge-noise sweep model
Date: 2026-10-15
Decision: Judge noise flips each condition bit (c0/c1/c2) INDEPENDENTLY with prob p (p in {0,.05,.1,.15,.2,.3}), 200 seeded trials/level; outputs mean±std accuracy and macro-F1.
Rationale: models per-verdict judge error directly on frozen triples; independent flips keep the sweep analysis-only (no re-generation, no re-judging).

## D015 - Frozen baseline reuse across exp06
Date: 2026-10-15
Decision: exp06's frozen baseline curves reuse exp04 recall-threshold predictions (they are gold-independent by construction, so gold degradation cannot move them).
Rationale: avoids re-running a baseline whose inputs the experiment manipulates; honest as long as documented (it is, in eval.json note).

## D016 - P6 extra models + threshold transfer
Date: 2026-10-15
Decision: exp07 reruns the test protocol under openai/gpt-oss-120b and qwen/qwen3.8-27b with the SAME frozen tau_r (pre-registered transfer; no per-model tuning). Availability gated on live Groq /models (skip-and-report, never substitute). weak_model-fault tasks keep the WEAK_MODEL override by definition -> identical across models (cache-shared); noted in eval.json.
Rationale: transfer is the honest multi-model claim; per-model tuning would spend test information; availability gate prevents silent model swaps.

## D017 - P7 repair set + policies
Date: 2026-10-15
Decision: Failure set = exp03 test items with c2==0. Repairs (1 cached call each): topk3 (deployable, TF-IDF top-3 over effective pool), grounding (deployable, strict-grounding system prompt over CURRENT retrieved context), gold_rerank (ORACLE-ASSISTED: gold row forced first when in pool - labeled as such everywhere). Policies compared on the same set: our_fix (frozen-label selection), random_fix (seeded), always_topk3, always_grounding, gold_rerank; sign tests = McNemar exact + Holm.
Rationale: utility claim needs a falsifiable comparator (mismatched repairs must not do as well); oracle repair shown separately so upper bound is never sold as deployable.

## D018 - Paper artifact choices
Date: 2026-10-15
Decision: IEEEtran conference format compiled with tools/tectonic.exe; author byline = repo git identity; bibliography is a hand-written thebibliography with the 8 ledger-verified entries (MIRAGE title corrected to "A Metric-Intensive Benchmark for RAG Evaluation"); pending exp03-exp08 numbers marked \todo{} with their source artifact paths instead of being guessed.
Rationale: viva deadline; CLAUDE.md no-invented-numbers rule; \todo markers make the final fill-in mechanical once chains complete.

## D019 - Judge ECE bugfix (red-team catch, 2026-10-15)
Decision: judge.agreement_report computed per-bin "accuracy" as the RATE OF "correct" VERDICTS (84/150=0.56), not agreement with the EM reference (0.973), inflating ECE to 0.4265 and fueling a wrong "judge is severely overconfident" claim in intent P1 and the paper. Fixed (accuracy = agreement vs EM; key renamed judge_accuracy -> agreement_vs_em); regression test added (balanced 10/10 perfect-agreement case must give ECE=0.01, not 0.49); agreement.json regenerated from cache (free); paper/intent updated: ECE = 0.0131, judge well calibrated against EM with the honest caveat that EM itself is imperfect (3 judge-only-correct rescues) and no human annotation exists.
Rationale: the erroneous number was load-bearing for a paper narrative; the P11 red-team pass caught it before submission. ECE needs a reference; ours is EM, stated explicitly wherever the number appears.
Impact: results/exp01_judge/agreement.json (ece 0.4265 -> 0.0131), paper abstract/6.1/9, intent.md P1 line.

## D020 - Fault activation + failure-subset metric (pre-specified from DEV, 2026-10-16, BEFORE test completion)
Decision: Two complementary views of attribution accuracy, both reported:
(1) PRIMARY (pre-registered, exp03b eval.json): accuracy/macro-F1 over ALL faulted items vs injected-component truth. Unchanged.
(2) SECONDARY (pre-specified HERE from dev evidence, computed post-hoc from stored items by experiments/exp03c_failure_subset.py): accuracy on the FAILURE SUBSET (c2==0) - items where the retrieved-condition answer was actually wrong. Justification: attribution is invoked to explain an observed failure; when the injected fault was benign (answer still correct, c2=1), the rule's NONE ("no failure to attribute") is behaviorally correct but scored against the intervention label.
Dev evidence motivating this split: activation rate = 22/59 (37/59 injections benign at k=1); rule acc = 0.545 on c2==0 vs 0.135 on c2==1; overall 0.254.
Also recorded: the dev tau sweep is DEGENERATE (R is binary in-pool at k=1, so all tau in 0.1..0.9 give identical macro-F1 0.2163; tau_r=0.1 = first argmax). Methods note for the paper; no test information used.
Rationale: without this split the benchmark silently conflates "fault did not activate" with "attribution wrong"; the split is derived on dev only and committed BEFORE test eval exists (pre-registration by commit timestamp).
Impact: results/exp03/eval_subset.json (new, analysis-only); paper Sec VII reports both numbers with the activation-rate statistic; exp08 failure set (c2==0) already matches the secondary definition.
