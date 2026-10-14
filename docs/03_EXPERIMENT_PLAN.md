# Experiment Plan (pre-register thresholds before running test split)

| Exp | Goal | Design | Metrics |
|---|---|---|---|
| exp01 | Reproduce MIRAGE-style numbers (sanity) | Run O1 on MIRAGE subset with 2 LLMs; compare to published trends | trend agreement, not exact match |
| exp02 | Metric implementation tests | Unit tests vs toy cases; spot-check vs MIRAGE repo | pass/fail |
| exp03 | Fault injection main study | Inject faults with known source (see below); run rule v1/v2; compare to baselines | accuracy, macro-F1, per-class P/R, confusion matrix, risk-coverage |
| exp04 | Baselines | recall-threshold, RAGChecker-metric rule, RAGVUE, LLM-diagnoser (GPT/Claude/open model given q, ctx, answer, ref) | same as exp03; McNemar / bootstrap CI |
| exp05 | Judge reliability | 300+ items, 2 human annotators, kappa; judge vs human agreement; calibration | Cohen kappa, Krippendorff alpha, ECE |
| exp06 | Robustness | gold degraded: 0/25/50/75% passages dropped, replaced by near-miss, pseudo-gold (SEARA-style); noise in judge; k, top-k sweeps | accuracy vs degradation curves |
| exp07 | Generalization | MIRAGE, GaRAGe, HotpotQA-like, custom corpus; 3 retrievers (BM25, dense, hybrid) x 3 LLMs (1 open-weight small, 1 open-weight large, 1 API) | per-dataset macro-F1, mixed-effects model for factors |
| exp08 | Utility | For failures labelled RETRIEVER apply retrieval fix (rerank/hybrid/top-k up); for GENERATOR apply prompt/model fix; compare vs random-fix and always-retriever-fix and always-generator-fix | accuracy gain, paired test |
| exp09 | Human case study / usability (optional) | 5-8 users diagnose with vs without tool | time-to-diagnosis, SUS |

## Fault injection catalogue (ground truth = injected source)
Retriever faults: remove gold passage from index; replace gold with near-miss distractors; truncate chunk so gold span is cut; shrink top-k; use weak embedder.
Generator faults: weak/small LLM with gold present; context-ignoring system prompt; adversarial prompt telling model to rely on memory; context shuffled/long-context dilution (lost-in-the-middle).
Interaction faults: gold present plus high-similarity contradictory distractor.
Control: no fault (expected NONE or genuine failure -> excluded or labelled via human).
Caveat: injected faults may be easier than natural faults; validate on the human-annotated natural-failure set (exp05) and report both.

## Statistics
Bootstrap 95% CI (10k resamples), paired McNemar for classifier comparison, Holm correction, effect sizes. >=3 seeds where sampling matters. Report cost.

## Reproducibility
Release code, configs, FaultBench, prompts, annotation guideline, model versions, cached outputs where license permits.
