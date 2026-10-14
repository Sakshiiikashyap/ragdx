# Claude Code Prompts (paste one at a time; start each session with: "Read CLAUDE.md and docs/*.md first.")

## P0 Scaffold
```
Read CLAUDE.md and docs/. Create the repo layout from CLAUDE.md with empty modules, pyproject.toml, pytest config, a Makefile (targets: test, lint, exp EXP=exp03), and the LLM client with disk cache + cost logging + manifest writer. Do not implement experiments yet. Add tests for cache and manifest. Show me the file tree and test output.
```
## P1 Audit existing Test Runner (O1)
```
Audit src/ragdx/runner/three_condition.py against docs/02_ATTRIBUTION_SPEC.md. Verify it records per question: answers, exact context used, retrieved ids+scores, gold ids, per-condition tokens/cost. List gaps, then fix them with tests. Do not change public function signatures without asking.
```
## P2 Metrics (O2)
```
Implement metrics/: retrieval (recall@k, precision@k, MRR, nDCG), generation (EM, token F1, LLM-judge correctness via judge/judge.py), mirage_metrics (Noise Vulnerability, Context Acceptability, Context Insensitivity, Context Misinterpretation). Fetch the MIRAGE paper/repo and copy the definitions exactly; put the formula and source section in each docstring and in docs/07_CITATION_LEDGER.md. Write unit tests with hand-computed toy cases. Ask me if any definition is unclear.
```
## P3 Attribution rule v1 + v2 (O3, O6)
```
Implement attribution/rule.py exactly per docs/02 (table). Pure function: (c0,c1,c2,R,flags,thresholds)->Label+evidence dict. Then probabilistic.py for rule v2 with Wilson CIs and ABSTAIN. Exhaustive tests over all 8 correctness combos x R above/below tau. Add a CLI `python -m ragdx.attribution run --exp exp03`.
```
## P4 Fault injectors (O7)
```
Implement faults/retriever_faults.py and generator_faults.py per the catalogue in docs/03. Each injector: deterministic given seed, returns modified (index|context|prompt|model) plus a ground-truth label. Build a script that generates RAG-FaultBench from a MIRAGE subset with dev/test split (70/30, stratified by fault type), writes a dataset card (data/faultbench/README.md), and prints class balance. Tests: injector actually changes the intended component only (e.g., removed gold id is absent from retrieved set).
```
## P5 Baselines (O8)
```
Implement attribution/baselines.py: (a) recall-threshold rule, (b) RAGChecker-metric rule (use the official package if installable; map claim recall/context precision/faithfulness/noise sensitivity to labels with documented thresholds tuned on dev), (c) RAGVUE adapter if feasible, (d) LLM-as-diagnoser with a fixed prompt in llm/prompts/diagnoser.md (input q, context, answer, reference; output label+rationale JSON). All baselines get the same dev/test split. Do not tune on test.
```
## P6 Judge reliability (O9)
```
Create judge/calibration.py and an annotation export/import tool: sample N natural failures stratified by condition pattern, export to CSV for 2 annotators with guideline docs/annotation_guideline.md, import and compute Cohen's kappa, Krippendorff's alpha, judge-vs-human agreement, and ECE. Generate a table in results/exp05/.
```
## P7 Experiments + analysis
```
Write experiments/exp03_faultbench.py and exp04_baselines.py driven by configs/*.yaml. Output per-item predictions CSV, then analysis/tables.py to produce: accuracy, macro-F1, per-class P/R, confusion matrix, bootstrap 95% CI, McNemar vs each baseline with Holm correction, risk-coverage plot. Everything reproducible via `make exp EXP=exp03`. Do not run paid API calls without printing an estimated cost and asking me.
```
## P8 Robustness, generalization, utility (O10-O12)
```
Implement exp06 (gold degradation 0/25/50/75%, near-miss replacement, pseudo-gold via retrieval/pseudo_gold.py SEARA-style, judge-noise injection), exp07 (4 datasets x 3 retrievers x 3 LLMs, mixed-effects analysis in analysis/stats.py), and exp08 (attribution-guided repair vs random/always-retriever/always-generator repair; fixes: hybrid retrieval, reranker, larger top-k for retriever; stricter grounding prompt/ larger LLM for generator). Provide a cost estimate first.
```
## P9 Dashboard (O5)
```
Build a read-only Streamlit app over results/: run list, config comparison table, per-question drill-down (3 conditions side by side, retrieved passages with gold highlighted, label + evidence), failure-type distribution, risk-coverage view. No computation in the UI beyond loading CSV/JSON.
```
## P10 Paper generation
```
Using docs/04_PAPER_OUTLINE.md and ONLY numbers in results/, generate paper/tables/*.tex and paper/figures/* via analysis scripts, then draft paper/sections/*.tex section by section. Mark every number with a macro bound to a results file. Insert \todo{} wherever evidence is missing. Never fabricate a citation: use only keys present in refs.bib that are marked verified in docs/07_CITATION_LEDGER.md.
```
## P11 Red-team review
```
Act as a skeptical ACL reviewer. Read paper/ and results/. List the 10 strongest rejection reasons, rank by severity, and for each state what experiment or edit would fix it. Check: injected-vs-natural gap, judge dependence, baseline fairness, test-set leakage in threshold tuning, overclaiming of "causal".
```
