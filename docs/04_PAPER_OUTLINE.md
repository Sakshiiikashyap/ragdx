# Paper Outline (ACL-style, ~8 pages + appendix)

Target: pick after pilot. Options: a *CL workshop on RAG/evaluation, a *ACL Findings/ARR cycle, ECIR/SIGIR resource-style, or arXiv first. VERIFY deadlines and page limits on official sites; do not trust this file.

**Title**: Attributing RAG Failures to Retriever or Generator: Controlled-Condition Diagnosis with Validated Ground Truth

**Abstract (150-200 words)**: problem; method (3-condition + probabilistic attribution w/ abstention); FaultBench; results (numbers from results/); utility; release.

**1 Introduction (~1 p)**: Failure-source ambiguity; end-to-end accuracy hides it; gap; 4 contributions; RQs.

**2 Related Work (~0.75 p)**: RAG eval benchmarks (MIRAGE, GaRAGe); metric frameworks (RAGChecker, RAGVUE, RAGAS-type); retriever eval w/o gold (SEARA); error analysis/fault localization in LLM pipelines; LLM-as-judge reliability. End with a positioning table.

**3 Problem Formulation (~0.5 p)**: definitions, labels, assumptions, what attribution does/doesn't claim.

**4 Method (~1.5 p)**: 4.1 three-condition protocol; 4.2 metrics; 4.3 rule v1; 4.4 probabilistic attribution + abstention; 4.5 imperfect-gold handling; 4.6 system overview (short; figure of pipeline).

**5 RAG-FaultBench & Annotation (~1 p)**: fault catalogue, construction, stats table, human-annotated natural-failure set, agreement.

**6 Experimental Setup (~0.75 p)**: datasets, retrievers, LLMs, judge, baselines, metrics, stats, compute/cost.

**7 Results (~2 p)**: 7.1 attribution accuracy vs baselines (main table); 7.2 confusion matrix & per-fault analysis; 7.3 risk-coverage curve; 7.4 robustness to gold/judge noise; 7.5 cross-dataset/config generalization; 7.6 utility study.

**8 Analysis & Discussion (~0.5 p)**: when it fails (parametric leakage, multi-hop gold insufficiency), error taxonomy, cost.

**9 Limitations & Ethics (~0.3 p)**: injected-vs-natural gap, judge dependence, English-only, API model drift.

**10 Conclusion (~0.2 p)**

**Appendix**: prompts; judge/annotation guideline; full tables; hyperparameters (tau_R, delta, theta); fault injector details; dashboard screenshots; reproducibility checklist.

## Figures/Tables (planned)
F1 pipeline; F2 attribution decision flow; F3 confusion matrices; F4 risk-coverage; F5 degradation curves; F6 utility bars.
T1 positioning vs related work; T2 FaultBench stats; T3 main results; T4 robustness; T5 generalization; T6 utility.
