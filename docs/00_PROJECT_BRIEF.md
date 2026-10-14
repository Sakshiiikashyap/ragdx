# Project Brief
**Title (working):** Attributing RAG Failures: A Validated Controlled-Condition Method for Retriever-vs-Generator Diagnosis

## Status
- Objective 1 (Test Runner: No-Context / Gold-Context / Real-Retrieval) is DONE.
- Objectives 2-5 in the synopsis (metrics, attribution, SEARA-style retriever eval, dashboard) are partially specified; this plan hardens them and adds O6-O12.

## Why the original synopsis is not yet a strong paper
- The 3-condition design is MIRAGE's Base/Oracle/Mixed design. Novelty must come from the attribution method + its validation, not the conditions.
- RAGChecker, RAGVUE already give retriever/generator-split diagnostics. We must beat or complement them empirically.
- "Gold context" requires gold passages; custom corpora lack them. We must quantify sensitivity to imperfect gold.
- No ground truth for "failure source" exists => we must construct one (fault injection + human annotation).

## Research questions
- RQ1: Can the pattern of correctness across No/Gold/Real conditions identify the failure source accurately? (attribution accuracy vs injected ground truth)
- RQ2: How does it compare with RAGChecker-style metric rules, RAGVUE, and an LLM-as-diagnoser baseline?
- RQ3: How robust is attribution to judge noise, sampling noise, and imperfect/pseudo gold context?
- RQ4: Does it generalize across datasets, retrievers, and LLM families?
- RQ5: Do attribution-guided fixes improve RAG accuracy more than unguided fixes? (utility)

## Hypotheses
- H1: Rule-based 3-condition attribution beats a retrieval-recall-threshold baseline on macro-F1 under injected faults.
- H2: Probabilistic (k-sample) attribution with abstention improves precision at a controlled coverage.
- H3: Attribution accuracy degrades gracefully (< X points) when gold is replaced by SEARA-style pseudo-gold. (X set after pilot, pre-registered in docs/03.)
- H4: Attribution-guided repair yields larger accuracy gain than random-component repair.

## Contributions to claim (max 4)
1. Formal attribution framework with abstention and uncertainty.
2. RAG-FaultBench: fault-injected benchmark with known failure-source labels (+ small human-annotated set).
3. Empirical comparison vs RAGChecker/RAGVUE/LLM-diagnoser baselines across datasets.
4. Utility study + open-source tool (ragdx) with dashboard.
