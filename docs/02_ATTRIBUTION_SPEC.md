# Attribution Specification

## Notation
For question q: c0, c1, c2 in {0,1} = answer correctness under No-Context, Gold-Context, Real-Retrieval. R = gold recall@k of retrieved set (needs gold ids; pseudo-gold allowed). Judge returns correctness (EM/F1 threshold + LLM judge, see O9).

## Rule v1 (deterministic, temp 0)
| c0 | c1 | c2 | Condition | Label |
|---|---|---|---|---|
| any | any | 1 | success | NONE (flag PARAMETRIC if c0=1) |
| any | 1 | 0 | R < tau_R | RETRIEVER |
| any | 1 | 0 | R >= tau_R | INTERACTION (noise vulnerability: gold present but distracted) |
| 1 | 0 | 0 | knew it, context hurt | GENERATOR (context misinterpretation) |
| 0 | 0 | 0 | fails even with gold | GENERATOR or QUESTION_ISSUE; if gold-audit flags insufficiency -> QUESTION_ISSUE |
| 0 | 0 | 1 | impossible-ish | flag ANOMALY (lucky/judge noise) |
Labels: RETRIEVER, GENERATOR, INTERACTION(Both), NONE, QUESTION_ISSUE, ABSTAIN.
Add GROUNDING flag (over-summarisation / ungrounded claims, GaRAGe-motivated) as an orthogonal tag, not a primary label.

## Rule v2 (probabilistic)
Run k samples (k>=5, temperature>0) per condition. Estimate p_i = mean correctness with Wilson/Beta CI. Decide using CI comparisons:
- RETRIEVER if p1 - p2 > delta with CI excluding 0 and R < tau_R.
- INTERACTION if p1 - p2 > delta and R >= tau_R.
- GENERATOR if p1 < theta.
- NONE if p2 >= theta.
- Otherwise ABSTAIN. Report coverage vs accuracy curve (risk-coverage).
tau_R, delta, theta tuned on a dev split ONLY; test split touched once.

## Rule v3 (optional learned)
Features: (p0,p1,p2,R, MIRAGE metrics, claim recall/faithfulness) -> small classifier (logistic/GBDT). Compare to rules. Check overfitting via cross-dataset transfer.

## Known threats (must be discussed in paper)
- Parametric knowledge inflates c0.
- Gold passage may be insufficient for multi-hop.
- Judge error propagates; calibrate (O9).
- Attribution is evidence-based, not causal proof; note interventions in O12 as the causal check.
