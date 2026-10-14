# Objectives (synopsis O1-O5 + new O6-O12)

| ID | Objective | Output | Status |
|---|---|---|---|
| O1 | Three-condition Test Runner | `runner/three_condition.py` | DONE |
| O2 | Retrieval, generation, MIRAGE-inspired metrics (+ GaRAGe-style grounding check) | `metrics/*` | TODO |
| O3 | Per-question failure category (rule-based v1) | `attribution/rule.py` | TODO |
| O4 | SEARA-style retriever evaluation without full gold | `retrieval/pseudo_gold.py` | TODO |
| O5 | Interactive dashboard | `app/` | TODO (late) |
| O6 | Formal attribution framework: probabilistic k-sample version, abstention, taxonomy incl. over-summarisation & question/label-error | `attribution/probabilistic.py`, `docs/02` | NEW |
| O7 | Fault-injection benchmark (RAG-FaultBench) giving ground-truth failure source | `faults/*`, `data/faultbench/` | NEW |
| O8 | Baselines: recall-threshold rule, RAGChecker-metric rule, RAGVUE, LLM-as-diagnoser | `attribution/baselines.py` | NEW |
| O9 | Judge reliability: human-annotated subset (>=300 items, 2 annotators), kappa/alpha, judge calibration | `judge/*`, `data/human/` | NEW |
| O10 | Robustness: imperfect gold (drop/replace/partial), judge noise, sampling noise, top-k | `experiments/exp06*` | NEW |
| O11 | Generalization: MIRAGE + GaRAGe + one multi-hop set (e.g., HotpotQA) + one custom domain corpus; >=3 retrievers x >=3 LLMs (incl. open-weight) | `experiments/exp07*` | NEW |
| O12 | Utility study: attribution-guided repair vs random/oracle-component repair | `experiments/exp08*` | NEW |

## Sub-projects (workstreams)
- P1 Core engine (O1-O3) 
- P2 RAG-FaultBench dataset (O7, O9)
- P3 Evaluation study (O8, O10, O11)
- P4 Utility/repair study (O12)
- P5 Tool + dashboard + packaging (O5)
- P6 Paper + artifact release

## Possible outputs
- Main paper (method + validation + utility).
- Optional short demo/system paper for tool (only if main paper avoids tool-centric framing).
- Dataset card for RAG-FaultBench (release with paper).
