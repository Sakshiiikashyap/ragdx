# Citation Ledger (verify each before use)
| Key | Source (from synopsis) | Verified? | URL checked | Notes |
|---|---|---|---|---|
| park2025mirage | Park et al., MIRAGE, Findings NAACL 2025 | [x] 2026-10-14 | arxiv 2504.17137 + github.com/nlpai-lab/MIRAGE | Metric definitions (EM-loose/strict, token-F1, NV/CA/CI/CM + elif priority order) copied exactly from upstream LLM.py / evaluation.py into src/ragdx/metrics/; docstrings cite source file+method |
| murugaraj2026ragvue | RAGVUE, EACL 2026 demos | [x] 2026-10-14 | aclanthology.org/2026.eacl-demo.35 | CONFIRMED: EACL 2026 System Demonstrations, pp. 512–526, Rabat, DOI 10.18653/v1/2026.eacl-demo.35; arxiv 2601.04196 |
| sorodoc2025garage | GaRAGe, Findings ACL 2025 | [ ] | | TODO: verify author/venue (Phase 0 remaining) |
| ru2024ragchecker | RAGChecker, NeurIPS 2024 D&B | [ ] | arxiv 2408.08067 | TODO: confirm venue + authors (Phase 0 remaining) |
| zou2025seara | SEARA, EMNLP 2025 Industry | [x] 2026-10-14 | 2025.emnlp.org/program/ind_papers | CONFIRMED: "SEARA: An Automated Approach for Obtaining Optimal Retrievers", Zou Yuheng, EMNLP 2025 Industry Track |
| sahu2026mirage_gen | MiRAGE (Sahu et al.) from concept PDF | [ ] | | NOT in synopsis; verify exists or drop — do not cite until confirmed |

To add (Phase 0 remaining): RAGAS, ARES, TruLens, lost-in-the-middle (Liu et al., 2023), LLM-as-judge reliability (MT-Bench/Zheng et al.), Self-RAG/CRAG (for repair baselines).

## Spec-conflict log (per CLAUDE.md rule 5/8)
- 2026-10-14 docs/02 table: row "any|any|1 -> NONE" overlaps row "0|0|1 -> ANOMALY". Implemented the specific row (0_0_1 -> ANOMALY) so ANOMALY is reachable; general row covers other c2=1 patterns. PENDING user confirmation; proposed diff: add note to docs/02 that 0_0_1 takes precedence.
