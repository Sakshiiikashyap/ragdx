# Phases & Checklist (about 10-12 weeks, adjust)

## Phase 0 - Hygiene (wk 0-1)
- [ ] Verify all 5 references and add 10-15 more (judge reliability, fault localization, lost-in-the-middle, RAGAS) in `paper/refs.bib`; fill `docs/07_CITATION_LEDGER.md`
- [ ] Inspect existing competitors (RAGChecker, RAGVUE, ragDx repo) and write 1-page gap note
- [ ] Freeze RQs/hypotheses; commit pre-registration of thresholds procedure
## Phase 1 - Core (wk 1-3)
- [ ] O1 audit + tests; add caching/manifest
- [ ] O2 metrics + tests
- [ ] O3 rule v1 + tests
## Phase 2 - Ground truth (wk 3-5)
- [ ] O7 fault injectors + FaultBench generation + dev/test split
- [ ] O9 annotation guideline, sample natural failures, 2 annotators, kappa
## Phase 3 - Study (wk 5-8)
- [ ] O8 baselines; exp03/exp04
- [ ] O6 probabilistic version; risk-coverage
- [ ] O10 robustness; O4 pseudo-gold
## Phase 4 - Generalize + utility (wk 8-10)
- [ ] O11 exp07; O12 exp08
## Phase 5 - Tool + paper (wk 9-12)
- [ ] O5 dashboard (read-only over results/)
- [ ] Write paper sections from `docs/04`; auto-generate tables from results
- [ ] Internal review with supervisor; arXiv; submit
## Definition of done per experiment
Script + config + manifest + results CSV + table/figure script + 5-line interpretation in `results/<exp>/NOTES.md`.
