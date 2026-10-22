# Red-Team Review (P11) — paper/main.tex draft

Date: 2026-10-15. Stance: skeptical ACL reviewer. Scope: full draft as of
commit b30f351+ (results sections still `\todo`-marked pending exp03–exp08
completion). Ranked by severity; each item lists the fix and its status.

**2026-10-22 re-check (final state, 7 pp):** all results sections are now
filled from `results/*/eval.json` (main study, D020 decomposition,
baselines+McNemar, exp06 robustness incl. the honest non-monotone-aggregate
explanation, exp07 multi-model κ≈0.83, exp08 utility incl. the honest
"always-topk3 is competitive, n.s." result). Item 1 is resolved. Layout:
0 overfulls (4 were fixed: 130pt math-set, T1–T3 tables); T-numbering made
self-consistent (T1–T6, F1–F3, F5–F7; F4 deferred with rule v2). The paper
reports every negative/null result (no baseline separates; matched repair
does not beat the dominant single repair) — the residual viva risks below
are unchanged and are now answerable with data rather than promises.

## Ranked rejection reasons

1. **No primary results in the draft (fatal at submission).**
   Main-study, robustness, multi-model, and utility sections are `\todo`
   stubs naming their source artifacts. → FIXED MECHANICALLY once
   run_phase4_chain + run_phase567_chain complete; every number is a
   transcription from `results/*/eval.json` via committed scripts.

2. **Benchmark circularity**: the injectors *define* the truth labels and the
   rule's conditions resemble the injector semantics; a reviewer can claim
   the rule is validated on faults it was designed for.
   → FIXED IN TEXT (§V): the rule never sees fault identity, only
   (c0,c1,c2,R); supported by the honest blind-spot results (§VII-C) where
   the rule *fails* systematically — a designed-for benchmark would not
   expose those failures.

3. **Injected-vs-natural gap**: system-prompt "ignore context" faults are
   artificial. → FIXED IN TEXT (§V framing as validation instrument,
   §IX limitation, §VII-E repair study as behavioral evidence beyond
   injection). Not fixable in scope tonight; stated, not hidden.

4. **Correctness-signal dependence**: EM-loose misses paraphrases; the judge
   is calibrated only against EM (no human annotator). D019 additionally
   found and fixed the ECE miscalculation (0.4265 → 0.0131) that had
   produced an untrue "judge overconfident" claim.
   → FIXED IN TEXT (§VI-A, §IX) + quantified by the judge-noise sweep
   (§VII-C): label flips under verdict noise are measured, not assumed.

5. **Baseline fairness**: is the LLM-diagnoser under-informed (it lacks
   c0/c1/c2 and R) or over-informed (it reads the reference)?
   → FIXED IN TEXT (§VI): explicit information-parity statement (rule sees
   outcomes, diagnoser sees texts, neither sees fault identity or dev
   labels) + prompt reproduced in the appendix.

6. **Test-set leakage**: exp07/exp08 re-analyze test records.
   → FIXED IN TEXT (§VI): re-analysis without parameter tuning; only
   τ_R and γ are tuned (dev, frozen pre-test); repairs are fixed heuristics.

7. **Small n, single corpus, lexical retriever**: 61 test items, TF-IDF
   pools, one dataset → wide CIs, limited external validity.
   → FIXED IN TEXT (bootstrap CIs everywhere, §IX; multi-model transfer as
   partial mitigation). Scope change would require new data collection.

8. **INTERACTION underpowered** (7 dev / 8 test instances) + top-1
   observability argument means natural interaction failures cannot arise
   in the pilot. → FIXED IN TEXT (descriptive-only caveat §VI; structural
   argument reported as a finding §VI-B; bench supplies the fault).

9. **Rule v2 / risk–coverage unevaluated** (k-sampling deferred for API
   cost). → PENDING/DEFERRED: marked `\todo` + explicit limitation §IX;
   not dropped silently (intent P6b contingency honored).

10. **Reproducibility under model drift**: API-hosted generators change.
    → FIXED IN TEXT (§IX): manifests pin model/date/git-hash; committed
    cache replays all calls deterministically; 130 offline tests.

## Applied edits this pass
- §VI-A judge paragraph rewritten with corrected ECE 0.0131 (D019).
- Abstract/§II/§IX overconfidence claims removed/replaced.
- §V non-circularity sentence; §VI information-parity + test-reuse
  statements + INTERACTION caveat; appendix diagnoser prompt reproduction.
- Recompile: clean (0 undefined citations/refs, no overfull >10pt).

## Residual risks for the viva
- If exp03 test accuracy lands low, frame as "validated measurement
  apparatus exposing a real blind spot" (analysis §VIII) — never as a
  solved problem.
- Utility study: present gold_rerank as an *oracle ceiling*, never as a
  deployable fix.
