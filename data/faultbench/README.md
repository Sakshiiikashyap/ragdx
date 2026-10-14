# RAG-FaultBench (v0, pilot)

Fault-injected RAG failure benchmark with KNOWN failure-source labels.

- Source questions: `nlpai-lab/mirage` (first 120 train rows), HF datasets.
- Faults: retriever (remove_gold, weak_topk, truncate_gold), generator
  (ignore_context, adversarial_prompt, weak_model, lost_in_middle),
  interaction (contradiction), plus NONE controls.
- Ground truth = the injected fault's component; natural (uninjected) failures
  are labelled NONE-control and may contain unforced errors (documented caveat).
- Splits: dev (threshold tuning) / test (single use), stratified by truth label,
  fixed seed 42. Class balance in manifest.json.
- Known limitation: injected faults may be easier than natural faults
  (docs/03 caveat); validate on human-annotated natural failures later.
