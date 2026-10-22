"""exp03_v2: k-sample data collection for probabilistic rule v2 and its
frozen-parameter test evaluation with abstention + risk-coverage (F4).

PRE-REGISTERED as D024 (2026-10-22) before any v2 test evaluation exists:
  k=5 samples per condition at T=0.7 (non-degenerate sampling; T=0 would
  make k samples identical). EM-loose per sample, no judge rescue (rates
  are EM estimates; judge x5 would triple cost and the judge is itself
  calibrated to EM). tau_R stays frozen at 0.1 (v1 dev tuning, reused).
  delta/theta grid-tuned on DEV only:
    delta in {0.0, 0.1, 0.2, 0.3, 0.4}, theta in {0.3, 0.4, 0.5, 0.6, 0.7}
  objective (lexicographic, pre-registered): dev selective accuracy on
  decided items -> coverage -> macro-F1 over decided. Test touched once.

Condition prompts are byte-identical to the exp03b protocol (fault styles
included); retrieval is deterministic TF-IDF so R is recomputed offline.

Outputs: results/exp03_v2/{dev_items,test_items,eval,risk_coverage,manifest}.json
         configs/frozen_thresholds_v2.json (dev-tuned, written at runtime)
Resume: each split's items file is rewritten after every finished question;
reruns skip already-collected query_ids (server-restart safe).

Usage: python experiments/exp03v2_ksample.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragdx.attribution.probabilistic import risk_coverage, rule_v2
from ragdx.attribution.rule import AttributionThresholds
from ragdx.faults.loader import load_rows, rows_to_tasks
from ragdx.faults.runner import _prompt_messages, build_retrieved_context
from ragdx.llm.client import LLMClient
from ragdx.metrics import generation as gen
from ragdx.metrics.retrieval import recall_at_k
from ragdx.retrieval import TfidfRetriever

ROOT = Path(__file__).resolve().parents[1]
K = 5
TEMPERATURE = 0.7
DELTAS = [0.0, 0.1, 0.2, 0.3, 0.4]
THETAS = [0.3, 0.4, 0.5, 0.6, 0.7]


def _git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def collect(split: str, tasks, client: LLMClient) -> list[dict]:
    od = ROOT / "results" / "exp03_v2"
    od.mkdir(parents=True, exist_ok=True)
    out = od / f"{split}_items.json"
    records = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {r["query_id"] for r in records}
    todo = [t for t in tasks if t.item.query_id not in done]
    print(f"exp03_v2 {split}: {len(todo)}/{len(tasks)} to collect "
          f"(k={K}, T={TEMPERATURE})")
    retriever = TfidfRetriever()
    for i, task in enumerate(todo):
        item = task.item
        counts, texts = [0, 0, 0], [[], [], []]
        ret_ctx, ret_ids, _ = build_retrieved_context(task, retriever, 1)
        r = recall_at_k(ret_ids, item.gold_chunk_ids, max(1, len(ret_ids)))
        contexts = [None, task.effective_oracle(), ret_ctx]
        for ci, ctx in enumerate(contexts):
            msgs = _prompt_messages(task, item.query, ctx)
            for _ in range(K):
                resp = client.chat(msgs)
                texts[ci].append(resp.text)
                if gen.em_loose(item.answers, resp.text) >= 1.0:
                    counts[ci] += 1
        records.append({"query_id": item.query_id, "fault": task.fault,
                        "ground_truth": task.ground_truth,
                        "counts": counts, "ns": [K, K, K], "R": r,
                        "texts_sample": [t[0] for t in texts]})
        out.write_text(json.dumps(records, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"[{i+1}/{len(todo)}] {item.query_id[:8]} {task.fault:18s} "
              f"counts={counts} R={r:.2f}")
    return records


def metrics(records: list[dict], delta: float, theta: float,
            tau_r: float) -> tuple[dict, list[dict]]:
    th = AttributionThresholds(tau_r=tau_r, delta=delta, theta=theta)
    rows, pts_conf = [], []
    for rec in records:
        res = rule_v2(tuple(rec["counts"]), tuple(rec["ns"]), rec["R"], th)
        ev = res.evidence
        ok = res.label.value == rec["ground_truth"]
        decided = res.coverage == 1.0
        if res.label.value in ("RETRIEVER", "INTERACTION"):
            conf = ev["diff_ci"][0]
        elif res.label.value == "GENERATOR":
            conf = theta - ev["p1"]
        elif res.label.value == "NONE":
            conf = ev["p2"] - theta
        else:
            conf = -1.0
        rows.append({"query_id": rec["query_id"], "label": res.label.value,
                     "decided": decided, "correct": ok, "conf": conf,
                     "ground_truth": rec["ground_truth"]})
        pts_conf.append((conf, ok))
    n = len(rows)
    dec = [r for r in rows if r["decided"]]
    cov = len(dec) / n if n else 0.0
    sel = sum(r["correct"] for r in dec) / len(dec) if dec else 0.0
    labels = sorted({r["label"] for r in dec} | {r["ground_truth"] for r in dec})
    f1s = []
    for lab in labels:
        tp = sum(1 for r in dec if r["label"] == lab and r["ground_truth"] == lab)
        fp = sum(1 for r in dec if r["label"] == lab and r["ground_truth"] != lab)
        fn = sum(1 for r in dec if r["label"] != lab and r["ground_truth"] == lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec_ = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0)
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    rep = {"n": n, "decided": len(dec), "abstains": n - len(dec),
           "coverage": round(cov, 4), "selective_accuracy": round(sel, 4),
           "overall_accuracy": round(sum(r["correct"] for r in dec) / n, 4)
           if n else 0.0, "macro_f1_decided": round(macro, 4)}
    curve = [{"coverage": round(c, 4), "accuracy": round(a, 4)}
             for c, a in risk_coverage([p[0] for p in pts_conf],
                                       [p[1] for p in pts_conf])]
    return rep, curve


def main() -> None:
    frozen = json.loads((ROOT / "configs" / "frozen_thresholds.json")
                        .read_text(encoding="utf-8"))
    tau = frozen["tau_r"]
    dev = rows_to_tasks(load_rows(ROOT / "data" / "faultbench" / "dev.jsonl"))
    test = rows_to_tasks(load_rows(ROOT / "data" / "faultbench" / "test.jsonl"))
    client = LLMClient(model="openai/gpt-oss-20b", cache_dir=ROOT / ".llm_cache",
                       log_path=ROOT / "results" / "_llm_calls.jsonl",
                       temperature=TEMPERATURE)
    t0 = time.time()

    dev_recs = collect("dev", dev, client)
    grid = []
    for d in DELTAS:
        for t in THETAS:
            rep, _ = metrics(dev_recs, d, t, tau)
            grid.append({"delta": d, "theta": t, **rep})
    best = max(grid, key=lambda g: (g["selective_accuracy"], g["coverage"],
                                    g["macro_f1_decided"]))
    d_star, t_star = best["delta"], best["theta"]
    (ROOT / "configs" / "frozen_thresholds_v2.json").write_text(json.dumps(
        {"tau_r": tau, "delta": d_star, "theta": t_star, "k": K,
         "temperature": TEMPERATURE, "tuned_on": "dev",
         "objective": "selective acc -> coverage -> macro-F1"}, indent=2),
        encoding="utf-8")
    print(f"dev-tuned: delta*={d_star} theta*={t_star} "
          f"(dev selective acc {best['selective_accuracy']}, "
          f"coverage {best['coverage']})")

    test_recs = collect("test", test, client)
    th = AttributionThresholds(tau_r=tau, delta=d_star, theta=t_star)
    rep, curve = metrics(test_recs, d_star, t_star, tau)

    e3_items = json.loads((ROOT / "results" / "exp03" / "items.json")
                          .read_text(encoding="utf-8"))
    v1 = {r["query_id"]: r for r in e3_items}
    fail_ids = {r["query_id"] for r in e3_items if r["c2"] == 0}
    test_rows = []
    for rec in test_recs:
        res = rule_v2(tuple(rec["counts"]), tuple(rec["ns"]), rec["R"], th)
        test_rows.append((res.coverage == 1.0,
                          res.label.value == rec["ground_truth"],
                          rec["query_id"]))
    dec_ids = {q for dcd, _, q in test_rows if dcd}
    f_sub = [(dcd, ok, q) for dcd, ok, q in test_rows if q in fail_ids]
    f_dec = [x for x in f_sub if x[0]]
    v1_on_dec = sum(1 for q in dec_ids
                    if v1[q]["pred_label_frozen"] == v1[q]["ground_truth"])
    eval_doc = {
        "tau_r": tau, "delta": d_star, "theta": t_star, "k": K,
        "temperature": TEMPERATURE,
        "dev": best, "test": rep,
        "test_failure_subset": {"n": len(f_sub), "decided": len(f_dec),
            "coverage": round(len(f_dec) / len(f_sub), 4) if f_sub else 0.0,
            "selective_accuracy": round(sum(ok for _, ok, _ in f_dec)
                                        / len(f_dec), 4) if f_dec else 0.0},
        "v1_comparison": {"v1_overall_acc":
            round(sum(1 for r in e3_items
                      if r["pred_label_frozen"] == r["ground_truth"])
                  / len(e3_items), 4),
            "v1_acc_on_v2_decided": round(v1_on_dec / len(dec_ids), 4)
            if dec_ids else 0.0,
            "v2_overall_acc": rep["overall_accuracy"]},
        "grid_dev": grid,
        "note": "k-sample EM rates at T=0.7, no judge; dev-only tuning; "
                "confidence: RETRIEVER/INTERACTION = Wald-diff lower bound, "
                "GENERATOR = theta-p1, NONE = p2-theta, ABSTAIN = -1"}
    od = ROOT / "results" / "exp03_v2"
    (od / "eval.json").write_text(json.dumps(eval_doc, indent=2),
                                  encoding="utf-8")
    (od / "risk_coverage.json").write_text(json.dumps(curve, indent=2),
                                           encoding="utf-8")
    (od / "manifest.json").write_text(json.dumps({
        "exp_id": "exp03_v2", "k": K, "temperature": TEMPERATURE,
        "n_dev": len(dev_recs), "n_test": len(test_recs),
        "delta": d_star, "theta": t_star, "tau_r": tau,
        "model": client.model, "git_hash": _git_hash(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_s": round(time.time() - t0, 1)}, indent=2),
        encoding="utf-8")
    print(json.dumps({k: eval_doc[k] for k in ("dev", "test",
          "test_failure_subset", "v1_comparison")}, indent=1))
    print(f"wrote {od / 'eval.json'}")


if __name__ == "__main__":
    main()
