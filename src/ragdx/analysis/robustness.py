"""P5 assembly helpers: per-level attribution reports and the exp06 eval
document (schema consumed by analysis/plots.fig_degradation). Fully offline:
works over recorded items, no API."""

from __future__ import annotations

from ragdx.analysis.noise import judge_noise_sweep
from ragdx.analysis.stats import macro_f1

DEFAULT_FLIP_PS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]


def _acc(truth: list[str], preds: list[str]) -> float:
    return round(sum(t == p for t, p in zip(truth, preds)) / len(truth), 4) \
        if truth else 0.0


def level_report(records: list[dict], level: str) -> dict | None:
    """Attribution accuracy + macro-F1 of rule v1 on one degradation level."""
    recs = [r for r in records if r.get("level") == level]
    if not recs:
        return None
    truth = [r["ground_truth"] for r in recs]
    preds = [r["pred_label"] for r in recs]
    return {"n": len(recs), "accuracy": _acc(truth, preds),
            "macro_f1": round(macro_f1(truth, preds), 4)}


def build_eval(records: list[dict], levels: list[str], tau_r: float,
               n_tasks: int, skips: dict[str, int],
               baseline_name: str | None = None,
               baseline_preds: dict[str, str] | None = None,
               flip_ps: list[float] | None = None,
               n_rep: int = 200, seed: int = 0) -> dict:
    """Assemble results/exp06/eval.json.

    Baselines are gold-independent: for each level we score the baseline's
    per-query predictions (frozen from exp04) over the records present at
    that level, so any variation across levels comes only from skips —
    never from the gold reference itself.
    """
    clean = [r for r in records if r.get("level") == "clean"]
    by_system: dict[str, dict] = {"rule_v1": {}}
    for lv in levels:
        rep = level_report(records, lv)
        if rep:
            by_system["rule_v1"][lv] = rep

    if baseline_name and baseline_preds:
        base: dict[str, dict] = {}
        for lv in levels:
            recs = [r for r in records if r.get("level") == lv]
            if not recs:
                continue
            truth = [r["ground_truth"] for r in recs]
            preds = [baseline_preds[r["query_id"]] for r in recs]
            base[lv] = {"n": len(recs), "accuracy": _acc(truth, preds),
                        "macro_f1": round(macro_f1(truth, preds), 4)}
        by_system[baseline_name] = base

    c1_rate = {}
    for lv in levels:
        recs = [r for r in records if r.get("level") == lv]
        if recs:
            c1_rate[lv] = round(sum(r["c1"] for r in recs) / len(recs), 4)

    # rule accuracy per ground-truth class x level (why the curve moves)
    by_fault_class: dict[str, dict] = {}
    for r in records:
        cls = r["ground_truth"]
        by_fault_class.setdefault(cls, {}).setdefault(r["level"], []).append(
            int(r["pred_label"] == cls))
    by_fault_class = {cls: {lv: round(sum(v) / len(v), 4)
                            for lv, v in per_lv.items()}
                      for cls, per_lv in by_fault_class.items()}

    noise = judge_noise_sweep(clean, tau_r=tau_r,
                              flip_ps=flip_ps or DEFAULT_FLIP_PS,
                              n_rep=n_rep, seed=seed) if clean else []

    return {"n_tasks": n_tasks, "tau_r": tau_r, "conditions": levels,
            "by_system": by_system, "best_baseline": baseline_name,
            "c1_rate": c1_rate, "by_fault_class": by_fault_class,
            "skips": skips, "judge_noise": noise,
            "n_records": len(records)}
