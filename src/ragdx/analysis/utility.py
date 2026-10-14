"""P7 repair-utility assembly (fully offline): policy gains on the failure set
plus paired sign tests (McNemar exact + Holm) of our fix vs each baseline.
Schema consumed by analysis/plots.fig_utility. Honest framing inputs: repairs
are heuristics, gold_rerank is oracle-assisted and labelled as such."""

from __future__ import annotations

import random

from ragdx.analysis.stats import holm_correction, mcnemar_exact

# what each policy applies on a failure item
POLICIES = ["our_fix", "random_fix", "always_topk3", "always_grounding",
            "gold_rerank"]


def select_repair(pred_label: str) -> str:
    """Repair triggered by OUR attribution label (deployment-side; no oracle)."""
    if pred_label == "RETRIEVER":
        return "topk3"
    if pred_label == "GENERATOR":
        return "grounding"
    return "no_op"  # INTERACTION / NONE / ABSTAIN: no automated repair


def _random_choice(query_id: str, seed: int) -> str:
    rng = random.Random(f"{seed}:{query_id}")
    return rng.choice(["topk3", "grounding"])


def build_utility_eval(rows: list[dict], seed: int = 0) -> dict:
    """rows: [{"query_id", "pred_label", "fixed": {topk3, gold_rerank?,
    grounding}}] over the failure set (every item failed before repair, so
    gains = post-repair accuracy on that set)."""
    if not rows:
        raise ValueError("empty failure set")
    ours, rnd, top3, ground, rerank = [], [], [], [], []
    for r in rows:
        fixed = r["fixed"]
        ours.append(float(fixed.get(select_repair(r["pred_label"]), 0.0)))
        rnd.append(float(fixed.get(_random_choice(r["query_id"], seed), 0.0)))
        top3.append(float(fixed.get("topk3", 0.0)))
        ground.append(float(fixed.get("grounding", 0.0)))
        rerank.append(float(fixed.get("gold_rerank", 0.0)))

    def gain(v: list[float]) -> float:
        return round(sum(v) / len(v), 4)

    fixes = [{"name": n, "gain": g} for n, g in zip(
        POLICIES, [gain(ours), gain(rnd), gain(top3), gain(ground),
                   gain(rerank)])]

    # paired sign tests: ours vs each baseline (discordant pairs)
    mcn = {}
    for name, other in [("random_fix", rnd), ("always_topk3", top3),
                        ("always_grounding", ground)]:
        b = sum(1 for o, x in zip(ours, other) if o == 1 and x == 0)
        c = sum(1 for o, x in zip(ours, other) if o == 0 and x == 1)
        mcn[name] = {"b_ours_only": b, "c_baseline_only": c,
                     "p_raw": mcnemar_exact(b, c)}
    adjusted = holm_correction({k: v["p_raw"] for k, v in mcn.items()})
    for k in mcn:
        mcn[k]["p_holm"] = adjusted[k]
        mcn[k]["p_raw"] = round(mcn[k]["p_raw"], 6)
        mcn[k]["p_holm"] = round(mcn[k]["p_holm"], 6)

    from collections import Counter
    sel = Counter(select_repair(r["pred_label"]) for r in rows)
    return {"n_failure": len(rows), "fixes": fixes,
            "mcnemar_ours_vs": mcn,
            "repair_selected_by_ours": dict(sel),
            "n_gold_in_pool": sum(1 for r in rows
                                  if "gold_rerank" in r["fixed"]),
            "seed": seed}
