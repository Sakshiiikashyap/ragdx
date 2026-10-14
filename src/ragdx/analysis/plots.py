"""analysis/plots.py — paper figures ONLY from results/ (CLAUDE.md rule 1).

Every function takes (root, out_path) -> bool: True if the figure was written,
False if its input artifact is missing (graceful skip — no \todo magic numbers
come out of thin air; the paper only embeds figures that were actually made).

Usage:
  python -m ragdx.analysis.plots all           # everything available
  python -m ragdx.analysis.plots confusion     # single figure by name
  python -m ragdx.analysis.plots confusion baselines
Writes paper/figures/<name>.png by default (root/paper/figures).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless (tests + CI)
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]

# IEEE double-column friendly
FW, FH, DPI = 3.4, 2.5, 200
CLASS_COLORS = {"RETRIEVER": "#d62728", "GENERATOR": "#1f77b4",
                "INTERACTION": "#ff7f0e", "NONE": "#7f7f7f",
                "QUESTION_ISSUE": "#9467bd", "ANOMALY": "#8c564b"}
LABEL_ORDER = ["RETRIEVER", "GENERATOR", "INTERACTION", "NONE",
               "QUESTION_ISSUE", "ANOMALY"]


def _load(root: Path, rel: str):
    p = root / rel
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _save(fig, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------- figures
def fig_tau_sweep(root: Path, out: Path) -> bool:
    """F: dev tau sweep with frozen operating point (results/exp03_dev/tuning.json)."""
    t = _load(root, "results/exp03_dev/tuning.json")
    if not t:
        return False
    sweep = t["sweep"]
    xs = [s["tau_r"] for s in sweep]
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.plot(xs, [s["macro_f1"] for s in sweep], "o-", label="macro-F1",
            color=CLASS_COLORS["GENERATOR"], ms=3)
    ax.plot(xs, [s["accuracy"] for s in sweep], "s--", label="accuracy",
            color=CLASS_COLORS["RETRIEVER"], ms=3)
    ax.axvline(t["tau_r"], color="k", lw=1, ls=":",
               label=f"frozen $\\tau$*={t['tau_r']}")
    ax.set_xlabel(r"retrieval threshold $\tau_R$")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=0.3)
    return _save(fig, out)


def fig_confusion(root: Path, out: Path) -> bool:
    """F3a: test confusion matrix (results/exp03/eval.json)."""
    e = _load(root, "results/exp03/eval.json")
    if not e:
        return False
    cm = e["confusion_truth_rows_pred_cols"]
    present = set(cm) | {k for row in cm.values() for k in row}
    labels = [l for l in LABEL_ORDER if l in present] + sorted(present - set(LABEL_ORDER))
    n = len(labels)
    mat = [[cm.get(t, {}).get(p, 0) for p in labels] for t in labels]

    fig, ax = plt.subplots(figsize=(FW, FH * 1.1))
    im = ax.imshow(mat, cmap="Blues")
    for i in range(n):
        for j in range(n):
            v = mat[i][j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                        color="white" if v > max(map(max, mat)) * 0.6 else "black")
    ax.set_xticks(range(n), labels, rotation=30, ha="right", fontsize=6)
    ax.set_yticks(range(n), labels, fontsize=6)
    ax.set_xlabel("predicted", fontsize=7)
    ax.set_ylabel("ground truth", fontsize=7)
    ax.set_title(f"n={e['n']}  acc={e['accuracy']:.2f}  "
                 f"macro-F1={e['macro_f1']:.2f}", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, out)


def fig_per_fault(root: Path, out: Path) -> bool:
    """F3b: attribution accuracy per injected fault (results/exp03/items.json)."""
    items = _load(root, "results/exp03/items.json")
    if not items:
        return False
    faults: dict[str, list[bool]] = {}
    truth: dict[str, str] = {}
    for r in items:
        faults.setdefault(r["fault"], []).append(bool(r.get(
            "correct_attribution", r.get("pred_label") == r.get("ground_truth"))))
        truth[r["fault"]] = r["ground_truth"]
    rows = sorted(((f, sum(v) / len(v)) for f, v in faults.items()),
                  key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(FW, FH * 1.2))
    ax.barh([f for f, _ in rows], [a for _, a in rows],
            color=[CLASS_COLORS.get(truth[f], "#333") for f, _ in rows])
    for i, (_, a) in enumerate(rows):
        ax.text(a + 0.01, i, f"{a:.0%}", va="center", fontsize=6)
    ax.set_xlabel("attribution accuracy", fontsize=7)
    ax.set_xlim(0, 1.08)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, out)


def fig_baselines(root: Path, out: Path) -> bool:
    """Main results bar: our rule vs 3 baselines, accuracy +/- bootstrap CI
    (results/exp04/eval.json)."""
    e = _load(root, "results/exp04/eval.json")
    if not e:
        return False
    systems = e["systems"]
    names = list(systems)
    accs = [systems[k]["accuracy"] for k in names]
    lo = [accs[i] - systems[k]["bootstrap_95ci_acc"][0] for i, k in enumerate(names)]
    hi = [systems[k]["bootstrap_95ci_acc"][1] - accs[i] for i, k in enumerate(names)]
    colors = ["#2ca02c" if k == "rule_v1" else "#9ecae1" for k in names]

    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.bar(names, accs, yerr=[lo, hi], capsize=2, color=colors, width=0.6)
    for i, k in enumerate(names):
        ax.text(i, accs[i] + hi[i] + 0.02, f"{accs[i]:.2f}", ha="center",
                fontsize=6)
    ax.set_ylabel("attribution accuracy", fontsize=7)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelsize=6, rotation=15)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, out)


def fig_risk_coverage(root: Path, out: Path) -> bool:
    """F4: risk-coverage for rule v2 (results/exp03_v2/risk_coverage.json =
    [{"coverage": .., "accuracy": ..}]); empty until the k-sample run exists."""
    rc = _load(root, "results/exp03_v2/risk_coverage.json")
    if not rc:
        return False
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.plot([r["coverage"] for r in rc], [r["accuracy"] for r in rc], "-o",
            ms=3, color=CLASS_COLORS["GENERATOR"])
    ax.set_xlabel("coverage (fraction answered)", fontsize=7)
    ax.set_ylabel("accuracy of answered", fontsize=7)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    return _save(fig, out)


def fig_degradation(root: Path, out: Path) -> bool:
    """F5: gold-degradation robustness curves (results/exp06/eval.json, schema
    {"conditions": [ordered names], "by_system": {sys: {cond: {"macro_f1": ..}}}})."""
    e = _load(root, "results/exp06/eval.json")
    if not e or not e.get("by_system"):
        return False
    conds = e["conditions"]
    fig, ax = plt.subplots(figsize=(FW, FH))
    markers = ["o", "s", "^", "v"]
    for i, (sysname, per_cond) in enumerate(e["by_system"].items()):
        ys = [per_cond.get(c, {}).get("macro_f1") for c in conds]
        ax.plot(range(len(conds)), ys, "-"+markers[i % len(markers)], ms=4,
                label=sysname)
    ax.set_xticks(range(len(conds)), conds, rotation=20, fontsize=6)
    ax.set_ylabel("macro-F1", fontsize=7)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=0.3)
    return _save(fig, out)


def fig_utility(root: Path, out: Path) -> bool:
    """F6: repair-utility gains (results/exp08/eval.json, schema
    {"fixes": [{"name": .., "gain": ..}]} = accuracy gain on failed set)."""
    e = _load(root, "results/exp08/eval.json")
    if not e or not e.get("fixes"):
        return False
    fixes = e["fixes"]
    fig, ax = plt.subplots(figsize=(FW, FH))
    names = [f["name"] for f in fixes]
    gains = [f["gain"] for f in fixes]
    colors = ["#2ca02c" if g > 0 else "#d62728" for g in gains]
    ax.bar(names, gains, color=colors, width=0.6)
    ax.axhline(0, color="k", lw=0.8)
    for i, g in enumerate(gains):
        ax.text(i, g + (0.01 if g >= 0 else -0.03), f"{g:+.2f}", ha="center",
                fontsize=6)
    ax.set_ylabel("accuracy gain on failed set", fontsize=7)
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, out)


def fig_multimodel(root: Path, out: Path) -> bool:
    """T5 support: per-model macro-F1 (results/exp07/eval.json, schema
    {"models": {model: {"macro_f1": ..}}})."""
    e = _load(root, "results/exp07/eval.json")
    if not e or not e.get("models"):
        return False
    models = e["models"]
    fig, ax = plt.subplots(figsize=(FW, FH))
    names = list(models)
    short = [m.split("/")[-1] for m in names]
    ax.bar(short, [models[m]["macro_f1"] for m in names],
           color="#1f77b4", width=0.6)
    for i, m in enumerate(names):
        ax.text(i, models[m]["macro_f1"] + 0.015,
                f"{models[m]['macro_f1']:.2f}", ha="center", fontsize=6)
    ax.set_ylabel("macro-F1", fontsize=7)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelsize=6, rotation=20)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, out)


FIGURES = {
    "tau_sweep": fig_tau_sweep,
    "confusion": fig_confusion,
    "per_fault": fig_per_fault,
    "baselines": fig_baselines,
    "risk_coverage": fig_risk_coverage,
    "degradation": fig_degradation,
    "utility": fig_utility,
    "multimodel": fig_multimodel,
}


def run_all(root: Path, outdir: Path) -> list[str]:
    written = []
    for name, fn in FIGURES.items():
        if fn(root, outdir / f"{name}.png"):
            written.append(name)
            print(f"wrote {outdir / (name + '.png')}")
        else:
            print(f"skip {name} (input missing)")
    return written


def main(argv: list[str]) -> None:
    root, outdir = ROOT, ROOT / "paper" / "figures"
    names = argv[1:] or ["all"]
    if names == ["all"]:
        run_all(root, outdir)
        return
    for name in names:
        if name not in FIGURES:
            print(f"unknown figure '{name}' (choose from {', '.join(FIGURES)})")
            continue
        ok = FIGURES[name](root, outdir / f"{name}.png")
        print(f"{'wrote' if ok else 'skip'} {name}")


if __name__ == "__main__":
    main(sys.argv)
