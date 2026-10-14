"""analysis/tables.py — generate paper tables/figures ONLY from results/.

Rule 1 of CLAUDE.md: every number in the paper comes from here.

Usage: python -m ragdx.analysis.tables exp01
Writes paper/tables/exp01_summary.tex and prints a Markdown summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load(exp: str) -> tuple[dict, dict, list]:
    d = ROOT / "results" / exp
    return (json.loads((d / "summary.json").read_text(encoding="utf-8")),
            json.loads((d / "manifest.json").read_text(encoding="utf-8")),
            json.loads((d / "items.json").read_text(encoding="utf-8")))


def summary_markdown(exp: str) -> str:
    summary, manifest, items = load(exp)
    acc, mm = summary["accuracy"], summary["mirage_metrics"]
    dist = summary["attribution_distribution"]
    lines = [
        f"| metric | value |", "|---|---|",
        f"| n | {summary['n']} |",
        f"| base acc | {acc['base']:.3f} |",
        f"| gold acc | {acc['gold']:.3f} |",
        f"| retrieved acc | {acc['retrieved']:.3f} |",
        f"| NV | {mm['nv']:.3f} |", f"| CA | {mm['ca']:.3f} |",
        f"| CI | {mm['ci']:.3f} |", f"| CM | {mm['cm']:.3f} |",
        f"| mean R@k | {summary['mean_recall_at_k']:.3f} |",
        f"| tokens | {manifest['total_tokens']} |",
    ]
    for lab, cnt in sorted(dist.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {lab} | {cnt} ({cnt/summary['n']:.0%}) |")
    return "\n".join(lines)


def tex_table(exp: str) -> str:
    summary, manifest, _ = load(exp)
    acc, mm = summary["accuracy"], summary["mirage_metrics"]
    rows = {
        "Base acc.": acc["base"], "Gold acc.": acc["gold"],
        "Retrieved acc.": acc["retrieved"], "NV": mm["nv"], "CA": mm["ca"],
        "CI": mm["ci"], "CM": mm["cm"], "Mean R@k": summary["mean_recall_at_k"],
    }
    body = "\n".join(f"{k} & {v:.3f} \\\\" for k, v in rows.items())
    return (f"% AUTO-GENERATED from results/{exp}/summary.json — do not edit\n"
            f"% model={manifest['model']} n={manifest['n_items']} date={manifest['date']}\n"
            f"\\begin{{tabular}}{{lr}}\\toprule\nMetric & Value \\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n")


if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else "exp01"
    print(summary_markdown(exp))
    out = ROOT / "paper" / "tables" / f"{exp}_summary.tex"
    out.parent.mkdir(exist_ok=True)
    out.write_text(tex_table(exp), encoding="utf-8")
    print(f"\nwrote {out}")
