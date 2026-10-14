"""ragdx viva demo - Rich-based TUI (P9, isolated from core repo).

Views ONLY artifacts under ../results, ../data/faultbench, ../paper
(no computation, no writes). Optional live-diagnose screen calls the
cached LLM client if GROQ_API_KEY is available; everything else is offline.

Launch:  viva_demo\\run.bat  |  viva_demo\\run.ps1  |  make -C viva_demo tui
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# legacy Windows consoles default to cp1252; force UTF-8 stdio with fallback
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]          # repo root
SRC = ROOT / "src"
RESULTS = ROOT / "results"
console = Console()

LABEL_STYLE = {
    "RETRIEVER": "bold red", "GENERATOR": "bold blue", "INTERACTION": "bold yellow",
    "NONE": "bold green", "QUESTION_ISSUE": "magenta", "ANOMALY": "white",
    "ABSTAIN": "dim",
}


def load_json(rel: str):
    p = ROOT / rel
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def fmt_pct(x) -> str:
    return f"{x:.1%}" if isinstance(x, (int, float)) else str(x)


# ---------------------------------------------------------------- screens

def screen_overview() -> None:
    table = Table(title="ragdx - experiment overview", header_style="bold cyan")
    table.add_column("artifact")
    table.add_column("n", justify="right")
    table.add_column("key numbers")

    e1 = load_json("results/exp01/summary.json")
    if e1:
        acc = e1["accuracy"]
        table.add_row("exp01 pilot (50q)", str(e1["n"]),
                      f"acc base/gold/retr = {acc['base']:.2f}/{acc['gold']:.2f}/{acc['retrieved']:.2f}"
                      f"  mean R@1={e1['mean_recall_at_k']:.2f}")
    j = load_json("results/exp01_judge/agreement.json")
    if j:
        table.add_row("judge calibration", str(j["n"]),
                      f"judge-vs-EM agreement {fmt_pct(j['agreement'])}  (ECE {j['ece']:.2f}, "
                      f"{j['n_unparseable']} unparseable)")
    d = load_json("results/exp03_dev/manifest.json")
    if d:
        table.add_row("exp03 dev (FaultBench)", str(d["n"]),
                      f"faults: {len(d.get('fault_distribution', {}))} classes")
    e3 = load_json("results/exp03/eval.json")
    if e3:
        ci = e3["bootstrap_95ci"]
        table.add_row("exp03 test (single-use)", str(e3["n"]),
                      f"attribution accuracy {fmt_pct(e3['accuracy'])} [CI {ci[0]:.2f}-{ci[1]:.2f}]"
                      f"  macro-F1 {e3['macro_f1']:.3f}  (tau*={e3['tau_r_frozen']})")
    e4 = load_json("results/exp04/eval.json")
    if e4:
        rows = "  ".join(f"{k}:{v['macro_f1']:.2f}" for k, v in e4["systems"].items())
        table.add_row("exp04 baselines", str(e4["n"]), f"macro-F1 -> {rows}")
    console.print(table)
    m = load_json("results/exp03/manifest.json") or load_json("results/exp01/manifest.json")
    if m:
        console.print(Panel(Text(
            f"model {m.get('model')} · dataset {m.get('dataset', 'faultbench/mirage')} · "
            f"git {str(m.get('git_hash', 'n/a'))[:10]} · {m.get('date')}",
            style="dim"), title="provenance", border_style="dim"))


def screen_questions() -> None:
    items = load_json("results/exp01/items.json")
    if not items:
        console.print("[red]results/exp01/items.json missing - run exp01 first[/red]")
        return
    idx = 0
    while True:
        it = items[idx]
        tbl = Table.grid(expand=True)
        tbl.add_column(justify="right", style="dim", width=4)
        tbl.add_column()
        for cond, title in (("base", "No context"), ("gold", "Gold context"),
                            ("retrieved", "Retrieved context")):
            ok = "OK" if it[cond]["em_loose"] else "XX"
            style = "green" if it[cond]["em_loose"] else "red"
            body = it[cond]["text"].strip()
            if cond == "retrieved":
                body += (f"\n[dim]R@1={it[cond]['recall_at_k']:.2f}  "
                         f"gold_idx={it.get('gold_index')}[/dim]")
            tbl.add_row(title, f"[{style}]{ok}[/{style}] {body[:400]}")
        header = (f"[bold]{idx+1}/{len(items)}[/bold]  {it['question']}\n"
                  f"[dim]expected:[/dim] {', '.join(it['answers'][:3])}")
        lab = it["label"]
        footer = (f"\n[bold]attribution:[/bold] [{LABEL_STYLE.get(lab, 'white')}]{lab}"
                  f"[/]  [dim]{it['evidence'].get('reason', '')}[/dim]")
        console.print(Panel(tbl, title=header, subtitle=footer, border_style="cyan"))
        cmd = Prompt.ask("[dim]n/p[/dim] next/prev - q back", default="n")
        if cmd == "q":
            break
        if cmd == "n":
            idx = (idx + 1) % len(items)
        elif cmd == "p":
            idx = (idx - 1) % len(items)


def screen_confusion() -> None:
    e3 = load_json("results/exp03/eval.json")
    if not e3:
        console.print("[red]results/exp03/eval.json missing - run exp03b first[/red]")
        return
    cm = e3["confusion_truth_rows_pred_cols"]
    labels = sorted({k for row in cm.values() for k in row} | set(cm))
    tbl = Table(title=f"FaultBench test confusion (truth->pred), n={e3['n']}, "
                      f"acc {fmt_pct(e3['accuracy'])}, macro-F1 {e3['macro_f1']}",
                header_style="bold cyan")
    tbl.add_column("truth/pred", style="dim")
    for lab in labels:
        tbl.add_column(lab, justify="right")
    for t in labels:
        cells = []
        for p in labels:
            v = cm.get(t, {}).get(p, 0)
            style = "bold green" if t == p and v else ("red" if v else "dim")
            cells.append(f"[{style}]{v}[/{style}]")
        tbl.add_row(f"[bold]{t}[/bold]", *cells)
    console.print(tbl)
    pf = Table(title="per-fault attribution accuracy", header_style="bold cyan")
    pf.add_column("fault"); pf.add_column("acc", justify="right")
    for f, a in sorted(e3["per_fault_accuracy"].items(), key=lambda kv: -kv[1]):
        pf.add_row(f, fmt_pct(a))
    console.print(pf)
    pc = Table(title="per-class P/R/F1", header_style="bold cyan")
    pc.add_column("class"); pc.add_column("P", justify="right")
    pc.add_column("R", justify="right"); pc.add_column("F1", justify="right")
    pc.add_column("support", justify="right")
    for lab, s in e3["per_class"].items():
        pc.add_row(lab, f"{s['precision']:.2f}", f"{s['recall']:.2f}",
                   f"{s['f1']:.2f}", str(s["support"]))
    console.print(pc)


def screen_baselines() -> None:
    e4 = load_json("results/exp04/eval.json")
    if not e4:
        console.print("[red]results/exp04/eval.json missing - run exp04 first[/red]")
        return
    tbl = Table(title=f"Baselines on FaultBench test (n={e4['n']}, tau*={e4['tau_r']}, "
                      f"gamma*={e4['gamma']})", header_style="bold cyan")
    tbl.add_column("system"); tbl.add_column("acc", justify="right")
    tbl.add_column("macro-F1", justify="right"); tbl.add_column("95% CI (acc)")
    for name, s in e4["systems"].items():
        ci = s["bootstrap_95ci_acc"]
        style = "bold green" if name == "rule_v1" else ""
        tbl.add_row(f"[{style}]{name}[/{style}]", fmt_pct(s["accuracy"]),
                    f"{s['macro_f1']:.3f}", f"{ci[0]:.2f}-{ci[1]:.2f}")
    console.print(tbl)
    mc = Table(title="McNemar vs rule_v1 (ours)", header_style="bold cyan")
    mc.add_column("baseline"); mc.add_column("ours-only right", justify="right")
    mc.add_column("baseline-only right", justify="right")
    mc.add_column("p (Holm)", justify="right")
    for name, s in e4["mcnemar_vs_rule_v1"].items():
        verdict = "significant" if s["p_holm"] < 0.05 else "n.s."
        mc.add_row(name, str(s["b_ours_only"]), str(s["c_baseline_only"]),
                   f"{s['p_holm']:.4f} [{verdict}]")
    console.print(mc)


def screen_faultbench() -> None:
    man = load_json("data/faultbench/manifest.json")
    card = ROOT / "data" / "faultbench" / "README.md"
    if not man:
        console.print("[red]data/faultbench missing - run build_faultbench.py[/red]")
        return
    for split in ("dev_truth", "test_truth"):
        tbl = Table(title=f"FaultBench {split.replace('_truth','')} class balance",
                    header_style="bold cyan")
        tbl.add_column("truth label"); tbl.add_column("n", justify="right")
        for k, v in sorted(man["balance"][split].items(), key=lambda kv: -kv[1]):
            tbl.add_row(f"[{LABEL_STYLE.get(k, 'white')}]{k}[/]", str(v))
        console.print(tbl)
    ft = Table(title="fault catalogue (test)", header_style="bold cyan")
    ft.add_column("fault"); ft.add_column("n", justify="right")
    for k, v in sorted(man["balance"]["test_fault"].items(), key=lambda kv: -kv[1]):
        ft.add_row(k, str(v))
    console.print(ft)
    if card.exists():
        console.print(Panel(card.read_text(encoding="utf-8")[:1200],
                            title="dataset card", border_style="dim"))


def screen_provenance() -> None:
    tbl = Table(title="Results provenance (manifests)", header_style="bold cyan")
    tbl.add_column("exp"); tbl.add_column("date"); tbl.add_column("n")
    tbl.add_column("tokens", justify="right"); tbl.add_column("git")
    for exp in ("exp01", "exp01_judge", "exp03_dev", "exp03", "exp04"):
        m = load_json(f"results/{exp}/manifest.json")
        if m:
            tbl.add_row(exp, str(m.get("date", "?")), str(m.get("n_items", m.get("n", "?"))),
                        str(m.get("total_tokens", "-")), str(m.get("git_hash", "-"))[:10])
    console.print(tbl)
    pdf = ROOT / "paper" / "main.pdf"
    if pdf.exists():
        console.print(f"paper: {pdf} [green](exists, {pdf.stat().st_size} bytes)[/green]")
    else:
        console.print("paper: not compiled yet")


def screen_live() -> None:
    console.print(Panel(Text(
        "Live diagnose: type a question. If GROQ_API_KEY is set, ragdx runs the\n"
        "3 conditions (cached) and applies attribution rule v1 live. Offline-safe:\n"
        "without a key it explains what would happen.", style="cyan"),
        title="live demo", border_style="cyan"))
    import os
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    q = Prompt.ask("question").strip()
    if not q:
        return
    if not os.getenv("GROQ_API_KEY"):
        console.print("[yellow]GROQ_API_KEY not set - live mode needs it "
                      "(.env is loaded by ragdx automatically).[/yellow]")
        return
    from ragdx.datasets.mirage import load_mirage
    from ragdx.llm.client import LLMClient
    from ragdx.runner.three_condition import RunConfig, run_three_condition
    items = load_mirage(n=30)
    # pick the MIRAGE item whose question lexically matches best
    best = max(items, key=lambda it: len(set(q.lower().split()) &
                                         set(it.query.lower().split())))
    console.print(f"[dim]closest MIRAGE question:[/dim] {best.query}")
    client = LLMClient(model="openai/gpt-oss-20b",
                       cache_dir=ROOT / ".llm_cache",
                       log_path=ROOT / "results" / "_llm_calls.jsonl")
    cfg = RunConfig(exp_id="live", model="openai/gpt-oss-20b", top_k=1, tau_r=0.5)
    with console.status("running 3 conditions (cached when possible)..."):
        res = run_three_condition(cfg, [best], client, ROOT / "results" / "_live_tmp")
    rec = res["records"][0]
    for cond, title in (("base", "No context"), ("gold", "Gold context"),
                        ("retrieved", "Retrieved")):
        ok = "OK" if rec[cond]["em_loose"] else "XX"
        console.print(f"[bold]{title}[/bold]: {ok} {rec[cond]['text'][:200]}")
    lab = rec["label"]
    console.print(Panel(f"[{LABEL_STYLE.get(lab, 'white')}]{lab}[/] - "
                        f"{rec['evidence'].get('reason','')}",
                        title="attribution", border_style="cyan"))


SCREENS = [
    ("1", "Overview & KPIs", screen_overview),
    ("2", "Question browser (3 conditions)", screen_questions),
    ("3", "FaultBench results (confusion, per-fault)", screen_confusion),
    ("4", "Baselines comparison (exp04)", screen_baselines),
    ("5", "FaultBench explorer (balance, card)", screen_faultbench),
    ("6", "Live diagnose (cached Groq)", screen_live),
    ("7", "Provenance & paper", screen_provenance),
]


def main() -> None:
    console.clear()
    console.print(Panel(
        Align.center(Text.from_markup(
            "[bold cyan]ragdx[/] - Attributing RAG Failures\n"
            "[dim]controlled-condition diagnosis - viva demo[/]"),
        ), border_style="cyan", height=7))
    while True:
        menu = Table.grid(padding=(0, 2))
        for key, title, _ in SCREENS:
            menu.add_row(f"[bold cyan]{key}[/]", title)
        menu.add_row("[bold red]q[/]", "quit")
        console.print(menu)
        choice = Prompt.ask("select", default="1")
        if choice == "q":
            console.print("[dim]bye - every number shown came from results/[/dim]")
            return
        for key, _, fn in SCREENS:
            if choice == key:
                console.clear()
                try:
                    fn()
                except Exception as e:  # demo must never die
                    console.print(f"[red]screen error: {e}[/red]")
                Prompt.ask("\n[dim]enter to return to menu[/dim]", default="")
                console.clear()
                break


if __name__ == "__main__":
    main()
