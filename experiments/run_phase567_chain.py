"""Orchestrate P5-P7: exp06 (robustness) -> exp07 (multi-model) -> exp08
(utility), with the same rate-limit-aware retry logic as run_phase4_chain.

Run AFTER the P4 chain completes: exp06 needs results/exp04, exp07 needs
results/exp03, exp08 needs results/exp03. Usage:
    python experiments/run_phase567_chain.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase4_chain import run_phase  # noqa: E402  (same retry semantics)

PHASES = [
    "experiments/exp06_robustness.py",
    "experiments/exp07_multimodel.py",
    "experiments/exp08_utility.py",
]


def main() -> None:
    for ph in PHASES:
        run_phase(ph)
    print("\nALL PHASES COMPLETE (exp06, exp07, exp08)", flush=True)


if __name__ == "__main__":
    main()
