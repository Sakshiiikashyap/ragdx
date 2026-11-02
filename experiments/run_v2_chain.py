"""Orchestrate the pre-registered exp03_v2 k-sample study (D024) with the
same rate-limit-aware retry semantics as the P4/P567 chains.

Single phase, but the runner is resume-safe (per-question checkpointing):
a 429-exhaustion crash resumes where it left off. Usage:
    python experiments/run_v2_chain.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase4_chain import run_phase  # noqa: E402  (same retry semantics)


def main() -> None:
    run_phase("experiments/exp03v2_ksample.py")
    print("\nV2 CHAIN COMPLETE (exp03_v2 eval + risk_coverage written)\n")


if __name__ == "__main__":
    main()
