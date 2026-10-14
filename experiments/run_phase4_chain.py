"""Orchestrate P4: exp03a (dev+tune+freeze) -> exp03b (test single-use) -> exp04 (baselines).

Rate-limit aware: on Groq 429 (rate_limit_exceeded) sleeps 6 min and retries
(cheap: finished LLM calls replay from .llm_cache); aborts on any other error.
Run once: python experiments/run_phase4_chain.py

Exit codes: 0 all phases done, 1 failure (see log).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
PHASES = [
    "experiments/exp03a_faultbench_dev.py",
    "experiments/exp03b_faultbench_test.py",
    "experiments/exp04_baselines.py",
]
MAX_ATTEMPTS = 20
RATE_SLEEP_S = 360


def run_phase(rel: str) -> bool:
    """Run one phase; True on success, False on hard failure (raises SystemExit
    after retries are exhausted)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               PYTHONPATH=str(ROOT / "src"))
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n=== {rel} attempt {attempt}/{MAX_ATTEMPTS} ===", flush=True)
        t0 = time.time()
        proc = subprocess.Popen(
            [PY, str(ROOT / rel)], cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace",
        )
        tail: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail.append(line)
        rc = proc.wait()
        if rc == 0:
            print(f"=== {rel} OK in {time.time() - t0:.0f}s ===", flush=True)
            return True
        text = "".join(tail)
        if "rate_limit_exceeded" in text:
            print(f"[orchestrator] Groq daily rate limit; "
                  f"sleeping {RATE_SLEEP_S}s before retry", flush=True)
            time.sleep(RATE_SLEEP_S)
            continue
        print(f"[orchestrator] HARD FAILURE in {rel} (exit {rc}) - aborting chain",
              flush=True)
        raise SystemExit(1)
    print(f"[orchestrator] {rel}: {MAX_ATTEMPTS} attempts exhausted - aborting",
          flush=True)
    raise SystemExit(1)


def main() -> None:
    for ph in PHASES:
        run_phase(ph)
    print("\nALL PHASES COMPLETE (exp03a, exp03b, exp04)", flush=True)


if __name__ == "__main__":
    main()
