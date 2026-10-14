# viva_demo — Rich TUI for the viva (P9)

Isolated from the core repo: reads `../results`, `../data/faultbench`, `../paper`
read-only; never writes. Screens: overview/KPIs, per-question 3-condition
browser, FaultBench confusion + per-fault accuracy, baselines comparison
(with McNemar), fault catalog + dataset card, live diagnose (cached Groq,
optional), provenance (manifests, git hash, paper PDF).

## Run

```bat
viva_demo\run.bat           (CMD)
powershell -File viva_demo\run.ps1
make -C viva_demo tui       (make)
```

Keys: pick a screen number, `q` quits, `n`/`p` browses questions.

## Notes

- Everything displayed is loaded from JSON artifacts produced by
  `experiments/*.py` — no numbers are computed in the UI (O5 rule).
- Live diagnose needs `GROQ_API_KEY` (via repo-root `.env`); all other
  screens work offline.
- Launchers default to the MIRAGE-main venv (D001); override `PY` in the
  Makefile or edit the `.ps1`/`.bat` if you create a dedicated venv.
