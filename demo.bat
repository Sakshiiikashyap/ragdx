@echo off
REM Demo script for the morning walkthrough (run from the repo root, in the MIRAGE venv)
set PY=C:\Users\pc\Desktop\MIRAGE-main\.venv\Scripts\python.exe
set ROOT=C:\Users\pc\OneDrive\Documents\Default Project
cd /d %ROOT%
set PYTHONPATH=%ROOT%\src

echo === 1/4 unit tests (offline) ===
%PY% -m pytest -q

echo === 2/4 exp01 three-condition run (cached if repeated) ===
%PY% experiments\exp01_three_condition.py 50

echo === 3/4 dashboard ===
%PY% -m ragdx.app.dashboard exp01

echo === 4/4 paper tables ===
%PY% -m ragdx.analysis.tables exp01

echo.
echo Open results\exp01\dashboard.html in a browser, and paper\main.tex for the paper.
pause
