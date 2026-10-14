@echo off
REM ragdx viva demo launcher (CMD)
REM usage:  viva_demo\run.bat
setlocal
set DEMO_DIR=%~dp0
set ROOT=%~dp0..
set PY=C:\Users\pc\Desktop\MIRAGE-main\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
set PYTHONPATH=%ROOT%\src
cd /d "%DEMO_DIR%"
"%PY%" app.py
endlocal
