# ragdx viva demo launcher (PowerShell)
# usage:  .\viva_demo\run.ps1
$ErrorActionPreference = "Stop"
$DemoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $DemoDir
$Py = "C:\Users\pc\Desktop\MIRAGE-main\.venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
$env:PYTHONPATH = Join-Path $Root "src"
Set-Location $DemoDir
& $Py app.py
