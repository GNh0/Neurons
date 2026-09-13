@echo off
cd /d "%~dp0"
if exist "Neurons.exe" (
  "Neurons.exe" --configure-postgres
  pause
  exit /b
)
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.ps1 first to prepare the source environment.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 scripts\configure_postgres.py
pause
