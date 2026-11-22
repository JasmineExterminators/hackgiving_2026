@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run SETUP_AND_TEST.cmd first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" turkey_keyboard.py --simulate --pads 9
if errorlevel 1 pause
