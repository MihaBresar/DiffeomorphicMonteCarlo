@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" START_HERE.py
    goto end
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3 START_HERE.py
    goto end
)
where python >nul 2>nul
if not errorlevel 1 (
    python START_HERE.py
    goto end
)
echo Install Python 3.12 from python.org, enable Add python.exe to PATH, then reopen this file.
:end
if errorlevel 1 pause
