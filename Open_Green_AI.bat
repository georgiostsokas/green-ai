@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe py -3 -m venv .venv
.venv\Scripts\python.exe -c "import green_ai.dashboard" >nul 2>&1
if errorlevel 1 .venv\Scripts\python.exe -m pip install -e ".[app]"
echo Open http://127.0.0.1:8501 - Ctrl+C stops the app. Nothing is published.
.venv\Scripts\python.exe -m green_ai.cli app
pause
