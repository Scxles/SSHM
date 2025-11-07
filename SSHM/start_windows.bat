@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>&1
if errorlevel 1 (
  echo Python not found. Please install Python 3.10+ from https://www.python.org/ and re-run.
  pause
  exit /b 1
)
echo Installing Python packages (first run only)...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo Starting Hardware Monitor...
python app.py
