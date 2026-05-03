@echo off
REM ============================================================
REM StrategyFactory - Environment setup (Windows)
REM ============================================================

setlocal enabledelayedexpansion

cd /d "%~dp0\.."

echo.
echo ============================================================
echo  StrategyFactory - Environment Setup
echo ============================================================
echo.

REM Check Python
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH. Install Python 3.11+ first.
    exit /b 1
)

python --version

REM Create venv if missing
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Creating venv...
    python -m venv venv
)

call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip --quiet

REM Install requirements
echo [INFO] Installing dependencies...
pip install -r requirements.txt --quiet

REM Verify config
if not exist "config\mt5_paths.yaml" (
    echo [ERROR] config\mt5_paths.yaml missing.
    exit /b 1
)

echo.
echo ============================================================
echo  Environment ready.
echo ============================================================
echo.
echo Next steps:
echo   1. Edit config\mt5_paths.yaml — verify MT5 install path
echo      and replace REPLACE_ME for data_folder.
echo   2. Try the example spec:
echo        python automation\spec_validator.py strategy_specs\_EXAMPLE_asian_mr_fx.yaml
echo   3. Read CLAUDE.md and docs\WORKFLOW.md
echo.

endlocal
