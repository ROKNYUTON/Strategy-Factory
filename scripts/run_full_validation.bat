@echo off
REM ============================================================
REM StrategyFactory - Run full test suite + environment health check
REM ============================================================

setlocal enabledelayedexpansion

cd /d "%~dp0\.."

call venv\Scripts\activate.bat 2>nul

echo.
echo [1/3] Running pytest...
python -m pytest tests/ -v --tb=short
if errorlevel 1 (
    echo [ERROR] Tests failed.
    exit /b 1
)

echo.
echo [2/3] Verifying MT5 paths config...
python -c "import yaml; d=yaml.safe_load(open('config/mt5_paths.yaml',encoding='utf-8')); paths=[d['mt5']['terminal_exe'],d['mt5']['metaeditor_exe']]; import os; [print('OK' if os.path.exists(p) else 'MISSING', p) for p in paths]"

echo.
echo [3/3] Verifying example spec validates...
python automation\spec_validator.py strategy_specs\_EXAMPLE_asian_mr_fx.yaml

echo.
echo ============================================================
echo  Validation complete.
echo ============================================================

endlocal
