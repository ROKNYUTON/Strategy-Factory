@echo off
REM ============================================================
REM StrategyFactory - 1-click strategy runner (Windows)
REM Usage: run_strategy.bat <spec_filename.yaml>
REM ============================================================

setlocal enabledelayedexpansion

cd /d "%~dp0\.."

if "%~1"=="" (
    echo Usage: run_strategy.bat ^<spec_filename.yaml^>
    echo Example: run_strategy.bat _EXAMPLE_asian_mr_fx.yaml
    exit /b 1
)

set "SPEC=strategy_specs\%~1"
if not exist "%SPEC%" (
    echo Spec not found: %SPEC%
    exit /b 1
)

call venv\Scripts\activate.bat 2>nul
if errorlevel 1 (
    echo [WARN] venv not found. Run scripts\setup_env.bat first.
)

python automation\pipeline.py full "%SPEC%"

if errorlevel 1 (
    echo.
    echo [ERROR] Pipeline failed. Check logs\ and console output above.
    exit /b 1
)

echo.
echo ============================================================
echo  DONE. Verdict logged to docs\HYPOTHESIS_LOG.md
echo ============================================================

endlocal
