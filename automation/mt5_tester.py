"""
StrategyFactory — MT5 Strategy Tester Wrapper
=============================================
Launches `terminal64.exe /config:tester.ini` for a backtest run.

Usage:
    python automation/mt5_tester.py STR_001_asian_mr_fx --period is
    python automation/mt5_tester.py STR_001_asian_mr_fx --period all
"""

from __future__ import annotations

import sys
import time
import shutil
import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, List

import yaml
from rich.console import Console
from rich.panel import Panel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from automation.spec_validator import validate_spec  # noqa: E402
from automation.tester_ini_builder import build_ini  # noqa: E402

console = Console()


@dataclass
class BacktestResult:
    success: bool
    period: str
    report_html: Path | None = None
    report_xml: Path | None = None
    log_file: Path | None = None
    runtime_sec: float = 0.0
    error: str = ""


def load_paths() -> dict:
    cfg = ROOT / "config" / "mt5_paths.yaml"
    with cfg.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_spec_for_strategy(strategy_id: str) -> Path:
    """Locate the YAML spec for a strategy_id."""
    candidates = list((ROOT / "strategy_specs").rglob("*.yaml"))
    for p in candidates:
        try:
            spec = validate_spec(p)
            if spec.meta.strategy_id == strategy_id:
                return p
        except Exception:
            continue
    raise FileNotFoundError(f"No valid spec found for strategy_id={strategy_id}")


def find_compiled_ex5(strategy_id: str) -> Path:
    p = ROOT / "mql5" / "compiled" / f"{strategy_id}.ex5"
    if not p.exists():
        raise FileNotFoundError(
            f"Compiled .ex5 not found at {p}. Run mt5_compiler first."
        )
    return p


def run_one_period(spec, strategy_id: str, period: Literal["is", "oos", "forward"]) -> BacktestResult:
    """Run one backtest for one period."""
    paths = load_paths()
    bt_cfg = paths.get("backtest", {})
    timeout = bt_cfg.get("timeout_seconds", 1800)
    poll = bt_cfg.get("poll_interval", 5)

    terminal = Path(paths["mt5"]["terminal_exe"])
    if not terminal.exists():
        return BacktestResult(success=False, period=period,
                              error=f"terminal not found: {terminal}")

    ex5_path = find_compiled_ex5(strategy_id)

    # Copy the .ex5 into MT5 Experts/StrategyFactory/ subfolder
    data_folder = Path(paths["mt5"]["data_folder"])
    if not data_folder.exists() or "REPLACE_ME" in str(data_folder):
        return BacktestResult(success=False, period=period,
                              error="config/mt5_paths.yaml: data_folder not configured.")
    experts_target = data_folder / "MQL5" / "Experts" / "StrategyFactory"
    experts_target.mkdir(parents=True, exist_ok=True)
    target_ex5 = experts_target / ex5_path.name
    shutil.copy2(ex5_path, target_ex5)

    # Build .ini
    ini_path = build_ini(spec, period, ex5_path)
    console.print(f"[cyan]Built tester.ini:[/cyan] {ini_path}")

    # MT5 expects ini path as absolute
    cmd = [str(terminal), f"/config:{str(ini_path.resolve())}"]
    if paths["mt5"].get("portable_mode", False):
        cmd.append("/portable")

    console.print(f"[cyan]Launching MT5 tester for period={period}...[/cyan]")
    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd)
    except Exception as e:
        return BacktestResult(success=False, period=period, error=f"Launch failed: {e}")

    # Poll for tester report
    reports_dir = data_folder / "tester" / "Reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    expected_report_html = reports_dir / f"{strategy_id}_{period}.htm"
    # MT5 can also be told to write reports — easier route: detect ANY new report
    # In practice we name the report by adding /report:<name> to terminal config.
    # Here, we wait for terminal exit (ShutdownTerminal=1 in the .ini).

    deadline = time.time() + timeout
    while True:
        ret = proc.poll()
        if ret is not None:
            break
        if time.time() > deadline:
            proc.kill()
            return BacktestResult(success=False, period=period,
                                  error=f"Tester timeout after {timeout}s")
        time.sleep(poll)

    elapsed = time.time() - t0

    # Find newest report (.htm) generated since t0
    candidates_html = sorted(reports_dir.glob("*.htm"), key=lambda x: x.stat().st_mtime, reverse=True)
    candidates_xml  = sorted(reports_dir.glob("*.xml"), key=lambda x: x.stat().st_mtime, reverse=True)
    report_html = next((p for p in candidates_html if p.stat().st_mtime >= t0 - 5), None)
    report_xml  = next((p for p in candidates_xml  if p.stat().st_mtime >= t0 - 5), None)

    # Move to backtests/raw_reports/{strategy_id}/{period}_{ts}/
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "backtests" / "raw_reports" / strategy_id / f"{period}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    final_html = final_xml = None
    if report_html:
        final_html = out_dir / report_html.name
        shutil.copy2(report_html, final_html)
    if report_xml:
        final_xml = out_dir / report_xml.name
        shutil.copy2(report_xml, final_xml)

    success = bool(final_html or final_xml)
    return BacktestResult(
        success=success,
        period=period,
        report_html=final_html,
        report_xml=final_xml,
        runtime_sec=elapsed,
        error="" if success else "No report produced — check MT5 journal."
    )


def run_backtest(strategy_id: str, period: str = "all") -> List[BacktestResult]:
    spec_path = find_spec_for_strategy(strategy_id)
    spec = validate_spec(spec_path)

    periods = [period] if period != "all" else ["is", "oos", "forward"]
    results = []
    for p in periods:
        console.print(Panel(f"[bold]Running backtest — period={p}[/bold]", border_style="cyan"))
        r = run_one_period(spec, strategy_id, p)
        if r.success:
            console.print(f"[green]✅ Period {p} done in {r.runtime_sec:.1f}s. Report: {r.report_html}[/green]")
        else:
            console.print(f"[red]❌ Period {p} failed: {r.error}[/red]")
        results.append(r)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy_id")
    ap.add_argument("--period", default="all", choices=["is", "oos", "forward", "all"])
    args = ap.parse_args()

    try:
        results = run_backtest(args.strategy_id, args.period)
    except Exception as e:
        console.print(Panel(f"[red]{e}[/red]", title="❌ ERROR"))
        return 1
    return 0 if all(r.success for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
