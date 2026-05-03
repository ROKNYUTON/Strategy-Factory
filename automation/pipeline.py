"""
StrategyFactory — Master Pipeline Orchestrator
==============================================
End-to-end CLI from hypothesis spec to verdict.

Commands:
  prepare    Validate spec + generate EA prompt + skeleton
  compile    Compile the (manually completed) EA
  backtest   Run MT5 Strategy Tester for IS / OOS / forward / all
  analyze    Run all analysis modules (parser, metrics, bootstrap, etc.)
  verdict    Run acceptance check, log to HYPOTHESIS_LOG.md
  full       Orchestrate all of the above with manual pause at EA generation

Usage:
  python automation/pipeline.py prepare strategy_specs/my_spec.yaml
  python automation/pipeline.py compile STR_001_asian_mr_fx
  python automation/pipeline.py backtest STR_001_asian_mr_fx --period all
  python automation/pipeline.py analyze STR_001_asian_mr_fx
  python automation/pipeline.py verdict STR_001_asian_mr_fx
  python automation/pipeline.py full strategy_specs/my_spec.yaml
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from automation import ea_generator
from automation import mt5_compiler
from automation import mt5_tester
from automation import hypothesis_logger
from automation.spec_validator import validate_spec

from analysis import report_parser
from analysis import metrics_calculator
from analysis import bootstrap_validator
from analysis import walk_forward
from analysis import parameter_sensitivity
from analysis import pnl_decomposer
from analysis import acceptance_check

console = Console()


def setup_logging() -> Path:
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_{datetime.utcnow().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return log_file


def banner(text: str, color: str = "cyan") -> None:
    console.print(Panel.fit(text, border_style=color))


@click.group()
def cli():
    """StrategyFactory pipeline."""
    setup_logging()


# ============================================================
# Commands
# ============================================================
@cli.command()
@click.argument("spec_path", type=click.Path(exists=True, dir_okay=False))
def prepare(spec_path: str):
    """Validate YAML spec and generate the Claude Code EA prompt."""
    p = Path(spec_path)
    banner(f"PREPARE — {p.name}")

    spec = validate_spec(p)
    console.print(f"[green]✅ Spec validated:[/green] {spec.meta.strategy_id}")

    res = ea_generator.generate(p)
    console.print(Panel(
        f"EA skeleton:    {res['skeleton_mq5']}\n"
        f"Generation prompt:  {res['prompt_md']}\n\n"
        f"[yellow]NEXT MANUAL STEP:[/yellow]\n"
        f"  1. Open the prompt file in VS Code.\n"
        f"  2. Send its full content to Claude Code.\n"
        f"  3. Save the AI-completed .mq5 over the skeleton.\n"
        f"  4. Run: pipeline.py compile {res['strategy_id']}",
        title="PREPARE — DONE", border_style="green"
    ))


@cli.command()
@click.argument("strategy_id")
def compile(strategy_id: str):
    """Compile the AI-completed .mq5."""
    banner(f"COMPILE — {strategy_id}")
    mq5_path = ROOT / "mql5" / "generated" / f"{strategy_id}.mq5"
    if not mq5_path.exists():
        console.print(f"[red]Not found: {mq5_path}[/red]")
        sys.exit(1)

    res = mt5_compiler.compile_ea(mq5_path)
    mt5_compiler.print_result(res, mq5_path)
    if not res.success:
        sys.exit(1)


@cli.command()
@click.argument("strategy_id")
@click.option("--period", default="all", type=click.Choice(["is", "oos", "forward", "all"]))
def backtest(strategy_id: str, period: str):
    """Run MT5 Strategy Tester."""
    banner(f"BACKTEST — {strategy_id} (period={period})")
    results = mt5_tester.run_backtest(strategy_id, period)
    n_ok = sum(1 for r in results if r.success)
    console.print(f"[bold]{n_ok}/{len(results)} period(s) succeeded.[/bold]")
    if n_ok < len(results):
        sys.exit(1)


@cli.command()
@click.argument("strategy_id")
def analyze(strategy_id: str):
    """Parse reports + run all analysis modules."""
    banner(f"ANALYZE — {strategy_id}")

    for period in ["is", "oos", "forward"]:
        report = report_parser.find_latest_report(strategy_id, period)
        if not report:
            console.print(f"[yellow]No report for period={period} — skipping[/yellow]")
            continue
        console.print(f"[cyan]Parsing {period}...[/cyan]")
        report_parser.parse_report(report, strategy_id, period)

        console.print(f"[cyan]Computing metrics for {period}...[/cyan]")
        parsed = metrics_calculator.load_parsed(strategy_id, period)
        m = metrics_calculator.compute_metrics(parsed)
        metrics_calculator.save_metrics(strategy_id, period, m)

        console.print(f"[cyan]Bootstrap p-value for {period}...[/cyan]")
        bootstrap_validator.run(strategy_id, period)

        console.print(f"[cyan]PnL decomposition for {period}...[/cyan]")
        pnl_decomposer.run(strategy_id, period)

    # Walk-forward needs both IS + OOS
    if (ROOT / "backtests" / "parsed_results" / strategy_id / "is_parsed.json").exists() and \
       (ROOT / "backtests" / "parsed_results" / strategy_id / "oos_parsed.json").exists():
        console.print("[cyan]Walk-forward analysis...[/cyan]")
        walk_forward.run(strategy_id)

    console.print("[cyan]Sensitivity (stub mode)...[/cyan]")
    sens_payload = parameter_sensitivity.stub_from_metrics(strategy_id)
    parameter_sensitivity.save(strategy_id, sens_payload)

    console.print("[green]✅ All analyses complete.[/green]")


@cli.command()
@click.argument("strategy_id")
@click.option("--auto-confirm", is_flag=True, help="Skip confirmation prompt on PASS.")
def verdict(strategy_id: str, auto_confirm: bool):
    """Run acceptance check and write to HYPOTHESIS_LOG.md."""
    banner(f"VERDICT — {strategy_id}")
    v = acceptance_check.evaluate(strategy_id)
    acceptance_check.print_verdict(v)

    if v["verdict"] == "PASS":
        if auto_confirm or Confirm.ask("Log this PASS to HYPOTHESIS_LOG.md?", default=True):
            hypothesis_logger.append_entry(v)
            console.print("[green]✅ Logged to docs/HYPOTHESIS_LOG.md[/green]")
            console.print("[cyan]Next step: deploy to demo for 30-day forward test.[/cyan]")
    else:
        hypothesis_logger.append_entry(v)
        console.print("[red]❌ FAIL logged. Hypothesis archived for research history.[/red]")


@cli.command()
@click.argument("spec_path", type=click.Path(exists=True, dir_okay=False))
def full(spec_path: str):
    """End-to-end pipeline with pause at manual EA generation."""
    p = Path(spec_path)
    banner("FULL PIPELINE", color="magenta")

    spec = validate_spec(p)
    sid = spec.meta.strategy_id

    # 1. Prepare
    console.print("[bold]Step 1/5 — PREPARE[/bold]")
    res = ea_generator.generate(p)
    console.print(Panel(
        f"[yellow]MANUAL STEP REQUIRED.[/yellow]\n\n"
        f"  Prompt:   {res['prompt_md']}\n"
        f"  Skeleton: {res['skeleton_mq5']}\n\n"
        f"Feed the prompt to Claude Code, then save the completed .mq5 over the skeleton.\n",
        title="PAUSE", border_style="yellow"
    ))
    if not Confirm.ask("Have you completed and saved the EA?"):
        console.print("[red]Aborted at manual step.[/red]")
        sys.exit(1)

    # 2. Compile
    console.print("\n[bold]Step 2/5 — COMPILE[/bold]")
    cres = mt5_compiler.compile_ea(ROOT / "mql5" / "generated" / f"{sid}.mq5")
    mt5_compiler.print_result(cres, ROOT / "mql5" / "generated" / f"{sid}.mq5")
    if not cres.success:
        sys.exit(1)

    # 3. Backtest all periods
    console.print("\n[bold]Step 3/5 — BACKTEST (IS, OOS, FORWARD)[/bold]")
    bres = mt5_tester.run_backtest(sid, "all")
    if not all(b.success for b in bres):
        console.print("[red]Some backtests failed. Investigate then re-run.[/red]")
        sys.exit(1)

    # 4. Analyze
    console.print("\n[bold]Step 4/5 — ANALYZE[/bold]")
    for period in ["is", "oos", "forward"]:
        report = report_parser.find_latest_report(sid, period)
        if not report: continue
        report_parser.parse_report(report, sid, period)
        parsed = metrics_calculator.load_parsed(sid, period)
        m = metrics_calculator.compute_metrics(parsed)
        metrics_calculator.save_metrics(sid, period, m)
        bootstrap_validator.run(sid, period)
        pnl_decomposer.run(sid, period)
    if (ROOT / "backtests" / "parsed_results" / sid / "is_parsed.json").exists() and \
       (ROOT / "backtests" / "parsed_results" / sid / "oos_parsed.json").exists():
        walk_forward.run(sid)
    parameter_sensitivity.save(sid, parameter_sensitivity.stub_from_metrics(sid))

    # 5. Verdict
    console.print("\n[bold]Step 5/5 — VERDICT[/bold]")
    v = acceptance_check.evaluate(sid)
    acceptance_check.print_verdict(v)
    hypothesis_logger.append_entry(v)
    console.print("[green]✅ Pipeline complete. Verdict logged.[/green]")


if __name__ == "__main__":
    cli()
