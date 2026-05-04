"""Tests for ``python_engine.report_generator``.

Verifies:
1. ``generate_final_report`` writes ``final_report.md`` with all required
   sections, the top-N table, and the summary block.
2. The verdict logic correctly tags clean PASS rows, borderline rows, and
   reject rows from the CSV columns alone (no re-run needed).
3. Equity-curve plotting and per-symbol breakdown succeed when a runnable
   strategy + cached data are present, and degrade gracefully without them.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from automation.generate_strategy import generate_strategy_skeleton  # noqa: E402
from python_engine.report_generator import (  # noqa: E402
    DEFAULT_ACCEPTANCE,
    _row_verdict,
    generate_final_report,
    generate_per_symbol_breakdown,
    plot_equity_curves,
)
from python_engine.vectorized_backtest import SymbolSpec  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bootstrapped_strategy(tmp_path: Path) -> Path:
    """Create a strategies/<name>/ skeleton plus a runnable strategy.py."""
    paper = tmp_path / "paper.txt"
    paper.write_text("Paper content.")
    folder = generate_strategy_skeleton(
        paper_path=paper,
        name="report_demo",
        logic_description="rsi mean rev max 8 bars",
        symbols=["EURUSD"],
        is_period=(date(2018, 1, 1), date(2018, 6, 30)),
        oos_period=(date(2018, 7, 1), date(2018, 12, 31)),
        base_dir=tmp_path / "strategies",
    )
    # Replace the NotImplementedError stub with a trivial runnable signal
    # generator so the report can re-run the OOS slice for detail.
    (folder / "strategy.py").write_text(dedent("""
        import pandas as pd

        DEFAULT_PARAMS = {"step": 50}

        def signals(df, **params):
            n = len(df)
            sl = pd.Series(0.0010, index=df.index)
            tp = pd.Series(0.0010, index=df.index)
            sig_long = pd.Series(False, index=df.index)
            sig_short = pd.Series(False, index=df.index)
            step = max(1, int(params.get("step", DEFAULT_PARAMS["step"])))
            for i in range(0, n - 2, step):
                sig_long.iloc[i] = True
            return {
                "signal_long": sig_long,
                "signal_short": sig_short,
                "sl_distance": sl,
                "tp_distance": tp,
            }
    """).lstrip())
    return folder


def _write_oos_parquet(folder: Path, symbol: str = "EURUSD") -> None:
    """Drop a synthetic OOS-window parquet into ``data_cache/``."""
    idx = pd.date_range("2018-07-01", "2018-12-31", freq="1h")
    n = len(idx)
    base = 1.10 + np.arange(n) * 1e-5
    df = pd.DataFrame({
        "open": base, "high": base + 0.0005, "low": base - 0.0001,
        "close": base + 0.0002, "spread": np.zeros(n),
    }, index=idx)
    out = folder / "data_cache" / f"{symbol}_M1_20180701_20181231.parquet"
    df.to_parquet(out)


def _toy_top(n: int = 3) -> pd.DataFrame:
    """Synthetic top-N optimizer output (matching get_top_n columns)."""
    return pd.DataFrame([
        {"rank": 1, "trial_id": 847, "params": '{"step": 50}',
         "is_sharpe_avg": 1.61, "oos_sharpe_avg": 1.42,
         "is_sharpe_min": 1.4, "oos_sharpe_min": 1.1,
         "max_dd_pct": -8.3, "total_trades": 487, "objective_value": 1.42},
        {"rank": 2, "trial_id": 312, "params": '{"step": 80}',
         "is_sharpe_avg": 1.50, "oos_sharpe_avg": 1.10,
         "is_sharpe_min": 1.2, "oos_sharpe_min": 0.9,
         "max_dd_pct": -10.0, "total_trades": 320, "objective_value": 1.10},
        {"rank": 3, "trial_id": 100, "params": '{"step": 200}',
         "is_sharpe_avg": 1.00, "oos_sharpe_avg": 0.60,
         "is_sharpe_min": 0.7, "oos_sharpe_min": 0.4,
         "max_dd_pct": -14.0, "total_trades": 110, "objective_value": 0.60},
    ][:n])


def _toy_all(n_total: int = 40, pruned_every: int = 4) -> pd.DataFrame:
    return pd.DataFrame([{
        "trial_id": i,
        "pruned": i % pruned_every == 0,
        "objective_value": 0.1, "params": "{}",
        "is_sharpe_avg": 0.5, "oos_sharpe_avg": 0.3,
        "is_sharpe_min": 0.2, "oos_sharpe_min": 0.1,
        "max_dd_pct": -5.0, "total_trades": 50,
    } for i in range(n_total)])


# ---------------------------------------------------------------------------
# 1. _row_verdict
# ---------------------------------------------------------------------------

def test_row_verdict_recommend():
    row = {"is_sharpe_avg": 1.5, "oos_sharpe_avg": 1.2, "max_dd_pct": -8.0,
           "total_trades": 400}
    verdict, reasons = _row_verdict(row, DEFAULT_ACCEPTANCE)
    assert verdict == "Recommend" and reasons == []


def test_row_verdict_reject_when_sharpe_low():
    row = {"is_sharpe_avg": 0.2, "oos_sharpe_avg": 0.1, "max_dd_pct": -3.0,
           "total_trades": 400}
    verdict, reasons = _row_verdict(row, DEFAULT_ACCEPTANCE)
    assert verdict == "Reject"
    assert any("OOS Sharpe" in r for r in reasons)


def test_row_verdict_conditional_when_one_check_misses():
    """One miss + OOS Sharpe ≥ 70% threshold + DD within 1.5x = Conditional."""
    row = {"is_sharpe_avg": 1.5, "oos_sharpe_avg": 0.5,  # 0.5 ≥ 0.7 * 0.6 = 0.42
           "max_dd_pct": -10.0, "total_trades": 400}
    verdict, reasons = _row_verdict(row, DEFAULT_ACCEPTANCE)
    assert verdict == "Conditional" and len(reasons) == 1


def test_row_verdict_wti_guard_blocks():
    row = {"is_sharpe_avg": 1.5, "oos_sharpe_avg": 1.2, "max_dd_pct": -8.0,
           "total_trades": 400}
    # A clean row — but directional 30% blows the WTI guard.
    verdict, reasons = _row_verdict(row, DEFAULT_ACCEPTANCE, directional_pct=30.0)
    assert verdict in ("Conditional", "Reject")
    assert any("directional" in r for r in reasons)


# ---------------------------------------------------------------------------
# 2. generate_final_report — minimal mode (no re-run)
# ---------------------------------------------------------------------------

def test_generate_final_report_minimal_writes_all_sections(tmp_path: Path):
    folder = _bootstrapped_strategy(tmp_path)
    out = generate_final_report(
        folder, _toy_top(3), _toy_all(40),
        optimization_summary={"wall_time_seconds": 1114,
                               "trials_run": 1000,
                               "trials_pruned": 230},
        plot_curves=False, write_per_symbol=False,
    )
    text = out.read_text(encoding="utf-8")
    for header in (
        "## 1. Strategy Hypothesis",
        "## 2. Optimization Summary",
        "## 3. Top-3 Performers",
        "## 4. Top-3 Detail",
        "## 5. Recommendation",
        "## 6. Next Steps",
    ):
        assert header in text, f"missing section {header!r}"
    assert "Trials run" in text and "1000" in text
    assert "18m 34s" in text  # 1114 seconds → 18m34s
    # The top-1 row (Sharpe 1.42, 487 trades, DD 8.3%) must appear.
    assert "1.42" in text and "487" in text
    # Without per-symbol re-run, detail sections still render the placeholder.
    assert "(unavailable" in text or "Per-symbol Sharpe" in text


def test_generate_final_report_handles_empty_top(tmp_path: Path):
    folder = _bootstrapped_strategy(tmp_path)
    empty_top = pd.DataFrame(columns=[
        "rank", "trial_id", "params", "is_sharpe_avg", "oos_sharpe_avg",
        "is_sharpe_min", "oos_sharpe_min", "max_dd_pct", "total_trades",
        "objective_value",
    ])
    out = generate_final_report(
        folder, empty_top, _toy_all(10),
        plot_curves=False, write_per_symbol=False,
    )
    text = out.read_text(encoding="utf-8")
    assert "No top performers" in text or "No surviving" in text


# ---------------------------------------------------------------------------
# 3. plot_equity_curves + per-symbol breakdown — runnable strategy + cache
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_symbol_spec(monkeypatch):
    """Stub SymbolSpec.from_config so tests don't need the real config files."""
    def _fake_from_config(symbol: str, *_, **__):
        return SymbolSpec(
            symbol=symbol,
            point_size=0.00001,
            usd_per_price_unit=100_000.0,
            commission_per_lot=7.0,
            swap_long=0.0,
            swap_short=0.0,
            triple_swap_weekday=2,
        )
    monkeypatch.setattr(SymbolSpec, "from_config", staticmethod(_fake_from_config))


def test_plot_equity_curves_writes_pngs(tmp_path: Path):
    folder = _bootstrapped_strategy(tmp_path)
    _write_oos_parquet(folder)
    # The function reads top_*_performers.csv from results/
    (folder / "results").mkdir(exist_ok=True)
    _toy_top(2).to_csv(folder / "results" / "top_2_performers.csv", index=False)

    written = plot_equity_curves(folder, top_n=2)
    assert len(written) >= 1
    for p in written:
        assert p.exists() and p.suffix == ".png" and p.stat().st_size > 0


def test_plot_equity_curves_no_strategy_returns_empty(tmp_path: Path):
    """If no strategy.py exists, the function logs and returns []."""
    empty = tmp_path / "empty"
    (empty / "results").mkdir(parents=True)
    _toy_top(1).to_csv(empty / "results" / "top_1_performers.csv", index=False)
    assert plot_equity_curves(empty, top_n=1) == []


def test_per_symbol_breakdown_writes_csvs(tmp_path: Path):
    folder = _bootstrapped_strategy(tmp_path)
    _write_oos_parquet(folder)
    written = generate_per_symbol_breakdown(
        folder, _toy_top(1).to_dict(orient="records"),
    )
    assert any(p.name == "EURUSD_trades.csv" for p in written)
    assert any(p.name == "EURUSD_monthly.csv" for p in written)
    assert any(p.name == "EURUSD_drawdowns.csv" for p in written)


def test_full_report_with_rerun_includes_decomposition(tmp_path: Path):
    folder = _bootstrapped_strategy(tmp_path)
    _write_oos_parquet(folder)
    out = generate_final_report(
        folder, _toy_top(3), _toy_all(40),
        optimization_summary={"trials_run": 1000, "trials_pruned": 230},
        plot_curves=False, write_per_symbol=False,
    )
    text = out.read_text(encoding="utf-8")
    # When the OOS re-run produces results, decomposition lines populate.
    assert "PnL decomposition" in text
    assert ("directional" in text.lower()) or ("unavailable" in text)
    # YAML period block was reused → both IS and OOS dates should appear.
    assert "2018-07-01" in text
