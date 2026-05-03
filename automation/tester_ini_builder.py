"""
StrategyFactory — Tester.ini Builder
====================================
Generates a tester.ini file for MT5 Strategy Tester batch runs.

Reference: https://www.mql5.com/en/docs/runtime/testing
"""

from __future__ import annotations

from pathlib import Path
from datetime import date
from typing import Literal

import yaml

ROOT = Path(__file__).parent.parent

TF_MAP = {
    "M1":  "M1",  "M5":  "M5",  "M15": "M15", "M30": "M30",
    "H1":  "H1",  "H4":  "H4",  "D1":  "Daily",
}

MODELING_MAP = {
    "every_tick_real": 0,  # MODE_EVERY_TICK_REAL
    "1m_ohlc":         1,  # MODE_1M_OHLC
    "open_prices":     2,  # MODE_OPEN_PRICES
}


def load_paths() -> dict:
    cfg = ROOT / "config" / "mt5_paths.yaml"
    with cfg.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_ini(spec, period: Literal["is", "oos", "forward"], ex5_path: Path) -> Path:
    """
    Build tester.ini from a validated spec.
    Returns path to the generated .ini file.
    """
    paths = load_paths()
    bt_defaults = paths.get("backtest", {})
    portable = paths["mt5"].get("portable_mode", False)

    if period == "is":
        p = spec.backtest.is_period
    elif period == "oos":
        p = spec.backtest.oos_period
    elif period == "forward":
        p = spec.backtest.forward_period
    else:
        raise ValueError(f"Unknown period: {period}")

    symbol = spec.universe.symbols[0]  # MT5 tester is single-symbol
    timeframe = TF_MAP[spec.universe.timeframe]
    modeling_id = MODELING_MAP[spec.backtest.modeling]

    expert_relative = f"StrategyFactory\\{ex5_path.stem}.ex5"

    # Tester wants Expert relative to MQL5/Experts/
    ini_lines = [
        "[Tester]",
        f"Expert={expert_relative}",
        f"Symbol={symbol}",
        f"Period={timeframe}",
        f"Optimization=0",
        f"Model={modeling_id}",
        f"FromDate={p.start.strftime('%Y.%m.%d')}",
        f"ToDate={p.end.strftime('%Y.%m.%d')}",
        f"ForwardMode=0",
        f"Deposit={int(spec.backtest.initial_balance)}",
        f"Currency={spec.backtest.currency}",
        f"ProfitInPips=0",
        f"Leverage={spec.backtest.leverage}",
        f"ExecutionMode=0",
        f"OptimizationCriterion=6",  # Custom (uses OnTester)
        f"Visual=0",
        f"ShutdownTerminal=1",
        f"UseLocal=1",
        f"Replace=1",
        "",
        "[TesterInputs]",
    ]

    # All EA inputs from the spec are passed as overrides
    inputs = build_input_lines(spec)
    ini_lines.extend(inputs)

    # Save to a known location so mt5_tester can find it
    ini_dir = ROOT / "backtests" / "raw_reports" / spec.meta.strategy_id
    ini_dir.mkdir(parents=True, exist_ok=True)
    ini_path = ini_dir / f"tester_{period}.ini"
    ini_path.write_text("\n".join(ini_lines), encoding="utf-16-le")
    # MT5 wants UTF-16 LE with BOM
    raw = open(ini_path, "rb").read()
    with open(ini_path, "wb") as f:
        f.write(b"\xff\xfe" + raw)
    return ini_path


def build_input_lines(spec) -> list[str]:
    """Convert spec inputs to TesterInputs lines.

    MT5 tester input format: Name=value||from||step||to||use_in_optimization
    For single-run: Name=value||value||0||value||N
    """
    lines = []
    risk = spec.risk
    sess = spec.universe.trading_session_utc

    sizing_int = {"fixed_lot": 0, "risk_per_trade_pct": 1, "kelly_quarter": 2}[risk.position_sizing]

    pairs = {
        "InpStrategyId":            f"{spec.meta.strategy_id}",
        "InpMagic":                 None,  # taken from compiled-in default
        "InpLogLevel":              1,     # INFO
        "InpSizingMethod":          sizing_int,
        "InpFixedLot":              risk.fixed_lot if risk.fixed_lot else 0.01,
        "InpRiskPerTradePct":       risk.risk_per_trade_pct if risk.risk_per_trade_pct else 0.5,
        "InpMaxConcurrent":         risk.max_concurrent_positions,
        "InpMaxConcurrentPerSym":   risk.max_concurrent_per_symbol,
        "InpMaxDailyLossPct":       risk.max_daily_loss_pct,
        "InpMaxDDPct":              risk.max_drawdown_pct_circuit_breaker,
        "InpMaxSlippagePts":        risk.max_slippage_points,
        "InpSessionStartUTC":       sess.start,
        "InpSessionEndUTC":         sess.end,
        "InpSessionCrossesMidnight": "true" if sess.crosses_midnight else "false",
        "InpStopLossATRMult":       spec.exit_rules.stop_loss.value if spec.exit_rules.stop_loss.type == "atr_multiple" else 1.5,
        "InpTakeProfitATRMult":     spec.exit_rules.take_profit.value if spec.exit_rules.take_profit.type == "atr_multiple" else 2.0,
        "InpATRPeriod":             spec.exit_rules.stop_loss.atr_period or 14,
        "InpTimeExitEnabled":       "true" if spec.exit_rules.time_exit.enabled else "false",
        "InpMaxHoldingBars":        spec.exit_rules.time_exit.max_holding_bars or 16,
        "InpDrawLevels":            "false",
    }

    for k, v in pairs.items():
        if v is None:
            continue
        if isinstance(v, str) and v not in ("true", "false"):
            # String input: quote
            lines.append(f"{k}={v}||{v}||0||{v}||N")
        else:
            lines.append(f"{k}={v}||{v}||0||{v}||N")

    return lines
