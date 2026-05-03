"""
StrategyFactory — Metrics Calculator
====================================
Computes annualized risk/return metrics from parsed trade data.

Output: backtests/parsed_results/{strategy_id}/{period}_metrics.json
"""

from __future__ import annotations

import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).parent.parent
console = Console()
ENGINE_VERSION = "StrategyFactory-1.0"

# Trading days in a year (annualization factor for daily returns)
TRADING_DAYS = 252


def load_parsed(strategy_id: str, period: str) -> dict:
    p = ROOT / "backtests" / "parsed_results" / strategy_id / f"{period}_parsed.json"
    return json.loads(p.read_text(encoding="utf-8"))


def trades_to_daily_returns(trades: list[dict], initial_balance: float) -> pd.DataFrame:
    """Convert trade list to daily P&L series, then daily return %."""
    if not trades:
        return pd.DataFrame(columns=["date", "pnl", "balance", "ret"])

    df = pd.DataFrame(trades)
    df["close_time"] = pd.to_datetime(df["close_time"], errors="coerce")
    df = df.dropna(subset=["close_time"])
    df["date"] = df["close_time"].dt.date

    daily = df.groupby("date")["profit_total"].sum().reset_index()
    daily.columns = ["date", "pnl"]

    # Build complete date index between min and max date
    full_idx = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D").date
    daily = daily.set_index("date").reindex(full_idx, fill_value=0.0).reset_index()
    daily.columns = ["date", "pnl"]

    daily["balance"] = initial_balance + daily["pnl"].cumsum()
    daily["ret"] = daily["balance"].pct_change().fillna(0.0)
    return daily


def compute_metrics(parsed: dict) -> dict:
    summary = parsed["summary"]
    trades = parsed["trades"]
    initial_balance = summary.get("initial_balance", 10000.0)

    daily = trades_to_daily_returns(trades, initial_balance)

    if len(daily) < 2:
        return {
            "warning": "Not enough data to compute metrics.",
            "trade_count": len(trades),
        }

    rets = daily["ret"].values
    mean_r = np.mean(rets)
    std_r  = np.std(rets, ddof=1) if len(rets) > 1 else 0.0
    downside = rets[rets < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0

    sharpe  = (mean_r / std_r) * np.sqrt(TRADING_DAYS) if std_r > 0 else 0.0
    sortino = (mean_r / down_std) * np.sqrt(TRADING_DAYS) if down_std > 0 else 0.0

    # Max drawdown (from balance series)
    bal = daily["balance"].values
    peak = np.maximum.accumulate(bal)
    dd = (peak - bal) / peak
    max_dd = float(dd.max()) if len(dd) > 0 else 0.0
    max_dd_idx = int(np.argmax(dd)) if len(dd) > 0 else 0

    # DD duration (in days from prev peak to current point)
    if max_dd_idx > 0:
        peak_idx = int(np.argmax(bal[:max_dd_idx + 1]))
        max_dd_duration = max_dd_idx - peak_idx
    else:
        max_dd_duration = 0

    # Calmar
    days = (daily["date"].iloc[-1] - daily["date"].iloc[0]).days
    years = max(days / 365.25, 0.0001)
    total_return = bal[-1] / bal[0] - 1.0
    cagr = (1 + total_return) ** (1 / years) - 1
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    # MAR (annualized return / max DD)
    mar = cagr / max_dd if max_dd > 0 else 0.0

    # Win/loss
    wins = [t for t in trades if t.get("profit_total", 0) > 0]
    losses = [t for t in trades if t.get("profit_total", 0) < 0]
    avg_win = np.mean([t["profit_total"] for t in wins]) if wins else 0.0
    avg_loss = np.mean([t["profit_total"] for t in losses]) if losses else 0.0
    win_rate = len(wins) / len(trades) if trades else 0.0
    profit_factor = (sum(t["profit_total"] for t in wins)
                     / abs(sum(t["profit_total"] for t in losses))) if losses else float("inf")

    # Skew, kurt of daily returns (Fisher: normal kurt = 0)
    skew = float(pd.Series(rets).skew()) if len(rets) > 2 else 0.0
    kurt = float(pd.Series(rets).kurt()) if len(rets) > 3 else 0.0

    # Avg holding (use bars if available, else trade count proxy)
    holding_days = 0.0
    times_open = pd.to_datetime([t.get("open_time") for t in trades], errors="coerce")
    times_close = pd.to_datetime([t.get("close_time") for t in trades], errors="coerce")
    if len(times_open) and not pd.isna(times_open).all():
        durations = (times_close - times_open).total_seconds() / 86400.0
        holding_days = float(np.nanmean(durations))

    # Time in market %
    if days > 0:
        days_with_trades = (daily["pnl"] != 0).sum()
        time_in_market = days_with_trades / len(daily)
    else:
        time_in_market = 0.0

    metrics = {
        "trade_count": len(trades),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": win_rate,
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "avg_win_loss_ratio": float(abs(avg_win / avg_loss)) if avg_loss != 0 else 0.0,
        "profit_factor": float(profit_factor) if profit_factor != float("inf") else None,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "mar_ratio": float(mar),
        "max_drawdown_pct": float(max_dd * 100.0),
        "max_drawdown_duration_days": int(max_dd_duration),
        "skewness_daily": skew,
        "kurtosis_daily_excess": kurt,
        "avg_holding_days": holding_days,
        "time_in_market_pct": float(time_in_market * 100.0),
        "first_date": str(daily["date"].iloc[0]),
        "last_date": str(daily["date"].iloc[-1]),
    }
    return metrics


def save_metrics(strategy_id: str, period: str, metrics: dict) -> Path:
    out_dir = ROOT / "backtests" / "parsed_results" / strategy_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "strategy_id": strategy_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "engine_version": ENGINE_VERSION,
        "period": period,
        "metrics": metrics,
    }
    p = out_dir / f"{period}_metrics.json"
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return p


def print_metrics(metrics: dict) -> None:
    t = Table(title="Metrics", show_header=True)
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="white")
    for k, v in metrics.items():
        if isinstance(v, float):
            t.add_row(k, f"{v:.4f}")
        else:
            t.add_row(k, str(v))
    console.print(t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy_id")
    ap.add_argument("--period", default="is", choices=["is", "oos", "forward"])
    args = ap.parse_args()

    parsed = load_parsed(args.strategy_id, args.period)
    metrics = compute_metrics(parsed)
    out = save_metrics(args.strategy_id, args.period, metrics)
    print_metrics(metrics)
    console.print(f"[green]Saved:[/green] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
