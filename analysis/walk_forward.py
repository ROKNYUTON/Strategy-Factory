"""
StrategyFactory — Walk-Forward Analyzer
=======================================
Computes WFA efficiency: ratio of OOS Sharpe to IS Sharpe across rolling windows.
Note: This is a LIGHTWEIGHT WFA — it splits the parsed trades by date windows
and computes Sharpe per window. Real "optimization-then-test" WFA requires
re-running MT5 tester per window (handled separately by future feature).

Output: backtests/parsed_results/{strategy_id}/walk_forward.json
"""

from __future__ import annotations

import json
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).parent.parent
console = Console()
ENGINE_VERSION = "StrategyFactory-1.0"
TRADING_DAYS = 252


def load_parsed(strategy_id: str, period: str) -> dict:
    p = ROOT / "backtests" / "parsed_results" / strategy_id / f"{period}_parsed.json"
    return json.loads(p.read_text(encoding="utf-8"))


def trades_to_daily(trades: list[dict], initial: float) -> pd.DataFrame:
    df = pd.DataFrame(trades)
    if df.empty:
        return df
    df["close_time"] = pd.to_datetime(df["close_time"], errors="coerce")
    df = df.dropna(subset=["close_time"])
    df["date"] = df["close_time"].dt.date
    daily = df.groupby("date")["profit_total"].sum().reset_index()
    daily.columns = ["date", "pnl"]
    daily["balance"] = initial + daily["pnl"].cumsum()
    daily["ret"] = daily["balance"].pct_change().fillna(0.0)
    return daily


def sharpe(rets: np.ndarray) -> float:
    if len(rets) < 2: return 0.0
    s = rets.std(ddof=1)
    return (rets.mean() / s) * np.sqrt(TRADING_DAYS) if s > 0 else 0.0


def walk_forward(trades_is: list[dict], trades_oos: list[dict],
                 initial: float, window_months: int = 6,
                 step_months: int = 3) -> dict:
    """
    Split IS trades into rolling windows and compute Sharpe per window.
    Then compute the OOS Sharpe and the WFA efficiency = OOS / IS_avg.
    """
    daily_is = trades_to_daily(trades_is, initial)
    daily_oos = trades_to_daily(trades_oos, initial)
    if daily_is.empty:
        return {"error": "No IS trades."}

    # Build rolling windows
    start = pd.to_datetime(daily_is["date"].iloc[0])
    end = pd.to_datetime(daily_is["date"].iloc[-1])
    windows = []
    cur = start
    while cur + pd.DateOffset(months=window_months) <= end:
        w_start = cur.date()
        w_end = (cur + pd.DateOffset(months=window_months)).date()
        sub = daily_is[(daily_is["date"] >= w_start) & (daily_is["date"] < w_end)]
        if len(sub) >= 20:
            windows.append({
                "start": str(w_start),
                "end": str(w_end),
                "n_days": int(len(sub)),
                "sharpe": float(sharpe(sub["ret"].values)),
            })
        cur = cur + pd.DateOffset(months=step_months)

    is_sharpes = [w["sharpe"] for w in windows]
    is_avg_sharpe = float(np.mean(is_sharpes)) if is_sharpes else 0.0
    is_min_sharpe = float(np.min(is_sharpes)) if is_sharpes else 0.0
    oos_sharpe = float(sharpe(daily_oos["ret"].values)) if not daily_oos.empty else 0.0
    efficiency = (oos_sharpe / is_avg_sharpe) if is_avg_sharpe > 0 else 0.0

    consistency = float((np.array(is_sharpes) > 0).sum() / len(is_sharpes)) if is_sharpes else 0.0

    return {
        "n_windows": len(windows),
        "windows": windows,
        "is_avg_sharpe": is_avg_sharpe,
        "is_min_sharpe": is_min_sharpe,
        "oos_sharpe": oos_sharpe,
        "wfa_efficiency": efficiency,
        "consistency_pct_positive_windows": consistency,
    }


def run(strategy_id: str) -> dict:
    is_p = load_parsed(strategy_id, "is")
    oos_p = load_parsed(strategy_id, "oos")
    initial = is_p["summary"].get("initial_balance", 10000.0)
    res = walk_forward(is_p["trades"], oos_p["trades"], initial)

    out = {
        "strategy_id": strategy_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "engine_version": ENGINE_VERSION,
        "walk_forward": res,
    }
    out_dir = ROOT / "backtests" / "parsed_results" / strategy_id
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "walk_forward.json"
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy_id")
    args = ap.parse_args()
    out = run(args.strategy_id)
    wf = out["walk_forward"]

    t = Table(title="Walk-Forward Analysis")
    t.add_column("Metric"); t.add_column("Value")
    t.add_row("n_windows", str(wf.get("n_windows", 0)))
    t.add_row("is_avg_sharpe", f"{wf.get('is_avg_sharpe', 0):.3f}")
    t.add_row("is_min_sharpe", f"{wf.get('is_min_sharpe', 0):.3f}")
    t.add_row("oos_sharpe", f"{wf.get('oos_sharpe', 0):.3f}")
    t.add_row("wfa_efficiency (target > 0.5)", f"{wf.get('wfa_efficiency', 0):.3f}")
    t.add_row("consistency", f"{wf.get('consistency_pct_positive_windows', 0):.2%}")
    console.print(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
