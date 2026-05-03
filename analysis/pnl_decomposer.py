"""
StrategyFactory — PnL Decomposer
================================
Aggregates per-trade decomposition into directional / swap / commission totals.
Triggers the WTI-lesson guard: if directional < 60% of positive PnL, FLAG.

Output: backtests/parsed_results/{strategy_id}/{period}_pnl_decomp.json
"""

from __future__ import annotations

import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).parent.parent
console = Console()
ENGINE_VERSION = "StrategyFactory-1.0"


def load_parsed(strategy_id: str, period: str) -> dict:
    p = ROOT / "backtests" / "parsed_results" / strategy_id / f"{period}_parsed.json"
    return json.loads(p.read_text(encoding="utf-8"))


def decompose(parsed: dict) -> dict:
    trades = parsed.get("trades", [])
    if not trades:
        return {"warning": "No trades to decompose."}

    df = pd.DataFrame(trades)
    cols = ["profit_directional", "profit_swap", "profit_commission", "profit_total"]
    for c in cols:
        if c not in df.columns: df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    total = float(df["profit_total"].sum())
    direct = float(df["profit_directional"].sum())
    swap = float(df["profit_swap"].sum())
    comm = float(df["profit_commission"].sum())

    abs_total = abs(total) if total != 0 else 1.0
    pct = {
        "directional_pct": (direct / abs_total) * 100.0,
        "swap_pct": (swap / abs_total) * 100.0,
        "commission_pct": (comm / abs_total) * 100.0,
    }

    # Per-symbol breakdown
    per_symbol = (df.groupby("symbol")[cols].sum().reset_index().to_dict(orient="records"))

    # Per-month breakdown
    df["close_time"] = pd.to_datetime(df["close_time"], errors="coerce")
    df["month"] = df["close_time"].dt.to_period("M").astype(str)
    per_month = (df.groupby("month")[cols].sum().reset_index().to_dict(orient="records"))

    # WTI-lesson guard
    flag = False
    flag_reason = ""
    if total > 0:
        positive_total = direct + (swap if swap > 0 else 0)
        directional_share = direct / positive_total if positive_total > 0 else 0
        if directional_share < 0.60:
            flag = True
            flag_reason = (
                f"Directional contribution is {directional_share*100:.1f}% of positive PnL "
                f"(< 60% threshold). Strategy may be a swap/carry trade in disguise. WTI lesson."
            )

    return {
        "totals": {
            "directional": direct,
            "swap": swap,
            "commission": comm,
            "total": total,
        },
        "percentages": pct,
        "per_symbol": per_symbol,
        "per_month": per_month,
        "wti_guard": {
            "flagged": flag,
            "reason": flag_reason,
        },
    }


def run(strategy_id: str, period: str) -> dict:
    parsed = load_parsed(strategy_id, period)
    res = decompose(parsed)

    out = {
        "strategy_id": strategy_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "engine_version": ENGINE_VERSION,
        "period": period,
        "decomposition": res,
    }
    out_dir = ROOT / "backtests" / "parsed_results" / strategy_id
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{period}_pnl_decomp.json"
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy_id")
    ap.add_argument("--period", default="is", choices=["is", "oos", "forward"])
    args = ap.parse_args()
    out = run(args.strategy_id, args.period)
    d = out["decomposition"]

    if "warning" in d:
        console.print(f"[yellow]{d['warning']}[/yellow]")
        return 1

    t = Table(title=f"PnL Decomposition — {args.strategy_id} / {args.period}")
    t.add_column("Component"); t.add_column("Value $"); t.add_column("% of |total|")
    totals = d["totals"]; pcts = d["percentages"]
    t.add_row("Directional", f"{totals['directional']:.2f}", f"{pcts['directional_pct']:.1f}%")
    t.add_row("Swap", f"{totals['swap']:.2f}", f"{pcts['swap_pct']:.1f}%")
    t.add_row("Commission", f"{totals['commission']:.2f}", f"{pcts['commission_pct']:.1f}%")
    t.add_row("[bold]TOTAL[/bold]", f"[bold]{totals['total']:.2f}[/bold]", "100.0%")
    console.print(t)

    if d["wti_guard"]["flagged"]:
        console.print(f"[red]⚠ WTI GUARD: {d['wti_guard']['reason']}[/red]")
    else:
        console.print("[green]✅ WTI guard passed.[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
