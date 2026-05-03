"""
StrategyFactory — Parameter Sensitivity (placeholder + framework)
================================================================
This module provides the FRAMEWORK to run sensitivity analysis.
True sensitivity requires re-running MT5 tester with perturbed inputs;
the orchestrator (pipeline.py) handles that loop.

This file:
  - Defines the perturbation grid generator (numeric inputs only).
  - Aggregates per-perturbation Sharpe results into a sensitivity score.

Output: backtests/parsed_results/{strategy_id}/sensitivity.json
"""

from __future__ import annotations

import json
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).parent.parent
console = Console()
ENGINE_VERSION = "StrategyFactory-1.0"


def perturbation_grid(base_value: float, pct: float = 0.20, steps: int = 5) -> List[float]:
    """Symmetric perturbation grid around base_value, ±pct in `steps` linear steps."""
    if base_value == 0:
        return [0.0]
    deltas = np.linspace(-pct, pct, steps)
    return [base_value * (1.0 + d) for d in deltas]


def aggregate_results(base_sharpe: float, perturbed_sharpes: Dict[str, List[float]]) -> dict:
    """
    Aggregate sensitivity results.
    perturbed_sharpes: {parameter_name: [sharpe_at_perturbation_1, ...]}
    """
    per_param = {}
    overall_min_retained = 1.0

    for name, sharpes in perturbed_sharpes.items():
        if not sharpes or base_sharpe <= 0:
            per_param[name] = {"min_retained": 0.0, "verdict": "no_data"}
            continue
        min_sharpe = min(sharpes)
        retained = min_sharpe / base_sharpe if base_sharpe > 0 else 0.0
        per_param[name] = {
            "base_sharpe": base_sharpe,
            "min_perturbed_sharpe": min_sharpe,
            "max_perturbed_sharpe": max(sharpes),
            "min_retained": float(retained),
            "verdict": "robust" if retained >= 0.5 else "fragile",
        }
        overall_min_retained = min(overall_min_retained, retained)

    return {
        "per_param": per_param,
        "overall_min_retained": float(overall_min_retained),
        "overall_verdict": "robust" if overall_min_retained >= 0.5 else "fragile",
    }


def save(strategy_id: str, payload: dict) -> Path:
    out_dir = ROOT / "backtests" / "parsed_results" / strategy_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "strategy_id": strategy_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "engine_version": ENGINE_VERSION,
        "sensitivity": payload,
    }
    p = out_dir / "sensitivity.json"
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return p


def stub_from_metrics(strategy_id: str) -> dict:
    """
    Stub mode: when full perturbation runs aren't available, derive a coarse
    proxy by partitioning trades by year and computing per-year Sharpe.
    Used when pipeline doesn't have time/resources to do full perturbation.
    """
    parsed_path = ROOT / "backtests" / "parsed_results" / strategy_id / "is_parsed.json"
    if not parsed_path.exists():
        return {"warning": "No IS parsed data — sensitivity stub unavailable."}

    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    import pandas as pd
    df = pd.DataFrame(parsed["trades"])
    if df.empty: return {"warning": "No trades."}
    df["close_time"] = pd.to_datetime(df["close_time"], errors="coerce")
    df = df.dropna(subset=["close_time"])
    df["year"] = df["close_time"].dt.year
    initial = parsed["summary"].get("initial_balance", 10000)

    by_year = []
    for y, sub in df.groupby("year"):
        rets = sub["profit_total"] / initial
        s = rets.std(ddof=1)
        sharpe = (rets.mean() / s) * np.sqrt(252) if s > 0 else 0.0
        by_year.append({"year": int(y), "sharpe": float(sharpe), "trades": int(len(sub))})

    sharpes = [r["sharpe"] for r in by_year]
    if not sharpes: return {"warning": "No yearly data."}
    avg = np.mean(sharpes); mn = min(sharpes)
    return {
        "stub_mode": True,
        "by_year": by_year,
        "avg_yearly_sharpe": float(avg),
        "min_yearly_sharpe": float(mn),
        "consistency_score": float(mn / avg) if avg > 0 else 0.0,
        "verdict": "robust" if mn / avg >= 0.5 else "fragile",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy_id")
    args = ap.parse_args()
    payload = stub_from_metrics(args.strategy_id)
    p = save(args.strategy_id, payload)
    console.print(f"[green]Saved (stub mode):[/green] {p}")
    if "by_year" in payload:
        t = Table(title="Per-Year Sharpe (Sensitivity Proxy)")
        t.add_column("Year"); t.add_column("Sharpe"); t.add_column("Trades")
        for r in payload["by_year"]:
            t.add_row(str(r["year"]), f"{r['sharpe']:.3f}", str(r["trades"]))
        console.print(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
