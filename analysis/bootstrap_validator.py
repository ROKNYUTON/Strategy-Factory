"""
StrategyFactory — Bootstrap Validator
=====================================
Computes bootstrap p-value for the Sharpe ratio under the null hypothesis
of zero mean trade return. Robust to fat tails (essential for Kurt > 3 series).

Also performs Benjamini-Hochberg multiple-testing correction across
hypotheses tested in the rolling window (default 90 days).

Output: backtests/parsed_results/{strategy_id}/{period}_bootstrap.json
"""

from __future__ import annotations

import json
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.panel import Panel

ROOT = Path(__file__).parent.parent
console = Console()
ENGINE_VERSION = "StrategyFactory-1.0"

DEFAULT_ITER = 5000
TRADING_DAYS = 252


def load_factory_defaults() -> dict:
    cfg = ROOT / "config" / "factory_defaults.yaml"
    return yaml.safe_load(cfg.read_text(encoding="utf-8"))


def load_parsed(strategy_id: str, period: str) -> dict:
    p = ROOT / "backtests" / "parsed_results" / strategy_id / f"{period}_parsed.json"
    return json.loads(p.read_text(encoding="utf-8"))


def trade_returns(parsed: dict) -> np.ndarray:
    initial = parsed["summary"].get("initial_balance", 10000.0)
    profits = np.array([t.get("profit_total", 0.0) for t in parsed["trades"]])
    if len(profits) == 0 or initial <= 0:
        return np.array([])
    return profits / initial  # per-trade return as % of initial balance


def bootstrap_pvalue(returns: np.ndarray,
                     n_iter: int = DEFAULT_ITER,
                     rng_seed: int = 42) -> dict:
    """
    Test H0: mean trade return = 0 vs H1: mean > 0.
    Centered bootstrap: resample from (returns - mean) under null.
    """
    if len(returns) < 30:
        return {
            "p_value": None,
            "warning": f"Sample too small ({len(returns)} trades) for bootstrap.",
        }

    rng = np.random.default_rng(rng_seed)
    obs_mean = returns.mean()
    obs_sharpe_per_trade = obs_mean / returns.std(ddof=1) if returns.std(ddof=1) > 0 else 0.0

    centered = returns - obs_mean  # null distribution
    n = len(returns)

    boot_means = np.empty(n_iter)
    boot_sharpes = np.empty(n_iter)
    for i in range(n_iter):
        sample = rng.choice(centered, size=n, replace=True)
        boot_means[i] = sample.mean()
        s = sample.std(ddof=1)
        boot_sharpes[i] = sample.mean() / s if s > 0 else 0.0

    # p-value: % of bootstrap means >= observed
    p_mean = float((boot_means >= obs_mean).sum() / n_iter)
    p_sharpe = float((boot_sharpes >= obs_sharpe_per_trade).sum() / n_iter)

    # 95% CI (percentile bootstrap, NOT centered)
    rng2 = np.random.default_rng(rng_seed + 1)
    naive_means = np.empty(n_iter)
    naive_sharpes = np.empty(n_iter)
    for i in range(n_iter):
        sample = rng2.choice(returns, size=n, replace=True)
        naive_means[i] = sample.mean()
        s = sample.std(ddof=1)
        naive_sharpes[i] = sample.mean() / s if s > 0 else 0.0

    return {
        "n_trades": int(n),
        "n_iterations": int(n_iter),
        "observed_mean_return": float(obs_mean),
        "observed_sharpe_per_trade": float(obs_sharpe_per_trade),
        "p_value_mean": p_mean,
        "p_value_sharpe": p_sharpe,
        "ci95_mean_low": float(np.percentile(naive_means, 2.5)),
        "ci95_mean_high": float(np.percentile(naive_means, 97.5)),
        "ci95_sharpe_low": float(np.percentile(naive_sharpes, 2.5)),
        "ci95_sharpe_high": float(np.percentile(naive_sharpes, 97.5)),
    }


def benjamini_hochberg_adjust(p_values: List[float], alpha: float = 0.01) -> List[float]:
    """Return BH-adjusted p-values."""
    if not p_values:
        return []
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = np.empty(n)
    p_sorted = np.array(p_values)[order]
    adj_sorted = np.minimum.accumulate(p_sorted[::-1] * n / np.arange(n, 0, -1))[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    ranked[order] = adj_sorted
    return ranked.tolist()


def collect_recent_pvalues(window_days: int = 90) -> List[dict]:
    """Aggregate all recent bootstrap p-values from log."""
    log_path = ROOT / "docs" / "HYPOTHESIS_LOG.md"
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8")
    import re
    cutoff = datetime.utcnow() - timedelta(days=window_days)
    entries = []
    for m in re.finditer(r"date:\s*([0-9\-]+).*?strategy_id:\s*([^\s]+).*?p_value_sharpe:\s*([0-9\.]+)",
                          text, re.S):
        try:
            d = datetime.fromisoformat(m.group(1))
            if d >= cutoff:
                entries.append({
                    "date": m.group(1),
                    "strategy_id": m.group(2),
                    "p": float(m.group(3)),
                })
        except Exception:
            continue
    return entries


def run(strategy_id: str, period: str, n_iter: int = DEFAULT_ITER) -> dict:
    parsed = load_parsed(strategy_id, period)
    returns = trade_returns(parsed)
    raw = bootstrap_pvalue(returns, n_iter=n_iter)

    # Multiple testing correction
    recent = collect_recent_pvalues(window_days=90)
    if raw.get("p_value_sharpe") is not None:
        all_p = [r["p"] for r in recent] + [raw["p_value_sharpe"]]
        adj = benjamini_hochberg_adjust(all_p)
        raw["p_value_sharpe_adjusted_bh"] = float(adj[-1])
        raw["multiple_testing_n"] = len(all_p)
    else:
        raw["p_value_sharpe_adjusted_bh"] = None

    out = {
        "strategy_id": strategy_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "engine_version": ENGINE_VERSION,
        "period": period,
        "bootstrap": raw,
    }
    out_dir = ROOT / "backtests" / "parsed_results" / strategy_id
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{period}_bootstrap.json"
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy_id")
    ap.add_argument("--period", default="is", choices=["is", "oos", "forward"])
    ap.add_argument("--iter", type=int, default=DEFAULT_ITER)
    args = ap.parse_args()

    out = run(args.strategy_id, args.period, args.iter)
    b = out["bootstrap"]
    msg = (
        f"Trades: {b.get('n_trades')}\n"
        f"Observed Sharpe (per-trade): {b.get('observed_sharpe_per_trade'):.4f}\n"
        f"p-value (mean): {b.get('p_value_mean')}\n"
        f"p-value (Sharpe): {b.get('p_value_sharpe')}\n"
        f"BH-adjusted p (Sharpe): {b.get('p_value_sharpe_adjusted_bh')}\n"
        f"95% CI Sharpe: [{b.get('ci95_sharpe_low'):.4f}, {b.get('ci95_sharpe_high'):.4f}]"
    )
    console.print(Panel(msg, title="BOOTSTRAP RESULT"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
