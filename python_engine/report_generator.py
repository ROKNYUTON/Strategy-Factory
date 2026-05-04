"""Final report generator for a Python-engine optimization run.

Reads the optimizer artifacts from ``strategies/<name>/results/`` (or accepts
them as DataFrames) and emits ``final_report.md`` plus per-config equity-curve
PNGs and per-symbol trade CSVs.

Design notes
------------
* The optimizer's ``optimization_results.csv`` and ``top_N_performers.csv``
  carry only *aggregate* metrics. Per-symbol breakdowns and PnL decomposition
  (the WTI guard) are not in those CSVs, so this module **re-runs** the top
  configurations against the cached data when more detail is needed.
* All re-running is best-effort: if ``strategies/<name>/strategy.py`` or the
  parquet cache aren't present, the report degrades gracefully — the
  corresponding sections show ``_(unavailable)_`` instead of failing the
  whole report.
* Plot style is intentionally austere — black background, gold equity line,
  grey grid — so PNGs land cleanly in a slide deck without restyling.

Public API
----------
* :func:`generate_final_report`
* :func:`plot_equity_curves`
* :func:`generate_per_symbol_breakdown`
"""
from __future__ import annotations

import importlib.util
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from python_engine.vectorized_backtest import (
    BacktestResult,
    SymbolSpec,
    VectorizedBacktest,
)

logger = logging.getLogger(__name__)

ENGINE_VERSION = "StrategyFactory Python-first v1.0"

# Acceptance thresholds — kept in sync with config/factory_defaults.yaml so
# the report's verdict matches the rest of the pipeline. Override per-call
# via ``acceptance`` if the spec differs.
DEFAULT_ACCEPTANCE: Dict[str, float] = {
    "is_min_sharpe": 0.8,
    "oos_min_sharpe": 0.6,
    "max_drawdown_pct": 15.0,
    "min_trades": 100,
    "min_directional_pnl_pct": 60.0,
}


# ---------------------------------------------------------------------------
# Internal helpers — strategy + data loading
# ---------------------------------------------------------------------------

@dataclass
class _StrategyContext:
    """All re-run inputs for one strategy folder. Any field may be None."""
    strategy_module: Optional[ModuleType] = None
    data_dict: Dict[str, pd.DataFrame] = field(default_factory=dict)
    is_period: Optional[Tuple[date, date]] = None
    oos_period: Optional[Tuple[date, date]] = None
    symbols: List[str] = field(default_factory=list)
    optimization_summary: Dict[str, Any] = field(default_factory=dict)


def _load_strategy_module(strategy_dir: Path) -> Optional[ModuleType]:
    path = strategy_dir / "strategy.py"
    if not path.exists():
        logger.warning("No strategy.py in %s", strategy_dir)
        return None
    spec = importlib.util.spec_from_file_location(
        f"strategy_{strategy_dir.name}", path
    )
    if spec is None or spec.loader is None:
        logger.warning("Could not spec-load %s", path)
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to import %s: %s", path, exc)
        return None
    if not hasattr(mod, "signals"):
        logger.warning("%s defines no top-level signals()", path)
        return None
    return mod


def _load_data_dict(
    strategy_dir: Path, symbols: List[str], timeframe: str = "M1",
) -> Dict[str, pd.DataFrame]:
    """Read any ``{SYM}_{TF}_*.parquet`` in ``data_cache/`` for the given symbols."""
    cache = strategy_dir / "data_cache"
    out: Dict[str, pd.DataFrame] = {}
    if not cache.exists():
        return out
    for sym in symbols:
        candidates = sorted(cache.glob(f"{sym}_{timeframe}_*.parquet"))
        if not candidates:
            logger.info("No parquet for %s in %s", sym, cache)
            continue
        # Pick the widest-coverage file: largest end-date in filename.
        chosen = candidates[-1]
        try:
            out[sym] = pd.read_parquet(chosen)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s", chosen, exc)
    return out


def _load_periods(strategy_dir: Path) -> Tuple[
    Optional[Tuple[date, date]], Optional[Tuple[date, date]], List[str]
]:
    """Pull IS/OOS windows + symbols out of ``hypothesis.yaml`` if present."""
    hyp = strategy_dir / "hypothesis.yaml"
    if not hyp.exists():
        return None, None, []
    try:
        data = yaml.safe_load(hyp.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse %s: %s", hyp, exc)
        return None, None, []
    bt = data.get("backtest", {}) or {}
    is_p = _period_from_yaml(bt.get("is_period"))
    oos_p = _period_from_yaml(bt.get("oos_period"))
    symbols = (data.get("universe", {}) or {}).get("symbols", []) or []
    return is_p, oos_p, list(symbols)


def _period_from_yaml(node: Optional[Dict]) -> Optional[Tuple[date, date]]:
    if not node:
        return None
    s, e = node.get("start"), node.get("end")
    if not s or not e:
        return None
    return _coerce_date(s), _coerce_date(e)


def _coerce_date(v: Any) -> date:
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    return datetime.strptime(str(v), "%Y-%m-%d").date()


def _build_context(strategy_dir: Path) -> _StrategyContext:
    is_p, oos_p, symbols = _load_periods(strategy_dir)
    ctx = _StrategyContext(
        strategy_module=_load_strategy_module(strategy_dir),
        is_period=is_p,
        oos_period=oos_p,
        symbols=symbols,
    )
    if symbols:
        ctx.data_dict = _load_data_dict(strategy_dir, symbols)
    return ctx


def _slice_period(df: pd.DataFrame, period: Tuple[date, date]) -> pd.DataFrame:
    """Return rows between ``period[0]`` (inclusive) and ``period[1]+1d`` (exclusive)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df = df.set_index(pd.DatetimeIndex(df["time"])).drop(columns=["time"])
        else:
            return df
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    start = pd.Timestamp(period[0])
    end = pd.Timestamp(period[1]) + pd.Timedelta(days=1)
    return df.sort_index().loc[start:end]


def _run_one_symbol(
    strategy_module: ModuleType,
    df: pd.DataFrame,
    symbol: str,
    params: Dict[str, Any],
    initial_balance: float = 10_000.0,
    risk_per_trade_pct: float = 0.5,
) -> Optional[BacktestResult]:
    if df is None or df.empty:
        return None
    try:
        payload = strategy_module.signals(df, **params)
    except NotImplementedError:
        logger.warning(
            "%s.signals() raised NotImplementedError — strategy not yet coded",
            getattr(strategy_module, "__name__", "?"),
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("signals() failed on %s: %s", symbol, exc)
        return None
    for k in ("signal_long", "signal_short", "sl_distance", "tp_distance"):
        if k not in payload:
            logger.warning("signals() missing %s for %s", k, symbol)
            return None
    try:
        spec = SymbolSpec.from_config(symbol)
    except (KeyError, FileNotFoundError) as exc:
        logger.warning("No SymbolSpec for %s (%s) — skipping re-run", symbol, exc)
        return None
    engine = VectorizedBacktest(
        data=df,
        symbol_spec=spec,
        initial_balance=initial_balance,
        risk_per_trade_pct=risk_per_trade_pct,
    )
    return engine.run(
        signal_long=payload["signal_long"],
        signal_short=payload["signal_short"],
        sl_distance=payload["sl_distance"],
        tp_distance=payload["tp_distance"],
        session_mask=payload.get("session_mask"),
        max_holding_bars=payload.get("max_holding_bars", 16),
    )


def _rerun_config(
    ctx: _StrategyContext,
    params: Dict[str, Any],
    period: Tuple[date, date],
    initial_balance: float = 10_000.0,
    risk_per_trade_pct: float = 0.5,
) -> Dict[str, BacktestResult]:
    """Re-run one parameter set across every symbol with cached data."""
    out: Dict[str, BacktestResult] = {}
    if ctx.strategy_module is None:
        return out
    for sym, df in ctx.data_dict.items():
        sliced = _slice_period(df, period)
        res = _run_one_symbol(
            ctx.strategy_module, sliced, sym, params,
            initial_balance=initial_balance,
            risk_per_trade_pct=risk_per_trade_pct,
        )
        if res is not None:
            out[sym] = res
    return out


def _aggregate_decomposition(
    per_symbol: Dict[str, BacktestResult],
) -> Dict[str, float]:
    """Combine PnL decompositions across symbols into one summary."""
    if not per_symbol:
        return {}
    directional = sum(r.pnl_decomposition["directional"] for r in per_symbol.values())
    swap = sum(r.pnl_decomposition["swap"] for r in per_symbol.values())
    commission = sum(r.pnl_decomposition["commission"] for r in per_symbol.values())
    spread = sum(r.pnl_decomposition["spread_cost"] for r in per_symbol.values())
    abs_total = abs(directional) + abs(swap) + abs(commission) + abs(spread)
    if abs_total == 0:
        d_pct = s_pct = c_pct = sp_pct = 0.0
    else:
        d_pct = abs(directional) / abs_total * 100.0
        s_pct = abs(swap) / abs_total * 100.0
        c_pct = abs(commission) / abs_total * 100.0
        sp_pct = abs(spread) / abs_total * 100.0
    return {
        "directional": directional,
        "swap": swap,
        "commission": commission,
        "spread_cost": spread,
        "directional_pct": d_pct,
        "swap_pct": s_pct,
        "commission_pct": c_pct,
        "spread_pct": sp_pct,
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def _row_verdict(row: Dict[str, Any], thresholds: Dict[str, float],
                 directional_pct: Optional[float] = None) -> Tuple[str, List[str]]:
    """Return ``("Recommend"|"Conditional"|"Reject", [failure reasons])``."""
    reasons: List[str] = []

    def _check(name: str, ok: bool, descr: str) -> None:
        if not ok:
            reasons.append(descr)

    is_sharpe = float(row.get("is_sharpe_avg", 0.0))
    oos_sharpe = float(row.get("oos_sharpe_avg", 0.0))
    max_dd = abs(float(row.get("max_dd_pct", 0.0)))
    n_trades = int(row.get("total_trades", 0))

    _check("is_sharpe", is_sharpe >= thresholds["is_min_sharpe"],
           f"IS Sharpe {is_sharpe:.2f} < {thresholds['is_min_sharpe']}")
    _check("oos_sharpe", oos_sharpe >= thresholds["oos_min_sharpe"],
           f"OOS Sharpe {oos_sharpe:.2f} < {thresholds['oos_min_sharpe']}")
    _check("max_dd", max_dd <= thresholds["max_drawdown_pct"],
           f"|max DD| {max_dd:.2f}% > {thresholds['max_drawdown_pct']}%")
    _check("min_trades", n_trades >= thresholds["min_trades"],
           f"trades {n_trades} < {thresholds['min_trades']}")
    if directional_pct is not None:
        _check(
            "directional",
            directional_pct >= thresholds["min_directional_pnl_pct"],
            f"directional {directional_pct:.1f}% "
            f"< {thresholds['min_directional_pnl_pct']}%",
        )

    if not reasons:
        return "Recommend", reasons
    # Borderline: missed at most one criterion AND OOS Sharpe is at least
    # 70% of the threshold. Otherwise reject.
    if (
        len(reasons) == 1
        and oos_sharpe >= 0.7 * thresholds["oos_min_sharpe"]
        and max_dd <= 1.5 * thresholds["max_drawdown_pct"]
    ):
        return "Conditional", reasons
    return "Reject", reasons


# ---------------------------------------------------------------------------
# Equity-curve plotting
# ---------------------------------------------------------------------------

def _plot_style() -> None:
    """Set the matplotlib hedge-fund-clean style locally."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "axes.facecolor": "#0a0a0a",
        "figure.facecolor": "#0a0a0a",
        "axes.edgecolor": "#666666",
        "axes.labelcolor": "#dddddd",
        "xtick.color": "#cccccc",
        "ytick.color": "#cccccc",
        "axes.titlecolor": "#f0c060",
        "grid.color": "#333333",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "axes.grid": True,
        "savefig.facecolor": "#0a0a0a",
        "savefig.edgecolor": "#0a0a0a",
        "font.family": "DejaVu Sans",
    })


def _draw_equity(equities: Dict[str, pd.Series], title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    _plot_style()
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
    palette = ["#f0c060", "#a06030", "#609060", "#406090", "#a04060",
               "#90a040", "#604090", "#308090", "#806020", "#508080"]
    for i, (label, eq) in enumerate(equities.items()):
        if eq is None or eq.empty:
            continue
        ax.plot(eq.index, eq.values, color=palette[i % len(palette)],
                linewidth=1.4, label=label)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity (USD)")
    if len(equities) > 1:
        ax.legend(loc="best", facecolor="#1a1a1a", edgecolor="#444444",
                  labelcolor="#dddddd", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _combined_equity(per_symbol: Dict[str, BacktestResult]) -> pd.Series:
    """Sum per-symbol equity into one curve, aligned on the union timeline."""
    if not per_symbol:
        return pd.Series(dtype=float)
    series = []
    for r in per_symbol.values():
        s = r.equity.copy()
        # Subtract initial balance so symbols can be summed without
        # double-counting the starting capital.
        series.append(s - r.initial_balance)
    union = (
        pd.concat(series, axis=1).sort_index().ffill().fillna(0.0).sum(axis=1)
    )
    seed = sum(r.initial_balance for r in per_symbol.values())
    return union + seed


# ---------------------------------------------------------------------------
# Public: equity curves
# ---------------------------------------------------------------------------

def plot_equity_curves(strategy_dir: Path, top_n: int = 10) -> List[Path]:
    """For each top-N config in ``results/top_N_performers.csv`` re-run the
    backtest on cached data and write per-symbol + combined equity-curve PNGs
    under ``results/equity_curves/``.

    Returns the list of written PNG paths (may be empty if data or strategy.py
    are missing).
    """
    strategy_dir = Path(strategy_dir)
    results_dir = strategy_dir / "results"
    out_dir = results_dir / "equity_curves"
    written: List[Path] = []

    top_csv = _find_top_csv(results_dir, top_n)
    if top_csv is None:
        logger.warning("No top_N_performers.csv in %s", results_dir)
        return written
    top_df = pd.read_csv(top_csv)
    if top_df.empty:
        return written

    ctx = _build_context(strategy_dir)
    if ctx.strategy_module is None or not ctx.data_dict or ctx.oos_period is None:
        logger.warning(
            "Cannot plot equity curves for %s: missing strategy.py, data_cache, "
            "or hypothesis.yaml backtest periods.", strategy_dir,
        )
        return written

    for _, row in top_df.head(top_n).iterrows():
        rank = int(row["rank"])
        params = _parse_params(row.get("params"))
        per_symbol = _rerun_config(ctx, params, ctx.oos_period)
        if not per_symbol:
            continue
        per_sym_eq = {sym: r.equity for sym, r in per_symbol.items()}
        combined_eq = _combined_equity(per_symbol)
        per_sym_path = out_dir / f"rank_{rank:02d}_per_symbol.png"
        combined_path = out_dir / f"rank_{rank:02d}_combined.png"
        _draw_equity(
            per_sym_eq,
            f"Rank #{rank} — per-symbol equity (OOS)",
            per_sym_path,
        )
        _draw_equity(
            {"combined": combined_eq},
            f"Rank #{rank} — combined equity (OOS)",
            combined_path,
        )
        written.extend([per_sym_path, combined_path])
    return written


# ---------------------------------------------------------------------------
# Public: per-symbol breakdown
# ---------------------------------------------------------------------------

def generate_per_symbol_breakdown(
    strategy_dir: Path, top_3_configs: List[Dict[str, Any]],
) -> List[Path]:
    """For the supplied top-3 configurations, write per-symbol trade CSVs and
    a monthly-returns summary under ``results/per_symbol/rank_NN/``.

    ``top_3_configs`` is a list of dicts with at least ``rank`` and ``params``
    keys (``params`` may be a dict or a JSON string — both accepted). Returns
    every CSV written.
    """
    strategy_dir = Path(strategy_dir)
    written: List[Path] = []
    ctx = _build_context(strategy_dir)
    if ctx.strategy_module is None or not ctx.data_dict or ctx.oos_period is None:
        logger.warning("Cannot generate per-symbol breakdown for %s", strategy_dir)
        return written

    base_out = strategy_dir / "results" / "per_symbol"
    for cfg in top_3_configs[:3]:
        rank = int(cfg.get("rank", 0))
        params = _parse_params(cfg.get("params"))
        per_symbol = _rerun_config(ctx, params, ctx.oos_period)
        if not per_symbol:
            continue
        rank_dir = base_out / f"rank_{rank:02d}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        for sym, res in per_symbol.items():
            trades_path = rank_dir / f"{sym}_trades.csv"
            res.export_trades_csv(trades_path)
            written.append(trades_path)

            monthly_path = rank_dir / f"{sym}_monthly.csv"
            monthly = _monthly_summary(res)
            monthly.to_csv(monthly_path)
            written.append(monthly_path)

            dd_path = rank_dir / f"{sym}_drawdowns.csv"
            dd = _drawdown_periods(res.equity)
            dd.to_csv(dd_path, index=False)
            written.append(dd_path)
    return written


def _monthly_summary(res: BacktestResult) -> pd.DataFrame:
    if not res.trades:
        return pd.DataFrame(columns=["pnl", "n_trades", "wins", "losses"])
    trades_df = res.trades_to_df()
    trades_df["close_time"] = pd.to_datetime(trades_df["close_time"])
    trades_df["month"] = trades_df["close_time"].dt.to_period("M")
    grouped = trades_df.groupby("month")
    return pd.DataFrame({
        "pnl": grouped["profit_total"].sum(),
        "n_trades": grouped.size(),
        "wins": grouped["profit_total"].apply(lambda s: int((s > 0).sum())),
        "losses": grouped["profit_total"].apply(lambda s: int((s < 0).sum())),
    })


def _drawdown_periods(equity: pd.Series) -> pd.DataFrame:
    """Return one row per drawdown excursion: peak/trough/recovery + depth."""
    if equity is None or equity.empty:
        return pd.DataFrame(columns=["peak_time", "trough_time", "recovery_time",
                                      "peak_equity", "trough_equity",
                                      "drawdown_abs", "drawdown_pct"])
    rolling_max = equity.cummax()
    in_dd = equity < rolling_max
    rows = []
    i = 0
    eq_vals = equity.values
    eq_idx = equity.index
    rmax_vals = rolling_max.values
    n = len(equity)
    while i < n:
        if not in_dd.iloc[i]:
            i += 1
            continue
        peak_eq = rmax_vals[i]
        peak_time = eq_idx[i - 1] if i > 0 else eq_idx[i]
        trough_eq = eq_vals[i]
        trough_time = eq_idx[i]
        j = i
        while j < n and eq_vals[j] < peak_eq:
            if eq_vals[j] < trough_eq:
                trough_eq = eq_vals[j]
                trough_time = eq_idx[j]
            j += 1
        recovery_time = eq_idx[j] if j < n else None
        rows.append({
            "peak_time": peak_time,
            "trough_time": trough_time,
            "recovery_time": recovery_time,
            "peak_equity": float(peak_eq),
            "trough_equity": float(trough_eq),
            "drawdown_abs": float(trough_eq - peak_eq),
            "drawdown_pct": float((trough_eq - peak_eq) / peak_eq * 100.0)
                            if peak_eq else 0.0,
        })
        i = j + 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public: final report
# ---------------------------------------------------------------------------

def generate_final_report(
    strategy_dir: Path,
    top_results: pd.DataFrame,
    all_results: pd.DataFrame,
    *,
    optimization_summary: Optional[Dict[str, Any]] = None,
    acceptance: Optional[Dict[str, float]] = None,
    plot_curves: bool = True,
    write_per_symbol: bool = True,
) -> Path:
    """Render ``strategies/<name>/results/final_report.md`` and return its path.

    Parameters
    ----------
    strategy_dir
        Folder created by :func:`automation.generate_strategy.generate_strategy_skeleton`.
    top_results, all_results
        DataFrames matching the columns produced by
        :meth:`MultiAssetOptimizer.get_top_n` / ``export_to_csv``.
    optimization_summary
        Optional dict with keys ``trials_run``, ``trials_pruned``,
        ``wall_time_seconds``, ``symbols``, ``is_period``, ``oos_period``.
        Missing keys are filled from the data we can derive locally.
    acceptance
        Override the verdict thresholds. Defaults to :data:`DEFAULT_ACCEPTANCE`.
    plot_curves
        Re-run + plot equity curves for the top-10 (best effort).
    write_per_symbol
        Re-run + write per-symbol CSVs for the top-3 (best effort).
    """
    strategy_dir = Path(strategy_dir)
    results_dir = strategy_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    thresholds = {**DEFAULT_ACCEPTANCE, **(acceptance or {})}
    summary = dict(optimization_summary or {})

    # Re-run helpers populate richer top-3 metrics (per-symbol + decomposition).
    ctx = _build_context(strategy_dir)
    if "symbols" not in summary and ctx.symbols:
        summary["symbols"] = ctx.symbols
    if "is_period" not in summary and ctx.is_period:
        summary["is_period"] = ctx.is_period
    if "oos_period" not in summary and ctx.oos_period:
        summary["oos_period"] = ctx.oos_period

    summary.setdefault("trials_run", _safe_int(len(all_results)))
    summary.setdefault(
        "trials_pruned",
        _safe_int(int(all_results["pruned"].sum())) if "pruned" in all_results
        else 0,
    )

    top_details = _gather_top3_details(ctx, top_results)

    if plot_curves:
        try:
            plot_equity_curves(strategy_dir, top_n=min(10, len(top_results)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Equity-curve plotting failed: %s", exc)

    if write_per_symbol and not top_results.empty:
        try:
            generate_per_symbol_breakdown(
                strategy_dir,
                top_results.head(3).to_dict(orient="records"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Per-symbol breakdown failed: %s", exc)

    report = _render_report(
        strategy_dir=strategy_dir,
        top_results=top_results,
        thresholds=thresholds,
        summary=summary,
        top_details=top_details,
    )
    out_path = results_dir / "final_report.md"
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote final report → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_report(
    *,
    strategy_dir: Path,
    top_results: pd.DataFrame,
    thresholds: Dict[str, float],
    summary: Dict[str, Any],
    top_details: List[Dict[str, Any]],
) -> str:
    name = strategy_dir.name
    parts: List[str] = []
    parts.append(f"# {name} — Optimization Report\n")
    parts.append(
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC  \n"
        f"**Engine:** {ENGINE_VERSION}\n"
    )

    # 1. Hypothesis
    parts.append("## 1. Strategy Hypothesis\n")
    parts.append(_read_readme_summary(strategy_dir) or "_(no README.md found)_")
    parts.append("")

    # 2. Optimization summary
    parts.append("## 2. Optimization Summary\n")
    parts.append(_render_summary_table(summary, total_trials=len(top_results.index)))
    parts.append("")

    # 3. Top-N table
    parts.append(f"## 3. Top-{len(top_results)} Performers\n")
    parts.append(_render_top_table(top_results, thresholds, top_details))
    parts.append("")

    # 4. Top-3 detail
    parts.append("## 4. Top-3 Detail\n")
    if top_results.empty:
        parts.append("_No surviving configs._\n")
    else:
        for detail in top_details[:3]:
            parts.append(_render_top_detail(detail, thresholds))

    # 5. Recommendation
    parts.append("## 5. Recommendation\n")
    parts.append(_render_recommendation(top_details, thresholds))
    parts.append("")

    # 6. Next steps
    parts.append("## 6. Next Steps\n")
    parts.append(_render_next_steps(name))

    return "\n".join(parts)


def _render_summary_table(summary: Dict[str, Any], total_trials: int) -> str:
    trials = summary.get("trials_run", total_trials)
    pruned = summary.get("trials_pruned", 0)
    pruned_pct = (pruned / trials * 100.0) if trials else 0.0
    wall = summary.get("wall_time_seconds")
    wall_str = _format_seconds(wall) if wall is not None else "—"
    symbols = summary.get("symbols") or []
    is_p = summary.get("is_period")
    oos_p = summary.get("oos_period")
    rows = [
        ("Trials run", f"{trials}"),
        ("Trials pruned", f"{pruned} ({pruned_pct:.1f}%)"),
        ("Wall time", wall_str),
        ("Symbols", ", ".join(symbols) if symbols else "—"),
        ("IS Period", _fmt_period(is_p)),
        ("OOS Period", _fmt_period(oos_p)),
    ]
    out = ["| Field | Value |", "|---|---|"]
    for k, v in rows:
        out.append(f"| {k} | {v} |")
    return "\n".join(out)


def _render_top_table(
    top: pd.DataFrame,
    thresholds: Dict[str, float],
    details: List[Dict[str, Any]],
) -> str:
    if top.empty:
        return "_No top performers._\n"
    detail_by_rank = {d["rank"]: d for d in details}
    out = [
        "| Rank | Trial | OOS Sharpe (avg) | IS Sharpe (avg) | Robustness | "
        "Max DD% | Trades | Verdict |",
        "|------|-------|------------------|------------------|------------|"
        "---------|--------|---------|",
    ]
    for _, row in top.iterrows():
        rank = int(row.get("rank", 0))
        trial = int(row.get("trial_id", 0))
        oos_sh = float(row.get("oos_sharpe_avg", 0.0))
        is_sh = float(row.get("is_sharpe_avg", 0.0))
        robustness = (oos_sh / is_sh) if is_sh > 0 else 0.0
        max_dd = abs(float(row.get("max_dd_pct", 0.0)))
        trades = int(row.get("total_trades", 0))
        det = detail_by_rank.get(rank, {})
        verdict, _ = _row_verdict(
            row.to_dict(), thresholds,
            directional_pct=det.get("directional_pct"),
        )
        glyph = {
            "Recommend": "PASS",
            "Conditional": "WARN",
            "Reject": "FAIL",
        }[verdict]
        out.append(
            f"| {rank} | {trial} | {oos_sh:.2f} | {is_sh:.2f} | "
            f"{robustness:.2f} | {max_dd:.1f} | {trades} | {glyph} |"
        )
    return "\n".join(out)


def _render_top_detail(detail: Dict[str, Any], thresholds: Dict[str, float]) -> str:
    rank = detail["rank"]
    params = detail.get("params", {})
    per_sym = detail.get("per_symbol_sharpe", {})
    decomp = detail.get("decomposition", {})
    directional_pct = decomp.get("directional_pct")
    wti_status = (
        "PASSED" if directional_pct is not None
        and directional_pct >= thresholds["min_directional_pnl_pct"]
        else "FAILED" if directional_pct is not None else "N/A"
    )
    pnl_line = (
        f"directional {decomp.get('directional_pct', 0):.1f}%, "
        f"spread {-decomp.get('spread_pct', 0):.1f}%, "
        f"swap {decomp.get('swap_pct', 0):.1f}%, "
        f"commission {-decomp.get('commission_pct', 0):.1f}%"
        if decomp else "_(unavailable — re-run failed)_"
    )
    params_str = json.dumps(params, sort_keys=True, default=str)
    per_sym_line = (
        ", ".join(f"{k}: {v:.2f}" for k, v in sorted(per_sym.items()))
        if per_sym else "_(unavailable — re-run failed)_"
    )
    img_combined = f"equity_curves/rank_{rank:02d}_combined.png"
    img_per_sym = f"equity_curves/rank_{rank:02d}_per_symbol.png"
    return (
        f"### #{rank} — Trial {detail.get('trial_id')}\n"
        f"- Parameters: `{params_str}`\n"
        f"- Per-symbol Sharpe: {per_sym_line}\n"
        f"- Equity curve (combined): ![]({img_combined})\n"
        f"- Equity curve (per symbol): ![]({img_per_sym})\n"
        f"- PnL decomposition: {pnl_line}\n"
        f"- WTI guard: {wti_status}\n"
    )


def _render_recommendation(
    details: List[Dict[str, Any]], thresholds: Dict[str, float],
) -> str:
    if not details:
        return "_No surviving configurations to evaluate._"
    verdicts = [d.get("verdict", "Reject") for d in details[:10]]
    n_recommend = sum(1 for v in verdicts if v == "Recommend")
    n_conditional = sum(1 for v in verdicts if v == "Conditional")
    n_reject = sum(1 for v in verdicts if v == "Reject")
    headline = (
        f"- **Top-3 → proceed to MQL5 translation** "
        f"({n_recommend} recommend / {n_conditional} conditional / "
        f"{n_reject} reject in top-10)\n"
        if n_recommend >= 1 else
        f"- **No clear winner.** {n_conditional} conditional, {n_reject} reject "
        "in top-10 — refine logic or grid before another search.\n"
    )
    backup = "- **Top 4-10 → backup candidates, archive results**"
    return headline + backup


def _render_next_steps(name: str) -> str:
    return (
        "```cmd\n"
        f"python automation/python_to_mql5_translator.py strategies/{name}/ --rank 1\n"
        f"python automation/pipeline.py compile STR_XXX_{name}\n"
        f"python automation/pipeline.py backtest STR_XXX_{name} --period all\n"
        "```\n"
    )


# ---------------------------------------------------------------------------
# Top-3 detail gathering — re-runs the OOS slice for each row
# ---------------------------------------------------------------------------

def _gather_top3_details(
    ctx: _StrategyContext, top: pd.DataFrame,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if top.empty:
        return out
    can_rerun = bool(
        ctx.strategy_module and ctx.data_dict and ctx.oos_period
    )
    for _, row in top.iterrows():
        rank = int(row.get("rank", 0))
        trial_id = int(row.get("trial_id", 0))
        params = _parse_params(row.get("params"))
        per_sym_sharpe: Dict[str, float] = {}
        decomp: Dict[str, float] = {}
        if can_rerun:
            per_sym_results = _rerun_config(ctx, params, ctx.oos_period)  # type: ignore[arg-type]
            per_sym_sharpe = {
                sym: float(r.metrics.get("sharpe", 0.0))
                for sym, r in per_sym_results.items()
            }
            decomp = _aggregate_decomposition(per_sym_results)
        verdict, _ = _row_verdict(
            row.to_dict(), DEFAULT_ACCEPTANCE,
            directional_pct=decomp.get("directional_pct"),
        )
        out.append({
            "rank": rank,
            "trial_id": trial_id,
            "params": params,
            "per_symbol_sharpe": per_sym_sharpe,
            "decomposition": decomp,
            "directional_pct": decomp.get("directional_pct"),
            "verdict": verdict,
        })
    return out


# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------

def _parse_params(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _read_readme_summary(strategy_dir: Path) -> str:
    readme = strategy_dir / "README.md"
    if not readme.exists():
        return ""
    text = readme.read_text(encoding="utf-8")
    # Pull the "## Hypothesis" section if present, else the first 600 chars.
    marker = "## Hypothesis"
    if marker in text:
        rest = text.split(marker, 1)[1]
        next_section = rest.find("\n## ")
        chunk = rest[:next_section] if next_section > 0 else rest
        return chunk.strip()
    return text[:600].strip()


def _find_top_csv(results_dir: Path, top_n: int) -> Optional[Path]:
    if not results_dir.exists():
        return None
    exact = results_dir / f"top_{top_n}_performers.csv"
    if exact.exists():
        return exact
    candidates = sorted(results_dir.glob("top_*_performers.csv"))
    return candidates[-1] if candidates else None


def _fmt_period(period: Optional[Tuple[date, date]]) -> str:
    if not period:
        return "—"
    return f"{period[0]} → {period[1]}"


def _format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0
