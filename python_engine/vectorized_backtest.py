"""Vectorised Python backtester with explicit cost model.

Replicates MQL5 ``OnTick`` semantics on M1 (or any uniform-bar) data:

- Entry signal computed AT CLOSE of bar ``i`` → position opened AT OPEN of bar
  ``i + 1`` (one-bar lag, identical to how an EA reacts to ``new bar`` events).
- ``sl`` / ``tp`` checked intra-bar via ``high`` / ``low``. When both are
  touched in the same bar we assume **SL fills first** (pessimistic worst case).
- Spread is baked into the actual entry / exit price: long enters at
  ``mid + spread/2``, exits at ``mid - spread/2``; short is symmetric. The
  resulting ``profit_spread_cost`` is therefore *negative* and equals the
  full round-turn spread on a typical trade.
- Swap is accrued at every 00:00 broker-time crossing strictly between entry
  and exit. The Wednesday rollover charges 3× the daily rate (FX convention)
  — configurable via ``triple_swap_weekday``.
- Round-turn commission is applied once per trade at close.

Single-position-only initially (the MQL5 EA template is single-position too).
The engine deliberately does NOT cross-validate against a second position; if
you need basket / hedging logic, build it on top.

Outputs
-------
``BacktestResult`` exposes:
    * ``trades``      – list of ``TradeFill`` (one per closed position)
    * ``equity``      – ``pd.Series`` indexed by trade-close time (with the
      initial balance as the first point), suitable for plotting and feeding
      into ``analysis/bootstrap_validator.py`` later.
    * ``metrics``     – Sharpe, Sortino, Calmar, max DD, win-rate, profit
      factor, total return.
    * ``pnl_decomposition`` – four buckets (directional / spread / swap /
      commission) with absolute amounts and percentage of |total|. This is the
      “WTI lesson” guard surface (see CLAUDE.md §2 rule 2).

CLI
---
::

    python -m python_engine.vectorized_backtest \\
        --strategy strategies/test_basic.py \\
        --data ./tmp_cache/EURUSD_M1_20240101_20241231.parquet \\
        --symbol EURUSD
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SWAP_RATES_PATH = ROOT / "config" / "swap_rates.yaml"
SYMBOLS_MAP_PATH = ROOT / "config" / "symbols_map.yaml"

_WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TradeFill:
    """One closed round-turn trade with full cost decomposition."""
    side: str                          # "long" | "short"
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    open_price: float                  # mid at entry bar open
    close_price: float                 # mid at exit (SL/TP/close)
    open_price_with_spread: float      # actual fill: ask for long, bid for short
    close_price_with_spread: float     # actual fill: bid for long, ask for short
    sl: float
    tp: float
    lots: float
    holding_bars: int                  # span: exit_idx - entry_idx + 1
    profit_directional: float          # mid-to-mid PnL
    profit_swap: float
    profit_commission: float           # negative (round-turn)
    profit_spread_cost: float          # negative (full round-turn spread)
    profit_total: float
    exit_reason: str                   # "tp"|"sl"|"time_exit"|"session_close"|"reverse_signal"

    def as_dict(self) -> Dict:
        return {
            "side": self.side,
            "open_time": self.open_time.isoformat(),
            "close_time": self.close_time.isoformat(),
            "open_price": self.open_price,
            "close_price": self.close_price,
            "open_price_with_spread": self.open_price_with_spread,
            "close_price_with_spread": self.close_price_with_spread,
            "sl": self.sl,
            "tp": self.tp,
            "lots": self.lots,
            "holding_bars": self.holding_bars,
            "profit_directional": self.profit_directional,
            "profit_swap": self.profit_swap,
            "profit_commission": self.profit_commission,
            "profit_spread_cost": self.profit_spread_cost,
            "profit_total": self.profit_total,
            "exit_reason": self.exit_reason,
        }


@dataclass
class BacktestResult:
    """Full output of one backtest run."""
    trades: List[TradeFill]
    equity: pd.Series
    metrics: Dict[str, float]
    pnl_decomposition: Dict[str, float]
    initial_balance: float

    def trades_to_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=list(TradeFill.__dataclass_fields__.keys())
            )
        return pd.DataFrame([t.as_dict() for t in self.trades])

    def export_trades_csv(self, path: Path) -> None:
        self.trades_to_df().to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Symbol metadata
# ---------------------------------------------------------------------------

@dataclass
class SymbolSpec:
    """All cost / unit-conversion info needed by the backtester for one symbol.

    Decoupled from the YAML loaders so tests can build a SymbolSpec inline
    without going through ``config/`` files.
    """
    symbol: str
    point_size: float                  # price units per MT5 "point"
    usd_per_price_unit: float          # USD per +1.0 price-unit move per 1 lot
    commission_per_lot: float          # round-turn USD per lot
    swap_long: float                   # daily, USD per lot
    swap_short: float                  # daily, USD per lot
    triple_swap_weekday: int = 2       # Mon=0; Wednesday=2 by FX convention

    @classmethod
    def from_config(
        cls,
        symbol: str,
        symbols_map_path: Path = SYMBOLS_MAP_PATH,
        swap_rates_path: Path = SWAP_RATES_PATH,
    ) -> "SymbolSpec":
        with open(symbols_map_path, "r", encoding="utf-8") as fh:
            sym_map = yaml.safe_load(fh)
        with open(swap_rates_path, "r", encoding="utf-8") as fh:
            swap_map = yaml.safe_load(fh)

        if symbol not in sym_map.get("symbols", {}):
            raise KeyError(f"Symbol '{symbol}' not in {symbols_map_path}")
        info = sym_map["symbols"][symbol]
        asset_class = info["asset_class"]

        if "pip_value_usd" in info:
            pip_value = float(info["pip_value_usd"])
            pip_size = 0.01 if "JPY" in symbol else 0.0001
            point_size = pip_size / 10.0      # standard 5-digit (3-digit JPY) broker
            upu = pip_value / pip_size
        elif "point_value_usd" in info:
            pt_value = float(info["point_value_usd"])
            if asset_class in ("commodity_metal", "commodity_energy"):
                point_size = 0.01
            else:
                point_size = 1.0
            upu = pt_value / point_size
        else:
            raise KeyError(
                f"Symbol '{symbol}' has neither pip_value_usd nor point_value_usd"
            )

        commission_table = swap_map.get("commission_per_lot", {}) or {}
        commission_per_lot = float(commission_table.get(asset_class, 0.0))

        swap_table = swap_map.get("swap_rates", {}) or {}
        if symbol in swap_table:
            swap_long = float(swap_table[symbol]["long"])
            swap_short = float(swap_table[symbol]["short"])
        else:
            logger.warning("No swap rates for %s — defaulting to 0", symbol)
            swap_long = 0.0
            swap_short = 0.0

        triple_day_name = swap_map.get("triple_swap_day", "Wednesday")
        try:
            triple_weekday = _WEEKDAYS.index(triple_day_name)
        except ValueError as exc:
            raise ValueError(
                f"Unknown triple_swap_day '{triple_day_name}' in {swap_rates_path}"
            ) from exc

        return cls(
            symbol=symbol,
            point_size=point_size,
            usd_per_price_unit=upu,
            commission_per_lot=commission_per_lot,
            swap_long=swap_long,
            swap_short=swap_short,
            triple_swap_weekday=triple_weekday,
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class VectorizedBacktest:
    """Single-position vectorised backtester.

    Parameters
    ----------
    data
        DataFrame with at least ``open``, ``high``, ``low``, ``close``. ``spread``
        is recommended (in MT5 points); a missing column is treated as 0. Index
        must be a ``DatetimeIndex`` or have a ``time`` column the engine can
        promote.
    symbol
        Used to load a ``SymbolSpec`` from ``config/`` if ``symbol_spec`` is not
        passed directly.
    initial_balance
        Account starting equity in USD. Sizing is computed off this fixed value
        (no compounding) — a research-engine simplification.
    risk_per_trade_pct
        Fraction (in percent, so ``0.5`` means 0.5%) of initial balance to risk
        on each trade. Lots are sized so that an SL hit costs exactly this
        amount, ignoring spread / swap. Overridden by ``fixed_lots``.
    fixed_lots
        If set, every trade uses this lot size and ``risk_per_trade_pct`` is
        ignored.
    symbol_spec
        Inject a ``SymbolSpec`` directly (tests / scripted research). When
        passed, the YAML config files are not read.
    swap_config
        ``{"long": float, "short": float}`` overrides for daily swap. Useful
        when calibrating against a fresh broker swap stream without editing
        the YAML.
    commission_per_lot
        Override the asset-class default commission.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        symbol: Optional[str] = None,
        initial_balance: float = 10_000.0,
        risk_per_trade_pct: float = 0.5,
        fixed_lots: Optional[float] = None,
        symbol_spec: Optional[SymbolSpec] = None,
        swap_config: Optional[Dict[str, float]] = None,
        commission_per_lot: Optional[float] = None,
    ):
        if data is None or len(data) == 0:
            raise ValueError("data must be a non-empty DataFrame")
        for col in ("open", "high", "low", "close"):
            if col not in data.columns:
                raise KeyError(f"data missing required column '{col}'")

        df = data.copy()
        if "spread" not in df.columns:
            logger.warning("data has no 'spread' column — assuming 0 spread")
            df["spread"] = 0

        if not isinstance(df.index, pd.DatetimeIndex):
            if "time" in df.columns:
                df = df.set_index(pd.DatetimeIndex(df["time"]))
                df = df.drop(columns=["time"])
            else:
                raise KeyError(
                    "data needs a DatetimeIndex or a 'time' column"
                )
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
        self.data = df

        self.initial_balance = float(initial_balance)
        self.risk_per_trade_pct = float(risk_per_trade_pct)
        self.fixed_lots = float(fixed_lots) if fixed_lots is not None else None

        if symbol_spec is None:
            if symbol is None:
                raise ValueError("Provide either symbol_spec or symbol")
            symbol_spec = SymbolSpec.from_config(symbol)
        # Apply optional overrides on the chosen spec (without mutating the
        # caller's instance).
        if commission_per_lot is not None or swap_config:
            symbol_spec = SymbolSpec(
                symbol=symbol_spec.symbol,
                point_size=symbol_spec.point_size,
                usd_per_price_unit=symbol_spec.usd_per_price_unit,
                commission_per_lot=(
                    float(commission_per_lot)
                    if commission_per_lot is not None
                    else symbol_spec.commission_per_lot
                ),
                swap_long=(
                    float(swap_config["long"]) if swap_config and "long" in swap_config
                    else symbol_spec.swap_long
                ),
                swap_short=(
                    float(swap_config["short"]) if swap_config and "short" in swap_config
                    else symbol_spec.swap_short
                ),
                triple_swap_weekday=symbol_spec.triple_swap_weekday,
            )
        self.spec = symbol_spec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        signal_long: pd.Series,
        signal_short: pd.Series,
        sl_distance: pd.Series,
        tp_distance: pd.Series,
        session_mask: Optional[pd.Series] = None,
        max_holding_bars: int = 16,
    ) -> BacktestResult:
        """Run the backtest.

        All input series are aligned to ``self.data.index`` via reindex; missing
        values default to ``False`` (signals / session) or ``NaN`` (distances).
        A bar with ``NaN`` SL or TP cannot trigger an entry.
        """
        if max_holding_bars < 1:
            raise ValueError("max_holding_bars must be ≥ 1")

        n = len(self.data)
        idx = self.data.index
        if session_mask is None:
            session_mask = pd.Series(True, index=idx)

        sig_long = (
            signal_long.reindex(idx).fillna(False).astype(bool).to_numpy()
        )
        sig_short = (
            signal_short.reindex(idx).fillna(False).astype(bool).to_numpy()
        )
        sl_dist = sl_distance.reindex(idx).to_numpy(dtype=float)
        tp_dist = tp_distance.reindex(idx).to_numpy(dtype=float)
        sess = (
            session_mask.reindex(idx).fillna(True).astype(bool).to_numpy()
        )

        opens = self.data["open"].to_numpy(dtype=float)
        highs = self.data["high"].to_numpy(dtype=float)
        lows = self.data["low"].to_numpy(dtype=float)
        closes = self.data["close"].to_numpy(dtype=float)
        spreads_pts = self.data["spread"].to_numpy(dtype=float)
        times = idx

        spec = self.spec
        point_size = spec.point_size
        upu = spec.usd_per_price_unit

        trades: List[TradeFill] = []
        i = 0
        while i < n - 1:
            if not sess[i]:
                i += 1
                continue
            if sig_long[i] and not sig_short[i]:
                side = "long"
            elif sig_short[i] and not sig_long[i]:
                side = "short"
            else:
                i += 1
                continue

            sl_d = sl_dist[i]
            tp_d = tp_dist[i]
            if not (np.isfinite(sl_d) and np.isfinite(tp_d) and sl_d > 0 and tp_d > 0):
                i += 1
                continue

            entry_idx = i + 1
            if entry_idx >= n:
                break
            mid_open = opens[entry_idx]
            entry_spread_price = spreads_pts[entry_idx] * point_size

            if side == "long":
                actual_open = mid_open + entry_spread_price / 2.0
                sl_price = mid_open - sl_d
                tp_price = mid_open + tp_d
                direction = 1.0
            else:
                actual_open = mid_open - entry_spread_price / 2.0
                sl_price = mid_open + sl_d
                tp_price = mid_open - tp_d
                direction = -1.0

            if self.fixed_lots is not None:
                lots = self.fixed_lots
            else:
                risk_usd = self.initial_balance * self.risk_per_trade_pct / 100.0
                denom = sl_d * upu
                if denom <= 0:
                    i += 1
                    continue
                lots = risk_usd / denom

            exit_reason: Optional[str] = None
            exit_idx: Optional[int] = None
            exit_mid: Optional[float] = None

            for j in range(entry_idx, n):
                # 1. Intra-bar SL / TP. Pessimistic SL-first when both touched.
                if side == "long":
                    hit_sl = lows[j] <= sl_price
                    hit_tp = highs[j] >= tp_price
                else:
                    hit_sl = highs[j] >= sl_price
                    hit_tp = lows[j] <= tp_price

                if hit_sl:
                    exit_reason = "sl"
                    exit_mid = sl_price
                    exit_idx = j
                    break
                if hit_tp:
                    exit_reason = "tp"
                    exit_mid = tp_price
                    exit_idx = j
                    break

                # 2. End-of-bar checks (only reached if neither SL nor TP hit).
                bars_held = j - entry_idx + 1
                if bars_held >= max_holding_bars:
                    exit_reason = "time_exit"
                    exit_mid = closes[j]
                    exit_idx = j
                    break
                # Session close: exit at close of last in-session bar.
                if j + 1 < n and not sess[j + 1]:
                    exit_reason = "session_close"
                    exit_mid = closes[j]
                    exit_idx = j
                    break
                # Reverse signal evaluated at this bar's close.
                if side == "long" and sig_short[j] and not sig_long[j]:
                    exit_reason = "reverse_signal"
                    exit_mid = closes[j]
                    exit_idx = j
                    break
                if side == "short" and sig_long[j] and not sig_short[j]:
                    exit_reason = "reverse_signal"
                    exit_mid = closes[j]
                    exit_idx = j
                    break

            if exit_reason is None:
                # Fell off the end of the dataset.
                exit_reason = "time_exit"
                exit_idx = n - 1
                exit_mid = closes[n - 1]

            entry_time = pd.Timestamp(times[entry_idx])
            exit_time = pd.Timestamp(times[exit_idx])
            holding_bars = exit_idx - entry_idx + 1
            exit_spread_price = spreads_pts[exit_idx] * point_size
            if side == "long":
                actual_exit = exit_mid - exit_spread_price / 2.0
            else:
                actual_exit = exit_mid + exit_spread_price / 2.0

            profit_directional = (exit_mid - mid_open) * direction * lots * upu
            avg_spread_price = (entry_spread_price + exit_spread_price) / 2.0
            profit_spread_cost = -avg_spread_price * lots * upu
            profit_swap = self._compute_swap(entry_time, exit_time, side, lots)
            profit_commission = -spec.commission_per_lot * lots
            profit_total = (
                profit_directional
                + profit_spread_cost
                + profit_swap
                + profit_commission
            )

            trades.append(TradeFill(
                side=side,
                open_time=entry_time,
                close_time=exit_time,
                open_price=float(mid_open),
                close_price=float(exit_mid),
                open_price_with_spread=float(actual_open),
                close_price_with_spread=float(actual_exit),
                sl=float(sl_price),
                tp=float(tp_price),
                lots=float(lots),
                holding_bars=int(holding_bars),
                profit_directional=float(profit_directional),
                profit_swap=float(profit_swap),
                profit_commission=float(profit_commission),
                profit_spread_cost=float(profit_spread_cost),
                profit_total=float(profit_total),
                exit_reason=exit_reason,
            ))

            i = exit_idx + 1

        equity = self._equity_curve(trades, times)
        metrics = self._metrics(trades, equity)
        decomposition = self._pnl_decomposition(trades)
        return BacktestResult(
            trades=trades,
            equity=equity,
            metrics=metrics,
            pnl_decomposition=decomposition,
            initial_balance=self.initial_balance,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _compute_swap(
        self,
        entry_time: pd.Timestamp,
        exit_time: pd.Timestamp,
        side: str,
        lots: float,
    ) -> float:
        """Sum daily swap charges for every 00:00 strictly between entry and exit.

        Triple-swap day applies when the midnight transitions *to* the
        configured weekday (default Wednesday → weekday=2 in Python's
        Mon=0 indexing).
        """
        if exit_time <= entry_time:
            return 0.0
        spec = self.spec
        rate = spec.swap_long if side == "long" else spec.swap_short
        if rate == 0.0:
            return 0.0
        first_midnight = entry_time.normalize() + pd.Timedelta(days=1)
        total = 0.0
        cur = first_midnight
        while cur < exit_time:
            mult = 3.0 if cur.weekday() == spec.triple_swap_weekday else 1.0
            total += rate * mult * lots
            cur = cur + pd.Timedelta(days=1)
        return total

    def _equity_curve(
        self, trades: List[TradeFill], times: pd.DatetimeIndex
    ) -> pd.Series:
        start_idx = pd.Timestamp(times[0])
        if not trades:
            return pd.Series([self.initial_balance], index=[start_idx], name="equity")
        close_times = [t.close_time for t in trades]
        pnls = np.cumsum([t.profit_total for t in trades])
        equity_after = pd.Series(
            self.initial_balance + pnls, index=close_times, name="equity"
        )
        # Prepend the starting equity at the first bar's timestamp so the curve
        # begins on the data window, not on the first close.
        if close_times[0] == start_idx:
            return equity_after
        start = pd.Series([self.initial_balance], index=[start_idx], name="equity")
        return pd.concat([start, equity_after])

    def _metrics(
        self, trades: List[TradeFill], equity: pd.Series
    ) -> Dict[str, float]:
        if not trades:
            return {
                "n_trades": 0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "calmar": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_return": 0.0,
                "total_return_pct": 0.0,
                "avg_holding_bars": 0.0,
            }
        pnls = np.array([t.profit_total for t in trades], dtype=float)
        n = len(pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        win_rate = len(wins) / n
        gross_loss = -losses.sum()
        if gross_loss > 0:
            profit_factor = float(wins.sum() / gross_loss)
        elif wins.sum() > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0
        total_pnl = float(pnls.sum())
        total_return_pct = total_pnl / self.initial_balance * 100.0

        # Daily resample for Sharpe / Sortino. Forward-fill to carry equity
        # across non-trading days so std isn't biased by missing weekends.
        eq_daily = equity.resample("1D").last().ffill()
        daily_ret = eq_daily.pct_change().dropna()
        if len(daily_ret) > 1 and daily_ret.std(ddof=0) > 0:
            sharpe = (
                daily_ret.mean() / daily_ret.std(ddof=0) * np.sqrt(252)
            )
            downside = daily_ret[daily_ret < 0]
            if len(downside) > 0 and downside.std(ddof=0) > 0:
                sortino = (
                    daily_ret.mean() / downside.std(ddof=0) * np.sqrt(252)
                )
            else:
                sortino = float("inf") if daily_ret.mean() > 0 else 0.0
        else:
            sharpe = 0.0
            sortino = 0.0

        rolling_max = equity.cummax()
        dd = equity - rolling_max
        max_dd = float(dd.min())
        max_dd_pct = float((dd / rolling_max).min() * 100.0)

        # Calmar = CAGR / |max DD|. Span < 1 day → return raw total return ratio.
        days_span = (equity.index[-1] - equity.index[0]).total_seconds() / 86400.0
        years = max(days_span / 365.25, 1e-9)
        final_eq = float(equity.iloc[-1])
        if final_eq > 0 and years > 0:
            cagr = (final_eq / self.initial_balance) ** (1.0 / years) - 1.0
        else:
            cagr = -1.0
        if max_dd_pct < 0:
            calmar = float(cagr / abs(max_dd_pct / 100.0))
        else:
            calmar = 0.0

        return {
            "n_trades": int(n),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "calmar": float(calmar),
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd_pct,
            "win_rate": float(win_rate),
            "profit_factor": profit_factor,
            "total_return": total_pnl,
            "total_return_pct": float(total_return_pct),
            "avg_holding_bars": float(np.mean([t.holding_bars for t in trades])),
        }

    def _pnl_decomposition(self, trades: List[TradeFill]) -> Dict[str, float]:
        if not trades:
            return {
                "directional": 0.0,
                "swap": 0.0,
                "commission": 0.0,
                "spread_cost": 0.0,
                "directional_pct": 0.0,
                "swap_pct": 0.0,
                "commission_pct": 0.0,
                "spread_pct": 0.0,
            }
        directional = float(sum(t.profit_directional for t in trades))
        swap = float(sum(t.profit_swap for t in trades))
        commission = float(sum(t.profit_commission for t in trades))
        spread_cost = float(sum(t.profit_spread_cost for t in trades))
        # Percentages are taken on the absolute magnitudes so the four buckets
        # sum to 100% regardless of sign — this is the "where did the money
        # come from / go to" view the WTI guard relies on.
        abs_total = (
            abs(directional) + abs(swap) + abs(commission) + abs(spread_cost)
        )
        if abs_total == 0:
            d_pct = s_pct = c_pct = sp_pct = 0.0
        else:
            d_pct = abs(directional) / abs_total * 100.0
            s_pct = abs(swap) / abs_total * 100.0
            c_pct = abs(commission) / abs_total * 100.0
            sp_pct = abs(spread_cost) / abs_total * 100.0
        return {
            "directional": directional,
            "swap": swap,
            "commission": commission,
            "spread_cost": spread_cost,
            "directional_pct": float(d_pct),
            "swap_pct": float(s_pct),
            "commission_pct": float(c_pct),
            "spread_pct": float(sp_pct),
        }


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _load_strategy(path: Path):
    """Dynamically import a Python file expected to expose ``signals(df)``.

    The function must return a dict with keys ``signal_long``, ``signal_short``,
    ``sl_distance``, ``tp_distance`` (all ``pd.Series`` aligned to ``df``), plus
    optional ``session_mask`` and ``max_holding_bars``.
    """
    spec_module = importlib.util.spec_from_file_location("strategy_module", path)
    if spec_module is None or spec_module.loader is None:
        raise ImportError(f"Cannot load strategy from {path}")
    mod = importlib.util.module_from_spec(spec_module)
    spec_module.loader.exec_module(mod)
    if not hasattr(mod, "signals"):
        raise AttributeError(
            f"{path} must define a top-level signals(df) function"
        )
    return mod.signals


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m python_engine.vectorized_backtest",
        description="Run a vectorised Python backtest with explicit cost model.",
    )
    p.add_argument("--strategy", required=True, type=Path,
                   help="Path to a .py file exposing signals(df) -> dict.")
    p.add_argument("--data", required=True, type=Path,
                   help="Parquet file with OHLC + spread (from data_fetcher).")
    p.add_argument("--symbol", required=True, help="e.g. EURUSD")
    p.add_argument("--initial-balance", type=float, default=10_000.0)
    p.add_argument("--risk-pct", type=float, default=0.5)
    p.add_argument("--fixed-lots", type=float, default=None)
    p.add_argument("--max-holding-bars", type=int, default=16)
    p.add_argument("--export-trades", type=Path, default=None,
                   help="Write closed trades to this CSV.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    df = pd.read_parquet(args.data)
    signals_fn = _load_strategy(args.strategy)
    payload = signals_fn(df)
    required = ("signal_long", "signal_short", "sl_distance", "tp_distance")
    for key in required:
        if key not in payload:
            raise KeyError(f"strategy.signals(df) missing key '{key}'")

    engine = VectorizedBacktest(
        data=df,
        symbol=args.symbol,
        initial_balance=args.initial_balance,
        risk_per_trade_pct=args.risk_pct,
        fixed_lots=args.fixed_lots,
    )
    result = engine.run(
        signal_long=payload["signal_long"],
        signal_short=payload["signal_short"],
        sl_distance=payload["sl_distance"],
        tp_distance=payload["tp_distance"],
        session_mask=payload.get("session_mask"),
        max_holding_bars=payload.get("max_holding_bars", args.max_holding_bars),
    )

    print(json.dumps({
        "metrics": result.metrics,
        "pnl_decomposition": result.pnl_decomposition,
    }, indent=2))

    if args.export_trades is not None:
        result.export_trades_csv(args.export_trades)
        logger.info("Exported %d trades → %s", len(result.trades), args.export_trades)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
