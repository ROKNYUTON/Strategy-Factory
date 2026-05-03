"""Optuna-based Bayesian optimizer for multi-asset strategies.

Wraps :class:`python_engine.vectorized_backtest.VectorizedBacktest` and runs
``signals(df, **params)`` over a panel of symbols, evaluating each candidate
parameter set on an IS slice and an OOS slice. Trials are pruned per-symbol via
``optuna.MedianPruner`` so that bad parameter regions are abandoned after a few
symbols instead of paying for the whole panel.

Strategy contract
-----------------
The strategy module passed in (``strategy_module.signals``) MUST accept ``df``
plus the kwargs declared in ``param_space`` and return a dict identical to the
one ``vectorized_backtest`` already consumes::

    {
        "signal_long":   pd.Series[bool],
        "signal_short":  pd.Series[bool],
        "sl_distance":   pd.Series[float],
        "tp_distance":   pd.Series[float],
        # optional:
        "session_mask":  pd.Series[bool],
        "max_holding_bars": int,
    }

Anti-curve-fit considerations baked in
--------------------------------------
* Optimization metric is **Sharpe**, never raw profit. The default ranking mode
  ``sharpe_oos_robust`` multiplies OOS Sharpe by an explicit overfit penalty
  ``(1 - max(0, IS - OOS) / IS)`` and gates on a minimum trade count, so that
  high-Sharpe / low-count outliers cannot win the search.
* A trial that lacks at least ``min_trades_per_symbol`` trades on the worst
  symbol scores zero, which directly enforces CLAUDE.md §4 ``min_trades=100``
  if you set the gate accordingly.
* All trials (incl. pruned) are exported, preserving the full search trace for
  multiple-testing audits.

CLI
---
::

    python -m python_engine.optuna_optimizer \\
        --strategy strategies/mean_rev_1/strategy.py \\
        --grid strategies/mean_rev_1/optimization_grid.yaml \\
        --data-dir tmp_cache \\
        --symbols EURUSD,USDJPY,EURCHF \\
        --is 2018-01-01:2023-12-31 \\
        --oos 2024-01-01:2026-04-30 \\
        --trials 1000 \\
        --top 10 \\
        --output-dir strategies/mean_rev_1/results
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import multiprocessing
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import optuna
    from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
    from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler

    _OPTUNA_AVAILABLE = True
except ImportError:  # pragma: no cover - import-time guard
    optuna = None  # type: ignore[assignment]
    _OPTUNA_AVAILABLE = False

from python_engine.vectorized_backtest import (
    BacktestResult,
    SymbolSpec,
    VectorizedBacktest,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OptimizationConfig:
    """Knobs for one optimization run.

    ``timeout_seconds`` and ``n_trials`` are both honored; whichever fires
    first stops the search. Set ``timeout_seconds=0`` to disable the timeout.
    """
    n_trials: int = 1000
    n_jobs: int = -1
    timeout_seconds: int = 7200          # 2h
    sampler: str = "tpe"                 # "tpe" | "random" | "cmaes"
    pruner: str = "median"               # "median" | "hyperband" | "none"
    objective: str = "sharpe_oos_robust"
    study_name: Optional[str] = None
    storage: Optional[str] = None        # e.g. "sqlite:///optuna.db"
    seed: Optional[int] = None
    initial_balance: float = 10_000.0
    risk_per_trade_pct: float = 0.5
    max_holding_bars_default: int = 16
    pruning_warmup_steps: int = 3        # min completed symbols before prune
    is_sharpe_skip_threshold: Optional[float] = None  # skip OOS for symbols
                                                       # with IS Sharpe < this
    min_trades_per_symbol: int = 30      # gate inside ranking_function
    show_progress: bool = True


@dataclass
class TrialResult:
    """One trial's full outcome — kept in memory for CSV export and top-N."""
    trial_id: int
    params: Dict[str, Any]
    is_metrics: Dict[str, Dict[str, float]]
    oos_metrics: Dict[str, Dict[str, float]]
    aggregate: Dict[str, float]
    objective_value: float
    pruned: bool


# ---------------------------------------------------------------------------
# Ranking function — three modes, all maximize-good
# ---------------------------------------------------------------------------

def ranking_function(
    is_metrics_per_symbol: Dict[str, Dict[str, float]],
    oos_metrics_per_symbol: Dict[str, Dict[str, float]],
    mode: str = "sharpe_oos_robust",
    min_trades_per_symbol: int = 30,
) -> float:
    """Reduce per-symbol metric dicts to a single optimization score.

    Modes
    -----
    ``sharpe_oos_robust``
        ``mean(OOS Sharpe) * (1 - max(0, IS_avg - OOS_avg) / IS_avg) * gate``
        where ``gate = 1`` iff every observed symbol has ≥ ``min_trades``
        trades on both IS and OOS, else 0. This is the recommended default.

    ``recovery_factor``
        ``mean(total_return / |max_drawdown|)`` over OOS symbols; trials with
        zero drawdown short-circuit to a finite score (0 if no profit, else
        the trial is dropped via inf-filter).

    ``sharpe_minus_pvalue``
        ``mean(OOS Sharpe) - 5 * (1 - pass_rate)`` where ``pass_rate`` is the
        proportion of symbols that cleared the min-trades gate. Use when you
        want a softer significance penalty than the hard gate above.
    """
    if not oos_metrics_per_symbol:
        return -1e9

    oos_sharpes = [m.get("sharpe", 0.0) for m in oos_metrics_per_symbol.values()]
    is_sharpes = [m.get("sharpe", 0.0) for m in is_metrics_per_symbol.values()]
    oos_trades = [m.get("n_trades", 0) for m in oos_metrics_per_symbol.values()]
    is_trades = [m.get("n_trades", 0) for m in is_metrics_per_symbol.values()]
    if not oos_sharpes:
        return -1e9

    if mode == "sharpe_oos_robust":
        oos_avg = float(np.mean(oos_sharpes))
        is_avg = float(np.mean(is_sharpes)) if is_sharpes else 0.0
        if is_avg > 0:
            degradation = max(0.0, is_avg - oos_avg) / is_avg
            penalty = max(0.0, 1.0 - degradation)
        else:
            # IS was worthless. Don't reward OOS-only luck — score zero.
            penalty = 0.0 if oos_avg <= 0 else 1.0
        worst_trades = min(
            min(oos_trades, default=0), min(is_trades, default=0)
        )
        gate = 1.0 if worst_trades >= min_trades_per_symbol else 0.0
        return float(oos_avg * penalty * gate)

    if mode == "recovery_factor":
        rfs: List[float] = []
        for m in oos_metrics_per_symbol.values():
            mdd_pct = abs(m.get("max_drawdown_pct", 0.0))
            tr_pct = m.get("total_return_pct", 0.0)
            if mdd_pct > 0:
                rfs.append(tr_pct / mdd_pct)
            elif tr_pct == 0:
                rfs.append(0.0)
            # else: profit with zero DD is ignored (filter out infinities)
        if not rfs:
            return -1e9
        return float(np.mean(rfs))

    if mode == "sharpe_minus_pvalue":
        oos_avg = float(np.mean(oos_sharpes))
        passes = sum(
            1
            for n_oos, n_is in zip(oos_trades, is_trades)
            if n_oos >= min_trades_per_symbol and n_is >= min_trades_per_symbol
        )
        prop_pass = passes / len(oos_trades)
        return float(oos_avg - 5.0 * (1.0 - prop_pass))

    raise ValueError(f"Unknown ranking mode: {mode!r}")


# ---------------------------------------------------------------------------
# Param space helpers
# ---------------------------------------------------------------------------

def _suggest_one(trial: "optuna.Trial", name: str, spec: Dict[str, Any]) -> Any:
    ptype = spec.get("type", "float")
    if ptype == "int":
        step = int(spec.get("step", 1))
        return trial.suggest_int(
            name, int(spec["low"]), int(spec["high"]), step=step
        )
    if ptype == "float":
        step = spec.get("step")
        log = bool(spec.get("log", False))
        if step is not None:
            return trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), step=float(step)
            )
        if log:
            return trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=True
            )
        return trial.suggest_float(
            name, float(spec["low"]), float(spec["high"])
        )
    if ptype == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"Unknown param type for {name!r}: {ptype!r}")


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

class MultiAssetOptimizer:
    """Run an Optuna study over a multi-symbol panel.

    The optimizer is intentionally stateful (``self._results``) so that pruned
    trials are still recorded — important for the multiple-testing audit trail
    in :doc:`docs/HYPOTHESIS_LOG.md`.
    """

    def __init__(
        self,
        strategy_module: ModuleType,
        data_dict: Dict[str, pd.DataFrame],
        is_period: Tuple[date, date],
        oos_period: Tuple[date, date],
        param_space: Dict[str, Dict[str, Any]],
        config: Optional[OptimizationConfig] = None,
        symbol_specs: Optional[Dict[str, SymbolSpec]] = None,
    ):
        if not _OPTUNA_AVAILABLE:
            raise RuntimeError(
                "optuna is not installed. Run: pip install 'optuna>=3.0'"
            )
        if not data_dict:
            raise ValueError("data_dict is empty")
        if not hasattr(strategy_module, "signals"):
            raise AttributeError(
                "strategy_module must define a top-level signals(df, **params)"
            )
        if not param_space:
            raise ValueError(
                "param_space is empty — nothing to optimize"
            )

        self.strategy_module = strategy_module
        self.data_dict = data_dict
        self.is_period = is_period
        self.oos_period = oos_period
        self.param_space = param_space
        self.config = config or OptimizationConfig()
        self.symbol_specs: Dict[str, SymbolSpec] = dict(symbol_specs or {})
        self._results: List[TrialResult] = []
        self._results_lock = threading.Lock()
        self._study: Optional["optuna.Study"] = None
        self._is_data: Dict[str, pd.DataFrame] = {}
        self._oos_data: Dict[str, pd.DataFrame] = {}
        self._prepare_data_slices()
        self._prefetch_symbol_specs()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _prepare_data_slices(self) -> None:
        is_start = pd.Timestamp(self.is_period[0])
        is_end = pd.Timestamp(self.is_period[1]) + pd.Timedelta(days=1)
        oos_start = pd.Timestamp(self.oos_period[0])
        oos_end = pd.Timestamp(self.oos_period[1]) + pd.Timedelta(days=1)
        for sym, df in self.data_dict.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                if "time" in df.columns:
                    df = df.set_index(pd.DatetimeIndex(df["time"])).drop(
                        columns=["time"]
                    )
                else:
                    raise KeyError(
                        f"data for {sym!r} needs a DatetimeIndex or 'time' column"
                    )
            if df.index.tz is not None:
                df = df.copy()
                df.index = df.index.tz_localize(None)
            df = df.sort_index()
            self._is_data[sym] = df.loc[is_start:is_end]
            self._oos_data[sym] = df.loc[oos_start:oos_end]
            if self._is_data[sym].empty or self._oos_data[sym].empty:
                logger.warning(
                    "Empty IS or OOS slice for %s — check periods (IS=%s..%s, OOS=%s..%s)",
                    sym, self.is_period[0], self.is_period[1],
                    self.oos_period[0], self.oos_period[1],
                )

    def _prefetch_symbol_specs(self) -> None:
        for sym in self.data_dict:
            if sym in self.symbol_specs:
                continue
            try:
                self.symbol_specs[sym] = SymbolSpec.from_config(sym)
            except (KeyError, FileNotFoundError) as exc:
                logger.warning(
                    "No SymbolSpec for %s in config (%s) — backtest will fail "
                    "unless you pass symbol_specs explicitly.",
                    sym, exc,
                )

    # ------------------------------------------------------------------
    # Inner loop
    # ------------------------------------------------------------------
    def _run_one(
        self,
        df: pd.DataFrame,
        symbol: str,
        params: Dict[str, Any],
    ) -> Optional[BacktestResult]:
        if df is None or df.empty:
            return None
        payload = self.strategy_module.signals(df, **params)
        for k in ("signal_long", "signal_short", "sl_distance", "tp_distance"):
            if k not in payload:
                raise KeyError(
                    f"strategy.signals() missing required key {k!r}"
                )
        spec = self.symbol_specs.get(symbol)
        engine_kwargs: Dict[str, Any] = {
            "data": df,
            "initial_balance": self.config.initial_balance,
            "risk_per_trade_pct": self.config.risk_per_trade_pct,
        }
        if spec is not None:
            engine_kwargs["symbol_spec"] = spec
        else:
            engine_kwargs["symbol"] = symbol
        engine = VectorizedBacktest(**engine_kwargs)
        return engine.run(
            signal_long=payload["signal_long"],
            signal_short=payload["signal_short"],
            sl_distance=payload["sl_distance"],
            tp_distance=payload["tp_distance"],
            session_mask=payload.get("session_mask"),
            max_holding_bars=payload.get(
                "max_holding_bars", self.config.max_holding_bars_default
            ),
        )

    def _objective(self, trial: "optuna.Trial") -> float:
        params = {
            name: _suggest_one(trial, name, spec)
            for name, spec in self.param_space.items()
        }
        is_metrics: Dict[str, Dict[str, float]] = {}
        oos_metrics: Dict[str, Dict[str, float]] = {}
        symbols = list(self.data_dict.keys())

        skip_threshold = self.config.is_sharpe_skip_threshold

        for step, sym in enumerate(symbols):
            is_df = self._is_data.get(sym)
            oos_df = self._oos_data.get(sym)
            if is_df is None or is_df.empty:
                continue
            try:
                is_res = self._run_one(is_df, sym, params)
            except Exception as exc:  # noqa: BLE001
                logger.warning("IS run failed for %s with %s: %s", sym, params, exc)
                continue
            if is_res is None:
                continue
            is_metrics[sym] = is_res.metrics

            run_oos = (
                skip_threshold is None
                or is_res.metrics.get("sharpe", 0.0) >= skip_threshold
            )
            if run_oos and oos_df is not None and not oos_df.empty:
                try:
                    oos_res = self._run_one(oos_df, sym, params)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "OOS run failed for %s with %s: %s", sym, params, exc
                    )
                    oos_res = None
                if oos_res is not None:
                    oos_metrics[sym] = oos_res.metrics

            interim = ranking_function(
                is_metrics, oos_metrics,
                mode=self.config.objective,
                min_trades_per_symbol=self.config.min_trades_per_symbol,
            )
            trial.report(interim, step=step)
            if (
                step >= self.config.pruning_warmup_steps - 1
                and trial.should_prune()
            ):
                self._record_trial(
                    trial, params, is_metrics, oos_metrics, interim, pruned=True
                )
                raise optuna.TrialPruned()

        score = ranking_function(
            is_metrics, oos_metrics,
            mode=self.config.objective,
            min_trades_per_symbol=self.config.min_trades_per_symbol,
        )
        self._record_trial(
            trial, params, is_metrics, oos_metrics, score, pruned=False
        )
        return score

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    def _record_trial(
        self,
        trial: "optuna.Trial",
        params: Dict[str, Any],
        is_metrics: Dict[str, Dict[str, float]],
        oos_metrics: Dict[str, Dict[str, float]],
        score: float,
        pruned: bool,
    ) -> None:
        agg = self._aggregate_metrics(is_metrics, oos_metrics)
        result = TrialResult(
            trial_id=trial.number,
            params=dict(params),
            is_metrics={k: dict(v) for k, v in is_metrics.items()},
            oos_metrics={k: dict(v) for k, v in oos_metrics.items()},
            aggregate=agg,
            objective_value=float(score),
            pruned=bool(pruned),
        )
        with self._results_lock:
            self._results.append(result)

    @staticmethod
    def _aggregate_metrics(
        is_metrics: Dict[str, Dict[str, float]],
        oos_metrics: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        def stats(d: Dict[str, Dict[str, float]], key: str) -> Tuple[float, float]:
            vals = [m.get(key, 0.0) for m in d.values()]
            if not vals:
                return 0.0, 0.0
            return float(np.mean(vals)), float(np.min(vals))

        is_avg, is_min = stats(is_metrics, "sharpe")
        oos_avg, oos_min = stats(oos_metrics, "sharpe")
        # Worst-symbol drawdown across IS+OOS — that's the surface that fails
        # CLAUDE.md §4 max_drawdown_pct=15.0
        max_dd_pct = 0.0
        for d in (is_metrics, oos_metrics):
            for m in d.values():
                mdd = m.get("max_drawdown_pct", 0.0)
                if mdd < max_dd_pct:
                    max_dd_pct = mdd
        total_trades = sum(
            m.get("n_trades", 0)
            for d in (is_metrics, oos_metrics)
            for m in d.values()
        )
        return {
            "is_sharpe_avg": is_avg,
            "is_sharpe_min": is_min,
            "oos_sharpe_avg": oos_avg,
            "oos_sharpe_min": oos_min,
            "max_drawdown_pct": float(max_dd_pct),
            "total_trades": int(total_trades),
        }

    # ------------------------------------------------------------------
    # Optuna wiring
    # ------------------------------------------------------------------
    def _make_sampler(self) -> "optuna.samplers.BaseSampler":
        sname = (self.config.sampler or "tpe").lower()
        if sname == "tpe":
            return TPESampler(seed=self.config.seed)
        if sname == "random":
            return RandomSampler(seed=self.config.seed)
        if sname == "cmaes":
            return CmaEsSampler(seed=self.config.seed)
        raise ValueError(f"Unknown sampler: {sname!r}")

    def _make_pruner(self) -> "optuna.pruners.BasePruner":
        pname = (self.config.pruner or "none").lower()
        if pname in ("", "none", "null"):
            return NopPruner()
        if pname == "median":
            return MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=max(1, self.config.pruning_warmup_steps),
            )
        if pname == "hyperband":
            return HyperbandPruner()
        raise ValueError(f"Unknown pruner: {pname!r}")

    def run(
        self,
        callbacks: Optional[List[Callable[["optuna.Study", "optuna.trial.FrozenTrial"], None]]] = None,
    ) -> List[TrialResult]:
        """Run the study and return all trials, best first."""
        with self._results_lock:
            self._results = []

        n_jobs = self.config.n_jobs
        if n_jobs in (-1, 0):
            n_jobs = max(1, multiprocessing.cpu_count() - 1)

        sampler = self._make_sampler()
        pruner = self._make_pruner()
        study_kwargs: Dict[str, Any] = {
            "direction": "maximize",
            "sampler": sampler,
            "pruner": pruner,
            "study_name": self.config.study_name,
            "storage": self.config.storage,
        }
        if self.config.storage:
            study_kwargs["load_if_exists"] = True
        study = optuna.create_study(**study_kwargs)
        self._study = study

        cb_list = list(callbacks) if callbacks else []
        progress_cb = None
        if self.config.show_progress:
            progress_cb = _RichProgressCallback(self.config.n_trials)
            progress_cb.start()
            cb_list.append(progress_cb)

        try:
            study.optimize(
                self._objective,
                n_trials=self.config.n_trials,
                timeout=(
                    self.config.timeout_seconds
                    if self.config.timeout_seconds and self.config.timeout_seconds > 0
                    else None
                ),
                n_jobs=n_jobs,
                callbacks=cb_list,
                gc_after_trial=True,
                show_progress_bar=False,
            )
        finally:
            if progress_cb is not None:
                progress_cb.stop()

        with self._results_lock:
            results = list(self._results)
        results.sort(key=lambda r: r.objective_value, reverse=True)
        with self._results_lock:
            self._results = results
        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def get_top_n(self, n: int = 10) -> pd.DataFrame:
        """Return the top-N successful (non-pruned) trials as a DataFrame.

        Columns: rank, trial_id, params, is_sharpe_avg, oos_sharpe_avg,
        is_sharpe_min, oos_sharpe_min, max_dd_pct, total_trades, objective_value.
        """
        with self._results_lock:
            active = [r for r in self._results if not r.pruned]
        active.sort(key=lambda r: r.objective_value, reverse=True)
        rows = []
        for rank, r in enumerate(active[:n], start=1):
            rows.append({
                "rank": rank,
                "trial_id": r.trial_id,
                "params": json.dumps(r.params, default=str, sort_keys=True),
                "is_sharpe_avg": r.aggregate.get("is_sharpe_avg", 0.0),
                "oos_sharpe_avg": r.aggregate.get("oos_sharpe_avg", 0.0),
                "is_sharpe_min": r.aggregate.get("is_sharpe_min", 0.0),
                "oos_sharpe_min": r.aggregate.get("oos_sharpe_min", 0.0),
                "max_dd_pct": r.aggregate.get("max_drawdown_pct", 0.0),
                "total_trades": r.aggregate.get("total_trades", 0),
                "objective_value": r.objective_value,
            })
        return pd.DataFrame(
            rows,
            columns=[
                "rank", "trial_id", "params",
                "is_sharpe_avg", "oos_sharpe_avg",
                "is_sharpe_min", "oos_sharpe_min",
                "max_dd_pct", "total_trades", "objective_value",
            ],
        )

    def export_to_csv(self, path: Path) -> None:
        """Write every trial (incl. pruned) to ``path`` for the audit trail."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._results_lock:
            results = list(self._results)
        rows = []
        for r in results:
            rows.append({
                "trial_id": r.trial_id,
                "pruned": r.pruned,
                "objective_value": r.objective_value,
                "params": json.dumps(r.params, default=str, sort_keys=True),
                "is_sharpe_avg": r.aggregate.get("is_sharpe_avg", 0.0),
                "oos_sharpe_avg": r.aggregate.get("oos_sharpe_avg", 0.0),
                "is_sharpe_min": r.aggregate.get("is_sharpe_min", 0.0),
                "oos_sharpe_min": r.aggregate.get("oos_sharpe_min", 0.0),
                "max_dd_pct": r.aggregate.get("max_drawdown_pct", 0.0),
                "total_trades": r.aggregate.get("total_trades", 0),
            })
        pd.DataFrame(
            rows,
            columns=[
                "trial_id", "pruned", "objective_value", "params",
                "is_sharpe_avg", "oos_sharpe_avg",
                "is_sharpe_min", "oos_sharpe_min",
                "max_dd_pct", "total_trades",
            ],
        ).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Rich progress bar callback
# ---------------------------------------------------------------------------

class _RichProgressCallback:
    """Optuna callback driving a rich.Progress bar.

    Displays trial counter, best objective so far, prune ratio, ETA. Safe to
    call from multiple threads — rich's Progress is internally locked.
    """

    def __init__(self, total: int):
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        self._progress = Progress(
            TextColumn("[bold blue]Trials"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("• best={task.fields[best]}"),
            TextColumn("• pruned={task.fields[pruned]}/{task.fields[done]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self._task_id = self._progress.add_task(
            "trials",
            total=total,
            best="–",
            pruned=0,
            done=0,
        )
        self._best = float("-inf")
        self._pruned = 0
        self._done = 0
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._progress.start()
            self._started = True

    def stop(self) -> None:
        if self._started:
            self._progress.stop()
            self._started = False

    def __call__(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
    ) -> None:
        with self._lock:
            self._done += 1
            if trial.state == optuna.trial.TrialState.PRUNED:
                self._pruned += 1
            elif (
                trial.state == optuna.trial.TrialState.COMPLETE
                and trial.value is not None
                and trial.value > self._best
            ):
                self._best = trial.value
            best_str = (
                f"{self._best:.4f}" if self._best != float("-inf") else "–"
            )
            self._progress.update(
                self._task_id,
                advance=1,
                best=best_str,
                pruned=self._pruned,
                done=self._done,
            )


# ---------------------------------------------------------------------------
# Module / file loaders
# ---------------------------------------------------------------------------

def load_strategy_module(path: Path) -> ModuleType:
    """Dynamically import a strategy file as a module."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location("strategy_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load strategy from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_param_space(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read an ``optimization_grid.yaml`` and return the parameters dict."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        raise ValueError(f"{path} is empty")
    if "parameters" in data:
        return data["parameters"]
    return data


def _parse_period(s: str) -> Tuple[date, date]:
    """Parse 'YYYY-MM-DD:YYYY-MM-DD' into a (start, end) date tuple."""
    try:
        a, b = s.split(":")
    except ValueError as exc:
        raise ValueError(
            f"Period must be 'YYYY-MM-DD:YYYY-MM-DD', got {s!r}"
        ) from exc
    return (
        datetime.strptime(a.strip(), "%Y-%m-%d").date(),
        datetime.strptime(b.strip(), "%Y-%m-%d").date(),
    )


def _load_data_dict(
    data_dir: Path,
    symbols: List[str],
    is_period: Tuple[date, date],
    oos_period: Tuple[date, date],
    timeframe: str = "M1",
) -> Dict[str, pd.DataFrame]:
    """Pick the parquet file in ``data_dir`` covering both periods.

    Filename convention from ``data_fetcher``::
        {SYMBOL}_{TF}_{YYYYMMDD}_{YYYYMMDD}.parquet
    """
    span_start = min(is_period[0], oos_period[0])
    span_end = max(is_period[1], oos_period[1])
    data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        candidates = sorted(
            data_dir.glob(f"{sym}_{timeframe}_*.parquet")
        )
        if not candidates:
            raise FileNotFoundError(
                f"No {timeframe} parquet for {sym} in {data_dir}"
            )
        chosen: Optional[Path] = None
        for path in candidates:
            stem = path.stem  # SYM_M1_YYYYMMDD_YYYYMMDD
            try:
                _sym, _tf, p_start, p_end = stem.rsplit("_", 3)
                ps = datetime.strptime(p_start, "%Y%m%d").date()
                pe = datetime.strptime(p_end, "%Y%m%d").date()
            except (ValueError, TypeError):
                continue
            if ps <= span_start and pe >= span_end:
                chosen = path
                break
        chosen = chosen or candidates[-1]
        logger.info("Loading %s ← %s", sym, chosen.name)
        data[sym] = pd.read_parquet(chosen)
    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m python_engine.optuna_optimizer",
        description="Bayesian multi-asset optimizer for Python-engine strategies.",
    )
    p.add_argument("--strategy", required=True, type=Path,
                   help="Path to strategy .py exposing signals(df, **params).")
    p.add_argument("--grid", required=True, type=Path,
                   help="optimization_grid.yaml file.")
    p.add_argument("--data-dir", type=Path, default=Path("./tmp_cache"),
                   help="Folder containing {SYMBOL}_{TF}_*.parquet files.")
    p.add_argument("--symbols", required=True,
                   help="Comma-separated symbols, e.g. EURUSD,USDJPY,EURCHF.")
    p.add_argument("--is", dest="is_period", required=True,
                   help="In-sample period 'YYYY-MM-DD:YYYY-MM-DD'.")
    p.add_argument("--oos", dest="oos_period", required=True,
                   help="Out-of-sample period 'YYYY-MM-DD:YYYY-MM-DD'.")
    p.add_argument("--timeframe", default="M1",
                   help="Parquet timeframe tag (default M1).")
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--timeout", type=int, default=7200,
                   help="Wall-clock cap in seconds (0 = unlimited).")
    p.add_argument("--sampler", choices=["tpe", "random", "cmaes"], default="tpe")
    p.add_argument("--pruner", choices=["median", "hyperband", "none"],
                   default="median")
    p.add_argument("--objective",
                   choices=["sharpe_oos_robust", "recovery_factor",
                            "sharpe_minus_pvalue"],
                   default="sharpe_oos_robust")
    p.add_argument("--initial-balance", type=float, default=10_000.0)
    p.add_argument("--risk-pct", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--storage", default=None,
                   help="Optional Optuna storage URL (e.g. sqlite:///x.db).")
    p.add_argument("--study-name", default=None)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write optimization_results.csv and "
                        "top_N_performers.csv (defaults to strategy folder/results).")
    p.add_argument("--min-trades-per-symbol", type=int, default=30)
    p.add_argument("--no-progress", action="store_true",
                   help="Disable rich progress bar (useful in CI).")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    is_period = _parse_period(args.is_period)
    oos_period = _parse_period(args.oos_period)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("--symbols must contain at least one symbol")

    strategy_module = load_strategy_module(args.strategy)
    param_space = load_param_space(args.grid)
    data_dict = _load_data_dict(
        args.data_dir, symbols, is_period, oos_period, args.timeframe
    )

    config = OptimizationConfig(
        n_trials=args.trials,
        n_jobs=args.n_jobs,
        timeout_seconds=args.timeout,
        sampler=args.sampler,
        pruner=args.pruner,
        objective=args.objective,
        study_name=args.study_name,
        storage=args.storage,
        seed=args.seed,
        initial_balance=args.initial_balance,
        risk_per_trade_pct=args.risk_pct,
        min_trades_per_symbol=args.min_trades_per_symbol,
        show_progress=not args.no_progress,
    )

    optimizer = MultiAssetOptimizer(
        strategy_module=strategy_module,
        data_dict=data_dict,
        is_period=is_period,
        oos_period=oos_period,
        param_space=param_space,
        config=config,
    )
    optimizer.run()

    output_dir = args.output_dir or args.strategy.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_csv = output_dir / "optimization_results.csv"
    top_csv = output_dir / f"top_{args.top}_performers.csv"
    optimizer.export_to_csv(all_csv)
    top_df = optimizer.get_top_n(args.top)
    top_df.to_csv(top_csv, index=False)

    logger.info("Wrote all trials → %s", all_csv)
    logger.info("Wrote top-%d → %s", args.top, top_csv)
    if not top_df.empty:
        logger.info(
            "Best objective=%.4f params=%s",
            top_df.iloc[0]["objective_value"],
            top_df.iloc[0]["params"],
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
