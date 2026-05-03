"""Tests for python_engine.optuna_optimizer.

Strategy:
1. Pin the ranking-function math with hand-computed expected values.
2. Validate ``_suggest_one`` for every supported parameter type.
3. End-to-end: run a 12-trial study with a mock strategy on a synthetic 2-day
   M1 panel and verify sorted top-N + complete CSV export.
4. Smoke-test the median pruner path (cannot guarantee a prune fires on a tiny
   mock, but the run must complete and the CSV must reflect any prunes that did).
5. Configuration guards: empty data dict, missing ``signals``, empty grid.
"""
from __future__ import annotations

import sys
import textwrap
from datetime import date
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

optuna = pytest.importorskip("optuna")  # noqa: E402

from python_engine.optuna_optimizer import (  # noqa: E402
    MultiAssetOptimizer,
    OptimizationConfig,
    _parse_period,
    _suggest_one,
    load_param_space,
    ranking_function,
)
from python_engine.vectorized_backtest import SymbolSpec  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _fx_spec() -> SymbolSpec:
    """EURUSD-like 5-digit FX, zero swap → keeps the test math clean."""
    return SymbolSpec(
        symbol="MOCK",
        point_size=0.00001,
        usd_per_price_unit=100_000.0,
        commission_per_lot=7.0,
        swap_long=0.0,
        swap_short=0.0,
        triple_swap_weekday=2,
    )


def _make_data(start: str, n_bars: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n_bars, freq="1min")
    # Slow upward drift so longs make money on average → trades produce
    # non-degenerate equity curves the engine can actually score.
    opens = 1.10000 + (np.arange(n_bars) * 1e-5)
    highs = opens + 0.00050
    lows = opens - 0.00010
    closes = opens + 0.00020
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows,
         "close": closes, "spread": np.zeros(n_bars)},
        index=idx,
    )


def _make_mock_strategy() -> ModuleType:
    """Build an in-memory strategy module accepting a ``magic`` int param."""
    code = textwrap.dedent("""
        import pandas as pd

        def signals(df, magic: int = 1, **kwargs):
            n = len(df)
            sl = pd.Series(0.0010, index=df.index)
            tp = pd.Series(0.0010, index=df.index)
            sig_long = pd.Series(False, index=df.index)
            sig_short = pd.Series(False, index=df.index)
            step = max(1, int(magic))
            for i in range(0, n - 2, step):
                sig_long.iloc[i] = True
            return {
                "signal_long": sig_long,
                "signal_short": sig_short,
                "sl_distance": sl,
                "tp_distance": tp,
            }
    """).strip() + "\n"
    mod = ModuleType("mock_strategy")
    exec(compile(code, "<mock_strategy>", "exec"), mod.__dict__)
    return mod


# ---------------------------------------------------------------------------
# 1. ranking_function — sharpe_oos_robust
# ---------------------------------------------------------------------------

def test_ranking_sharpe_oos_robust_basic():
    """IS_avg=2.0, OOS_avg=1.5, gate=1 → 1.5 * (1 - 0.5/2.0) * 1 = 1.125."""
    is_m = {
        "EURUSD": {"sharpe": 2.0, "n_trades": 100},
        "USDJPY": {"sharpe": 2.0, "n_trades": 100},
    }
    oos_m = {
        "EURUSD": {"sharpe": 1.5, "n_trades": 50},
        "USDJPY": {"sharpe": 1.5, "n_trades": 50},
    }
    assert ranking_function(
        is_m, oos_m, mode="sharpe_oos_robust", min_trades_per_symbol=30
    ) == pytest.approx(1.125, abs=1e-9)


def test_ranking_sharpe_oos_robust_overfit_caps_penalty_at_zero():
    """If OOS Sharpe ≤ 0 while IS is strongly positive, penalty floors at 0."""
    is_m = {"EURUSD": {"sharpe": 2.0, "n_trades": 100}}
    oos_m = {"EURUSD": {"sharpe": -1.0, "n_trades": 50}}
    # Hand: degradation = (2 - (-1))/2 = 1.5, penalty = max(0, 1 - 1.5) = 0.
    score = ranking_function(
        is_m, oos_m, mode="sharpe_oos_robust", min_trades_per_symbol=30
    )
    assert score == pytest.approx(0.0, abs=1e-9)


def test_ranking_sharpe_oos_robust_min_trades_gate_blocks():
    is_m = {"EURUSD": {"sharpe": 3.0, "n_trades": 100}}
    oos_m = {"EURUSD": {"sharpe": 3.0, "n_trades": 5}}
    score = ranking_function(
        is_m, oos_m, mode="sharpe_oos_robust", min_trades_per_symbol=30
    )
    assert score == 0.0


def test_ranking_sharpe_oos_robust_negative_is_blocks_score():
    """When IS Sharpe ≤ 0, the overfit penalty math is undefined; treat as
    worthless and refuse to score positively. Both negative → 0."""
    is_m = {"EURUSD": {"sharpe": -1.0, "n_trades": 100}}
    oos_m = {"EURUSD": {"sharpe": -2.0, "n_trades": 50}}
    score = ranking_function(
        is_m, oos_m, mode="sharpe_oos_robust", min_trades_per_symbol=30
    )
    assert score == 0.0


# ---------------------------------------------------------------------------
# 2. ranking_function — recovery_factor / sharpe_minus_pvalue
# ---------------------------------------------------------------------------

def test_ranking_recovery_factor():
    """RF = total_return_pct / |max_dd_pct|. Hand: (10/5 + 20/4)/2 = 3.5."""
    oos_m = {
        "A": {"max_drawdown_pct": -5.0, "total_return_pct": 10.0,
              "sharpe": 1.0, "n_trades": 100},
        "B": {"max_drawdown_pct": -4.0, "total_return_pct": 20.0,
              "sharpe": 1.0, "n_trades": 100},
    }
    assert ranking_function(
        {}, oos_m, mode="recovery_factor"
    ) == pytest.approx(3.5, abs=1e-9)


def test_ranking_sharpe_minus_pvalue():
    """oos_avg=1.0, half pass gate → 1.0 - 5*(1 - 0.5) = -1.5."""
    is_m = {
        "A": {"sharpe": 1.0, "n_trades": 100},
        "B": {"sharpe": 1.0, "n_trades": 5},
    }
    oos_m = {
        "A": {"sharpe": 1.0, "n_trades": 100},
        "B": {"sharpe": 1.0, "n_trades": 5},
    }
    assert ranking_function(
        is_m, oos_m, mode="sharpe_minus_pvalue", min_trades_per_symbol=30
    ) == pytest.approx(-1.5, abs=1e-9)


def test_ranking_unknown_mode_raises():
    with pytest.raises(ValueError):
        ranking_function({}, {"A": {"sharpe": 1.0}}, mode="bogus")


# ---------------------------------------------------------------------------
# 3. _suggest_one
# ---------------------------------------------------------------------------

def test_suggest_one_int_float_categorical():
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=42),
    )
    captured: dict = {}

    def objective(trial):
        captured["i"] = _suggest_one(
            trial, "i", {"type": "int", "low": 5, "high": 10}
        )
        captured["f"] = _suggest_one(
            trial, "f", {"type": "float", "low": 0.1, "high": 0.5}
        )
        captured["fs"] = _suggest_one(
            trial, "fs",
            {"type": "float", "low": 1.0, "high": 2.0, "step": 0.1},
        )
        captured["c"] = _suggest_one(
            trial, "c",
            {"type": "categorical", "choices": ["a", "b", "c"]},
        )
        return 0.0

    study.optimize(objective, n_trials=1)
    assert isinstance(captured["i"], int) and 5 <= captured["i"] <= 10
    assert isinstance(captured["f"], float) and 0.1 <= captured["f"] <= 0.5
    assert isinstance(captured["fs"], float) and 1.0 <= captured["fs"] <= 2.0
    assert captured["c"] in {"a", "b", "c"}


def test_suggest_one_unknown_type_raises():
    study = optuna.create_study()

    def objective(trial):
        _suggest_one(trial, "x", {"type": "bogus", "low": 0, "high": 1})
        return 0.0

    with pytest.raises(ValueError):
        study.optimize(objective, n_trials=1, catch=())


# ---------------------------------------------------------------------------
# 4. End-to-end: 12 trials, sorted top-N, full CSV
# ---------------------------------------------------------------------------

def test_optimizer_runs_and_produces_top_n(tmp_path):
    n_bars = 60 * 24 * 2   # 2 days of M1
    df = _make_data("2024-01-01", n_bars)
    data = {"MOCK1": df, "MOCK2": df.copy()}

    config = OptimizationConfig(
        n_trials=12, n_jobs=1, sampler="random", pruner="none",
        timeout_seconds=120, seed=7,
        min_trades_per_symbol=1,   # synthetic data → relax gate
        show_progress=False,
    )
    optimizer = MultiAssetOptimizer(
        strategy_module=_make_mock_strategy(),
        data_dict=data,
        is_period=(date(2024, 1, 1), date(2024, 1, 1)),
        oos_period=(date(2024, 1, 2), date(2024, 1, 2)),
        param_space={
            "magic": {"type": "int", "low": 5, "high": 60, "step": 5},
        },
        config=config,
        symbol_specs={"MOCK1": _fx_spec(), "MOCK2": _fx_spec()},
    )
    results = optimizer.run()
    assert len(results) == 12

    # Sorted descending by objective_value.
    objs = [r.objective_value for r in results]
    assert objs == sorted(objs, reverse=True)

    top = optimizer.get_top_n(5)
    assert list(top.columns) == [
        "rank", "trial_id", "params",
        "is_sharpe_avg", "oos_sharpe_avg",
        "is_sharpe_min", "oos_sharpe_min",
        "max_dd_pct", "total_trades", "objective_value",
    ]
    assert len(top) <= 5
    if len(top) >= 2:
        assert top.iloc[0]["objective_value"] >= top.iloc[1]["objective_value"]
        assert int(top.iloc[0]["rank"]) == 1

    out_csv = tmp_path / "trials.csv"
    optimizer.export_to_csv(out_csv)
    df_out = pd.read_csv(out_csv)
    assert len(df_out) == 12
    for col in ("trial_id", "pruned", "objective_value", "params",
                "is_sharpe_avg", "oos_sharpe_avg",
                "is_sharpe_min", "oos_sharpe_min",
                "max_dd_pct", "total_trades"):
        assert col in df_out.columns


# ---------------------------------------------------------------------------
# 5. Median pruner — at minimum, the run completes and the CSV agrees
# ---------------------------------------------------------------------------

def test_median_pruner_run_consistent(tmp_path):
    """Cannot guarantee a prune fires on a small mock, but the run must
    complete and any pruned trials must be recorded in the CSV."""
    n_bars = 60 * 24 * 2
    df = _make_data("2024-01-01", n_bars)
    data = {f"MOCK{i}": df.copy() for i in range(4)}
    specs = {f"MOCK{i}": _fx_spec() for i in range(4)}

    config = OptimizationConfig(
        n_trials=15, n_jobs=1, sampler="tpe", pruner="median",
        timeout_seconds=120, seed=11,
        min_trades_per_symbol=1,
        pruning_warmup_steps=2,
        show_progress=False,
    )
    optimizer = MultiAssetOptimizer(
        strategy_module=_make_mock_strategy(),
        data_dict=data,
        is_period=(date(2024, 1, 1), date(2024, 1, 1)),
        oos_period=(date(2024, 1, 2), date(2024, 1, 2)),
        param_space={
            "magic": {"type": "int", "low": 1, "high": 200},
        },
        config=config,
        symbol_specs=specs,
    )
    results = optimizer.run()
    assert len(results) == 15
    # Every result is either a completed trial with finite objective, or a
    # pruned trial recorded with whatever interim score it had.
    for r in results:
        assert isinstance(r.objective_value, float)

    out_csv = tmp_path / "all.csv"
    optimizer.export_to_csv(out_csv)
    df_out = pd.read_csv(out_csv)
    pruned_count = sum(1 for r in results if r.pruned)
    assert int(df_out["pruned"].sum()) == pruned_count


# ---------------------------------------------------------------------------
# 6. Helpers: load_param_space + _parse_period
# ---------------------------------------------------------------------------

def test_load_param_space_handles_both_layouts(tmp_path):
    flat = tmp_path / "flat.yaml"
    flat.write_text("rsi_period:\n  type: int\n  low: 5\n  high: 15\n")
    nested = tmp_path / "nested.yaml"
    nested.write_text(
        "parameters:\n  rsi_period:\n    type: int\n    low: 5\n    high: 15\n"
    )
    expected = {"rsi_period": {"type": "int", "low": 5, "high": 15}}
    assert load_param_space(flat) == expected
    assert load_param_space(nested) == expected


def test_parse_period_round_trip():
    a, b = _parse_period("2024-01-01:2024-12-31")
    assert a == date(2024, 1, 1)
    assert b == date(2024, 12, 31)
    with pytest.raises(ValueError):
        _parse_period("not-a-period")


# ---------------------------------------------------------------------------
# 7. Configuration guards
# ---------------------------------------------------------------------------

def test_optimizer_rejects_empty_data():
    with pytest.raises(ValueError):
        MultiAssetOptimizer(
            strategy_module=_make_mock_strategy(),
            data_dict={},
            is_period=(date(2024, 1, 1), date(2024, 1, 1)),
            oos_period=(date(2024, 1, 2), date(2024, 1, 2)),
            param_space={"magic": {"type": "int", "low": 1, "high": 10}},
            config=OptimizationConfig(show_progress=False),
        )


def test_optimizer_rejects_strategy_without_signals():
    bad = ModuleType("bad")
    with pytest.raises(AttributeError):
        MultiAssetOptimizer(
            strategy_module=bad,
            data_dict={"X": _make_data("2024-01-01", 10)},
            is_period=(date(2024, 1, 1), date(2024, 1, 1)),
            oos_period=(date(2024, 1, 2), date(2024, 1, 2)),
            param_space={"magic": {"type": "int", "low": 1, "high": 10}},
            config=OptimizationConfig(show_progress=False),
            symbol_specs={"X": _fx_spec()},
        )


def test_optimizer_rejects_empty_param_space():
    with pytest.raises(ValueError):
        MultiAssetOptimizer(
            strategy_module=_make_mock_strategy(),
            data_dict={"X": _make_data("2024-01-01", 10)},
            is_period=(date(2024, 1, 1), date(2024, 1, 1)),
            oos_period=(date(2024, 1, 2), date(2024, 1, 2)),
            param_space={},
            config=OptimizationConfig(show_progress=False),
            symbol_specs={"X": _fx_spec()},
        )
