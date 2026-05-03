"""Tests for python_engine.vectorized_backtest.

The engine has too much surface area to validate purely from a "compare to
external library" approach (it doesn't exist for our cost model). Instead each
test below pins one concrete invariant — the result is computed by hand in the
docstring or alongside the assertion so future-me can audit the maths quickly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from python_engine.vectorized_backtest import (  # noqa: E402
    SymbolSpec,
    TradeFill,
    VectorizedBacktest,
)


# ---------------------------------------------------------------------------
# Hand-built fixtures
# ---------------------------------------------------------------------------

def _fx_spec(swap_long: float = -3.5, swap_short: float = 0.8) -> SymbolSpec:
    """A vanilla EURUSD-like 5-digit FX symbol. Avoids YAML coupling."""
    return SymbolSpec(
        symbol="TESTFX",
        point_size=0.00001,         # 5-digit broker
        usd_per_price_unit=100_000.0,  # 1 lot * +1.0 price → $100,000
        commission_per_lot=7.0,
        swap_long=swap_long,
        swap_short=swap_short,
        triple_swap_weekday=2,      # Wednesday
    )


def _make_bars(
    start: str,
    n: int,
    *,
    freq: str = "1min",
    opens=None,
    highs=None,
    lows=None,
    closes=None,
    spreads=None,
) -> pd.DataFrame:
    """Build an OHLC+spread DataFrame with explicit per-bar values."""
    idx = pd.date_range(start, periods=n, freq=freq)
    if opens is None:
        opens = np.full(n, 1.10000)
    opens = np.asarray(opens, dtype=float)
    highs = np.asarray(highs if highs is not None else opens, dtype=float)
    lows = np.asarray(lows if lows is not None else opens, dtype=float)
    closes = np.asarray(closes if closes is not None else opens, dtype=float)
    spreads = np.asarray(spreads if spreads is not None else np.zeros(n), dtype=float)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "spread": spreads},
        index=idx,
    )


def _signals(
    df: pd.DataFrame,
    *,
    long_idx=(),
    short_idx=(),
    sl: float = 0.0010,
    tp: float = 0.0010,
    session=None,
):
    n = len(df)
    sl_arr = pd.Series(sl, index=df.index)
    tp_arr = pd.Series(tp, index=df.index)
    sig_long = pd.Series(False, index=df.index)
    sig_short = pd.Series(False, index=df.index)
    for i in long_idx:
        sig_long.iloc[i] = True
    for i in short_idx:
        sig_short.iloc[i] = True
    if session is None:
        sess = pd.Series(True, index=df.index)
    else:
        sess = pd.Series(session, index=df.index)
    return sig_long, sig_short, sl_arr, tp_arr, sess


# ---------------------------------------------------------------------------
# 1. Trivial directional check (time exit, zero spread, zero swap)
# ---------------------------------------------------------------------------

def test_buy_and_hold_directional_pnl():
    """Long trade times out after 8 bars on a clean +1 pip per minute ramp.

    Hand calculation: entry at bar 1 mid_open = 1.10010, time-exit at close
    of bar 8 = 1.10090 (since opens[i] = 1.10000 + i*0.00010 and the 8th bar
    after entry is bar 8, at which bars_held reaches 8).
    Directional PnL = (1.10090 - 1.10010) * 0.1 * 100_000 = $8.00.
    Commission round-turn = 7.0 * 0.1 = $0.70.
    Spread = 0, swap = 0 (no midnight crossed in 10 minutes).
    Total = $7.30.
    """
    n = 10
    opens = np.array([1.10000 + i * 0.00010 for i in range(n)])
    closes = np.array([1.10000 + (i + 1) * 0.00010 for i in range(n)])
    highs = np.maximum(opens, closes) + 0.00002
    lows = np.minimum(opens, closes) - 0.00002
    df = _make_bars("2026-05-04 09:00", n, opens=opens, highs=highs,
                    lows=lows, closes=closes)

    sig_long, sig_short, sl, tp, sess = _signals(
        df, long_idx=[0], sl=0.0050, tp=0.0050,
    )

    bt = VectorizedBacktest(
        data=df, symbol_spec=_fx_spec(), fixed_lots=0.1,
    )
    res = bt.run(sig_long, sig_short, sl, tp, sess, max_holding_bars=8)

    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.side == "long"
    assert t.exit_reason == "time_exit"
    assert t.holding_bars == 8
    assert t.open_price == pytest.approx(1.10010)
    assert t.close_price == pytest.approx(1.10090)
    assert t.profit_directional == pytest.approx(8.00, abs=1e-6)
    assert t.profit_swap == 0.0
    assert t.profit_spread_cost == 0.0
    assert t.profit_commission == pytest.approx(-0.70, abs=1e-6)
    assert t.profit_total == pytest.approx(7.30, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. SL hit, with explicit spread cost
# ---------------------------------------------------------------------------

def test_long_trade_sl_hit_with_spread():
    """Long stops out 50 pips below entry; spread = 10 points both sides.

    Hand calculation: entry mid 1.10000, SL distance 0.0005 → SL price
    1.09950. At bar 3 the low pierces 1.09940, so SL fills at exactly
    1.09950 (we exit at the level, never at the deeper spike — that's the
    standard "stop fills at SL" assumption).

    profit_directional = (1.09950 - 1.10000) * 0.1 * 100_000 = -$5.00
    profit_spread_cost = -((10 + 10)/2 * 0.00001) * 0.1 * 100_000 = -$1.00
        (full round-turn spread = 10 points = $1 on 0.1 lot)
    commission = -$0.70.  swap = 0 (intraday, no midnight crossed).
    total = -5.00 - 1.00 - 0.70 = -$6.70.
    """
    n = 6
    opens = np.full(n, 1.10000)
    highs = opens + 0.00002
    lows = opens - 0.00002
    lows[3] = 1.09940       # pierces below SL=1.09950
    closes = np.full(n, 1.09995)
    df = _make_bars(
        "2026-05-04 09:00", n,
        opens=opens, highs=highs, lows=lows, closes=closes,
        spreads=np.full(n, 10.0),
    )
    sig_long, sig_short, sl, tp, sess = _signals(
        df, long_idx=[0], sl=0.0005, tp=0.0050,
    )
    bt = VectorizedBacktest(
        data=df, symbol_spec=_fx_spec(), fixed_lots=0.1,
    )
    res = bt.run(sig_long, sig_short, sl, tp, sess, max_holding_bars=20)

    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.exit_reason == "sl"
    assert t.close_price == pytest.approx(1.09950)
    assert t.profit_directional == pytest.approx(-5.00, abs=1e-6)
    assert t.profit_spread_cost == pytest.approx(-1.00, abs=1e-6)
    assert t.profit_commission == pytest.approx(-0.70, abs=1e-6)
    assert t.profit_total == pytest.approx(-6.70, abs=1e-6)
    # actual fill: ask side at entry, bid side at exit (mid - 5 pts)
    assert t.open_price_with_spread == pytest.approx(1.10000 + 5e-5)
    assert t.close_price_with_spread == pytest.approx(1.09950 - 5e-5)


# ---------------------------------------------------------------------------
# 3. TP hit
# ---------------------------------------------------------------------------

def test_long_trade_tp_hit():
    """Long take-profit fills at exactly the TP level, regardless of overshoot."""
    n = 6
    opens = np.full(n, 1.10000)
    highs = opens + 0.00002
    lows = opens - 0.00002
    highs[2] = 1.10115       # overshoots TP=1.10100
    closes = np.full(n, 1.10005)
    df = _make_bars(
        "2026-05-04 09:00", n,
        opens=opens, highs=highs, lows=lows, closes=closes,
    )
    sig_long, sig_short, sl, tp, sess = _signals(
        df, long_idx=[0], sl=0.0050, tp=0.0010,
    )
    bt = VectorizedBacktest(
        data=df, symbol_spec=_fx_spec(), fixed_lots=0.1,
    )
    res = bt.run(sig_long, sig_short, sl, tp, sess, max_holding_bars=20)

    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.exit_reason == "tp"
    assert t.close_price == pytest.approx(1.10100)
    # +10 pips * 0.1 lot * 100k = $10
    assert t.profit_directional == pytest.approx(10.00, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Multi-day swap accumulation incl. Wednesday triple charge
# ---------------------------------------------------------------------------

def test_multi_day_swap_with_wednesday_triple():
    """Hold a long FX position from Tue 10:00 to Fri 10:00 (H1 bars).

    Midnights crossed:
      - Wed 00:00 (weekday=2 Wednesday) → 3× swap
      - Thu 00:00 (weekday=3) → 1× swap
      - Fri 00:00 (weekday=4) → 1× swap
    Total = 5 daily-swap units.

    swap_long = -3.5/day/lot. With 0.1 lot:
        profit_swap = -3.5 * 5 * 0.1 = -$1.75
    """
    # 76 H1 bars starting Tue 2026-04-28 09:00 → Fri 2026-05-01 12:00
    n = 76
    start = pd.Timestamp("2026-04-28 09:00")     # Tuesday
    times = pd.date_range(start, periods=n, freq="1h")
    # Flat prices so SL/TP never fire even with wide stops.
    opens = np.full(n, 1.10000)
    highs = opens + 0.00010
    lows = opens - 0.00010
    closes = np.full(n, 1.10000)
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "spread": np.zeros(n)},
        index=times,
    )

    sig_long, sig_short, sl, tp, sess = _signals(
        df, long_idx=[0], sl=0.10, tp=0.10,   # 10000 pips — never hit
    )
    bt = VectorizedBacktest(
        data=df, symbol_spec=_fx_spec(), fixed_lots=0.1,
    )
    # Entry at bar 1 (Tue 10:00); we want time-exit at bar 73 (Fri 10:00),
    # which means bars_held = 73 - 1 + 1 = 73.
    res = bt.run(sig_long, sig_short, sl, tp, sess, max_holding_bars=73)

    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.exit_reason == "time_exit"
    assert t.open_time == pd.Timestamp("2026-04-28 10:00")
    assert t.close_time == pd.Timestamp("2026-05-01 10:00")
    # 5 daily-swap units with Wednesday's tripled
    assert t.profit_swap == pytest.approx(-3.5 * 5 * 0.1, abs=1e-6)


def test_swap_zero_when_no_midnight_crossed():
    """Holding for less than a day with no 00:00 boundary → zero swap."""
    spec = _fx_spec()
    bt = VectorizedBacktest(
        data=_make_bars("2026-05-04 09:00", 5),
        symbol_spec=spec, fixed_lots=0.1,
    )
    swap = bt._compute_swap(
        pd.Timestamp("2026-05-04 09:00"),
        pd.Timestamp("2026-05-04 22:00"),
        side="long", lots=0.1,
    )
    assert swap == 0.0


def test_swap_short_uses_short_rate():
    """Short positions accrue swap_short, not swap_long."""
    spec = _fx_spec(swap_long=-3.5, swap_short=0.8)
    bt = VectorizedBacktest(
        data=_make_bars("2026-05-04 09:00", 5),
        symbol_spec=spec, fixed_lots=1.0,
    )
    # Tue 10:00 → Thu 10:00. Midnights: Wed 00:00 (triple), Thu 00:00 (1x)
    # Total = 4 daily-swap units.
    swap = bt._compute_swap(
        pd.Timestamp("2026-04-28 10:00"),
        pd.Timestamp("2026-04-30 10:00"),
        side="short", lots=1.0,
    )
    assert swap == pytest.approx(0.8 * 4 * 1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 5. Time exit
# ---------------------------------------------------------------------------

def test_time_exit_at_max_holding_bars():
    """max_holding_bars = 4 forces close at close of bar (entry_idx + 3)."""
    n = 10
    df = _make_bars("2026-05-04 09:00", n)
    sig_long, sig_short, sl, tp, sess = _signals(
        df, long_idx=[0], sl=0.0050, tp=0.0050,
    )
    bt = VectorizedBacktest(
        data=df, symbol_spec=_fx_spec(), fixed_lots=0.1,
    )
    res = bt.run(sig_long, sig_short, sl, tp, sess, max_holding_bars=4)

    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.exit_reason == "time_exit"
    assert t.holding_bars == 4
    # Entry bar = index 1, exit bar = index 4 (4 - 1 + 1 = 4 bars span).
    assert t.open_time == df.index[1]
    assert t.close_time == df.index[4]


# ---------------------------------------------------------------------------
# 6. Session close
# ---------------------------------------------------------------------------

def test_session_close_exits_at_last_in_session_bar():
    """Session flips False at bar 6 → position must close at close of bar 5."""
    n = 10
    df = _make_bars("2026-05-04 09:00", n)
    session = [True] * 6 + [False] * 4
    sig_long, sig_short, sl, tp, sess = _signals(
        df, long_idx=[0], sl=0.0050, tp=0.0050, session=session,
    )
    bt = VectorizedBacktest(
        data=df, symbol_spec=_fx_spec(), fixed_lots=0.1,
    )
    res = bt.run(sig_long, sig_short, sl, tp, sess, max_holding_bars=20)

    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.exit_reason == "session_close"
    assert t.close_time == df.index[5]


# ---------------------------------------------------------------------------
# 7. Pessimistic SL when SL and TP both touched in the same bar
# ---------------------------------------------------------------------------

def test_pessimistic_sl_when_both_touched_same_bar():
    n = 6
    opens = np.full(n, 1.10000)
    highs = opens + 0.00002
    lows = opens - 0.00002
    # Bar 2 hits both: high above TP, low below SL.
    highs[2] = 1.10120        # > TP at 1.10100
    lows[2] = 1.09940         # < SL at 1.09950
    closes = np.full(n, 1.10000)
    df = _make_bars("2026-05-04 09:00", n,
                    opens=opens, highs=highs, lows=lows, closes=closes)
    sig_long, sig_short, sl, tp, sess = _signals(
        df, long_idx=[0], sl=0.0005, tp=0.0010,
    )
    bt = VectorizedBacktest(
        data=df, symbol_spec=_fx_spec(), fixed_lots=0.1,
    )
    res = bt.run(sig_long, sig_short, sl, tp, sess, max_holding_bars=20)
    assert len(res.trades) == 1
    assert res.trades[0].exit_reason == "sl"


# ---------------------------------------------------------------------------
# 8. Reverse signal closes position at bar close
# ---------------------------------------------------------------------------

def test_reverse_signal_closes_long():
    n = 8
    df = _make_bars("2026-05-04 09:00", n)
    sig_long, sig_short, sl, tp, sess = _signals(
        df, long_idx=[0], short_idx=[3], sl=0.0050, tp=0.0050,
    )
    bt = VectorizedBacktest(
        data=df, symbol_spec=_fx_spec(), fixed_lots=0.1,
    )
    res = bt.run(sig_long, sig_short, sl, tp, sess, max_holding_bars=20)
    # First trade: long opens at bar 1, reverses out at close of bar 3.
    assert len(res.trades) >= 1
    first = res.trades[0]
    assert first.side == "long"
    assert first.exit_reason == "reverse_signal"
    assert first.close_time == df.index[3]


# ---------------------------------------------------------------------------
# 9. Short trade, basic directional accounting
# ---------------------------------------------------------------------------

def test_short_trade_tp_hit():
    """Short TP at -10 pips below entry; verify direction sign and exit price."""
    n = 6
    opens = np.full(n, 1.10000)
    highs = opens + 0.00002
    lows = opens - 0.00002
    lows[2] = 1.09885   # overshoots short TP at 1.09900
    closes = np.full(n, 1.09995)
    df = _make_bars("2026-05-04 09:00", n,
                    opens=opens, highs=highs, lows=lows, closes=closes)
    sig_long, sig_short, sl, tp, sess = _signals(
        df, short_idx=[0], sl=0.0050, tp=0.0010,
    )
    bt = VectorizedBacktest(
        data=df, symbol_spec=_fx_spec(), fixed_lots=0.1,
    )
    res = bt.run(sig_long, sig_short, sl, tp, sess, max_holding_bars=20)

    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.side == "short"
    assert t.exit_reason == "tp"
    assert t.close_price == pytest.approx(1.09900)
    # Short profit: (1.10000 - 1.09900) * 0.1 * 100_000 = $10
    assert t.profit_directional == pytest.approx(10.00, abs=1e-6)


# ---------------------------------------------------------------------------
# 10. PnL decomposition surface — intraday FX should be ≥90% directional
# ---------------------------------------------------------------------------

def test_pnl_decomposition_buckets_split_to_100pct():
    """Sanity: the four percentage buckets in the decomposition sum to 100."""
    n = 10
    opens = np.array([1.10000 + i * 0.00010 for i in range(n)])
    highs = opens + 0.00002
    lows = opens - 0.00002
    closes = np.array([1.10000 + (i + 1) * 0.00010 for i in range(n)])
    df = _make_bars("2026-05-04 09:00", n,
                    opens=opens, highs=highs, lows=lows, closes=closes,
                    spreads=np.full(n, 5.0))
    sig_long, sig_short, sl, tp, sess = _signals(
        df, long_idx=[0], sl=0.0050, tp=0.0050,
    )
    bt = VectorizedBacktest(
        data=df, symbol_spec=_fx_spec(), fixed_lots=0.1,
    )
    res = bt.run(sig_long, sig_short, sl, tp, sess, max_holding_bars=8)

    pct_sum = (
        res.pnl_decomposition["directional_pct"]
        + res.pnl_decomposition["swap_pct"]
        + res.pnl_decomposition["commission_pct"]
        + res.pnl_decomposition["spread_pct"]
    )
    assert pct_sum == pytest.approx(100.0, abs=1e-6)
    # Intraday-FX expectation from CLAUDE.md: directional dominates, swap ~0.
    assert res.pnl_decomposition["swap_pct"] == 0.0
    assert res.pnl_decomposition["directional_pct"] > 80.0


# ---------------------------------------------------------------------------
# 11. Empty result is safe (no signals at all)
# ---------------------------------------------------------------------------

def test_no_signals_produces_empty_result():
    df = _make_bars("2026-05-04 09:00", 5)
    sig_long, sig_short, sl, tp, sess = _signals(df, sl=0.001, tp=0.001)
    bt = VectorizedBacktest(
        data=df, symbol_spec=_fx_spec(), fixed_lots=0.1,
    )
    res = bt.run(sig_long, sig_short, sl, tp, sess, max_holding_bars=5)
    assert res.trades == []
    assert res.metrics["n_trades"] == 0
    assert res.equity.iloc[-1] == 10_000.0


# ---------------------------------------------------------------------------
# 12. Risk-based sizing matches hand calculation
# ---------------------------------------------------------------------------

def test_risk_based_lot_sizing():
    """0.5% of $10k = $50 risk per trade. With SL=50 pips and EURUSD-like
    pip-value $10/lot, lots = $50 / ($10 * 50 pips) = 0.10."""
    n = 6
    opens = np.full(n, 1.10000)
    highs = opens + 0.00002
    lows = opens - 0.00002
    lows[3] = 1.09940
    closes = np.full(n, 1.09990)
    df = _make_bars("2026-05-04 09:00", n,
                    opens=opens, highs=highs, lows=lows, closes=closes)
    sig_long, sig_short, sl, tp, sess = _signals(
        df, long_idx=[0], sl=0.0050, tp=0.0050,
    )
    bt = VectorizedBacktest(
        data=df, symbol_spec=_fx_spec(),
        initial_balance=10_000.0, risk_per_trade_pct=0.5,
    )
    res = bt.run(sig_long, sig_short, sl, tp, sess, max_holding_bars=20)
    assert res.trades[0].lots == pytest.approx(0.10, abs=1e-9)
