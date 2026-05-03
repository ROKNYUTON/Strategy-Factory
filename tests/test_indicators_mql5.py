"""Golden-value tests for python_engine.indicators_mql5.

Every assertion below is a value computed **by hand** from the MQL5 reference
formulas (see docstrings in indicators_mql5.py and the comments next to each
``assert``). If any of these tests starts failing, the indicator no longer
agrees with MQL5 and the regression must be investigated before any backtest
results from the Python research engine can be trusted.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from python_engine.indicators_mql5 import (
    atr,
    bollinger_bands,
    ema,
    lwma,
    macd,
    rsi,
    sma,
    smma,
    stochastic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ramp(start: float, n: int, step: float = 1.0) -> pd.Series:
    """Linear ramp; matches the data used in the by-hand calculations."""
    return pd.Series(np.arange(n, dtype=float) * step + start)


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------

def test_sma_basic_ramp():
    s = _ramp(10.0, 10)  # [10, 11, ..., 19]
    out = sma(s, period=5)
    # First 4 values undefined.
    assert out.iloc[:4].isna().all()
    # idx 4: mean(10..14) = 12.0
    assert out.iloc[4] == pytest.approx(12.0)
    # idx 5: mean(11..15) = 13.0
    assert out.iloc[5] == pytest.approx(13.0)
    # idx 9: mean(15..19) = 17.0
    assert out.iloc[9] == pytest.approx(17.0)


def test_sma_short_input_returns_all_nan():
    s = pd.Series([1.0, 2.0, 3.0])
    out = sma(s, period=5)
    assert out.isna().all()


def test_sma_rejects_nonpositive_period():
    with pytest.raises(ValueError):
        sma(pd.Series([1.0, 2.0]), period=0)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def test_ema_seed_and_recursion():
    """alpha = 2/6 = 1/3. Seed at idx 4 = SMA = 12.0.

    EMA[5] = (1/3)*15 + (2/3)*12 = 13.0
    EMA[6] = (1/3)*16 + (2/3)*13 = 14.0
    EMA[i] = i - (period-2) for the linear ramp. (Holds because the linear
    ramp is its own EMA, shifted by exactly 1.)"""
    s = _ramp(10.0, 10)
    out = ema(s, period=5)
    assert out.iloc[:4].isna().all()
    assert out.iloc[4] == pytest.approx(12.0)
    assert out.iloc[5] == pytest.approx(13.0)
    assert out.iloc[6] == pytest.approx(14.0)
    assert out.iloc[7] == pytest.approx(15.0)
    assert out.iloc[8] == pytest.approx(16.0)
    assert out.iloc[9] == pytest.approx(17.0)


def test_ema_constant_input():
    s = pd.Series([5.0] * 20)
    out = ema(s, period=4)
    # First 3 NaN; everything else == 5.0.
    assert out.iloc[:3].isna().all()
    np.testing.assert_allclose(out.iloc[3:].to_numpy(), 5.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# SMMA (Wilder)
# ---------------------------------------------------------------------------

def test_smma_matches_mql5_recursion():
    """Hand-calculated values for [10..19] period=5:

    seed at idx 4 = 12.0
    idx 5: (12.0*4 + 15)/5 = 12.6
    idx 6: (12.6*4 + 16)/5 = 13.28
    idx 7: (13.28*4 + 17)/5 = 14.024
    idx 8: (14.024*4 + 18)/5 = 14.8192
    idx 9: (14.8192*4 + 19)/5 = 15.65536
    """
    s = _ramp(10.0, 10)
    out = smma(s, period=5)
    assert out.iloc[:4].isna().all()
    assert out.iloc[4] == pytest.approx(12.0)
    assert out.iloc[5] == pytest.approx(12.6)
    assert out.iloc[6] == pytest.approx(13.28)
    assert out.iloc[7] == pytest.approx(14.024)
    assert out.iloc[8] == pytest.approx(14.8192)
    assert out.iloc[9] == pytest.approx(15.65536)


def test_smma_equivalent_to_ema_with_alpha_one_over_n():
    """SMMA = EMA with alpha = 1/period (well-known identity)."""
    rng = np.random.default_rng(42)
    s = pd.Series(rng.normal(100, 1, 50).cumsum())
    period = 7
    smma_out = smma(s, period)
    # Manual EMA with alpha = 1/period, seeded with SMA of first `period`.
    values = s.to_numpy()
    expected = np.full(len(values), np.nan)
    seed = values[:period].mean()
    expected[period - 1] = seed
    for i in range(period, len(values)):
        expected[i] = (expected[i - 1] * (period - 1) + values[i]) / period
    np.testing.assert_allclose(
        smma_out.to_numpy(), expected, rtol=1e-12, equal_nan=True
    )


# ---------------------------------------------------------------------------
# LWMA
# ---------------------------------------------------------------------------

def test_lwma_weights():
    """LWMA period=5, weights 1..5, denom=15.

    [10..19]:
    idx 4: (10*1 + 11*2 + 12*3 + 13*4 + 14*5)/15
         = (10 + 22 + 36 + 52 + 70)/15 = 190/15 ≈ 12.6666...
    idx 5: 205/15 ≈ 13.6666...
    idx 9: 265/15 ≈ 17.6666...
    """
    s = _ramp(10.0, 10)
    out = lwma(s, period=5)
    assert out.iloc[:4].isna().all()
    assert out.iloc[4] == pytest.approx(190 / 15)
    assert out.iloc[5] == pytest.approx(205 / 15)
    assert out.iloc[9] == pytest.approx(265 / 15)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def test_rsi_alternating_series_is_50():
    """Symmetric +/- moves → equal SMMA gain & loss → RSI = 50."""
    close = pd.Series([44.0, 44.5] * 7 + [44.0])  # 15 bars
    out = rsi(close, period=14)
    assert out.iloc[:14].isna().all()
    assert out.iloc[14] == pytest.approx(50.0)


def test_rsi_monotonic_up_is_100():
    """No losses ever → avg_loss == 0 → RSI saturates at 100."""
    close = _ramp(10.0, 15)  # 15 bars, all gains == 1
    out = rsi(close, period=14)
    assert out.iloc[14] == pytest.approx(100.0)


def test_rsi_zigzag_with_drift():
    """close = [10, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 17, 16, 18, 17, 19]

    diffs (after prepend):
      [0, 2, -1, 2, -1, 2, -1, 2, -1, 2, -1, 2, -1, 2, -1, 2]

    gain[1:15] mean = (7 * 2) / 14 = 1.0
    loss[1:15] mean = (7 * 1) / 14 = 0.5
    RSI[14] = 100 - 100/(1 + 1/0.5) = 200/3 ≈ 66.6666...

    Step at idx 15 (gain=2, loss=0):
      avg_gain = (1.0 * 13 + 2) / 14 = 15/14
      avg_loss = (0.5 * 13 + 0) / 14 = 13/28
      RS       = (15/14) / (13/28) = 30/13
      RSI[15]  = 100 - 100*13/43 = 3000/43 ≈ 69.7674418604...
    """
    close = pd.Series(
        [10, 12, 11, 13, 12, 14, 13, 15, 14, 16,
         15, 17, 16, 18, 17, 19],
        dtype=float,
    )
    out = rsi(close, period=14)
    assert out.iloc[14] == pytest.approx(200 / 3, rel=1e-12)
    assert out.iloc[15] == pytest.approx(3000 / 43, rel=1e-12)


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

def test_atr_with_known_true_ranges():
    """Hand-computed TR and SMMA(period=3):

    bars:
      i  high  low  close
      0   10    9     9
      1   12   10    11
      2   11    9    10
      3   14   11    12
      4   13   10    11
      5   16   12    14
      6   15   11    13

    TR[0] = high[0] - low[0]            = 1
    TR[1] = max(2, |12-9|, |10-9|)      = 3
    TR[2] = max(2, |11-11|, |9-11|)     = 2
    TR[3] = max(3, |14-10|, |11-10|)    = 4
    TR[4] = max(3, |13-12|, |10-12|)    = 3
    TR[5] = max(4, |16-11|, |12-11|)    = 5
    TR[6] = max(4, |15-14|, |11-14|)    = 4

    ATR period=3 (SMMA of TR):
      idx 2 (seed) = (1+3+2)/3 = 2.0
      idx 3 = (2*2 + 4)/3 = 8/3
      idx 4 = (8/3*2 + 3)/3 = 25/9
      idx 5 = (25/9*2 + 5)/3 = 95/27
      idx 6 = (95/27*2 + 4)/3 = 298/81
    """
    df = pd.DataFrame({
        "high":  [10, 12, 11, 14, 13, 16, 15],
        "low":   [9, 10,  9, 11, 10, 12, 11],
        "close": [9, 11, 10, 12, 11, 14, 13],
    }, dtype=float)
    out = atr(df, period=3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[3] == pytest.approx(8 / 3)
    assert out.iloc[4] == pytest.approx(25 / 9)
    assert out.iloc[5] == pytest.approx(95 / 27)
    assert out.iloc[6] == pytest.approx(298 / 81)


def test_atr_requires_ohlc():
    df = pd.DataFrame({"high": [1.0], "low": [1.0]})
    with pytest.raises(KeyError):
        atr(df, period=3)


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def test_bollinger_population_std():
    """Hand-computed for close = [2, 4, 4, 4, 5, 5, 7, 9], period=4, dev=2:

    idx 3: middle = 14/4 = 3.5
           var    = (4+16+16+16)/4 - 3.5^2 = 13 - 12.25 = 0.75
           dev    = sqrt(0.75)
           upper  = 3.5 + 2*sqrt(0.75) ≈ 5.232050807568877
           lower  = 3.5 - 2*sqrt(0.75) ≈ 1.767949192431123

    idx 7: middle = (5+5+7+9)/4 = 6.5
           var    = (25+25+49+81)/4 - 6.5^2 = 45 - 42.25 = 2.75
           dev    = sqrt(2.75)
           upper  = 6.5 + 2*sqrt(2.75) ≈ 9.816624900156904
           lower  = 6.5 - 2*sqrt(2.75) ≈ 3.183375099843096
    """
    close = pd.Series([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)
    middle, upper, lower = bollinger_bands(close, period=4, deviations=2.0)
    assert middle.iloc[:3].isna().all()
    assert middle.iloc[3] == pytest.approx(3.5)
    assert upper.iloc[3] == pytest.approx(3.5 + 2 * math.sqrt(0.75))
    assert lower.iloc[3] == pytest.approx(3.5 - 2 * math.sqrt(0.75))
    assert middle.iloc[7] == pytest.approx(6.5)
    assert upper.iloc[7] == pytest.approx(6.5 + 2 * math.sqrt(2.75))
    assert lower.iloc[7] == pytest.approx(6.5 - 2 * math.sqrt(2.75))


def test_bollinger_uses_population_not_sample_std():
    """Sanity: pandas .std() is sample (N-1) and would over-estimate the band
    width by sqrt(N/(N-1)). We must NOT use it."""
    close = pd.Series([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)
    _, upper, _ = bollinger_bands(close, period=4, deviations=2.0)
    # If we mistakenly used sample std, upper at idx 3 would be:
    # 3.5 + 2 * sqrt(1.0) = 5.5  (since sample var = 1.0 here).
    # Population: 3.5 + 2 * sqrt(0.75) ≈ 5.232.
    assert upper.iloc[3] != pytest.approx(5.5)
    assert upper.iloc[3] == pytest.approx(3.5 + 2 * math.sqrt(0.75))


def test_bollinger_shift_displaces_bands():
    close = pd.Series(np.arange(20, dtype=float))
    base_mid, _, _ = bollinger_bands(close, period=4, deviations=2.0, shift=0)
    shifted_mid, _, _ = bollinger_bands(
        close, period=4, deviations=2.0, shift=2
    )
    # Positive shift moves the band forward in time: shifted[i] = base[i-2].
    np.testing.assert_allclose(
        shifted_mid.iloc[5:].to_numpy(),
        base_mid.iloc[3:-2].to_numpy(),
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# Stochastic
# ---------------------------------------------------------------------------

def test_stochastic_slowing_one():
    """k_period=3, slowing=1, d_period=2.

    For the bars below:
      i  high  low  close
      0   2    1    1.5
      1   4    3    3.5
      2   6    5    5.5
      3   5    4    4.5
      4   7    6    6.5
      5   8    7    7.5

    LL_3:  idx 2..5 = [1, 3, 4, 4]
    HH_3:  idx 2..5 = [6, 6, 7, 8]
    %K (slowing=1):
      idx 2 = 100*(5.5-1)/(6-1)   = 100*4.5/5  = 90.0
      idx 3 = 100*(4.5-3)/(6-3)   = 100*1.5/3  = 50.0
      idx 4 = 100*(6.5-4)/(7-4)   = 100*2.5/3  = 83.333...
      idx 5 = 100*(7.5-4)/(8-4)   = 100*3.5/4  = 87.5

    %D = SMA(%K, 2):
      idx 3 = (90 + 50)/2     = 70.0
      idx 4 = (50 + 83.333)/2 ≈ 66.6666...
      idx 5 = (83.333 + 87.5)/2 ≈ 85.4166...
    """
    df = pd.DataFrame({
        "high":  [2, 4, 6, 5, 7, 8],
        "low":   [1, 3, 5, 4, 6, 7],
        "close": [1.5, 3.5, 5.5, 4.5, 6.5, 7.5],
    }, dtype=float)
    k, d = stochastic(df, k_period=3, d_period=2, slowing=1)
    assert k.iloc[:2].isna().all()
    assert k.iloc[2] == pytest.approx(90.0)
    assert k.iloc[3] == pytest.approx(50.0)
    assert k.iloc[4] == pytest.approx(250 / 3)
    assert k.iloc[5] == pytest.approx(87.5)
    assert d.iloc[:3].isna().all()
    assert d.iloc[3] == pytest.approx(70.0)
    assert d.iloc[4] == pytest.approx((50 + 250 / 3) / 2)
    assert d.iloc[5] == pytest.approx((250 / 3 + 87.5) / 2)


def test_stochastic_sum_over_sum_form():
    """Slowing=2 — verifies the MT5 ``sum(close-LL) / sum(HH-LL)`` form,
    which is NOT the same as ``SMA(raw_K, slowing)``.

    Same bars as above. With slowing=2:
      idx 3: num = (5.5-1) + (4.5-3) = 6.0;  den = (6-1) + (6-3) = 8;  %K = 75.0
      idx 4: num = (4.5-3) + (6.5-4) = 4.0;  den = (6-3) + (7-4) = 6;  %K = 200/3
      idx 5: num = (6.5-4) + (7.5-4) = 6.0;  den = (7-4) + (8-4) = 7;  %K = 600/7
    """
    df = pd.DataFrame({
        "high":  [2, 4, 6, 5, 7, 8],
        "low":   [1, 3, 5, 4, 6, 7],
        "close": [1.5, 3.5, 5.5, 4.5, 6.5, 7.5],
    }, dtype=float)
    k, _ = stochastic(df, k_period=3, d_period=2, slowing=2)
    assert k.iloc[:3].isna().all()
    assert k.iloc[3] == pytest.approx(75.0)
    assert k.iloc[4] == pytest.approx(200 / 3)
    assert k.iloc[5] == pytest.approx(600 / 7)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def test_macd_constant_series_is_zero():
    """Constant input → both EMAs converge → MACD line == 0 from idx slow-1."""
    close = pd.Series([10.0] * 30)
    macd_line, signal_line, hist = macd(close, fast=3, slow=5, signal=2)
    # First slow-1 = 4 entries are NaN on macd_line.
    assert macd_line.iloc[:4].isna().all()
    np.testing.assert_allclose(macd_line.iloc[4:].to_numpy(), 0.0, atol=1e-12)
    # Signal needs an extra 2-1 = 1 bar -> first valid at idx 5.
    np.testing.assert_allclose(signal_line.iloc[5:].to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(hist.iloc[5:].to_numpy(), 0.0, atol=1e-12)


def test_macd_linear_ramp_macd_line_is_one():
    """For close=[1..40], fast=3, slow=5:
        EMA3 self-stabilises to close (lag 0) for a linear ramp once seeded.
        Specifically EMA3[i] = i for i>=4, EMA5[i] = i-1 for i>=4.
        So MACD = 1 from idx 4 onwards. Signal = SMA(MACD,2) = 1 from idx 5.
        Histogram = 0 from idx 5.
    """
    close = pd.Series(np.arange(1, 41, dtype=float))
    macd_line, signal_line, hist = macd(close, fast=3, slow=5, signal=2)
    assert macd_line.iloc[:4].isna().all()
    np.testing.assert_allclose(macd_line.iloc[4:].to_numpy(), 1.0, rtol=1e-12)
    np.testing.assert_allclose(signal_line.iloc[5:].to_numpy(), 1.0, rtol=1e-12)
    np.testing.assert_allclose(hist.iloc[5:].to_numpy(), 0.0, atol=1e-12)


def test_macd_histogram_is_difference():
    """Invariant on any input: hist == macd_line - signal_line."""
    rng = np.random.default_rng(42)
    close = pd.Series(100 + rng.normal(0, 1, 200).cumsum())
    macd_line, signal_line, hist = macd(close, fast=12, slow=26, signal=9)
    diff = (macd_line - signal_line).dropna()
    np.testing.assert_allclose(
        hist.dropna().to_numpy(), diff.to_numpy(), rtol=1e-12
    )


def test_macd_rejects_invalid_periods():
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        macd(s, fast=26, slow=12, signal=9)


# ---------------------------------------------------------------------------
# Cross-indicator integration: deterministic seeded series
# ---------------------------------------------------------------------------

def test_no_indicator_silently_drops_index():
    """All indicators must preserve the input pandas index (UTC timestamps in
    real use). Regression guard for accidental .reset_index() calls."""
    idx = pd.date_range("2025-01-01", periods=30, freq="h", tz="UTC")
    rng = np.random.default_rng(7)
    close = pd.Series(100 + rng.normal(0, 1, 30).cumsum(), index=idx)
    df = pd.DataFrame(
        {
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        },
        index=idx,
    )
    for out in (
        sma(close, 5),
        ema(close, 5),
        smma(close, 5),
        lwma(close, 5),
        rsi(close, 14),
        atr(df, 14),
    ):
        assert out.index.equals(idx)

    mid, up, low = bollinger_bands(close, period=5)
    assert mid.index.equals(idx) and up.index.equals(idx) and low.index.equals(idx)
    k, d = stochastic(df, k_period=5, d_period=3, slowing=3)
    assert k.index.equals(idx) and d.index.equals(idx)
    m, s, h = macd(close)
    assert m.index.equals(idx) and s.index.equals(idx) and h.index.equals(idx)
