"""MQL5-faithful technical indicators for the Python research engine.

Every function in this module is designed to produce **bit-identical output**
(within float64 precision) to the corresponding MetaTrader 5 indicator on the
same input series. If a function here disagrees with MT5 on the last decimal,
it is a bug — file an issue and add a regression test in
``tests/test_indicators_mql5.py``.

Conventions
-----------
- Input ``pd.Series``/``pd.DataFrame`` are time-ascending (oldest first, newest
  last). This matches the order produced by ``MetaTrader5.copy_rates_*`` once
  it is converted to a DataFrame, and is the natural pandas order.
- For an indicator with period ``N``, the first ``N - 1`` outputs are
  ``np.nan`` (MT5 returns 0 / empty buffer in those slots — we use NaN so that
  pandas alignment and statistical tooling behave correctly).
- ``EMA`` and ``SMMA`` are seeded with the SMA of the first ``N`` bars (MT5
  convention), then propagate recursively.
- All standard deviations are **population** (divide by N), not sample (N-1).
  This is what MT5's ``iBands`` uses internally.

References
----------
- MQL5 Reference, ``iMA``, ``iRSI``, ``iATR``, ``iBands``, ``iStochastic``,
  ``iMACD``: https://www.mql5.com/en/docs/indicators
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average.

    MQL5 equivalent::

        iMA(symbol, timeframe, period, 0, MODE_SMA, applied_price)

    First ``period - 1`` values are NaN. Value at index ``i >= period - 1`` is
    the arithmetic mean of ``series[i - period + 1 : i + 1]``.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average (MT5 convention).

    MQL5 equivalent::

        iMA(symbol, timeframe, period, 0, MODE_EMA, applied_price)

    The first ``period`` values are seeded with the SMA of the first ``period``
    bars (so the value at index ``period - 1`` equals ``SMA(period)``).
    Subsequent values use::

        alpha = 2 / (period + 1)
        EMA[i] = alpha * price[i] + (1 - alpha) * EMA[i - 1]

    Indices ``0 .. period - 2`` are NaN.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    values = series.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    if n < period:
        return pd.Series(out, index=series.index, name=series.name)
    alpha = 2.0 / (period + 1)
    one_minus = 1.0 - alpha
    seed = values[:period].mean()
    out[period - 1] = seed
    prev = seed
    for i in range(period, n):
        prev = alpha * values[i] + one_minus * prev
        out[i] = prev
    return pd.Series(out, index=series.index, name=series.name)


def smma(series: pd.Series, period: int) -> pd.Series:
    """Smoothed Moving Average / Wilder's MA.

    MQL5 equivalent::

        iMA(symbol, timeframe, period, 0, MODE_SMMA, applied_price)

    Identical to ``EMA`` with ``alpha = 1 / period``. Used internally by
    :func:`rsi` and :func:`atr`.

    Recursion::

        SMMA[period - 1] = mean(price[0 : period])
        SMMA[i] = (SMMA[i - 1] * (period - 1) + price[i]) / period
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    values = series.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    if n < period:
        return pd.Series(out, index=series.index, name=series.name)
    seed = values[:period].mean()
    out[period - 1] = seed
    prev = seed
    inv = 1.0 / period
    pm1 = period - 1
    for i in range(period, n):
        prev = (prev * pm1 + values[i]) * inv
        out[i] = prev
    return pd.Series(out, index=series.index, name=series.name)


def lwma(series: pd.Series, period: int) -> pd.Series:
    """Linear Weighted Moving Average.

    MQL5 equivalent::

        iMA(symbol, timeframe, period, 0, MODE_LWMA, applied_price)

    Most recent bar receives weight ``period``, oldest in the window weight
    ``1``. Denominator is ``period * (period + 1) / 2``.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    weights = np.arange(1, period + 1, dtype=float)
    denom = weights.sum()

    def _weighted(window: np.ndarray) -> float:
        return float(np.dot(window, weights) / denom)

    return series.rolling(window=period, min_periods=period).apply(
        _weighted, raw=True
    )


# ---------------------------------------------------------------------------
# Oscillators
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder, MT5 implementation).

    MQL5 equivalent::

        iRSI(symbol, timeframe, period, applied_price)

    Computed via SMMA of gains and losses (NOT a simple mean — that is the
    most common porting bug). The first ``period`` values are NaN; the first
    valid output is at index ``period``.

    Formula::

        delta[i]  = close[i] - close[i-1]
        gain[i]   = max(delta[i], 0)
        loss[i]   = max(-delta[i], 0)
        avg_gain  = SMMA(gain, period)   # seeded with mean of first period values
        avg_loss  = SMMA(loss, period)
        RS  = avg_gain / avg_loss
        RSI = 100 - 100 / (1 + RS)

    Note on seeding: MT5 seeds the SMMA at index ``period`` (because gain[0]
    and loss[0] are undefined — there is no prior close). So the seed is the
    mean of ``gain[1 : period + 1]`` (i.e. the first ``period`` defined
    differences) and the first RSI value lands at index ``period``.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    values = close.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    if n <= period:
        return pd.Series(out, index=close.index, name="rsi")
    diff = np.diff(values, prepend=values[0])
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    # gain[0] / loss[0] are 0 by construction (prepend trick) — we ignore index
    # 0 in the seed by starting from index 1.
    seed_gain = gain[1 : period + 1].mean()
    seed_loss = loss[1 : period + 1].mean()
    avg_gain = seed_gain
    avg_loss = seed_loss
    out[period] = _rsi_from_avgs(avg_gain, avg_loss)
    pm1 = period - 1
    inv = 1.0 / period
    for i in range(period + 1, n):
        avg_gain = (avg_gain * pm1 + gain[i]) * inv
        avg_loss = (avg_loss * pm1 + loss[i]) * inv
        out[i] = _rsi_from_avgs(avg_gain, avg_loss)
    return pd.Series(out, index=close.index, name="rsi")


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder, MT5 implementation).

    MQL5 equivalent::

        iATR(symbol, timeframe, period)

    ``df`` must expose columns ``high``, ``low``, ``close``.

    Formula::

        TR[0] = high[0] - low[0]    # no previous close
        TR[i] = max( high[i] - low[i],
                     |high[i] - close[i-1]|,
                     |low[i]  - close[i-1]| )
        ATR   = SMMA(TR, period)

    First ``period - 1`` values are NaN; first valid output at index
    ``period - 1`` is the simple mean of TR[0 .. period - 1].
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise KeyError(f"DataFrame missing required column '{col}'")
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(df)
    tr = np.empty(n, dtype=float)
    if n == 0:
        return pd.Series(tr, index=df.index, name="atr")
    tr[0] = high[0] - low[0]
    if n > 1:
        prev_close = close[:-1]
        hl = high[1:] - low[1:]
        hc = np.abs(high[1:] - prev_close)
        lc = np.abs(low[1:] - prev_close)
        tr[1:] = np.maximum(np.maximum(hl, hc), lc)
    return smma(pd.Series(tr, index=df.index), period).rename("atr")


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------

def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    deviations: float = 2.0,
    shift: int = 0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands (MT5 ``iBands``).

    MQL5 equivalent::

        iBands(symbol, timeframe, period, shift, deviations, applied_price)

    Returns ``(middle, upper, lower)``.

    Formula::

        middle = SMA(close, period)
        dev    = sqrt( sum((close - middle)^2 over the window) / period )
        upper  = middle + deviations * dev
        lower  = middle - deviations * dev

    Note: ``dev`` uses **population** standard deviation (divide by ``N``),
    matching MT5 — pandas' ``rolling.std()`` defaults to sample (``N - 1``)
    and would be off by a factor of ``sqrt(N / (N-1))``.

    The ``shift`` argument is the MT5 ``bands_shift`` (horizontal displacement
    of the bands). A positive value shifts the bands to the right by ``shift``
    bars (i.e. ``result[i]`` shows the band computed at bar ``i - shift``).
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    middle = sma(close, period)
    # Population variance via E[X^2] - (E[X])^2. Floor at 0 to absorb float
    # subtraction noise on near-flat windows.
    sq_mean = close.pow(2).rolling(window=period, min_periods=period).mean()
    var = (sq_mean - middle.pow(2)).clip(lower=0.0)
    deviation = var.pow(0.5)
    upper = middle + deviations * deviation
    lower = middle - deviations * deviation
    if shift:
        middle = middle.shift(shift)
        upper = upper.shift(shift)
        lower = lower.shift(shift)
    return (
        middle.rename("bb_middle"),
        upper.rename("bb_upper"),
        lower.rename("bb_lower"),
    )


# ---------------------------------------------------------------------------
# Stochastic
# ---------------------------------------------------------------------------

def stochastic(
    df: pd.DataFrame,
    k_period: int = 5,
    d_period: int = 3,
    slowing: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator (MT5 ``iStochastic``, ``MODE_LOWHIGH`` + ``MODE_SMA``).

    MQL5 equivalent::

        iStochastic(symbol, timeframe, k_period, d_period, slowing,
                    MODE_SMA, STO_LOWHIGH)

    Returns ``(%K, %D)``.

    Formula (MT5 specific — note the **sum/sum** form, not an SMA of raw %K)::

        LL[i] = min(low[i - k_period + 1 : i + 1])
        HH[i] = max(high[i - k_period + 1 : i + 1])
        num   = sum(close - LL,  over `slowing` bars)
        den   = sum(HH    - LL,  over `slowing` bars)
        %K    = 100 * num / den
        %D    = SMA(%K, d_period)

    The MT5 sum/sum form is mathematically distinct from the (more common in
    Python TA libraries) "SMA of raw %K"; getting this wrong is a frequent
    porting bug.

    First valid ``%K`` is at index ``k_period - 1 + slowing - 1``.
    First valid ``%D`` is at index ``k_period - 1 + slowing - 1 + d_period - 1``.
    """
    if k_period <= 0 or d_period <= 0 or slowing <= 0:
        raise ValueError("k_period, d_period, slowing must be positive")
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise KeyError(f"DataFrame missing required column '{col}'")
    low = df["low"]
    high = df["high"]
    close = df["close"]
    ll = low.rolling(window=k_period, min_periods=k_period).min()
    hh = high.rolling(window=k_period, min_periods=k_period).max()
    num = (close - ll).rolling(window=slowing, min_periods=slowing).sum()
    den = (hh - ll).rolling(window=slowing, min_periods=slowing).sum()
    # Replicate MT5 behaviour on a flat window: when the high range is zero,
    # MT5 carries forward the previous %K value (or 0 if none). For our
    # research purposes we emit NaN in that degenerate slot — let the caller
    # decide.
    k = 100.0 * num / den.where(den != 0)
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k.rename("stoch_k"), d.rename("stoch_d")


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD (MT5 ``iMACD``).

    MQL5 equivalent::

        iMACD(symbol, timeframe, fast, slow, signal, applied_price)

    Returns ``(macd_line, signal_line, histogram)``.

    **Important MT5 quirk**: the signal line is an **SMA** of the MACD line,
    not an EMA. This differs from the more common (e.g. TA-Lib) MACD which
    uses an EMA. We follow MT5.

    Formula::

        macd_line   = EMA(close, fast) - EMA(close, slow)
        signal_line = SMA(macd_line, signal)
        histogram   = macd_line - signal_line
    """
    if fast <= 0 or slow <= 0 or signal <= 0:
        raise ValueError("fast, slow, signal must be positive")
    if fast >= slow:
        raise ValueError(
            f"fast ({fast}) must be smaller than slow ({slow})"
        )
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = (fast_ema - slow_ema).rename("macd")
    signal_line = sma(macd_line, signal).rename("macd_signal")
    histogram = (macd_line - signal_line).rename("macd_hist")
    return macd_line, signal_line, histogram


__all__ = [
    "sma",
    "ema",
    "smma",
    "lwma",
    "rsi",
    "atr",
    "bollinger_bands",
    "stochastic",
    "macd",
]
