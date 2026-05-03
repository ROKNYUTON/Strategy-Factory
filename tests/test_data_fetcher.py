"""Tests for python_engine.data_fetcher with MetaTrader5 mocked."""
from __future__ import annotations

import calendar
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


def _utc_seconds(dt: datetime) -> int:
    """Treat naive datetime as UTC and return unix seconds.

    Mirrors how MT5 reports broker server time (already in seconds, no local-tz shift).
    """
    return calendar.timegm(dt.utctimetuple())

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from python_engine.data_fetcher import (  # noqa: E402
    MT5ConnectionError,
    MT5DataFetcher,
    MT5SymbolError,
)


# ----------------------------------------------------------------------
# Fake MT5 module
# ----------------------------------------------------------------------
def _make_rates(start: datetime, n_bars: int, skip_weekends: bool = True):
    """Build a numpy structured array shaped like mt5.copy_rates_range output."""
    dtype = np.dtype([
        ("time", "i8"),
        ("open", "f8"),
        ("high", "f8"),
        ("low", "f8"),
        ("close", "f8"),
        ("tick_volume", "i8"),
        ("real_volume", "i8"),
        ("spread", "i4"),
    ])
    rows = []
    cur = start
    while len(rows) < n_bars:
        if skip_weekends and cur.weekday() >= 5:
            cur = cur + pd.Timedelta(minutes=1)
            continue
        ts = _utc_seconds(cur)
        rows.append((ts, 1.1, 1.2, 1.05, 1.15, 100, 0, 5))
        cur = cur + pd.Timedelta(minutes=1)
    return np.array(rows, dtype=dtype)


class FakeMT5:
    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_H1 = 60

    def __init__(self, init_ok: bool = True, symbol_known: bool = True,
                 symbol_select_ok: bool = True, rates=None):
        self._init_ok = init_ok
        self._symbol_known = symbol_known
        self._symbol_select_ok = symbol_select_ok
        self._rates = rates
        self.initialize_calls = 0
        self.shutdown_calls = 0
        self.copy_rates_calls = []
        self.symbol_select_calls = []

    def initialize(self):
        self.initialize_calls += 1
        return self._init_ok

    def shutdown(self):
        self.shutdown_calls += 1

    def last_error(self):
        return (-1, "fake error")

    def symbol_info(self, symbol):
        if self._symbol_known:
            return SimpleNamespace(name=symbol, visible=True)
        return None

    def symbol_select(self, symbol, enable):
        self.symbol_select_calls.append((symbol, enable))
        return self._symbol_select_ok

    def copy_rates_range(self, symbol, tf, start, end):
        self.copy_rates_calls.append((symbol, tf, start, end))
        return self._rates


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "data_cache"


@pytest.fixture
def rates_3_days():
    # Mon 2024-01-01 → enough bars covering weekday minutes
    return _make_rates(datetime(2024, 1, 1, 0, 0), n_bars=300)


# ----------------------------------------------------------------------
# Connection
# ----------------------------------------------------------------------
def test_initialize_failure_raises(cache_dir):
    fake = FakeMT5(init_ok=False)
    with pytest.raises(MT5ConnectionError):
        MT5DataFetcher(cache_dir=cache_dir, mt5_module=fake)


def test_context_manager_shuts_down(cache_dir, rates_3_days):
    fake = FakeMT5(rates=rates_3_days)
    with MT5DataFetcher(cache_dir=cache_dir, mt5_module=fake) as f:
        assert f._initialized is True
    assert fake.shutdown_calls == 1


# ----------------------------------------------------------------------
# Fetch + cache
# ----------------------------------------------------------------------
def test_fetch_returns_dataframe_with_required_columns(cache_dir, rates_3_days):
    fake = FakeMT5(rates=rates_3_days)
    f = MT5DataFetcher(cache_dir=cache_dir, mt5_module=fake)
    df = f.fetch("EURUSD", datetime(2024, 1, 1), datetime(2024, 1, 5))

    expected = ["time", "open", "high", "low", "close",
                "tick_volume", "real_volume", "spread"]
    assert list(df.columns) == expected
    assert pd.api.types.is_datetime64_any_dtype(df["time"])
    assert len(df) == 300


def test_fetch_writes_cache_first_call_and_reads_second_call(
    cache_dir, rates_3_days
):
    fake = FakeMT5(rates=rates_3_days)
    f = MT5DataFetcher(cache_dir=cache_dir, mt5_module=fake)

    start, end = datetime(2024, 1, 1), datetime(2024, 1, 5)
    df1 = f.fetch("EURUSD", start, end)
    assert len(fake.copy_rates_calls) == 1

    cache_files = list(cache_dir.glob("*.parquet"))
    assert len(cache_files) == 1

    df2 = f.fetch("EURUSD", start, end)
    # Cache hit: no new MT5 call
    assert len(fake.copy_rates_calls) == 1
    # Parquet round-trip may rescale datetime precision; dtypes can differ.
    pd.testing.assert_frame_equal(df1, df2, check_dtype=False)


def test_force_refresh_bypasses_cache(cache_dir, rates_3_days):
    fake = FakeMT5(rates=rates_3_days)
    f = MT5DataFetcher(cache_dir=cache_dir, mt5_module=fake)
    start, end = datetime(2024, 1, 1), datetime(2024, 1, 5)
    f.fetch("EURUSD", start, end)
    f.fetch("EURUSD", start, end, force_refresh=True)
    assert len(fake.copy_rates_calls) == 2


def test_fetch_invalid_range_raises(cache_dir):
    fake = FakeMT5()
    f = MT5DataFetcher(cache_dir=cache_dir, mt5_module=fake)
    with pytest.raises(ValueError):
        f.fetch("EURUSD", datetime(2024, 1, 5), datetime(2024, 1, 1))


def test_fetch_no_bars_raises(cache_dir):
    fake = FakeMT5(rates=np.array([], dtype=[("time", "i8")]))
    f = MT5DataFetcher(cache_dir=cache_dir, mt5_module=fake)
    with pytest.raises(RuntimeError):
        f.fetch("EURUSD", datetime(2024, 1, 1), datetime(2024, 1, 5))


# ----------------------------------------------------------------------
# Symbol selection
# ----------------------------------------------------------------------
def test_unknown_symbol_triggers_symbol_select(cache_dir, rates_3_days):
    fake = FakeMT5(symbol_known=False, symbol_select_ok=True, rates=rates_3_days)
    f = MT5DataFetcher(cache_dir=cache_dir, mt5_module=fake)
    f.fetch("EURUSD", datetime(2024, 1, 1), datetime(2024, 1, 5))
    assert ("EURUSD", True) in fake.symbol_select_calls


def test_unknown_symbol_select_failure_raises(cache_dir, rates_3_days):
    fake = FakeMT5(symbol_known=False, symbol_select_ok=False, rates=rates_3_days)
    f = MT5DataFetcher(cache_dir=cache_dir, mt5_module=fake)
    with pytest.raises(MT5SymbolError):
        f.fetch("WEIRDSYM", datetime(2024, 1, 1), datetime(2024, 1, 5))


# ----------------------------------------------------------------------
# Weekend gap handling
# ----------------------------------------------------------------------
def test_weekend_gaps_not_forward_filled(cache_dir):
    # Build rates that span a weekend: Fri last bar → Mon first bar.
    fri = datetime(2024, 1, 5, 22, 59)  # Friday late
    mon = datetime(2024, 1, 8, 0, 0)    # Monday open
    dtype = np.dtype([
        ("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
        ("close", "f8"), ("tick_volume", "i8"), ("real_volume", "i8"),
        ("spread", "i4"),
    ])
    rows = [
        (_utc_seconds(fri), 1.1, 1.2, 1.0, 1.15, 100, 0, 5),
        (_utc_seconds(mon), 1.16, 1.17, 1.14, 1.165, 120, 0, 6),
    ]
    rates = np.array(rows, dtype=dtype)
    fake = FakeMT5(rates=rates)
    f = MT5DataFetcher(cache_dir=cache_dir, mt5_module=fake)
    df = f.fetch("EURUSD", datetime(2024, 1, 5), datetime(2024, 1, 9))

    assert len(df) == 2
    weekend_rows = df[df["time"].dt.dayofweek >= 5]
    assert len(weekend_rows) == 0
    diff = df["time"].iloc[1] - df["time"].iloc[0]
    assert diff > pd.Timedelta(minutes=1)


# ----------------------------------------------------------------------
# Quality report
# ----------------------------------------------------------------------
def test_quality_report_basic(cache_dir, rates_3_days):
    fake = FakeMT5(rates=rates_3_days)
    f = MT5DataFetcher(cache_dir=cache_dir, mt5_module=fake)
    df = f.fetch("EURUSD", datetime(2024, 1, 1), datetime(2024, 1, 5))
    report = f.get_data_quality_report(df)
    assert report["total_bars"] == 300
    assert report["spread_avg"] == 5.0
    assert report["spread_max"] == 5.0
    assert report["weekends_handled"] is True


def test_quality_report_empty():
    f = MT5DataFetcher.__new__(MT5DataFetcher)  # bypass init for pure-fn test
    report = MT5DataFetcher.get_data_quality_report(f, pd.DataFrame())
    assert report["total_bars"] == 0
    assert report["first_bar"] is None


# ----------------------------------------------------------------------
# Multi
# ----------------------------------------------------------------------
def test_fetch_multi_returns_all_symbols(cache_dir, rates_3_days):
    fake = FakeMT5(rates=rates_3_days)
    f = MT5DataFetcher(cache_dir=cache_dir, mt5_module=fake)
    out = f.fetch_multi(
        ["EURUSD", "USDJPY"],
        datetime(2024, 1, 1), datetime(2024, 1, 5),
        n_workers=2,
    )
    assert set(out.keys()) == {"EURUSD", "USDJPY"}
    assert all(len(df) == 300 for df in out.values())


# ----------------------------------------------------------------------
# Large period guard
# ----------------------------------------------------------------------
def test_large_period_requires_confirmation(cache_dir, rates_3_days):
    fake = FakeMT5(rates=rates_3_days)
    confirm = MagicMock(return_value=True)
    f = MT5DataFetcher(
        cache_dir=cache_dir, mt5_module=fake, confirm_large_download=confirm
    )
    f.fetch("EURUSD", datetime(2018, 1, 1), datetime(2026, 1, 1))
    confirm.assert_called_once()


def test_large_period_decline_raises(cache_dir, rates_3_days):
    fake = FakeMT5(rates=rates_3_days)
    f = MT5DataFetcher(
        cache_dir=cache_dir, mt5_module=fake,
        confirm_large_download=lambda *a, **kw: False,
    )
    with pytest.raises(RuntimeError):
        f.fetch("EURUSD", datetime(2018, 1, 1), datetime(2026, 1, 1))
