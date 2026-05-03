"""MT5 M1 OHLC + spread fetcher with parquet cache.

Time stored is broker server time (matches MT5 Strategy Tester exactly).
Spread is in points. Weekend gaps are preserved (no forward fill).

CLI:
    python -m python_engine.data_fetcher --symbols EURUSD,USDJPY \
        --start 2018-01-01 --end 2026-04-30 --cache-dir ./tmp_cache
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


_TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}

_REQUIRED_COLS = ["time", "open", "high", "low", "close",
                  "tick_volume", "real_volume", "spread"]


def _import_mt5():
    try:
        import MetaTrader5 as mt5  # type: ignore
        return mt5
    except ImportError as exc:
        raise RuntimeError(
            "MetaTrader5 package not installed. Run: pip install MetaTrader5"
        ) from exc


class MT5ConnectionError(RuntimeError):
    pass


class MT5SymbolError(RuntimeError):
    pass


class MT5DataFetcher:
    """Fetches M1 OHLC + spread from MT5, caches to parquet.

    Cache layout: {cache_dir}/{symbol}_{tf}_{start}_{end}.parquet
    """

    LARGE_PERIOD_DAYS = 365 * 5

    def __init__(
        self,
        cache_dir: Path,
        mt5_module=None,
        auto_init: bool = True,
        confirm_large_download=None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._mt5 = mt5_module
        self._initialized = False
        self._owns_connection = False
        self._confirm_large_download = confirm_large_download or self._default_confirm

        if auto_init:
            self._ensure_initialized()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        if self._mt5 is None:
            self._mt5 = _import_mt5()

        if not self._mt5.initialize():
            err = None
            if hasattr(self._mt5, "last_error"):
                err = self._mt5.last_error()
            raise MT5ConnectionError(
                f"mt5.initialize() failed (error={err}). "
                f"Open MT5 terminal first and ensure 'Allow algorithmic trading' is enabled."
            )
        self._initialized = True
        self._owns_connection = True

    def close(self) -> None:
        if self._initialized and self._owns_connection and self._mt5 is not None:
            try:
                self._mt5.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self._initialized = False

    def __enter__(self):
        self._ensure_initialized()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def _cache_path(
        self, symbol: str, tf: str, start: datetime, end: datetime
    ) -> Path:
        s = start.strftime("%Y%m%d")
        e = end.strftime("%Y%m%d")
        return self.cache_dir / f"{symbol}_{tf}_{s}_{e}.parquet"

    # ------------------------------------------------------------------
    # Symbol selection
    # ------------------------------------------------------------------
    def _ensure_symbol(self, symbol: str) -> None:
        info = self._mt5.symbol_info(symbol)
        if info is None:
            if not self._mt5.symbol_select(symbol, True):
                raise MT5SymbolError(
                    f"Symbol '{symbol}' not found in Market Watch and "
                    f"symbol_select failed. Add it manually in MT5."
                )
            return
        if not getattr(info, "visible", True):
            if not self._mt5.symbol_select(symbol, True):
                raise MT5SymbolError(
                    f"Symbol '{symbol}' present but not visible; symbol_select failed."
                )

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------
    def fetch(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "M1",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        if end <= start:
            raise ValueError(f"end ({end}) must be after start ({start})")

        cache_file = self._cache_path(symbol, timeframe, start, end)
        if cache_file.exists() and not force_refresh:
            logger.info("Cache hit: %s", cache_file.name)
            return pd.read_parquet(cache_file)

        if (end - start).days > self.LARGE_PERIOD_DAYS:
            ok = self._confirm_large_download(symbol, start, end)
            if not ok:
                raise RuntimeError(
                    f"Large download for {symbol} ({start.date()} → {end.date()}) declined."
                )

        self._ensure_initialized()
        self._ensure_symbol(symbol)

        tf_attr = _TIMEFRAME_MAP.get(timeframe)
        if tf_attr is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        tf_const = getattr(self._mt5, tf_attr)

        rates = self._mt5.copy_rates_range(symbol, tf_const, start, end)
        if rates is None or len(rates) == 0:
            err = None
            if hasattr(self._mt5, "last_error"):
                err = self._mt5.last_error()
            raise RuntimeError(
                f"No bars returned for {symbol} {timeframe} "
                f"{start} → {end} (error={err})."
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=False)

        for col in _REQUIRED_COLS:
            if col not in df.columns:
                df[col] = 0
        df = df[_REQUIRED_COLS].copy()
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)

        df.to_parquet(cache_file, index=False)
        logger.info("Cached %d bars → %s", len(df), cache_file.name)
        return df

    def fetch_multi(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "M1",
        n_workers: int = 4,
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}
        errors: Dict[str, Exception] = {}

        with ThreadPoolExecutor(max_workers=max(1, n_workers)) as pool:
            future_map = {
                pool.submit(
                    self.fetch, sym, start, end, timeframe, force_refresh
                ): sym
                for sym in symbols
            }
            for fut in as_completed(future_map):
                sym = future_map[fut]
                try:
                    results[sym] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    errors[sym] = exc
                    logger.error("Fetch failed for %s: %s", sym, exc)

        if errors and not results:
            first = next(iter(errors.values()))
            raise first
        return results

    # ------------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------------
    def get_data_quality_report(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return {
                "total_bars": 0,
                "first_bar": None,
                "last_bar": None,
                "gaps": 0,
                "missing_bars": 0,
                "weekends_handled": True,
                "spread_avg": None,
                "spread_max": None,
            }

        df = df.sort_values("time").reset_index(drop=True)
        first = df["time"].iloc[0]
        last = df["time"].iloc[-1]

        deltas = df["time"].diff().dropna()
        one_min = pd.Timedelta(minutes=1)
        gaps = int((deltas > one_min).sum())

        # Expected bars excluding weekends (Sat=5, Sun=6 in pandas dt.dayofweek).
        full_range = pd.date_range(first, last, freq="1min")
        weekday_mask = full_range.dayofweek < 5
        expected_business_minutes = int(weekday_mask.sum())
        missing_bars = max(0, expected_business_minutes - len(df))

        weekend_bars = int((df["time"].dt.dayofweek >= 5).sum())
        weekends_handled = weekend_bars == 0 or weekend_bars / len(df) < 0.05

        return {
            "total_bars": int(len(df)),
            "first_bar": first.isoformat(),
            "last_bar": last.isoformat(),
            "gaps": gaps,
            "missing_bars": missing_bars,
            "weekends_handled": bool(weekends_handled),
            "spread_avg": float(df["spread"].mean()),
            "spread_max": float(df["spread"].max()),
        }

    # ------------------------------------------------------------------
    # Confirmations
    # ------------------------------------------------------------------
    @staticmethod
    def _default_confirm(symbol: str, start: datetime, end: datetime) -> bool:
        if not sys.stdin or not sys.stdin.isatty():
            logger.warning(
                "Large download for %s (%s → %s); non-interactive shell, proceeding.",
                symbol, start.date(), end.date(),
            )
            return True
        prompt = (
            f"\n[!] Large download requested: {symbol} {start.date()} → {end.date()} "
            f"(>5 years). Proceed? [y/N]: "
        )
        try:
            ans = input(prompt).strip().lower()
        except EOFError:
            return False
        return ans in ("y", "yes")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m python_engine.data_fetcher",
        description="Fetch M1 OHLC+spread from MT5 with parquet cache.",
    )
    p.add_argument("--symbols", required=True, help="Comma-separated, e.g. EURUSD,USDJPY")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--timeframe", default="M1", choices=list(_TIMEFRAME_MAP.keys()))
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def _print_quality_table(reports: Dict[str, dict]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print("\nQuality report (install 'rich' for prettier output):")
        for sym, r in reports.items():
            print(f"  {sym}: {r}")
        return

    table = Table(title="MT5 Data Quality Report")
    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Total Bars", justify="right")
    table.add_column("First Bar")
    table.add_column("Last Bar")
    table.add_column("Gaps", justify="right")
    table.add_column("Missing", justify="right")
    table.add_column("Weekends OK", justify="center")
    table.add_column("Spread Avg", justify="right")
    table.add_column("Spread Max", justify="right")

    for sym, r in reports.items():
        table.add_row(
            sym,
            f"{r['total_bars']:,}",
            str(r["first_bar"]) if r["first_bar"] else "-",
            str(r["last_bar"]) if r["last_bar"] else "-",
            str(r["gaps"]),
            str(r["missing_bars"]),
            "Y" if r["weekends_handled"] else "N",
            f"{r['spread_avg']:.2f}" if r["spread_avg"] is not None else "-",
            f"{r['spread_max']:.2f}" if r["spread_max"] is not None else "-",
        )
    Console().print(table)


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    with MT5DataFetcher(cache_dir=args.cache_dir) as fetcher:
        dfs = fetcher.fetch_multi(
            symbols, start, end,
            timeframe=args.timeframe,
            n_workers=args.workers,
            force_refresh=args.force_refresh,
        )
        reports = {sym: fetcher.get_data_quality_report(df) for sym, df in dfs.items()}

    _print_quality_table(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
