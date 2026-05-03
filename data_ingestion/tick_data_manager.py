"""
StrategyFactory — Tick Data Manager
===================================
Verifies tick data availability before launching backtests.

If MetaTrader5 Python package is available, queries the broker directly.
If not, prints clear manual instructions for the trader.

Cache: data_ingestion/cache/tick_data_status.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Dict

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data_ingestion" / "cache" / "tick_data_status.json"
console = Console()

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


def load_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {}


def save_cache(data: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def verify_symbol_period(symbol: str, start: date, end: date) -> Dict:
    """Verify ticks for one symbol/period. Returns dict with status."""
    if not MT5_AVAILABLE:
        return {
            "symbol": symbol,
            "start": str(start),
            "end": str(end),
            "status": "unknown_no_mt5_module",
            "message": "Install MetaTrader5 package to auto-verify, or check MT5 manually.",
        }

    if not mt5.initialize():
        return {
            "symbol": symbol,
            "status": "mt5_init_failed",
            "message": "Could not connect to MT5. Open the terminal and re-run.",
        }

    try:
        sel = mt5.symbol_select(symbol, True)
        if not sel:
            return {
                "symbol": symbol,
                "status": "symbol_unavailable",
                "message": f"Symbol {symbol} not in Market Watch.",
            }

        utc_from = datetime.combine(start, datetime.min.time())
        utc_to   = datetime.combine(end + timedelta(days=1), datetime.min.time())

        ticks = mt5.copy_ticks_range(symbol, utc_from, utc_to, mt5.COPY_TICKS_ALL)
        if ticks is None:
            return {
                "symbol": symbol,
                "start": str(start),
                "end": str(end),
                "status": "no_data",
                "message": f"copy_ticks_range returned None — likely no history loaded.",
            }

        n = len(ticks)
        return {
            "symbol": symbol,
            "start": str(start),
            "end": str(end),
            "status": "ok" if n > 1000 else "sparse",
            "tick_count": n,
        }
    finally:
        mt5.shutdown()


def verify_spec(spec_path: Path) -> List[Dict]:
    """Verify all symbols across IS/OOS/forward periods of a spec."""
    sys.path.insert(0, str(ROOT))
    from automation.spec_validator import validate_spec  # noqa: E402

    spec = validate_spec(spec_path)
    results = []

    for symbol in spec.universe.symbols:
        for period_name, p in [("is", spec.backtest.is_period),
                                ("oos", spec.backtest.oos_period),
                                ("forward", spec.backtest.forward_period)]:
            r = verify_symbol_period(symbol, p.start, p.end)
            r["period"] = period_name
            results.append(r)

    cache = load_cache()
    cache[spec.meta.strategy_id] = {
        "checked_at": datetime.utcnow().isoformat() + "Z",
        "results": results,
    }
    save_cache(cache)
    return results


def print_results(results: List[Dict]) -> None:
    t = Table(title="Tick Data Status")
    t.add_column("Symbol")
    t.add_column("Period")
    t.add_column("Range")
    t.add_column("Status")
    t.add_column("Details")
    for r in results:
        status_color = {
            "ok": "green",
            "sparse": "yellow",
            "no_data": "red",
            "symbol_unavailable": "red",
            "mt5_init_failed": "red",
            "unknown_no_mt5_module": "yellow",
        }.get(r.get("status", ""), "white")

        t.add_row(
            r.get("symbol", "?"),
            r.get("period", "?"),
            f"{r.get('start','?')} → {r.get('end','?')}",
            f"[{status_color}]{r.get('status', '?')}[/{status_color}]",
            r.get("message", f"ticks={r.get('tick_count', '?')}"),
        )
    console.print(t)


def main() -> int:
    if len(sys.argv) != 2:
        console.print("[red]Usage: python data_ingestion/tick_data_manager.py <spec.yaml>[/red]")
        return 2

    p = Path(sys.argv[1])
    if not p.exists():
        console.print(f"[red]Not found: {p}[/red]")
        return 2

    results = verify_spec(p)
    print_results(results)

    bad = [r for r in results if r.get("status") in ("no_data", "symbol_unavailable")]
    if bad:
        console.print(
            "\n[yellow]MANUAL FIX:[/yellow] Open MT5 → Tools → Options → Charts → 'Max bars in chart' = unlimited.\n"
            "Then for each missing symbol: View → Symbols → select symbol → 'Bars' tab → request the period.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
