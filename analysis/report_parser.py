"""
StrategyFactory — MT5 Report Parser
===================================
Parses MT5 Strategy Tester reports (.htm or .xml) into a normalized JSON.

Output schema (saved to backtests/parsed_results/{strategy_id}/{period}_parsed.json):
{
  "strategy_id": "...",
  "generated_at": "ISO-8601",
  "engine_version": "StrategyFactory-1.0",
  "period": "is|oos|forward",
  "summary": {
      "initial_balance": float,
      "net_profit": float,
      "gross_profit": float,
      "gross_loss": float,
      "profit_factor": float,
      "expected_payoff": float,
      "max_drawdown_abs": float,
      "max_drawdown_pct": float,
      "total_trades": int,
      "winning_trades": int,
      "losing_trades": int,
      "win_rate": float,
      "sharpe_ratio_mt5": float,
      "recovery_factor": float
  },
  "trades": [
      {
        "ticket": str,
        "open_time": "ISO-8601",
        "close_time": "ISO-8601",
        "symbol": str,
        "direction": "long|short",
        "lots": float,
        "entry_price": float,
        "exit_price": float,
        "profit_directional": float,
        "profit_swap": float,
        "profit_commission": float,
        "profit_total": float
      }, ...
  ]
}
"""

from __future__ import annotations

import sys
import json
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup
from rich.console import Console
from rich.panel import Panel

ROOT = Path(__file__).parent.parent
console = Console()
ENGINE_VERSION = "StrategyFactory-1.0"


def _to_float(s: str) -> float:
    if s is None: return 0.0
    s = str(s).strip()
    s = s.replace("\xa0", "").replace(" ", "").replace(",", ".")
    s = re.sub(r"[^\d\.\-eE]", "", s)
    if s in ("", "-", "."): return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _to_int(s: str) -> int:
    return int(_to_float(s))


def parse_html_report(html_path: Path) -> dict:
    """Parse an MT5 Strategy Tester .htm report into structured dict."""
    html = html_path.read_text(encoding="utf-16-le", errors="ignore")
    if not html.strip():
        html = html_path.read_text(encoding="utf-8", errors="ignore")

    soup = BeautifulSoup(html, "lxml")

    # Summary section: MT5 reports have key=value pairs in <tr> rows
    summary = {}

    # Walk all rows; collect <td> pairs that look like "Label:" / "value"
    for tr in soup.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 2:
            continue
        # MT5 layout: usually cell pairs like ["Label:", "Value", "Label:", "Value"]
        for i in range(0, len(cells) - 1, 2):
            key = cells[i].rstrip(":").strip().lower()
            val = cells[i + 1]
            if key:
                summary.setdefault(key, val)

    # Translate canonical names
    summary_norm = {
        "initial_balance":  _to_float(summary.get("initial deposit", 10000)),
        "net_profit":       _to_float(summary.get("total net profit", 0)),
        "gross_profit":     _to_float(summary.get("gross profit", 0)),
        "gross_loss":       _to_float(summary.get("gross loss", 0)),
        "profit_factor":    _to_float(summary.get("profit factor", 0)),
        "expected_payoff":  _to_float(summary.get("expected payoff", 0)),
        "max_drawdown_abs": _to_float(summary.get("balance drawdown maximal", 0)),
        "max_drawdown_pct": _parse_dd_pct(summary.get("balance drawdown maximal", "")),
        "total_trades":     _to_int(summary.get("total trades", 0)),
        "winning_trades":   _to_int(summary.get("profit trades (% of total)", 0)),
        "losing_trades":    _to_int(summary.get("loss trades (% of total)", 0)),
        "sharpe_ratio_mt5": _to_float(summary.get("sharpe ratio", 0)),
        "recovery_factor":  _to_float(summary.get("recovery factor", 0)),
    }
    if summary_norm["total_trades"] > 0:
        summary_norm["win_rate"] = summary_norm["winning_trades"] / summary_norm["total_trades"]
    else:
        summary_norm["win_rate"] = 0.0

    # Trades table — MT5 reports have a table with header containing "Time" and "Deal"
    trades = parse_trades_table(soup)

    return {"summary": summary_norm, "trades": trades}


def _parse_dd_pct(s: str) -> float:
    """MT5 reports DD like '500.00 (5.00%)' — extract the pct."""
    m = re.search(r"\(([\d\.]+)%\)", str(s))
    return float(m.group(1)) if m else 0.0


def parse_trades_table(soup: BeautifulSoup) -> list[dict]:
    """Locate and parse the deals/orders table."""
    trades = []
    # MT5 deals table header keywords
    deal_kw = ["time", "deal", "symbol", "type", "direction", "volume", "price", "profit"]
    tables = soup.find_all("table")

    deals_table = None
    for tbl in tables:
        headers = [th.get_text(strip=True).lower() for th in tbl.find_all("th")]
        if not headers:
            # Sometimes header row is <td><b>...</b>
            first_row = tbl.find("tr")
            if first_row:
                headers = [td.get_text(strip=True).lower() for td in first_row.find_all("td")]
        if any("deal" in h for h in headers) and any("profit" in h for h in headers):
            deals_table = tbl
            break

    if deals_table is None:
        return trades

    rows = deals_table.find_all("tr")[1:]  # skip header
    for row in rows:
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) < 8:
            continue
        # Heuristic mapping — column names vary by MT5 version. Adjust if needed.
        try:
            trade = {
                "ticket": cells[1] if len(cells) > 1 else "",
                "open_time": cells[0] if len(cells) > 0 else "",
                "close_time": cells[0] if len(cells) > 0 else "",
                "symbol": cells[2] if len(cells) > 2 else "",
                "direction": cells[3].lower() if len(cells) > 3 else "",
                "lots": _to_float(cells[5]) if len(cells) > 5 else 0.0,
                "entry_price": _to_float(cells[6]) if len(cells) > 6 else 0.0,
                "exit_price": 0.0,
                "profit_directional": _to_float(cells[-1]),
                "profit_swap": 0.0,
                "profit_commission": 0.0,
                "profit_total": _to_float(cells[-1]),
            }
            trades.append(trade)
        except Exception:
            continue
    return trades


def merge_with_csv_log(parsed: dict, strategy_id: str) -> dict:
    """If StrategyFactory CSV log is available, prefer its decomposition."""
    files_dir = ROOT / "backtests" / "raw_reports" / strategy_id
    csv_logs = sorted(files_dir.rglob(f"{strategy_id}_*.csv"), reverse=True)
    if not csv_logs:
        return parsed
    try:
        df = pd.read_csv(csv_logs[0])
    except Exception:
        return parsed

    # Replace trades with CSV rows that have full decomposition
    trades = []
    for _, r in df.iterrows():
        trades.append({
            "ticket": str(r.get("magic", "")),
            "open_time": str(r.get("ts_open", "")),
            "close_time": str(r.get("ts_close", "")),
            "symbol": str(r.get("symbol", "")),
            "direction": str(r.get("direction", "")),
            "lots": float(r.get("lots", 0)),
            "entry_price": float(r.get("entry_price", 0)),
            "exit_price": float(r.get("exit_price", 0)),
            "profit_directional": float(r.get("profit_directional", 0)),
            "profit_swap": float(r.get("profit_swap", 0)),
            "profit_commission": float(r.get("profit_commission", 0)),
            "profit_total": float(r.get("profit_total", 0)),
            "exit_reason": str(r.get("exit_reason", "")),
        })
    parsed["trades"] = trades
    return parsed


def parse_report(report_path: Path, strategy_id: str, period: str) -> dict:
    if report_path.suffix.lower() == ".htm":
        parsed = parse_html_report(report_path)
    else:
        # XML fallback — minimal implementation
        raise NotImplementedError(f"XML report parsing not implemented yet: {report_path.suffix}")

    parsed = merge_with_csv_log(parsed, strategy_id)

    out = {
        "strategy_id": strategy_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "engine_version": ENGINE_VERSION,
        "period": period,
        "source_report": str(report_path),
        "summary": parsed["summary"],
        "trades": parsed["trades"],
    }

    out_dir = ROOT / "backtests" / "parsed_results" / strategy_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{period}_parsed.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def find_latest_report(strategy_id: str, period: str) -> Optional[Path]:
    """Locate the most recent report for a strategy/period."""
    base = ROOT / "backtests" / "raw_reports" / strategy_id
    if not base.exists():
        return None
    candidates = sorted(base.glob(f"{period}_*/*.htm"), key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy_id")
    ap.add_argument("--period", default="is", choices=["is", "oos", "forward"])
    ap.add_argument("--report", help="Explicit report path; overrides auto-discovery")
    args = ap.parse_args()

    if args.report:
        report = Path(args.report)
    else:
        report = find_latest_report(args.strategy_id, args.period)

    if not report or not report.exists():
        console.print(f"[red]No report found for {args.strategy_id} period={args.period}[/red]")
        return 1

    out = parse_report(report, args.strategy_id, args.period)
    console.print(Panel(
        f"[green]Parsed {len(out['trades'])} trades; net profit {out['summary']['net_profit']:.2f}[/green]\n"
        f"Saved: backtests/parsed_results/{args.strategy_id}/{args.period}_parsed.json",
        title="REPORT PARSED"
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
