"""
StrategyFactory — Hypothesis Logger
===================================
Append-only writer for docs/HYPOTHESIS_LOG.md.
Every PASS or FAIL verdict is logged here as a research diary entry.

This is the "lab notebook" — failed hypotheses are research data and
become YouTube content. Always log.

Future: feeds Obsidian vault when integration is added.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
LOG = ROOT / "docs" / "HYPOTHESIS_LOG.md"


def append_entry(verdict: dict) -> None:
    """Append a markdown entry for one strategy verdict."""
    LOG.parent.mkdir(parents=True, exist_ok=True)

    if not LOG.exists():
        LOG.write_text(_header(), encoding="utf-8")

    block = _format_entry(verdict)
    with LOG.open("a", encoding="utf-8") as f:
        f.write("\n\n" + block)


def _header() -> str:
    return """# Hypothesis Log — StrategyFactory

> Append-only research diary. Every hypothesis tested ends here, PASS or FAIL.
> Failed hypotheses are research data — never delete entries.
>
> Format per entry:
> ```
> ## YYYY-MM-DD — strategy_id — VERDICT
>     date: ...
>     strategy_id: ...
>     verdict: PASS | FAIL
>     hypothesis: ...
>     key metrics, p-values, decomposition...
> ```
>
> This file is also designed to be ingested by Obsidian (future integration).
"""


def _format_entry(v: dict) -> str:
    sid = v.get("strategy_id", "UNKNOWN")
    verdict = v.get("verdict", "UNKNOWN")
    date = datetime.utcnow().strftime("%Y-%m-%d")
    color = "✅" if verdict == "PASS" else "❌"

    lines = [f"## {date} — {sid} — {verdict} {color}", ""]
    lines.append("```yaml")
    lines.append(f"date: {date}")
    lines.append(f"strategy_id: {sid}")
    lines.append(f"verdict: {verdict}")
    lines.append(f"checks_passed: {v.get('checks_passed', 0)}")
    lines.append(f"checks_failed: {v.get('checks_failed', 0)}")

    # Extract a few key numbers from checks for quick scanning
    for c in v.get("checks", []):
        name = c.get("name", "")
        val = c.get("value")
        if val is None: continue
        if "p-value" in name.lower():
            key = "p_value_sharpe" if "IS bootstrap" in name else "p_value_sharpe_oos"
            lines.append(f"{key}: {val}")
        if "IS Sharpe" in name:
            lines.append(f"is_sharpe: {val}")
        if "OOS Sharpe" in name:
            lines.append(f"oos_sharpe: {val}")
        if "WFA efficiency" in name:
            lines.append(f"wfa_efficiency: {val}")
        if "Directional PnL" in name:
            lines.append(f"directional_pct: {val}")

    lines.append("```")

    if verdict == "FAIL":
        failed = [c["name"] for c in v.get("checks", []) if not c["passed"]]
        lines.append("\n**Failed checks:**")
        for f in failed:
            lines.append(f"- {f}")

    lines.append("\n**Artifacts:**")
    lines.append(f"- Spec: `{v.get('spec_path', '?')}`")
    lines.append(f"- Verdict JSON: `backtests/parsed_results/{sid}/acceptance_verdict.json`")

    return "\n".join(lines)
