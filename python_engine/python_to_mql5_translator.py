"""Python → MQL5 strategy translator.

Translates ``strategies/<name>/strategy.py`` plus the winning parameter set
selected from ``results/top_N_performers.csv`` into a `mql5/` EA that drops
into the StrategyFactory MT5 pipeline.

The translator is deliberately **not** a generic AST converter: every project
that has tried full Python-to-MQL5 AST conversion has ended up either (a)
restricting Python to a tiny DSL or (b) producing brittle, untrusted MQL5
that needs human review anyway. We take the pragmatic middle road:

1. **Auto path** — ``auto_translate_simple`` recognises a small library of
   well-known patterns (today: RSI + Bollinger Bands + ATR mean-reversion with
   session filter). When a pattern matches, the translator deterministically
   fills the four AI-marker blocks in ``BaseEA_Template.mq5`` with the
   equivalent MQL5 indicator handles, inputs and conditions.
2. **Manual path** — ``translate_strategy`` is the fallback for any pattern
   we don't recognise. It writes a translation *prompt* the trader pastes
   into Claude Code, alongside a metadata-filled .mq5 skeleton so the trader
   only has to fill in the four AI-marker blocks.

Both paths emit a ``translation_report.md`` explaining what got translated
automatically and what needs review, and (when ``--compile`` is passed)
invoke ``automation.mt5_compiler`` to compile the result on Windows.

Translation rule reference (kept here so it's grep-able in one file)::

    Python (python_engine.indicators_mql5)         MQL5 equivalent
    -------------------------------------------    --------------------------------------------
    sma(close, period=N)                            iMA(_Symbol,_Period,N,0,MODE_SMA,PRICE_CLOSE)
    ema(close, period=N)                            iMA(_Symbol,_Period,N,0,MODE_EMA,PRICE_CLOSE)
    smma(close, period=N)                           iMA(_Symbol,_Period,N,0,MODE_SMMA,PRICE_CLOSE)
    lwma(close, period=N)                           iMA(_Symbol,_Period,N,0,MODE_LWMA,PRICE_CLOSE)
    rsi(close, period=N)                            iRSI(_Symbol,_Period,N,PRICE_CLOSE)
    atr(df,   period=N)                             iATR(_Symbol,_Period,N)
    bollinger_bands(close, period=N, deviations=D)  iBands(_Symbol,_Period,N,0,D,PRICE_CLOSE)
        BB buffers: BASE=0, UPPER=1, LOWER=2
    stochastic(df,k,d,slowing)                      iStochastic(_Symbol,_Period,k,d,slowing,
                                                                MODE_SMA, STO_LOWHIGH)
    macd(close, fast,slow,signal)                   iMACD(_Symbol,_Period,fast,slow,signal,PRICE_CLOSE)

Session filter mapping::

    (time.hour >= H1) | (time.hour < H2)           IsTradingSession() helper from the template
    cross-midnight is enabled via InpSessionCrossesMidnight=true

CLI::

    python -m python_engine.python_to_mql5_translator strategies/mean_rev_1/ --rank 1
    python -m python_engine.python_to_mql5_translator strategies/mean_rev_1/ --rank 1 --compile
    python -m python_engine.python_to_mql5_translator strategies/mean_rev_1/ --rank 1 --force-manual
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = ROOT / "mql5" / "_template" / "BaseEA_Template.mq5"

# Anchor strings used to slice the EA template into "before / placeholder / after"
# blocks. Order matters: blocks appear in this order in the template.
AI_BLOCK_TAGS = [
    "inputs",        # input declarations
    "handles",       # global indicator handles
    "init",          # OnInit indicator creation
    "deinit",        # OnDeinit indicator release
    "entry",         # OnTick entry decision
    "exit",          # ManageOpenPositions custom exits
    "checks",        # CheckLongEntry / CheckShortEntry bodies
]


# ---------------------------------------------------------------------------
# Magic number — deterministic, same algorithm as automation.ea_generator
# ---------------------------------------------------------------------------

def magic_from_strategy_id(strategy_id: str) -> int:
    import hashlib

    h = hashlib.sha256(strategy_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 99999999 + 10000000


# ---------------------------------------------------------------------------
# Translation rules — the lookup table the prompt + report both quote
# ---------------------------------------------------------------------------

TRANSLATION_RULES: Dict[str, str] = {
    "sma(close, period=N)":
        "iMA(_Symbol, _Period, N, 0, MODE_SMA, PRICE_CLOSE)",
    "ema(close, period=N)":
        "iMA(_Symbol, _Period, N, 0, MODE_EMA, PRICE_CLOSE)",
    "smma(close, period=N)":
        "iMA(_Symbol, _Period, N, 0, MODE_SMMA, PRICE_CLOSE)",
    "lwma(close, period=N)":
        "iMA(_Symbol, _Period, N, 0, MODE_LWMA, PRICE_CLOSE)",
    "rsi(close, period=N)":
        "iRSI(_Symbol, _Period, N, PRICE_CLOSE)",
    "atr(df, period=N)":
        "iATR(_Symbol, _Period, N)",
    "bollinger_bands(close, period=N, deviations=D)":
        "iBands(_Symbol, _Period, N, 0, D, PRICE_CLOSE)  // buffers BASE=0, UPPER=1, LOWER=2",
    "stochastic(df, k, d, slowing)":
        "iStochastic(_Symbol, _Period, k, d, slowing, MODE_SMA, STO_LOWHIGH)",
    "macd(close, fast, slow, signal)":
        "iMACD(_Symbol, _Period, fast, slow, signal, PRICE_CLOSE)",
    "session_mask = (time.hour >= H1) | (time.hour < H2)":
        "Built-in IsTradingSession() helper; configure InpSessionStartUTC, "
        "InpSessionEndUTC, InpSessionCrossesMidnight.",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TranslationResult:
    """What got produced. ``mode`` is ``"auto"`` or ``"manual"``."""

    mode: str
    strategy_id: str
    rank: int
    params: Dict[str, Any]
    mq5_path: Path
    report_path: Path
    prompt_path: Optional[Path] = None
    pattern: Optional[str] = None
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Inputs: top-N CSV + hypothesis.yaml + strategy.py
# ---------------------------------------------------------------------------

def _read_top_performers_csv(strategy_dir: Path) -> List[Dict[str, Any]]:
    """Load every available top-N CSV in ``results/`` and return rows.

    Generator writes either ``top_10_performers.csv`` (default) or
    ``top_{N}_performers.csv``. We accept any ``top_*_performers.csv`` file.
    The returned rows are sorted by ``rank`` ascending.
    """
    results_dir = strategy_dir / "results"
    if not results_dir.exists():
        raise FileNotFoundError(
            f"No results/ folder under {strategy_dir}. Run the optimizer first."
        )
    candidates = sorted(results_dir.glob("top_*_performers.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No top_*_performers.csv under {results_dir}. Run the optimizer first."
        )
    # Prefer the largest N if multiple exist (more entries, better rank coverage).
    candidates.sort(key=lambda p: _top_n(p), reverse=True)
    csv_path = candidates[0]
    logger.info("Reading top performers from %s", csv_path)
    rows = _parse_csv(csv_path)
    rows.sort(key=lambda r: int(r.get("rank", 0)))
    return rows


def _top_n(path: Path) -> int:
    m = re.search(r"top_(\d+)_performers", path.name)
    return int(m.group(1)) if m else 0


def _parse_csv(path: Path) -> List[Dict[str, Any]]:
    import csv

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        if "params" in row and isinstance(row["params"], str):
            try:
                row["params"] = json.loads(row["params"])
            except json.JSONDecodeError:
                pass  # leave as string; caller surfaces error later
    return rows


def _load_hypothesis(strategy_dir: Path) -> Dict[str, Any]:
    import yaml

    p = strategy_dir / "hypothesis.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"hypothesis.yaml not found under {strategy_dir}. "
            "Did you run automation/generate_strategy.py first?"
        )
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _load_strategy_source(strategy_dir: Path) -> str:
    p = strategy_dir / "strategy.py"
    if not p.exists():
        raise FileNotFoundError(f"strategy.py not found under {strategy_dir}")
    return p.read_text(encoding="utf-8")


def _strategy_id_from_hypothesis(spec: Dict[str, Any]) -> str:
    sid = spec.get("meta", {}).get("strategy_id")
    if not sid:
        raise ValueError("hypothesis.yaml is missing meta.strategy_id")
    return sid


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------

_PATTERN_RSI_BB_ATR = "rsi_bb_atr_mean_reversion"

# A "MQL5_TRANSLATION_HINTS" block lets a strategy author override pattern
# detection or force the prompt path. Example::
#
#     # MQL5_TRANSLATION_HINTS
#     # pattern: rsi_bb_atr_mean_reversion
#     # force_manual: false
#
_HINTS_RE = re.compile(
    r"#\s*MQL5_TRANSLATION_HINTS\s*\n((?:\s*#.*\n)+)",
    re.MULTILINE,
)


def parse_translation_hints(source: str) -> Dict[str, str]:
    """Parse the ``# MQL5_TRANSLATION_HINTS`` comment block if present.

    Returns an empty dict if absent. Each non-empty hint line is parsed as
    ``key: value`` (whitespace tolerant).
    """
    m = _HINTS_RE.search(source)
    if not m:
        return {}
    hints: Dict[str, str] = {}
    for raw in m.group(1).splitlines():
        raw = raw.strip().lstrip("#").strip()
        if not raw or ":" not in raw:
            continue
        key, _, val = raw.partition(":")
        hints[key.strip()] = val.strip()
    return hints


def detect_pattern(source: str) -> Optional[str]:
    """Return a pattern label if ``source`` matches a known auto-translatable
    layout, else ``None``.

    Detection is deliberately conservative — we'd rather drop into the manual
    prompt than translate something we don't fully recognise.
    """
    hints = parse_translation_hints(source)
    forced = hints.get("pattern")
    if forced:
        return forced
    if (
        re.search(r"\brsi\s*\(", source)
        and re.search(r"\bbollinger_bands\s*\(", source)
        and re.search(r"\batr\s*\(", source)
        and re.search(r"signal_long", source)
        and re.search(r"signal_short", source)
    ):
        # Heuristic: looks like the RSI<oversold & close<lower mean-reversion
        # sketch shipped with the default strategy.py template.
        return _PATTERN_RSI_BB_ATR
    return None


# ---------------------------------------------------------------------------
# Template rendering (AI markers replacement)
# ---------------------------------------------------------------------------

# Pre-compiled regex matching one AI-generated block. It is non-greedy so it
# matches the smallest possible content between start and end markers; the
# template uses these markers in 7 distinct locations.
_AI_BLOCK_RE = re.compile(
    r"(// === AI GENERATED LOGIC START ===\n)(.*?)(// === AI GENERATED LOGIC END ===)",
    re.DOTALL,
)


def _replace_ai_blocks(template_text: str, replacements: List[str]) -> str:
    """Replace every AI block in order with the strings in ``replacements``.

    ``len(replacements)`` must equal the number of AI blocks in the template.
    Each replacement is the *interior* of the block (no trailing newline; the
    closing marker is added back).
    """
    blocks = list(_AI_BLOCK_RE.finditer(template_text))
    expected = len(blocks)
    if len(replacements) != expected:
        raise ValueError(
            f"AI block count mismatch: template has {expected}, "
            f"got {len(replacements)} replacements."
        )
    out: List[str] = []
    last = 0
    for blk, repl in zip(blocks, replacements):
        out.append(template_text[last : blk.start()])
        out.append(blk.group(1))  # START marker line
        body = repl if repl.endswith("\n") else repl + "\n"
        out.append(body)
        out.append(blk.group(3))  # END marker
        last = blk.end()
    out.append(template_text[last:])
    return "".join(out)


def _load_template() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def _stub_replacements() -> List[str]:
    """Replacements that leave every AI block in its empty / stub state.

    Used for the manual-prompt path so that the metadata-filled skeleton
    compiles (it doesn't trade, of course) and the trader only has to edit
    the AI sections.
    """
    return [
        # 1. inputs
        "// AI EDIT: Add strategy-specific inputs here.\n",
        # 2. handles
        "// AI EDIT: Indicator handles.\n",
        # 3. init
        "// AI EDIT: Initialize strategy indicator handles.\n",
        # 4. deinit
        "// AI EDIT: Release strategy indicator handles.\n",
        # 5. entry decision in OnTick
        "// AI EDIT: Entry decisions go here.\n"
        "   if(CheckLongEntry())  OpenPosition(POSITION_TYPE_BUY,  current_atr);\n"
        "   if(CheckShortEntry()) OpenPosition(POSITION_TYPE_SELL, current_atr);\n",
        # 6. ManageOpenPositions custom exits
        "// AI EDIT (optional): custom exit logic.\n",
        # 7. CheckLongEntry / CheckShortEntry bodies
        "// AI EDIT: Replace stub bodies below.\n"
        "bool CheckLongEntry()  { return(false); }\n"
        "bool CheckShortEntry() { return(false); }\n",
    ]


# ---------------------------------------------------------------------------
# Auto translation: RSI + Bollinger + ATR mean-reversion
# ---------------------------------------------------------------------------

_RSI_BB_ATR_REQUIRED = (
    "rsi_period",
    "rsi_oversold",
    "rsi_overbought",
    "bb_period",
    "bb_dev",
)


def _format_param_int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _format_param_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rsi_bb_atr_blocks(params: Dict[str, Any]) -> List[str]:
    """Build the seven AI-block bodies for the RSI+BB+ATR pattern.

    All numeric defaults are pulled from the winning ``params``; if a key is
    missing the helper falls back to a sane default and the caller records
    the substitution in the report.
    """
    rsi_p = _format_param_int(params.get("rsi_period"), 14)
    rsi_os = _format_param_int(params.get("rsi_oversold"), 25)
    rsi_ob = _format_param_int(params.get("rsi_overbought"), 75)
    bb_p = _format_param_int(params.get("bb_period"), 20)
    bb_d = _format_param_float(params.get("bb_dev"), 2.0)

    inputs_block = (
        "// AI: Strategy inputs (RSI + Bollinger Bands).\n"
        "input group \"=== Strategy ===\"\n"
        f"input int     InpRSIPeriod        = {rsi_p};\n"
        f"input double  InpRSIOversold      = {rsi_os}.0;\n"
        f"input double  InpRSIOverbought    = {rsi_ob}.0;\n"
        f"input int     InpBBPeriod         = {bb_p};\n"
        f"input double  InpBBDeviations     = {bb_d:.2f};\n"
    )

    handles_block = (
        "// AI: Indicator handles for RSI and Bollinger Bands.\n"
        "int g_rsi_handle = INVALID_HANDLE;\n"
        "int g_bb_handle  = INVALID_HANDLE;\n"
        "#define BB_BASE  0\n"
        "#define BB_UPPER 1\n"
        "#define BB_LOWER 2\n"
    )

    init_block = (
        "// AI: Create RSI + BB handles.\n"
        "   g_rsi_handle = iRSI(_Symbol, _Period, InpRSIPeriod, PRICE_CLOSE);\n"
        "   if(g_rsi_handle == INVALID_HANDLE) return(INIT_FAILED);\n"
        "   g_bb_handle  = iBands(_Symbol, _Period, InpBBPeriod, 0, InpBBDeviations, PRICE_CLOSE);\n"
        "   if(g_bb_handle == INVALID_HANDLE) return(INIT_FAILED);\n"
    )

    deinit_block = (
        "// AI: Release RSI + BB handles.\n"
        "   if(g_rsi_handle != INVALID_HANDLE) IndicatorRelease(g_rsi_handle);\n"
        "   if(g_bb_handle  != INVALID_HANDLE) IndicatorRelease(g_bb_handle);\n"
    )

    entry_block = (
        "// AI: Mean-reversion entry decision.\n"
        "   if(CheckLongEntry())  OpenPosition(POSITION_TYPE_BUY,  current_atr);\n"
        "   if(CheckShortEntry()) OpenPosition(POSITION_TYPE_SELL, current_atr);\n"
    )

    exit_block = "// AI: No custom exit beyond SL/TP/time exit.\n"

    checks_block = (
        "// AI: CheckLongEntry / CheckShortEntry — match strategy.py logic.\n"
        "bool CheckLongEntry()\n"
        "  {\n"
        "   double rsi[];\n"
        "   if(CopyBuffer(g_rsi_handle, 0, 1, 1, rsi) <= 0) return(false);\n"
        "   double lower[];\n"
        "   if(CopyBuffer(g_bb_handle, BB_LOWER, 1, 1, lower) <= 0) return(false);\n"
        "   double close_prev = iClose(_Symbol, _Period, 1);\n"
        "   if(rsi[0] < InpRSIOversold && close_prev < lower[0]) return(true);\n"
        "   return(false);\n"
        "  }\n"
        "\n"
        "bool CheckShortEntry()\n"
        "  {\n"
        "   double rsi[];\n"
        "   if(CopyBuffer(g_rsi_handle, 0, 1, 1, rsi) <= 0) return(false);\n"
        "   double upper[];\n"
        "   if(CopyBuffer(g_bb_handle, BB_UPPER, 1, 1, upper) <= 0) return(false);\n"
        "   double close_prev = iClose(_Symbol, _Period, 1);\n"
        "   if(rsi[0] > InpRSIOverbought && close_prev > upper[0]) return(true);\n"
        "   return(false);\n"
        "  }\n"
    )

    return [
        inputs_block,
        handles_block,
        init_block,
        deinit_block,
        entry_block,
        exit_block,
        checks_block,
    ]


# ---------------------------------------------------------------------------
# Skeleton renderer (used by both auto + manual paths)
# ---------------------------------------------------------------------------

def _fill_metadata(template_text: str, strategy_id: str, magic: int) -> str:
    strategy_name = strategy_id.replace("_", " ").title()
    return (
        template_text
        .replace("{{STRATEGY_ID}}", strategy_id)
        .replace("{{STRATEGY_NAME}}", strategy_name)
        .replace("{{STRATEGY_MAGIC}}", str(magic))
    )


def _apply_exit_inputs(
    template_text: str,
    sl_atr_mult: Optional[float],
    tp_atr_mult: Optional[float],
    time_exit_bars: Optional[int],
) -> str:
    """Bake the SL/TP/time-exit defaults from the spec into the input section.

    Editing the ``input`` defaults is technically outside the AI-marker blocks
    so we limit ourselves to the four lines that are explicitly templated; if
    the template ever changes we fall back to leaving the defaults alone.
    """

    def _replace_default(text: str, name: str, new_value: str) -> str:
        # Match an "input <type>  <name>   = <default>;" line where <default>
        # is the original number. Width-preserving replacement keeps git diffs
        # tidy and avoids breaking column alignment.
        pattern = re.compile(
            rf"(input\s+\w+\s+{re.escape(name)}\s*=\s*)([0-9.\-]+)(\s*;)"
        )
        return pattern.sub(rf"\g<1>{new_value}\g<3>", text, count=1)

    out = template_text
    if sl_atr_mult is not None:
        out = _replace_default(out, "InpStopLossATRMult", f"{float(sl_atr_mult):.2f}")
    if tp_atr_mult is not None:
        out = _replace_default(out, "InpTakeProfitATRMult", f"{float(tp_atr_mult):.2f}")
    if time_exit_bars is not None:
        out = _replace_default(out, "InpMaxHoldingBars", str(int(time_exit_bars)))
    return out


def _apply_session_inputs(template_text: str, spec: Dict[str, Any]) -> str:
    sess = spec.get("universe", {}).get("trading_session_utc", {}) or {}
    start = sess.get("start")
    end = sess.get("end")
    crosses = sess.get("crosses_midnight")
    if not (start and end):
        return template_text
    out = template_text
    out = re.sub(
        r'(input\s+string\s+InpSessionStartUTC\s*=\s*)"[^"]*"',
        rf'\g<1>"{start}"',
        out,
        count=1,
    )
    out = re.sub(
        r'(input\s+string\s+InpSessionEndUTC\s*=\s*)"[^"]*"',
        rf'\g<1>"{end}"',
        out,
        count=1,
    )
    if crosses is not None:
        out = re.sub(
            r"(input\s+bool\s+InpSessionCrossesMidnight\s*=\s*)(true|false)",
            rf"\g<1>{'true' if crosses else 'false'}",
            out,
            count=1,
        )
    return out


# ---------------------------------------------------------------------------
# Prompt builder (manual path)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """# Translation prompt — `{strategy_id}` (rank {rank})

You are translating a vectorized Python research strategy into a MetaTrader 5
Expert Advisor that fits into the StrategyFactory pipeline.

## Hard rules
1. Edit **only** between `// === AI GENERATED LOGIC START ===` and
   `// === AI GENERATED LOGIC END ===` markers in the skeleton below.
2. Use **only** MQL5 built-ins (`iRSI`, `iBands`, `iATR`, `iMA`, `iMACD`,
   `iStochastic`). They have the same arithmetic as the Python
   `python_engine.indicators_mql5` functions used in the source — this is
   verified by `tests/test_indicators_mql5.py`.
3. Read indicator buffers at index **1** (last *closed* bar). Never read
   index 0 — that's the still-forming bar.
4. Do not modify includes, OnInit/OnDeinit/OnTick scaffolding, RiskManager,
   Logger, PnLDecomposer, or `IsTradingSession()`.
5. The winning parameter set is given in the table below. Bake those values
   into the `input` defaults you add; do not invent new parameters.

## Strategy metadata
- `strategy_id`: `{strategy_id}`
- `magic`: `{magic}`
- Source file: `strategies/{strategy_name}/strategy.py`
- Detected pattern: `{pattern}`

## Winning parameters (rank {rank})
```json
{params_json}
```

## Translation rules (Python → MQL5)
{rules_block}

## Source `strategy.py`
```python
{strategy_source}
```

## Skeleton (already metadata-filled; edit only AI markers)
```mql5
{skeleton}
```

## Deliverable
Save the completed file at:

    strategies/{strategy_name}/mql5_translation/{strategy_id}.mq5

Then run:

    python -m python_engine.python_to_mql5_translator strategies/{strategy_name}/ --rank {rank} --compile

…to compile via `metaeditor64.exe` (Windows VPS only).
"""


def _rules_block() -> str:
    return "\n".join(
        f"- `{python}` → `{mql5}`" for python, mql5 in TRANSLATION_RULES.items()
    )


def _build_prompt(
    *,
    strategy_id: str,
    strategy_name: str,
    rank: int,
    magic: int,
    params: Dict[str, Any],
    strategy_source: str,
    skeleton: str,
    pattern: Optional[str],
) -> str:
    return PROMPT_TEMPLATE.format(
        strategy_id=strategy_id,
        strategy_name=strategy_name,
        rank=rank,
        magic=magic,
        params_json=json.dumps(params, indent=2, sort_keys=True),
        rules_block=_rules_block(),
        strategy_source=strategy_source,
        skeleton=skeleton,
        pattern=pattern or "unknown (manual translation required)",
    )


# ---------------------------------------------------------------------------
# Report builder (both paths)
# ---------------------------------------------------------------------------

def _build_report(
    *,
    mode: str,
    strategy_id: str,
    rank: int,
    params: Dict[str, Any],
    pattern: Optional[str],
    notes: List[str],
    mq5_path: Path,
    prompt_path: Optional[Path],
) -> str:
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    auto_or_manual = (
        "**Auto translation** — pattern recognised, EA filled deterministically."
        if mode == "auto"
        else "**Manual translation** — prompt written for Claude Code review."
    )
    review_block = (
        "- Indicator periods, oversold/overbought levels, BB deviation\n"
        "- Session window matches `hypothesis.yaml universe.trading_session_utc`\n"
        "- SL/TP/ATR multiples and `InpMaxHoldingBars` match the winning trial\n"
    )
    if mode != "auto":
        review_block += (
            "- AI marker blocks: confirm they only contain logic, "
            "no scaffold edits\n"
            "- Indicator buffer reads use shift = 1 (last *closed* bar)\n"
        )
    notes_block = (
        "\n".join(f"- {note}" for note in notes) if notes else "_None._"
    )
    return f"""# Translation report — `{strategy_id}`

**Generated:** {timestamp}
**Rank used:** {rank}
**Pattern detected:** `{pattern or 'unknown'}`
**Mode:** {mode}

{auto_or_manual}

## Winning parameters (rank {rank})
```json
{json.dumps(params, indent=2, sort_keys=True)}
```

## Files produced
- EA: `{mq5_path}`
{('- Prompt: `' + str(prompt_path) + '`') if prompt_path else ''}

## Required human review
{review_block}

## Notes
{notes_block}

## Next steps
1. Open the EA file and verify every AI marker block.
2. Run `python -m python_engine.python_to_mql5_translator <strategy_dir> --rank {rank} --compile`
   on the Windows VPS to invoke `metaeditor64.exe`.
3. After compile success, hand off to the standard pipeline:
   `python automation/pipeline.py backtest {strategy_id} --period all`
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _select_winner(
    rows: List[Dict[str, Any]], rank: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Pick the row with ``rank == rank``, return ``(row, params_dict)``."""
    if not rows:
        raise ValueError("top performers CSV is empty")
    for r in rows:
        try:
            r_rank = int(r.get("rank", 0))
        except (TypeError, ValueError):
            continue
        if r_rank == rank:
            params = r.get("params") or {}
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except json.JSONDecodeError:
                    params = {}
            return r, dict(params)
    raise ValueError(
        f"rank {rank} not found in CSV (have ranks "
        f"{[r.get('rank') for r in rows]})"
    )


def _ensure_required_params(
    params: Dict[str, Any], required: Tuple[str, ...]
) -> List[str]:
    missing = [k for k in required if k not in params]
    return missing


def _strategy_name(strategy_dir: Path) -> str:
    return strategy_dir.resolve().name


def _output_paths(
    strategy_dir: Path, strategy_id: str
) -> Tuple[Path, Path, Path]:
    out_dir = strategy_dir / "mql5_translation"
    out_dir.mkdir(parents=True, exist_ok=True)
    return (
        out_dir / f"{strategy_id}.mq5",
        out_dir / "translation_prompt.md",
        out_dir / "translation_report.md",
    )


def translate_strategy(
    strategy_dir: Path,
    rank: int = 1,
    *,
    compile_after: bool = False,
    force_manual: bool = False,
) -> TranslationResult:
    """Entry point — try auto first, fall back to manual prompt.

    Side effects:
    - Writes ``mql5_translation/<strategy_id>.mq5`` (auto or skeleton).
    - For manual path: ``mql5_translation/translation_prompt.md``.
    - Always writes ``mql5_translation/translation_report.md``.
    - If ``compile_after`` and ``mode == auto``: invokes
      ``automation.mt5_compiler.compile_ea``.
    """
    strategy_dir = Path(strategy_dir).resolve()
    if not strategy_dir.exists():
        raise FileNotFoundError(strategy_dir)

    spec = _load_hypothesis(strategy_dir)
    strategy_id = _strategy_id_from_hypothesis(spec)
    rows = _read_top_performers_csv(strategy_dir)
    _, params = _select_winner(rows, rank)
    source = _load_strategy_source(strategy_dir)

    hints = parse_translation_hints(source)
    if not force_manual and str(hints.get("force_manual", "")).lower() == "true":
        force_manual = True

    pattern = None if force_manual else detect_pattern(source)
    if pattern == _PATTERN_RSI_BB_ATR:
        return auto_translate_simple(
            strategy_dir,
            rank=rank,
            compile_after=compile_after,
            _preloaded=(spec, rows, params, source),
        )
    # Manual path
    return _emit_manual_prompt(
        strategy_dir=strategy_dir,
        strategy_id=strategy_id,
        rank=rank,
        params=params,
        spec=spec,
        source=source,
        pattern=pattern,
    )


def auto_translate_simple(
    strategy_dir: Path,
    rank: int = 1,
    *,
    compile_after: bool = False,
    _preloaded: Optional[
        Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], str]
    ] = None,
) -> TranslationResult:
    """Deterministic translation for recognised patterns only.

    If the strategy doesn't match a known pattern, this function falls back
    to ``translate_strategy`` (manual prompt path).
    """
    strategy_dir = Path(strategy_dir).resolve()
    if _preloaded is None:
        spec = _load_hypothesis(strategy_dir)
        rows = _read_top_performers_csv(strategy_dir)
        _, params = _select_winner(rows, rank)
        source = _load_strategy_source(strategy_dir)
    else:
        spec, _rows, params, source = _preloaded

    strategy_id = _strategy_id_from_hypothesis(spec)
    pattern = detect_pattern(source)
    if pattern != _PATTERN_RSI_BB_ATR:
        logger.info(
            "auto_translate_simple: pattern %r not auto-translatable; "
            "falling back to manual prompt.",
            pattern,
        )
        return _emit_manual_prompt(
            strategy_dir=strategy_dir,
            strategy_id=strategy_id,
            rank=rank,
            params=params,
            spec=spec,
            source=source,
            pattern=pattern,
        )

    missing = _ensure_required_params(params, _RSI_BB_ATR_REQUIRED)
    notes: List[str] = []
    if missing:
        notes.append(
            "Missing params filled with defaults: " + ", ".join(missing)
        )

    blocks = _rsi_bb_atr_blocks(params)
    template = _load_template()
    magic = magic_from_strategy_id(strategy_id)
    template = _fill_metadata(template, strategy_id, magic)
    template = _apply_session_inputs(template, spec)
    template = _apply_exit_inputs(
        template,
        sl_atr_mult=params.get("sl_atr_mult"),
        tp_atr_mult=params.get("tp_atr_mult"),
        time_exit_bars=params.get("time_exit_bars"),
    )
    filled = _replace_ai_blocks(template, blocks)

    mq5_path, _, report_path = _output_paths(strategy_dir, strategy_id)
    mq5_path.write_text(filled, encoding="utf-8")
    report_path.write_text(
        _build_report(
            mode="auto",
            strategy_id=strategy_id,
            rank=rank,
            params=params,
            pattern=pattern,
            notes=notes,
            mq5_path=mq5_path,
            prompt_path=None,
        ),
        encoding="utf-8",
    )

    if compile_after:
        _try_compile(mq5_path, report_path)

    return TranslationResult(
        mode="auto",
        strategy_id=strategy_id,
        rank=rank,
        params=params,
        mq5_path=mq5_path,
        report_path=report_path,
        pattern=pattern,
        notes=notes,
    )


def _emit_manual_prompt(
    *,
    strategy_dir: Path,
    strategy_id: str,
    rank: int,
    params: Dict[str, Any],
    spec: Dict[str, Any],
    source: str,
    pattern: Optional[str],
) -> TranslationResult:
    template = _load_template()
    magic = magic_from_strategy_id(strategy_id)
    template = _fill_metadata(template, strategy_id, magic)
    template = _apply_session_inputs(template, spec)
    template = _apply_exit_inputs(
        template,
        sl_atr_mult=params.get("sl_atr_mult"),
        tp_atr_mult=params.get("tp_atr_mult"),
        time_exit_bars=params.get("time_exit_bars"),
    )
    skeleton = _replace_ai_blocks(template, _stub_replacements())

    prompt = _build_prompt(
        strategy_id=strategy_id,
        strategy_name=_strategy_name(strategy_dir),
        rank=rank,
        magic=magic,
        params=params,
        strategy_source=source,
        skeleton=skeleton,
        pattern=pattern,
    )

    mq5_path, prompt_path, report_path = _output_paths(strategy_dir, strategy_id)
    mq5_path.write_text(skeleton, encoding="utf-8")
    prompt_path.write_text(prompt, encoding="utf-8")
    report_path.write_text(
        _build_report(
            mode="manual",
            strategy_id=strategy_id,
            rank=rank,
            params=params,
            pattern=pattern,
            notes=[
                "Manual translation required — feed the prompt file to "
                "Claude Code in VS Code.",
            ],
            mq5_path=mq5_path,
            prompt_path=prompt_path,
        ),
        encoding="utf-8",
    )

    return TranslationResult(
        mode="manual",
        strategy_id=strategy_id,
        rank=rank,
        params=params,
        mq5_path=mq5_path,
        report_path=report_path,
        prompt_path=prompt_path,
        pattern=pattern,
    )


# ---------------------------------------------------------------------------
# Optional compile step
# ---------------------------------------------------------------------------

def _try_compile(mq5_path: Path, report_path: Path) -> None:
    """Invoke ``automation.mt5_compiler`` and append the outcome to the report.

    Compile only works on Windows (metaeditor64.exe). On macOS / Linux the
    config path is missing and we silently note that in the report.
    """
    try:
        from automation.mt5_compiler import compile_ea  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cannot import mt5_compiler: %s", exc)
        _append_to_report(
            report_path,
            f"\n## Compile attempt\n_Skipped — mt5_compiler import failed: {exc}_\n",
        )
        return

    try:
        result = compile_ea(mq5_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("compile_ea raised %s", exc)
        _append_to_report(
            report_path,
            f"\n## Compile attempt\n_Failed — exception: {exc}_\n",
        )
        return

    if result.success:
        body = f"✅ Compiled in {result.compile_time_sec:.2f}s → `{result.ex5_path}`"
    else:
        bullets = "\n".join(f"  - `{e}`" for e in result.errors[:10])
        body = f"❌ Compile failed.\n\n{bullets}"
    _append_to_report(report_path, f"\n## Compile attempt\n{body}\n")


def _append_to_report(report_path: Path, body: str) -> None:
    with report_path.open("a", encoding="utf-8") as f:
        f.write(body)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m python_engine.python_to_mql5_translator",
        description=(
            "Translate a Python strategy + winning parameter set into an "
            "MQL5 Expert Advisor for the StrategyFactory pipeline."
        ),
    )
    p.add_argument("strategy_dir", type=Path,
                   help="Path to strategies/<name>/")
    p.add_argument("--rank", type=int, default=1,
                   help="Which top performer to translate (1 = best). "
                        "Default: 1.")
    p.add_argument("--compile", dest="compile_after", action="store_true",
                   help="After writing the .mq5, invoke automation.mt5_compiler "
                        "(Windows VPS only).")
    p.add_argument("--force-manual", action="store_true",
                   help="Skip pattern detection and always emit a prompt.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        result = translate_strategy(
            strategy_dir=args.strategy_dir,
            rank=args.rank,
            compile_after=args.compile_after,
            force_manual=args.force_manual,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"Translated [{result.mode}] {result.strategy_id} rank={result.rank}\n"
        f"  EA:     {result.mq5_path}\n"
        f"  Report: {result.report_path}"
        + (f"\n  Prompt: {result.prompt_path}" if result.prompt_path else "")
    )
    if result.mode == "manual":
        print(
            "\nNext: open the prompt file, paste into Claude Code, save the "
            "completed EA at the path above, then re-run with --compile."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
