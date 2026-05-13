"""Tests for ``python_engine.python_to_mql5_translator``.

Covers:
1. Pattern detection on the RSI+BB+ATR mean-reversion sketch + on unknown code.
2. Translation-hints comment block parsing (forced pattern / force_manual).
3. ``auto_translate_simple`` end-to-end: skeleton fills with winning params,
   AI blocks all replaced, file compiles as MQL5 syntax (we don't have an
   MQL5 compiler in CI, so we only verify the structural invariants).
4. Manual prompt path: prompt file written, .mq5 stays in stub state,
   report mentions manual translation.
5. CSV loading: ranks are honored, missing files raise.
6. Magic-number determinism (matches ``automation.ea_generator``).
"""
from __future__ import annotations

import json
import sys
import textwrap
from datetime import date
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from automation.ea_generator import magic_from_strategy_id as ea_gen_magic  # noqa: E402
from automation.generate_strategy import generate_strategy_skeleton  # noqa: E402
from python_engine.python_to_mql5_translator import (  # noqa: E402
    _AI_BLOCK_RE,
    _PATTERN_RSI_BB_ATR,
    TRANSLATION_RULES,
    _replace_ai_blocks,
    auto_translate_simple,
    detect_pattern,
    magic_from_strategy_id,
    parse_translation_hints,
    translate_strategy,
)


def _real_block_count(ea_text: str) -> int:
    """Count real AI blocks in an EA file, ignoring marker references inside
    doc comments (the template header quotes the marker text)."""
    return len(_AI_BLOCK_RE.findall(ea_text))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def strategy_dir(tmp_path: Path) -> Path:
    """Spin up a full strategies/<name>/ skeleton via the official generator.

    Using the real generator (rather than hand-rolling YAML) keeps the test
    in sync with whatever the generator emits today — including the spec
    schema fields the translator depends on.
    """
    paper = tmp_path / "p.txt"
    paper.write_text(
        "Mean-reversion in the FX Asian session. Sell upper BB / buy lower BB.",
        encoding="utf-8",
    )
    folder = generate_strategy_skeleton(
        paper_path=paper,
        name="mean_rev_test",
        logic_description=(
            "RSI<25 + BB lower entry, intraday Asian session 22-06 UTC, "
            "max 16 bars"
        ),
        symbols=["EURUSD", "USDJPY"],
        is_period=(date(2018, 1, 1), date(2023, 12, 31)),
        oos_period=(date(2024, 1, 1), date(2026, 4, 30)),
        base_dir=tmp_path / "strategies",
    )
    return folder


@pytest.fixture
def winning_params() -> dict:
    """A reasonable best-trial param dict (mirrors what optuna_optimizer writes)."""
    return {
        "rsi_period": 14,
        "rsi_oversold": 22,
        "rsi_overbought": 78,
        "bb_period": 24,
        "bb_dev": 2.3,
        "sl_atr_mult": 1.6,
        "tp_atr_mult": 2.4,
        "time_exit_bars": 12,
    }


def _write_top_csv(folder: Path, params: dict, *, rank: int = 1) -> Path:
    """Write a minimal top-N CSV with one row, matching optuna_optimizer schema."""
    csv = folder / "results" / "top_10_performers.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "rank,trial_id,params,is_sharpe_avg,oos_sharpe_avg,is_sharpe_min,"
        "oos_sharpe_min,max_dd_pct,total_trades,objective_value\n"
    )
    params_json = json.dumps(params, sort_keys=True).replace('"', '""')
    row = (
        f'{rank},0,"{params_json}",1.5,1.2,1.0,0.9,7.2,250,1.20\n'
    )
    csv.write_text(header + row, encoding="utf-8")
    return csv


def _patch_strategy_py(folder: Path, body: str) -> None:
    """Overwrite strategy.py with a known-pattern body (RSI+BB+ATR sketch)."""
    (folder / "strategy.py").write_text(body, encoding="utf-8")


_RSI_BB_ATR_SOURCE = '''"""Strategy: mean_rev_test"""
from __future__ import annotations

import pandas as pd
from python_engine.indicators_mql5 import rsi, atr, bollinger_bands

DEFAULT_PARAMS = {
    "rsi_period": 14, "rsi_oversold": 25, "rsi_overbought": 75,
    "bb_period": 20, "bb_dev": 2.0,
    "sl_atr_mult": 1.5, "tp_atr_mult": 2.0, "time_exit_bars": 16,
}


def signals(df, **params):
    p = {**DEFAULT_PARAMS, **params}
    rsi_val = rsi(df["close"], period=p["rsi_period"])
    mid, upper, lower = bollinger_bands(
        df["close"], period=p["bb_period"], deviations=p["bb_dev"],
    )
    atr_val = atr(df, period=14)
    signal_long  = (rsi_val < p["rsi_oversold"])  & (df["close"] < lower)
    signal_short = (rsi_val > p["rsi_overbought"]) & (df["close"] > upper)
    return {
        "signal_long": signal_long.fillna(False).astype(bool),
        "signal_short": signal_short.fillna(False).astype(bool),
        "sl_distance": atr_val * p["sl_atr_mult"],
        "tp_distance": atr_val * p["tp_atr_mult"],
        "max_holding_bars": p["time_exit_bars"],
    }
'''


_UNKNOWN_SOURCE = '''"""Strategy: weird"""
import pandas as pd

def signals(df, **params):
    custom_series = (df["close"].pct_change() * 1000).abs()
    signal_long = custom_series > 5
    signal_short = ~signal_long
    return {"signal_long": signal_long, "signal_short": signal_short,
            "sl_distance": 0.001, "tp_distance": 0.001}
'''


# ---------------------------------------------------------------------------
# 1. Pattern detection
# ---------------------------------------------------------------------------

def test_detect_pattern_rsi_bb_atr():
    assert detect_pattern(_RSI_BB_ATR_SOURCE) == _PATTERN_RSI_BB_ATR


def test_detect_pattern_unknown_returns_none():
    assert detect_pattern(_UNKNOWN_SOURCE) is None


def test_parse_hints_extracts_keys():
    src = textwrap.dedent(
        """\
        # MQL5_TRANSLATION_HINTS
        # pattern: rsi_bb_atr_mean_reversion
        # force_manual: false
        # note: anything else
        def signals(): pass
        """
    )
    hints = parse_translation_hints(src)
    assert hints["pattern"] == "rsi_bb_atr_mean_reversion"
    assert hints["force_manual"] == "false"
    assert hints["note"] == "anything else"


def test_hints_override_pattern_detection():
    src = "# MQL5_TRANSLATION_HINTS\n# pattern: rsi_bb_atr_mean_reversion\n"
    assert detect_pattern(src) == _PATTERN_RSI_BB_ATR


# ---------------------------------------------------------------------------
# 2. AI-block replacement primitive
# ---------------------------------------------------------------------------

def test_replace_ai_blocks_counts_match():
    tpl_text = (ROOT / "mql5" / "_template" / "BaseEA_Template.mq5").read_text(
        encoding="utf-8"
    )
    # Real template has 7 AI blocks; mismatch must raise.
    with pytest.raises(ValueError, match="AI block count mismatch"):
        _replace_ai_blocks(tpl_text, ["only one"])


def test_replace_ai_blocks_substitutes_in_order():
    tpl = (
        "alpha\n"
        "// === AI GENERATED LOGIC START ===\n"
        "old1\n"
        "// === AI GENERATED LOGIC END ===\n"
        "beta\n"
        "// === AI GENERATED LOGIC START ===\n"
        "old2\n"
        "// === AI GENERATED LOGIC END ===\n"
        "gamma\n"
    )
    out = _replace_ai_blocks(tpl, ["NEW1\n", "NEW2\n"])
    assert "NEW1" in out and "NEW2" in out
    assert "old1" not in out and "old2" not in out
    assert out.count("// === AI GENERATED LOGIC START ===") == 2
    assert out.count("// === AI GENERATED LOGIC END ===") == 2


# ---------------------------------------------------------------------------
# 3. Auto translation (RSI + BB + ATR)
# ---------------------------------------------------------------------------

def test_auto_translate_simple_rsi_bb_atr(strategy_dir: Path, winning_params: dict):
    _write_top_csv(strategy_dir, winning_params)
    _patch_strategy_py(strategy_dir, _RSI_BB_ATR_SOURCE)

    result = auto_translate_simple(strategy_dir, rank=1)

    assert result.mode == "auto"
    assert result.pattern == _PATTERN_RSI_BB_ATR
    assert result.mq5_path.exists()
    assert result.report_path.exists()
    assert result.prompt_path is None  # no prompt on auto path

    ea = result.mq5_path.read_text(encoding="utf-8")
    # All AI blocks must be filled — no stub bodies remain. (The template
    # *header* mentions "AI EDIT" as documentation; we only care about the
    # bodies of the marker-delimited blocks.)
    block_bodies = [m.group(2) for m in _AI_BLOCK_RE.finditer(ea)]
    assert block_bodies, "no AI blocks found"
    for body in block_bodies:
        assert "AI EDIT:" not in body, f"stub body left in: {body[:100]!r}"
    assert "return(false); // stub" not in ea
    # Winning params baked into inputs
    assert "InpRSIPeriod        = 14" in ea
    assert "InpRSIOversold      = 22.0" in ea
    assert "InpRSIOverbought    = 78.0" in ea
    assert "InpBBPeriod         = 24" in ea
    assert "InpBBDeviations     = 2.30" in ea
    # Exit defaults applied
    assert "InpStopLossATRMult    = 1.60" in ea
    assert "InpTakeProfitATRMult  = 2.40" in ea
    assert "InpMaxHoldingBars     = 12" in ea
    # Session inherited from hypothesis.yaml
    assert 'InpSessionStartUTC    = "22:00"' in ea
    assert 'InpSessionEndUTC      = "06:00"' in ea
    assert "InpSessionCrossesMidnight = true" in ea
    # Indicator wiring
    assert "iRSI(_Symbol, _Period, InpRSIPeriod, PRICE_CLOSE)" in ea
    assert "iBands(_Symbol, _Period, InpBBPeriod, 0, InpBBDeviations, PRICE_CLOSE)" in ea
    # Strategy metadata
    assert 'STRATEGY_ID    "' in ea
    assert "STRATEGY_MAGIC" in ea
    # File is well-formed: still has all 7 real AI block pairs.
    # (The template comment header *quotes* the marker as documentation, so
    # naive .count() returns 8; we want the real block count.)
    assert _real_block_count(ea) == 7


def test_auto_translate_report_mentions_pattern(
    strategy_dir: Path, winning_params: dict,
):
    _write_top_csv(strategy_dir, winning_params)
    _patch_strategy_py(strategy_dir, _RSI_BB_ATR_SOURCE)
    result = auto_translate_simple(strategy_dir, rank=1)
    report = result.report_path.read_text(encoding="utf-8")
    assert _PATTERN_RSI_BB_ATR in report
    assert "Auto translation" in report
    assert json.dumps(winning_params["bb_period"]) in report


def test_auto_translate_missing_param_falls_back_to_default(
    strategy_dir: Path,
):
    """If the optimizer never tuned a param, the translator uses a default and
    notes it in the report."""
    minimal = {"rsi_period": 14, "bb_period": 20, "bb_dev": 2.0}
    _write_top_csv(strategy_dir, minimal)
    _patch_strategy_py(strategy_dir, _RSI_BB_ATR_SOURCE)
    result = auto_translate_simple(strategy_dir, rank=1)
    assert any("Missing params" in n for n in result.notes)
    ea = result.mq5_path.read_text(encoding="utf-8")
    # Defaults from _rsi_bb_atr_blocks: rsi_oversold=25, rsi_overbought=75
    assert "InpRSIOversold      = 25.0" in ea
    assert "InpRSIOverbought    = 75.0" in ea


# ---------------------------------------------------------------------------
# 4. Manual prompt fallback
# ---------------------------------------------------------------------------

def test_translate_strategy_unknown_pattern_emits_prompt(
    strategy_dir: Path, winning_params: dict,
):
    _write_top_csv(strategy_dir, winning_params)
    _patch_strategy_py(strategy_dir, _UNKNOWN_SOURCE)

    result = translate_strategy(strategy_dir, rank=1)

    assert result.mode == "manual"
    assert result.prompt_path is not None and result.prompt_path.exists()
    prompt = result.prompt_path.read_text(encoding="utf-8")
    # Prompt embeds the source code and a translation rules cheat-sheet.
    assert "def signals" in prompt
    assert "Translation rules" in prompt
    for python_call in TRANSLATION_RULES:
        assert python_call in prompt
    # Skeleton .mq5 was still written; AI blocks remain in stub form.
    ea = result.mq5_path.read_text(encoding="utf-8")
    assert "AI EDIT" in ea
    assert _real_block_count(ea) == 7


def test_force_manual_flag_skips_pattern_detection(
    strategy_dir: Path, winning_params: dict,
):
    _write_top_csv(strategy_dir, winning_params)
    _patch_strategy_py(strategy_dir, _RSI_BB_ATR_SOURCE)
    result = translate_strategy(strategy_dir, rank=1, force_manual=True)
    assert result.mode == "manual"
    assert result.prompt_path is not None


def test_hints_force_manual_blocks_auto(
    strategy_dir: Path, winning_params: dict,
):
    body = (
        "# MQL5_TRANSLATION_HINTS\n"
        "# force_manual: true\n\n"
        + _RSI_BB_ATR_SOURCE
    )
    _write_top_csv(strategy_dir, winning_params)
    _patch_strategy_py(strategy_dir, body)
    result = translate_strategy(strategy_dir, rank=1)
    assert result.mode == "manual"


# ---------------------------------------------------------------------------
# 5. CSV loading + rank selection
# ---------------------------------------------------------------------------

def test_translate_strategy_missing_csv_raises(strategy_dir: Path):
    _patch_strategy_py(strategy_dir, _RSI_BB_ATR_SOURCE)
    with pytest.raises(FileNotFoundError):
        translate_strategy(strategy_dir, rank=1)


def test_translate_strategy_rank_not_found(
    strategy_dir: Path, winning_params: dict,
):
    _write_top_csv(strategy_dir, winning_params, rank=1)
    _patch_strategy_py(strategy_dir, _RSI_BB_ATR_SOURCE)
    with pytest.raises(ValueError, match="rank 7 not found"):
        translate_strategy(strategy_dir, rank=7)


def test_translate_strategy_picks_correct_rank(
    strategy_dir: Path, winning_params: dict, tmp_path: Path,
):
    # Write two rows: rank 1 has aggressive params, rank 2 conservative ones.
    aggressive = {**winning_params, "rsi_period": 7,  "bb_period": 14}
    conservative = {**winning_params, "rsi_period": 21, "bb_period": 40}
    csv = strategy_dir / "results" / "top_10_performers.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "rank,trial_id,params,is_sharpe_avg,oos_sharpe_avg,is_sharpe_min,"
        "oos_sharpe_min,max_dd_pct,total_trades,objective_value\n"
    )
    j1 = json.dumps(aggressive, sort_keys=True).replace('"', '""')
    j2 = json.dumps(conservative, sort_keys=True).replace('"', '""')
    csv.write_text(
        header
        + f'1,0,"{j1}",1.5,1.2,1.0,0.9,7.2,250,1.20\n'
        + f'2,1,"{j2}",1.4,1.1,0.9,0.8,7.5,260,1.10\n',
        encoding="utf-8",
    )
    _patch_strategy_py(strategy_dir, _RSI_BB_ATR_SOURCE)

    r1 = translate_strategy(strategy_dir, rank=1)
    assert r1.params["rsi_period"] == 7

    r2 = translate_strategy(strategy_dir, rank=2)
    assert r2.params["rsi_period"] == 21


# ---------------------------------------------------------------------------
# 6. Magic-number determinism
# ---------------------------------------------------------------------------

def test_magic_matches_ea_generator():
    """Magic numbers must agree with ``automation.ea_generator`` so the same
    strategy_id resolves to the same magic in both EA-from-spec and
    EA-from-Python paths."""
    sid = "STR_007_test_magic"
    assert magic_from_strategy_id(sid) == ea_gen_magic(sid)


# ---------------------------------------------------------------------------
# 7. CLI smoke test
# ---------------------------------------------------------------------------

def test_cli_main_writes_files(
    strategy_dir: Path, winning_params: dict, capsys, monkeypatch,
):
    _write_top_csv(strategy_dir, winning_params)
    _patch_strategy_py(strategy_dir, _RSI_BB_ATR_SOURCE)
    from python_engine.python_to_mql5_translator import main

    rc = main([str(strategy_dir), "--rank", "1"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Translated [auto]" in captured.out
    assert (strategy_dir / "mql5_translation").exists()
    assert list((strategy_dir / "mql5_translation").glob("STR_*.mq5"))
