"""Tests for ``automation.generate_strategy``.

Covers:
1. Heuristic parsing of the logic string (session, edge source, max bars).
2. ``next_strategy_id`` numbering against an existing strategies/ tree.
3. End-to-end skeleton creation: every required artifact exists, the
   generated ``hypothesis.yaml`` validates against ``StrategySpec``, and the
   generated ``strategy.py`` is syntactically valid Python.
4. Refusal to overwrite an existing folder unless ``overwrite=True``.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from automation.generate_strategy import (  # noqa: E402
    build_hypothesis_spec,
    build_optimization_grid,
    extract_paper_text,
    generate_strategy_skeleton,
    infer_edge_source,
    infer_holding_period,
    infer_regime,
    next_strategy_id,
    parse_max_holding_bars,
    parse_session,
    render_strategy_py,
)
from automation.spec_validator import StrategySpec  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Logic-string heuristics
# ---------------------------------------------------------------------------

def test_parse_session_dash():
    assert parse_session("Asian session 22-06 UTC") == ("22:00", "06:00", True)


def test_parse_session_with_colons():
    assert parse_session("trade 09:30 to 16:00 NY hours") == ("09:30", "16:00", False)


def test_parse_session_default_when_no_match():
    assert parse_session("no time mentioned") == ("22:00", "06:00", True)


def test_parse_max_holding_bars():
    assert parse_max_holding_bars("max 16 bars then exit") == 16
    assert parse_max_holding_bars("plain text") == 16  # default


def test_infer_edge_source_mean_reversion():
    assert infer_edge_source("RSI<25 fade BB lower") == "mean_reversion"


def test_infer_edge_source_breakout():
    assert infer_edge_source("Donchian channel breakout") == "breakout"


def test_infer_edge_source_default_falls_back():
    assert infer_edge_source("no obvious keyword") == "mean_reversion"


def test_infer_regime_asian_low_vol():
    assert infer_regime("intraday Asian session") == "low_vol_sideways"


def test_infer_holding_period_intraday():
    assert infer_holding_period("intraday RSI") == "intraday"


def test_infer_holding_period_swing():
    assert infer_holding_period("swing trade across multi day moves") == "swing_1_5d"


# ---------------------------------------------------------------------------
# 2. next_strategy_id
# ---------------------------------------------------------------------------

def test_next_strategy_id_starts_at_1_in_empty_tree(tmp_path: Path):
    sid = next_strategy_id("foo", search_dirs=[tmp_path])
    assert sid == "STR_001_foo"


def test_next_strategy_id_picks_max_plus_1(tmp_path: Path):
    (tmp_path / "STR_002_a").mkdir()
    (tmp_path / "STR_005_b").mkdir()
    (tmp_path / "STR_003_c").mkdir()
    sid = next_strategy_id("new_one", search_dirs=[tmp_path])
    assert sid == "STR_006_new_one"


def test_next_strategy_id_sanitizes_name(tmp_path: Path):
    sid = next_strategy_id("Mixed CASE/Name!", search_dirs=[tmp_path])
    # Lowercased, non-alphanumeric collapsed to underscores, strips leading/trailing _.
    assert sid == "STR_001_mixed_case_name"


# ---------------------------------------------------------------------------
# 3. End-to-end skeleton creation
# ---------------------------------------------------------------------------

def test_skeleton_creates_all_artifacts_and_validates(tmp_path: Path):
    paper = tmp_path / "paper.txt"
    paper.write_text("Asian-session mean reversion paper. Plenty of citations.\n")
    folder = generate_strategy_skeleton(
        paper_path=paper,
        name="mean_rev_1",
        logic_description=(
            "RSI<25 + BB lower entry, intraday Asian session 22-06 UTC, max 16 bars"
        ),
        symbols=["EURUSD", "USDJPY"],
        is_period=(date(2018, 1, 1), date(2023, 12, 31)),
        oos_period=(date(2024, 1, 1), date(2026, 4, 30)),
        base_dir=tmp_path / "strategies",
    )
    assert folder.is_dir()
    for f in (
        "README.md",
        "hypothesis.yaml",
        "strategy.py",
        "optimization_grid.yaml",
        "data_cache/.gitkeep",
        "results/.gitkeep",
        "mql5_translation/.gitkeep",
    ):
        assert (folder / f).exists(), f"missing {f}"

    # The generated spec must validate cleanly.
    raw = yaml.safe_load((folder / "hypothesis.yaml").read_text())
    spec = StrategySpec(**raw)
    assert spec.meta.strategy_id.startswith("STR_")
    assert spec.universe.symbols == ["EURUSD", "USDJPY"]
    assert spec.universe.trading_session_utc.start == "22:00"
    assert spec.universe.trading_session_utc.end == "06:00"
    assert spec.universe.trading_session_utc.crosses_midnight is True
    assert spec.exit_rules.time_exit.max_holding_bars == 16
    assert spec.hypothesis.expected_edge_source == "mean_reversion"

    # strategy.py must be syntactically valid Python and define `signals`.
    code = (folder / "strategy.py").read_text()
    compile(code, str(folder / "strategy.py"), "exec")
    assert "def signals(" in code
    assert "DEFAULT_PARAMS" in code

    # Paper extract is referenced in README.
    readme = (folder / "README.md").read_text()
    assert "Asian-session mean reversion" in readme


def test_skeleton_refuses_existing_folder(tmp_path: Path):
    paper = tmp_path / "p.txt"
    paper.write_text("text")
    args = dict(
        paper_path=paper, name="x", logic_description="rsi mean rev",
        symbols=["EURUSD"],
        is_period=(date(2018, 1, 1), date(2019, 1, 1)),
        oos_period=(date(2019, 1, 2), date(2020, 1, 1)),
        base_dir=tmp_path / "strategies",
    )
    generate_strategy_skeleton(**args)
    with pytest.raises(FileExistsError):
        generate_strategy_skeleton(**args)
    # overwrite=True succeeds.
    folder = generate_strategy_skeleton(**args, overwrite=True)
    assert (folder / "hypothesis.yaml").exists()


def test_skeleton_rejects_overlapping_periods(tmp_path: Path):
    paper = tmp_path / "p.txt"
    paper.write_text("text")
    with pytest.raises(ValueError, match="must be <"):
        generate_strategy_skeleton(
            paper_path=paper, name="bad", logic_description="rsi",
            symbols=["EURUSD"],
            is_period=(date(2018, 1, 1), date(2024, 1, 1)),
            oos_period=(date(2023, 1, 1), date(2025, 1, 1)),
            base_dir=tmp_path / "strategies",
        )


# ---------------------------------------------------------------------------
# 4. Builder helpers in isolation
# ---------------------------------------------------------------------------

def test_build_hypothesis_spec_default_forward_window(tmp_path: Path):
    paper = tmp_path / "x.txt"
    paper.write_text("x")
    spec = build_hypothesis_spec(
        strategy_id="STR_001_t", name="t", paper_path=paper,
        logic_description="rsi mean rev intraday",
        symbols=["EURUSD"],
        is_period=(date(2018, 1, 1), date(2023, 12, 31)),
        oos_period=(date(2024, 1, 1), date(2024, 12, 31)),
    )
    fwd = spec["backtest"]["forward_period"]
    assert fwd["start"] == "2025-01-01"
    # +365 days from 2025-01-01 is 2025-12-31 (not a leap year inside).
    assert fwd["end"] == "2025-12-31" or fwd["end"] == "2026-01-01"
    StrategySpec(**spec)  # must validate


def test_build_optimization_grid_keys_present():
    grid = build_optimization_grid("rsi mean rev max 20 bars")
    params = grid["parameters"]
    for key in (
        "rsi_period", "rsi_oversold", "rsi_overbought",
        "bb_period", "bb_dev", "sl_atr_mult", "tp_atr_mult", "time_exit_bars",
    ):
        assert key in params
    # time_exit_bars range adapts to the parsed max-bars value.
    assert params["time_exit_bars"]["high"] >= 20


def test_render_strategy_py_handles_braces_in_logic(tmp_path: Path):
    """User-supplied logic may contain {}; replace() must not interpret them."""
    code = render_strategy_py(
        name="x",
        paper_path=tmp_path / "p.txt",
        logic_description='use dict {"k": 1} as input',
        default_params={"a": 1},
    )
    # Must compile.
    compile(code, "<test>", "exec")
    assert '{"k": 1}' in code
    assert "DEFAULT_PARAMS: Dict = {'a': 1}" in code


# ---------------------------------------------------------------------------
# 5. Paper extraction
# ---------------------------------------------------------------------------

def test_extract_paper_text_plain(tmp_path: Path):
    p = tmp_path / "x.md"
    p.write_text("# Title\n\nBody paragraph.")
    text = extract_paper_text(p)
    assert "Title" in text and "Body paragraph" in text


def test_extract_paper_text_missing_returns_empty(tmp_path: Path):
    text = extract_paper_text(tmp_path / "does_not_exist.pdf")
    assert text == ""


def test_extract_paper_text_truncates(tmp_path: Path):
    p = tmp_path / "big.txt"
    p.write_text("X" * 5000)
    text = extract_paper_text(p, max_chars=100)
    assert len(text) == 100
