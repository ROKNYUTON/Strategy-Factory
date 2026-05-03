"""Tests for spec_validator."""
import sys
from pathlib import Path
import copy

import pytest
import yaml
from pydantic import ValidationError

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from automation.spec_validator import StrategySpec, validate_spec  # noqa: E402


@pytest.fixture
def valid_spec_dict() -> dict:
    """Load the example spec which should always validate."""
    example_path = ROOT / "strategy_specs" / "_EXAMPLE_asian_mr_fx.yaml"
    with example_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_example_spec_validates(valid_spec_dict):
    """The shipped example spec must always validate cleanly."""
    spec = StrategySpec(**valid_spec_dict)
    assert spec.meta.strategy_id == "STR_001_asian_mr_fx"
    assert "EURUSD" in spec.universe.symbols


def test_template_placeholder_rejected():
    """Template strategy_id placeholder must be rejected."""
    bad = {
        "meta": {
            "strategy_id": "STR_XXX_descriptive_name",
            "author": "x",
            "created_date": "2026-01-01",
            "hypothesis_version": "1.0",
        },
    }
    with pytest.raises(ValidationError):
        StrategySpec(**bad)


def test_missing_required_field(valid_spec_dict):
    """Missing top-level field must raise ValidationError."""
    bad = copy.deepcopy(valid_spec_dict)
    del bad["hypothesis"]
    with pytest.raises(ValidationError):
        StrategySpec(**bad)


def test_invalid_timeframe_enum(valid_spec_dict):
    """Invalid timeframe must be rejected."""
    bad = copy.deepcopy(valid_spec_dict)
    bad["universe"]["timeframe"] = "M3"  # not in TIMEFRAMES
    with pytest.raises(ValidationError):
        StrategySpec(**bad)


def test_invalid_edge_source(valid_spec_dict):
    """Invalid edge source must be rejected."""
    bad = copy.deepcopy(valid_spec_dict)
    bad["hypothesis"]["expected_edge_source"] = "telepathy"
    with pytest.raises(ValidationError):
        StrategySpec(**bad)


def test_period_inversion_rejected(valid_spec_dict):
    """IS end after OOS start must be rejected."""
    bad = copy.deepcopy(valid_spec_dict)
    bad["backtest"]["is_period"]["end"] = "2025-12-31"  # overlaps OOS
    with pytest.raises(ValidationError):
        StrategySpec(**bad)


def test_period_start_after_end(valid_spec_dict):
    """Single-period start >= end must be rejected."""
    bad = copy.deepcopy(valid_spec_dict)
    bad["backtest"]["oos_period"]["start"] = "2024-12-31"
    bad["backtest"]["oos_period"]["end"] = "2024-01-01"
    with pytest.raises(ValidationError):
        StrategySpec(**bad)


def test_unrealistic_risk_per_trade(valid_spec_dict):
    """risk_per_trade > 5% must be rejected."""
    bad = copy.deepcopy(valid_spec_dict)
    bad["risk"]["risk_per_trade_pct"] = 25.0
    with pytest.raises(ValidationError):
        StrategySpec(**bad)


def test_negative_stop_loss(valid_spec_dict):
    """Stop loss value <= 0 must be rejected."""
    bad = copy.deepcopy(valid_spec_dict)
    bad["exit_rules"]["stop_loss"]["value"] = -1.0
    with pytest.raises(ValidationError):
        StrategySpec(**bad)


def test_both_directions_disabled(valid_spec_dict):
    """If both long and short are disabled the spec must fail."""
    bad = copy.deepcopy(valid_spec_dict)
    bad["entry_rules"]["long"]["enabled"] = False
    bad["entry_rules"]["short"]["enabled"] = False
    with pytest.raises(ValidationError):
        StrategySpec(**bad)


def test_invalid_session_time_format(valid_spec_dict):
    """Bad time format must be rejected."""
    bad = copy.deepcopy(valid_spec_dict)
    bad["universe"]["trading_session_utc"]["start"] = "25:00"
    with pytest.raises(ValidationError):
        StrategySpec(**bad)


def test_strategy_id_format_strict(valid_spec_dict):
    """strategy_id must match STR_<NUM>_<snake_case_name>."""
    bad = copy.deepcopy(valid_spec_dict)
    bad["meta"]["strategy_id"] = "MyStrategy123"
    with pytest.raises(ValidationError):
        StrategySpec(**bad)


def test_orthogonality_rationale_required(valid_spec_dict):
    """Orthogonality rationale placeholder must be rejected."""
    bad = copy.deepcopy(valid_spec_dict)
    bad["orthogonality_target"]["orthogonality_rationale"] = "REPLACE: TBD"
    with pytest.raises(ValidationError):
        StrategySpec(**bad)


def test_validate_spec_function(tmp_path, valid_spec_dict):
    """End-to-end validate_spec on a written file."""
    p = tmp_path / "test_spec.yaml"
    p.write_text(yaml.dump(valid_spec_dict), encoding="utf-8")
    spec = validate_spec(p)
    assert spec.meta.strategy_id == "STR_001_asian_mr_fx"
