"""
StrategyFactory — Spec Validator
================================
Validates strategy YAML specs against a Pydantic schema.

Usage:
    python automation/spec_validator.py strategy_specs/my_strategy.yaml

Exit codes:
    0 = valid
    1 = invalid (with detailed errors)
    2 = file not found / read error
"""

from __future__ import annotations

import sys
import re
from pathlib import Path
from datetime import date as Date
from typing import Optional, List, Dict, Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# ---------------------------------------------------------------
# Allowed enums
# ---------------------------------------------------------------
TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
EDGE_SOURCES = [
    "mean_reversion", "trend_following", "carry",
    "volatility", "arbitrage", "momentum", "breakout",
]
REGIMES = [
    "low_vol_sideways", "trending_bull", "trending_bear",
    "high_vol_crisis", "all_weather",
]
HOLDING_PERIODS = ["intraday", "swing_1_5d", "position_5_30d"]
SIZING_METHODS = ["fixed_lot", "risk_per_trade_pct", "kelly_quarter"]
SL_TYPES = ["atr_multiple", "fixed_pips", "percentage", "custom"]
MODELING = ["every_tick_real", "1m_ohlc", "open_prices"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ---------------------------------------------------------------
# Models
# ---------------------------------------------------------------
class Meta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_id: str
    author: str
    created_date: Date
    hypothesis_version: str
    tags: List[str] = []

    @field_validator("strategy_id")
    @classmethod
    def strategy_id_format(cls, v: str) -> str:
        if not re.match(r"^STR_\d{3,}_[a-z0-9_]+$", v):
            raise ValueError(
                f"strategy_id must match pattern STR_<NUM>_<snake_case_name>, got: {v}"
            )
        if v == "STR_XXX_descriptive_name":
            raise ValueError(
                "strategy_id is still the template placeholder. Replace it."
            )
        return v


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rationale: str
    expected_edge_source: str
    expected_regime_performance: str
    expected_failure_modes: str = ""
    expected_holding_period: str

    @field_validator("rationale")
    @classmethod
    def rationale_filled(cls, v: str) -> str:
        if "REPLACE" in v.upper() or len(v.strip()) < 50:
            raise ValueError(
                "rationale is empty or template placeholder. Provide real 3+ line mechanism."
            )
        return v

    @field_validator("expected_edge_source")
    @classmethod
    def edge_source_valid(cls, v: str) -> str:
        if v not in EDGE_SOURCES:
            raise ValueError(f"expected_edge_source must be one of {EDGE_SOURCES}")
        return v

    @field_validator("expected_regime_performance")
    @classmethod
    def regime_valid(cls, v: str) -> str:
        if v not in REGIMES:
            raise ValueError(f"expected_regime_performance must be one of {REGIMES}")
        return v

    @field_validator("expected_holding_period")
    @classmethod
    def holding_valid(cls, v: str) -> str:
        if v not in HOLDING_PERIODS:
            raise ValueError(f"expected_holding_period must be one of {HOLDING_PERIODS}")
        return v


class TradingSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: str
    end: str
    crosses_midnight: bool = False

    @field_validator("start", "end")
    @classmethod
    def hhmm_format(cls, v: str) -> str:
        if not re.match(r"^([01]\d|2[0-3]):[0-5]\d$", v):
            raise ValueError(f"Time must be HH:MM 24h format, got: {v}")
        return v


class NewsBlackout(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    minutes_before: int = 30
    minutes_after: int = 30


class Universe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbols: List[str]
    timeframe: str
    trading_session_utc: TradingSession
    trading_days: List[str]
    news_blackout: Optional[NewsBlackout] = None

    @field_validator("symbols")
    @classmethod
    def symbols_nonempty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("symbols list is empty.")
        return v

    @field_validator("timeframe")
    @classmethod
    def timeframe_valid(cls, v: str) -> str:
        if v not in TIMEFRAMES:
            raise ValueError(f"timeframe must be one of {TIMEFRAMES}")
        return v

    @field_validator("trading_days")
    @classmethod
    def trading_days_valid(cls, v: List[str]) -> List[str]:
        invalid = [d for d in v if d not in DAYS]
        if invalid:
            raise ValueError(f"trading_days contains invalid: {invalid}. Allowed: {DAYS}")
        return v


class Condition(BaseModel):
    model_config = ConfigDict(extra="allow")  # allow flexibility for indicator params
    type: str
    indicator: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    operator: Optional[str] = None
    value: Optional[float] = None
    reference: Optional[str] = None
    description: Optional[str] = None


class Direction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool
    conditions: List[Condition] = []


class EntryRules(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    long: Direction
    short: Direction
    filter_rules: List[Condition] = []

    @model_validator(mode="after")
    def at_least_one_direction(self):
        if not self.long.enabled and not self.short.enabled:
            raise ValueError("Both long and short are disabled — strategy never trades.")
        if self.long.enabled and not self.long.conditions:
            raise ValueError("long.enabled=True but no conditions specified.")
        if self.short.enabled and not self.short.conditions:
            raise ValueError("short.enabled=True but no conditions specified.")
        return self


class StopLoss(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    value: float
    atr_period: Optional[int] = None

    @field_validator("type")
    @classmethod
    def sl_type_valid(cls, v: str) -> str:
        if v not in SL_TYPES:
            raise ValueError(f"stop_loss.type must be one of {SL_TYPES}")
        return v

    @field_validator("value")
    @classmethod
    def sl_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("stop_loss.value must be > 0")
        return v


class TakeProfit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    value: float
    atr_period: Optional[int] = None

    @field_validator("type")
    @classmethod
    def tp_type_valid(cls, v: str) -> str:
        if v not in SL_TYPES:
            raise ValueError(f"take_profit.type must be one of {SL_TYPES}")
        return v

    @field_validator("value")
    @classmethod
    def tp_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("take_profit.value must be > 0")
        return v


class TimeExit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    max_holding_bars: Optional[int] = None


class Trailing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    type: Optional[str] = None
    activation_atr: Optional[float] = None
    trail_atr: Optional[float] = None


class ExitRules(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stop_loss: StopLoss
    take_profit: TakeProfit
    time_exit: TimeExit = TimeExit()
    trailing: Trailing = Trailing()
    reverse_on_opposite_signal: bool = False


class KellyInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_win_rate: float
    expected_avg_win_loss_ratio: float


class Risk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position_sizing: str
    fixed_lot: Optional[float] = None
    risk_per_trade_pct: Optional[float] = None
    kelly_inputs: Optional[KellyInputs] = None
    max_concurrent_positions: int = 1
    max_concurrent_per_symbol: int = 1
    max_daily_loss_pct: float = 2.0
    max_drawdown_pct_circuit_breaker: float = 10.0
    max_slippage_points: int = 20

    @field_validator("position_sizing")
    @classmethod
    def sizing_valid(cls, v: str) -> str:
        if v not in SIZING_METHODS:
            raise ValueError(f"position_sizing must be one of {SIZING_METHODS}")
        return v

    @field_validator("risk_per_trade_pct")
    @classmethod
    def risk_reasonable(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v <= 0 or v > 5):
            raise ValueError(
                f"risk_per_trade_pct={v} is unreasonable. Must be 0 < x <= 5 (in % of equity)."
            )
        return v

    @model_validator(mode="after")
    def sizing_consistency(self):
        if self.position_sizing == "fixed_lot" and self.fixed_lot is None:
            raise ValueError("position_sizing='fixed_lot' but fixed_lot not set.")
        if self.position_sizing == "risk_per_trade_pct" and self.risk_per_trade_pct is None:
            raise ValueError(
                "position_sizing='risk_per_trade_pct' but risk_per_trade_pct not set."
            )
        if self.position_sizing == "kelly_quarter" and self.kelly_inputs is None:
            raise ValueError(
                "position_sizing='kelly_quarter' but kelly_inputs not set."
            )
        return self


class Period(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: Date
    end: Date

    @model_validator(mode="after")
    def start_before_end(self):
        if self.start >= self.end:
            raise ValueError(f"period start={self.start} must be before end={self.end}")
        return self


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_period: Period
    oos_period: Period
    forward_period: Period
    initial_balance: float = 10000
    currency: str = "USD"
    leverage: int = 30
    modeling: str = "every_tick_real"
    use_real_spread: bool = True
    optimization: bool = False

    @field_validator("modeling")
    @classmethod
    def modeling_valid(cls, v: str) -> str:
        if v not in MODELING:
            raise ValueError(f"modeling must be one of {MODELING}")
        return v

    @model_validator(mode="after")
    def periods_ordered(self):
        if self.is_period.end >= self.oos_period.start:
            raise ValueError(
                f"is_period.end ({self.is_period.end}) must be < oos_period.start ({self.oos_period.start})"
            )
        if self.oos_period.end >= self.forward_period.start:
            raise ValueError(
                f"oos_period.end ({self.oos_period.end}) must be < forward_period.start ({self.forward_period.start})"
            )
        return self


class PnLDecomp(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_directional_pnl_pct: float = 60


class AcceptanceCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")
    is_min_sharpe: float = 0.8
    oos_min_sharpe: float = 0.6
    bootstrap_max_pvalue: float = 0.01
    max_drawdown_pct: float = 15.0
    min_trades: int = 100
    param_sensitivity_min_retained: float = 0.5
    pnl_decomposition: PnLDecomp = PnLDecomp()


class OrthogonalityTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_correlation_with_existing_book: float = 0.2
    target_uncovered_dimension: str
    orthogonality_rationale: str

    @field_validator("orthogonality_rationale")
    @classmethod
    def rationale_filled(cls, v: str) -> str:
        if "REPLACE" in v.upper() or len(v.strip()) < 30:
            raise ValueError(
                "orthogonality_rationale is empty or placeholder. Provide real reasoning."
            )
        return v


class StrategySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: Meta
    hypothesis: Hypothesis
    universe: Universe
    entry_rules: EntryRules
    exit_rules: ExitRules
    risk: Risk
    backtest: BacktestConfig
    acceptance_criteria: Optional[AcceptanceCriteria] = None
    orthogonality_target: OrthogonalityTarget
    notes: Optional[str] = ""


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------
def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_spec(path: Path) -> StrategySpec:
    """Validate a YAML spec file. Raises ValidationError on failure."""
    raw = load_yaml(path)
    return StrategySpec(**raw)


def print_summary(spec: StrategySpec, path: Path) -> None:
    table = Table(title=f"Spec Validation: {path.name}", show_header=True)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("strategy_id", spec.meta.strategy_id)
    table.add_row("author", spec.meta.author)
    table.add_row("created_date", str(spec.meta.created_date))
    table.add_row("edge_source", spec.hypothesis.expected_edge_source)
    table.add_row("regime", spec.hypothesis.expected_regime_performance)
    table.add_row("symbols", ", ".join(spec.universe.symbols))
    table.add_row("timeframe", spec.universe.timeframe)
    table.add_row(
        "session_utc",
        f"{spec.universe.trading_session_utc.start} → {spec.universe.trading_session_utc.end}",
    )
    table.add_row(
        "is_period",
        f"{spec.backtest.is_period.start} → {spec.backtest.is_period.end}",
    )
    table.add_row(
        "oos_period",
        f"{spec.backtest.oos_period.start} → {spec.backtest.oos_period.end}",
    )
    table.add_row(
        "forward_period",
        f"{spec.backtest.forward_period.start} → {spec.backtest.forward_period.end}",
    )
    table.add_row("position_sizing", spec.risk.position_sizing)
    table.add_row(
        "orthogonality_target",
        spec.orthogonality_target.target_uncovered_dimension,
    )

    console.print(table)


def main() -> int:
    if len(sys.argv) != 2:
        console.print("[red]Usage: python automation/spec_validator.py <spec.yaml>[/red]")
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        return 2

    try:
        spec = validate_spec(path)
    except yaml.YAMLError as e:
        console.print(Panel(f"[red]YAML parse error:[/red]\n{e}", title="❌ INVALID"))
        return 1
    except Exception as e:
        console.print(
            Panel(f"[red]Validation error:[/red]\n{e}", title="❌ INVALID")
        )
        return 1

    console.print(Panel("[green]Spec is valid.[/green]", title="✅ VALID"))
    print_summary(spec, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
