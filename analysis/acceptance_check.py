"""
StrategyFactory — Acceptance Check (the gate)
=============================================
Aggregates all analysis outputs and applies the acceptance criteria
(spec overrides ∪ factory defaults). Returns PASS or FAIL with detailed
per-criterion verdict.

Output: backtests/parsed_results/{strategy_id}/acceptance_verdict.json
"""

from __future__ import annotations

import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

ROOT = Path(__file__).parent.parent
console = Console()
ENGINE_VERSION = "StrategyFactory-1.0"

sys.path.insert(0, str(ROOT))
from automation.spec_validator import validate_spec  # noqa: E402


def load_factory_defaults() -> dict:
    cfg = ROOT / "config" / "factory_defaults.yaml"
    return yaml.safe_load(cfg.read_text(encoding="utf-8"))


def find_spec(strategy_id: str) -> Path:
    candidates = list((ROOT / "strategy_specs").rglob("*.yaml"))
    for p in candidates:
        try:
            spec = validate_spec(p)
            if spec.meta.strategy_id == strategy_id:
                return p
        except Exception:
            continue
    raise FileNotFoundError(f"No spec for {strategy_id}")


def load_json(path: Path) -> dict:
    if not path.exists(): return {}
    return json.loads(path.read_text(encoding="utf-8"))


def merge_criteria(spec, defaults: dict) -> dict:
    """Merge spec.acceptance_criteria with factory defaults."""
    base = defaults["acceptance_criteria"]
    crit = dict(base)
    if spec.acceptance_criteria is not None:
        from dataclasses import asdict
        try:
            spec_crit = spec.acceptance_criteria.model_dump()
        except AttributeError:
            spec_crit = spec.acceptance_criteria.dict()
        # Shallow merge
        for k, v in spec_crit.items():
            if isinstance(v, dict) and k in crit and isinstance(crit[k], dict):
                crit[k].update(v)
            else:
                crit[k] = v
    return crit


def evaluate(strategy_id: str) -> dict:
    spec_path = find_spec(strategy_id)
    spec = validate_spec(spec_path)
    defaults = load_factory_defaults()
    crit = merge_criteria(spec, defaults)

    base_dir = ROOT / "backtests" / "parsed_results" / strategy_id

    # Load all relevant artifacts
    is_metrics  = load_json(base_dir / "is_metrics.json").get("metrics", {})
    oos_metrics = load_json(base_dir / "oos_metrics.json").get("metrics", {})
    fwd_metrics = load_json(base_dir / "forward_metrics.json").get("metrics", {})
    boot_is     = load_json(base_dir / "is_bootstrap.json").get("bootstrap", {})
    boot_oos    = load_json(base_dir / "oos_bootstrap.json").get("bootstrap", {})
    wfa         = load_json(base_dir / "walk_forward.json").get("walk_forward", {})
    sens        = load_json(base_dir / "sensitivity.json").get("sensitivity", {})
    decomp_is   = load_json(base_dir / "is_pnl_decomp.json").get("decomposition", {})

    checks = []

    def check(name: str, value, op: str, threshold, weight: int = 1) -> dict:
        ok = False
        try:
            if op == ">=":  ok = value is not None and value >= threshold
            elif op == "<=": ok = value is not None and value <= threshold
            elif op == ">":  ok = value is not None and value > threshold
            elif op == "<":  ok = value is not None and value < threshold
            elif op == "==": ok = value == threshold
        except Exception:
            ok = False
        return {
            "name": name,
            "value": value,
            "operator": op,
            "threshold": threshold,
            "passed": bool(ok),
            "weight": weight,
        }

    # IS checks
    checks.append(check("IS Sharpe >= min", is_metrics.get("sharpe_ratio"), ">=", crit["is_min_sharpe"]))
    checks.append(check("IS trades >= min", is_metrics.get("trade_count"), ">=", crit["min_trades"]))
    checks.append(check("IS max DD <= limit", is_metrics.get("max_drawdown_pct"), "<=", crit["max_drawdown_pct"]))

    # OOS checks
    checks.append(check("OOS Sharpe >= min", oos_metrics.get("sharpe_ratio"), ">=", crit["oos_min_sharpe"]))
    checks.append(check("OOS max DD <= limit", oos_metrics.get("max_drawdown_pct"), "<=", crit["max_drawdown_pct"]))

    # Bootstrap (use BH-adjusted if available)
    p_is = boot_is.get("p_value_sharpe_adjusted_bh", boot_is.get("p_value_sharpe"))
    p_oos = boot_oos.get("p_value_sharpe_adjusted_bh", boot_oos.get("p_value_sharpe"))
    checks.append(check("IS bootstrap p-value <= max", p_is, "<=", crit["bootstrap_max_pvalue"]))
    if p_oos is not None:
        checks.append(check("OOS bootstrap p-value <= max", p_oos, "<=", crit["bootstrap_max_pvalue"]))

    # WFA efficiency
    wfa_eff = wfa.get("wfa_efficiency")
    checks.append(check("WFA efficiency >= min", wfa_eff, ">=", crit.get("wfa_efficiency_min", 0.5)))

    # Sensitivity
    sens_min = sens.get("overall_min_retained")
    if sens_min is None:
        sens_min = sens.get("consistency_score")  # stub mode
    checks.append(check("Sensitivity (min retained) >= min", sens_min, ">=", crit["param_sensitivity_min_retained"]))

    # PnL decomposition (IS)
    pcts = decomp_is.get("percentages", {})
    dir_pct = pcts.get("directional_pct")
    min_dir = crit.get("pnl_decomposition", {}).get("min_directional_pnl_pct", 60)
    checks.append(check("Directional PnL % >= min", dir_pct, ">=", min_dir))
    wti_flag = decomp_is.get("wti_guard", {}).get("flagged", False)
    checks.append(check("WTI guard not flagged", not wti_flag, "==", True))

    # Forward (informational, not blocking unless data present)
    if fwd_metrics:
        checks.append(check("Forward Sharpe >= 0.4 (info)", fwd_metrics.get("sharpe_ratio"), ">=", 0.4))

    passed = all(c["passed"] for c in checks)
    n_pass = sum(1 for c in checks if c["passed"])
    n_fail = sum(1 for c in checks if not c["passed"])

    verdict = {
        "strategy_id": strategy_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "engine_version": ENGINE_VERSION,
        "verdict": "PASS" if passed else "FAIL",
        "checks_passed": n_pass,
        "checks_failed": n_fail,
        "checks_total": len(checks),
        "criteria_used": crit,
        "checks": checks,
        "spec_path": str(spec_path),
    }

    out = ROOT / "backtests" / "parsed_results" / strategy_id / "acceptance_verdict.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict


def print_verdict(v: dict) -> None:
    color = "green" if v["verdict"] == "PASS" else "red"
    console.print(Panel(
        f"[bold {color}]{v['verdict']}[/bold {color}]\n"
        f"{v['checks_passed']} passed / {v['checks_failed']} failed / {v['checks_total']} total",
        title=f"VERDICT — {v['strategy_id']}"
    ))

    t = Table(show_header=True)
    t.add_column("Check", style="cyan")
    t.add_column("Value")
    t.add_column("Op", justify="center")
    t.add_column("Threshold")
    t.add_column("OK", justify="center")
    for c in v["checks"]:
        ok = "[green]✅[/green]" if c["passed"] else "[red]❌[/red]"
        val = c["value"]
        if isinstance(val, float):
            val_s = f"{val:.4f}"
        else:
            val_s = str(val)
        t.add_row(c["name"], val_s, c["operator"], str(c["threshold"]), ok)
    console.print(t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy_id")
    args = ap.parse_args()

    v = evaluate(args.strategy_id)
    print_verdict(v)
    return 0 if v["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
