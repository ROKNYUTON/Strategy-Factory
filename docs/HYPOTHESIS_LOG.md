# Hypothesis Log — StrategyFactory

> Append-only research diary. Every hypothesis tested ends here, PASS or FAIL.
> Failed hypotheses are research data — never delete entries.
>
> This file is also designed to be ingested by Obsidian (future integration). Each entry uses
> YAML front-matter style metadata so it can be parsed by Obsidian Dataview.

## Format per entry

```
## YYYY-MM-DD — STR_NNN_<name> — VERDICT (✅ or ❌)

```yaml
date: ...
strategy_id: ...
verdict: PASS | FAIL
checks_passed: N
checks_failed: M
is_sharpe: ...
oos_sharpe: ...
p_value_sharpe: ...
wfa_efficiency: ...
directional_pct: ...
```

**Failed checks:** (only when FAIL)
- ...

**Artifacts:**
- Spec: `path`
- Verdict JSON: `path`

**Lessons learned:** (free-form, optional)

---
```

## Example PASS entry

## 2026-05-15 — STR_001_asian_mr_fx — PASS ✅

```yaml
date: 2026-05-15
strategy_id: STR_001_asian_mr_fx
verdict: PASS
checks_passed: 11
checks_failed: 0
is_sharpe: 1.34
oos_sharpe: 0.91
p_value_sharpe: 0.003
wfa_efficiency: 0.68
directional_pct: 87.4
```

**Artifacts:**
- Spec: `strategy_specs/_EXAMPLE_asian_mr_fx.yaml`
- Verdict JSON: `backtests/parsed_results/STR_001_asian_mr_fx/acceptance_verdict.json`

**Lessons learned:** Asian session MR works as expected on EURUSD/GBPUSD. Strategy is intraday so WTI guard naturally passes. Next: monitor 30-day forward demo before live deploy at 1/4 Kelly.

---

## Example FAIL entry

## 2026-04-22 — STR_002_overnight_carry_jpy — FAIL ❌

```yaml
date: 2026-04-22
strategy_id: STR_002_overnight_carry_jpy
verdict: FAIL
checks_passed: 6
checks_failed: 5
is_sharpe: 1.78
oos_sharpe: 0.42
p_value_sharpe: 0.018
wfa_efficiency: 0.21
directional_pct: 31.2
```

**Failed checks:**
- OOS Sharpe >= min (0.42 < 0.6)
- IS bootstrap p-value <= max (0.018 > 0.01)
- WFA efficiency >= min (0.21 < 0.5)
- Directional PnL % >= min (31.2 < 60)
- WTI guard not flagged (FLAGGED — strategy is 69% swap)

**Artifacts:**
- Spec: `strategy_specs/_archive/rejected/STR_002_overnight_carry_jpy.yaml`
- Verdict JSON: `backtests/parsed_results/STR_002_overnight_carry_jpy/acceptance_verdict.json`

**Lessons learned:** Classic carry-trade trap. Looked great in IS due to JPY rate differential, dies in OOS when rate dynamics change. **The WTI lesson confirmed in real time.** Move on. Don't try variants of this — same failure mode.

---

## Real entries below this line


## 2026-05-02 — STR_999_e2e_pass_full — FAIL ❌

```yaml
date: 2026-05-02
strategy_id: STR_999_e2e_pass_full
verdict: FAIL
checks_passed: 9
checks_failed: 3
is_sharpe: 4.902620240062567
oos_sharpe: 4.902620240062567
p_value_sharpe: 0.018
p_value_sharpe_oos: 0.018
wfa_efficiency: 0.0
directional_pct: 124.5799562588447
```

**Failed checks:**
- IS bootstrap p-value <= max
- OOS bootstrap p-value <= max
- WFA efficiency >= min

**Artifacts:**
- Spec: `/home/claude/StrategyFactory/strategy_specs/STR_999_e2e_pass_full.yaml`
- Verdict JSON: `backtests/parsed_results/STR_999_e2e_pass_full/acceptance_verdict.json`