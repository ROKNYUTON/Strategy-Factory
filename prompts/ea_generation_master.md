# MASTER EA GENERATION PROMPT

> This is the master template fed to Claude Code by `automation/ea_generator.py`.
> Variables in `{{...}}` are replaced with content from the strategy spec.

---

You are generating an MQL5 Expert Advisor for **StrategyFactory**, a disciplined quant strategy development pipeline.

## CONTEXT

You are writing the strategy logic for an EA that will be backtested on MetaTrader 5 with **real tick data** and validated against strict statistical acceptance criteria (Sharpe IS > 0.8, OOS > 0.6, bootstrap p < 0.01, parameter sensitivity ±20%, PnL decomposition).

The EA must:
- Trade **only** what the YAML spec describes — no creative additions.
- Reuse the existing template scaffolding without modification.
- Be production-ready (no stub code, no `Print("TODO")`, no missing branches).

## STRATEGY SPEC

```yaml
{{SPEC_YAML_CONTENT}}
```

## FILE TO MODIFY

The template at `mql5/_template/BaseEA_Template.mq5` has been pre-populated with metadata (strategy_id, magic, name) and saved to:

```
mql5/generated/{{STRATEGY_ID}}.mq5
```

## YOUR TASK

Edit **only the regions** delimited by:
```
// === AI GENERATED LOGIC START ===
... your code here ...
// === AI GENERATED LOGIC END ===
```

There are **5 regions** to fill:

### Region 1 — Strategy-specific INPUTS
Add `input` declarations matching the indicators and parameters in the YAML.
Use **round numbers** (no curve-fit precision). Each input must have a clear name `Inp<Name>` and a sensible default from the spec.

### Region 2 — GLOBAL indicator handles
Declare global `int g_<name>_handle = INVALID_HANDLE;` for each indicator used.

### Region 3 — OnInit() indicator initialization
Initialize each handle. If any returns `INVALID_HANDLE`, return `INIT_FAILED` with a `Logger_Msg(LOG_ERROR, ...)` call.

### Region 4 — OnDeinit() indicator release
Release every handle initialized in Region 3.

### Region 5 — OnTick() entry decision
After the existing risk gates and ATR fetch, call `CheckLongEntry()` and `CheckShortEntry()` and open positions accordingly. Use the `OpenPosition(side, current_atr)` helper that already exists in the template.

### Region 6 (in ManageOpenPositions) — Optional custom exits
Leave empty unless the spec specifies trailing or custom exits.

### Region 7 — AI HOOKS: CheckLongEntry / CheckShortEntry function bodies
Implement these two functions to return `true` ONLY when **ALL** conditions in `entry_rules.long.conditions` (and `filter_rules`) are met. Same for short.

Use **only built-in MQL5 indicators** unless the YAML explicitly requires custom logic:
- `iRSI` for RSI
- `iBands` for Bollinger Bands (buffers: `BASE_LINE`, `UPPER_BAND`, `LOWER_BAND`)
- `iATR` for ATR
- `iMA` for Moving Averages
- `iMACD` for MACD
- `iStochastic` for Stochastic

## RULES — NON-NEGOTIABLE

1. **DO NOT** modify ANY code outside the marked regions.
2. **DO NOT** modify the includes, the `OnTradeTransaction` handler, the `OpenPosition` helper, the risk-gate sequence in `OnTick`, or any function that already exists fully implemented.
3. **DO NOT** invent indicators or parameters not present in the YAML.
4. **DO NOT** add print statements outside `Logger_Msg(...)` calls (Logger respects log levels).
5. **DO** add inline comments referencing the YAML rule each block implements:
   ```mql5
   // YAML entry_rules.long.conditions[0]: RSI(14) < 25
   ```
6. **DO** check `CopyBuffer(...) <= 0` before reading indicator data. Return false on failure.
7. **DO** read the indicator value at index `[1]` (last closed bar), NOT `[0]` (current forming bar), unless the spec requires intrabar decisions.

## OUTPUT FORMAT

Return ONLY the complete modified `.mq5` file content. No markdown code fences, no explanations, no preamble. Just the raw MQL5 source.

The trader will paste your output directly into `mql5/generated/{{STRATEGY_ID}}.mq5`.

## CHECKLIST BEFORE SUBMITTING

- [ ] All 5+ AI regions are filled (none left as stubs or comments-only).
- [ ] `CheckLongEntry()` and `CheckShortEntry()` return `true` only when ALL YAML conditions are met (AND logic).
- [ ] Filter rules from `filter_rules` are checked inside both Check functions.
- [ ] Indicator handles are released in OnDeinit.
- [ ] No code outside AI markers has been touched.
- [ ] Each conditional has a `// YAML <path>: <description>` comment.
- [ ] No use of `[0]` (current forming bar) for entry decisions — use `[1]`.
