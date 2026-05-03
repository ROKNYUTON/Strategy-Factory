# Workflow — From Hypothesis to Verdict

## Overview

```
hypothesis  →  YAML spec  →  AI-generated EA  →  3 backtests  →  analysis  →  verdict  →  log
   (you)         (you)         (Claude Code)         (MT5)        (auto)      (auto)    (auto)
```

Estimated time per hypothesis: **2–4 hours active + 30 min – 2 hours backtest CPU time** (tick-based, depends on symbol/period).

---

## Step-by-Step

### Step 0 — One-Time Setup

```cmd
:: Verify Python is installed
python --version    :: must be 3.11+

:: Initialize venv and install deps
scripts\setup_env.bat

:: Edit MT5 paths
notepad config\mt5_paths.yaml
:: Set: terminal_exe, metaeditor_exe, data_folder
:: To find data_folder: open MT5 → File → Open Data Folder → copy the path

:: Verify
scripts\run_full_validation.bat
```

### Step 1 — Write the Hypothesis

```cmd
:: Copy the template
copy strategy_specs\_TEMPLATE.yaml strategy_specs\my_hypothesis.yaml

:: Open and fill in
notepad strategy_specs\my_hypothesis.yaml
```

Critical fields to fill carefully:
- `meta.strategy_id` — unique, format `STR_<NUM>_<snake_case>`.
- `hypothesis.rationale` — 3-line economic mechanism. NOT "I think this might work."
- `hypothesis.expected_failure_modes` — be honest about when this breaks.
- `entry_rules` — only conditions you can defend mechanistically.
- `orthogonality_target` — which dimension of your existing book is this filling?

Validate before generating:
```cmd
python automation\spec_validator.py strategy_specs\my_hypothesis.yaml
```

### Step 2 — Generate the EA

```cmd
python automation\pipeline.py prepare strategy_specs\my_hypothesis.yaml
```

This produces:
- `mql5/generated/<id>.mq5` — skeleton with metadata + AI markers
- `prompts/generation_prompts/<id>_prompt.md` — full prompt for Claude Code

### Step 3 — Manual EA Generation in Claude Code

1. Open `prompts/generation_prompts/<id>_prompt.md` in VS Code.
2. Select all → copy → paste into Claude Code.
3. Claude Code returns the modified `.mq5` file content.
4. Save it over `mql5/generated/<id>.mq5`.
5. **Review the diff manually.** Don't trust blindly. Look for:
   - Are the conditions ANDed correctly?
   - Is the indicator buffer index right (`[1]` for last closed bar)?
   - Are filter rules applied?
   - No new indicators not in your spec?

### Step 4 — Compile

```cmd
python automation\pipeline.py compile <strategy_id>
```

If errors:
- Read the parsed errors in console.
- Open the .mq5 in MetaEditor manually to debug.
- Fix and re-run compile.

### Step 5 — Verify Tick Data

```cmd
python data_ingestion\tick_data_manager.py strategy_specs\my_hypothesis.yaml
```

If any period reports `no_data`:
1. Open MT5
2. Tools → Options → Charts → set "Max bars in chart" = unlimited
3. View → Symbols → select your symbol → "Bars" tab
4. Request the missing date range
5. Wait for download to complete (can take 10–30 min for years of tick data)
6. Re-run verification

### Step 6 — Backtest

```cmd
python automation\pipeline.py backtest <strategy_id> --period all
```

This runs 3 backtests sequentially: IS, OOS, forward. Each can take 10–60+ minutes depending on the symbol and period length. Tick-based testing is slow but the only way to be confident.

### Step 7 — Analyze

```cmd
python automation\pipeline.py analyze <strategy_id>
```

Runs all analysis modules. Output JSONs in `backtests/parsed_results/<strategy_id>/`.

### Step 8 — Verdict

```cmd
python automation\pipeline.py verdict <strategy_id>
```

This applies the acceptance criteria and PASS/FAIL the strategy.

### Step 9 — Log

If PASS, the verdict is logged automatically (with confirmation prompt) to `docs/HYPOTHESIS_LOG.md`. If FAIL, also logged. **Never delete entries.** Failed hypotheses are valuable research data.

---

## All-in-One

```cmd
scripts\run_strategy.bat my_hypothesis.yaml
```

This runs all steps with a manual pause after Step 2 for you to do the EA generation in Claude Code.

---

## Common Pitfalls

| Pitfall | Fix |
|---|---|
| MT5 tester hangs forever | MT5 needs `ShutdownTerminal=1` in the .ini (we set this automatically). If still hanging, the EA likely has a runtime infinite loop — check OnTick logic. |
| "Symbol not found" in backtest | Symbol code differs in your broker. Check `config/symbols_map.yaml` and the Market Watch in MT5. |
| Compile errors after AI generation | Open the generated .mq5 and check the AI markers. The AI may have left a syntax error — fix manually or re-prompt with the error message. |
| Bootstrap p-value = None | Less than 30 trades. Either lower `min_trades` in spec for prototype phase, or extend the IS period. |
| Forward result much worse than OOS | Possible regime change. Don't dismiss yet — log as FAIL with comment, monitor regime, retry in 30 days. |
| WTI guard flagged | Your strategy depends on swap accumulation. Either accept (it's a carry trade — rare to want this) or restructure entry/exit to capture price movement, not interest differential. |

---

## Decision Tree After Verdict

```
                        [VERDICT]
                            │
              ┌─────────────┴─────────────┐
              │                           │
            PASS                        FAIL
              │                           │
              ▼                           ▼
   Deploy to demo for           Read failed checks:
   30-day forward test          - IS Sharpe low?
              │                   → edge weak; refine entry
              ▼                 - OOS << IS?
   Forward demo PASS?            → likely overfit; add filters
              │                 - p-value high?
        ┌─────┴─────┐             → noise; try different signal
        │           │           - WFA fail?
       Yes         No             → unstable across periods
        │           │           - WTI flagged?
        ▼           ▼             → fix exit logic to capture price
   Live deploy   Reduce         - Sensitivity fragile?
   Quarter       size, monitor    → curve-fit; round parameters
   Kelly         additional 30
                 days
```

---

## Future: Obsidian Integration

Once enabled (`automation/obsidian_sync.py`, planned):
1. Each PASS or FAIL entry in `HYPOTHESIS_LOG.md` will be auto-mirrored as a note in your Obsidian vault.
2. Each spec will produce a backlinked note with metadata for graph view.
3. References to research papers in spec `notes:` will become first-class Obsidian links.
4. This makes your research diary part of your second brain.

For now, the markdown file is portable — you can already symlink `docs/HYPOTHESIS_LOG.md` into your Obsidian vault and it will render correctly with backlinks.
