# Architecture — StrategyFactory

## System Diagram

```
                    ┌─────────────────────────────┐
                    │       Trader (you)          │
                    │  writes hypothesis YAML     │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
       ┌───────────────────────────────────────────────────┐
       │           strategy_specs/<id>.yaml               │
       └──────────────┬─────────────────────┬──────────────┘
                      │                     │
                      ▼                     ▼
        ┌─────────────────────┐   ┌──────────────────────┐
        │  spec_validator.py  │   │  ea_generator.py     │
        │  (Pydantic schema)  │   │  → prompt + skeleton │
        └─────────┬───────────┘   └──────────┬───────────┘
                  │                          │
                  │                          ▼
                  │             ┌──────────────────────────┐
                  │             │   ★ MANUAL ★             │
                  │             │ Trader pastes prompt     │
                  │             │ into Claude Code in VSC  │
                  │             │ → completed .mq5         │
                  │             └──────────┬───────────────┘
                  │                        │
                  ▼                        ▼
       ┌────────────────────────────────────────────────┐
       │  mql5/generated/<id>.mq5                       │
       └────────────────────┬───────────────────────────┘
                            │
                            ▼
                ┌───────────────────────────────┐
                │       mt5_compiler.py         │
                │  metaeditor64.exe /compile    │
                └────────────────┬──────────────┘
                                 │
                                 ▼
                ┌───────────────────────────────┐
                │  mql5/compiled/<id>.ex5       │
                └────────────────┬──────────────┘
                                 │
                                 ▼
        ┌─────────────────────────────────────────┐
        │  tester_ini_builder.py                  │
        │  → IS / OOS / forward .ini per period   │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
              ┌──────────────────────────────┐
              │    mt5_tester.py             │
              │  terminal64.exe /config:.ini │
              │  (× 3 periods)               │
              └──────────────┬───────────────┘
                             │
                             ▼
        ┌──────────────────────────────────────────┐
        │  backtests/raw_reports/<id>/<period>/    │
        │   - report.htm  (MT5 native)             │
        │   - <id>_<ts>.csv (Logger.mqh output)    │
        └─────────────────┬────────────────────────┘
                          │
                          ▼
       ┌──────────────────────────────────────────────┐
       │           ANALYSIS LAYER                     │
       │ ┌────────────────────────────────────────┐   │
       │ │ report_parser    → parsed.json         │   │
       │ │ metrics_calc     → metrics.json        │   │
       │ │ bootstrap        → bootstrap.json      │   │
       │ │ walk_forward     → walk_forward.json   │   │
       │ │ pnl_decomposer   → pnl_decomp.json     │   │
       │ │ sensitivity      → sensitivity.json    │   │
       │ └────────────────────────────────────────┘   │
       └─────────────────┬────────────────────────────┘
                         │
                         ▼
            ┌──────────────────────────────┐
            │  acceptance_check.py         │
            │  → PASS or FAIL verdict      │
            └─────────────────┬────────────┘
                              │
                              ▼
            ┌──────────────────────────────┐
            │  hypothesis_logger.py        │
            │  → docs/HYPOTHESIS_LOG.md    │
            └──────────────────────────────┘
```

## Component Responsibilities

| Layer | Components | Responsibility |
|---|---|---|
| **Input** | `strategy_specs/*.yaml`, `spec_validator.py` | Capture & validate hypothesis. |
| **Code generation** | `ea_generator.py`, `prompts/ea_generation_master.md`, `mql5/_template/*` | Produce a Claude Code prompt + a parameterized EA skeleton. |
| **Compile** | `mt5_compiler.py` | Wrap MetaEditor; parse compile log. |
| **Backtest** | `tester_ini_builder.py`, `mt5_tester.py`, `data_ingestion/tick_data_manager.py` | Drive MT5 Strategy Tester via .ini configs; manage tick data prerequisites. |
| **Analysis** | `report_parser.py`, `metrics_calculator.py`, `bootstrap_validator.py`, `walk_forward.py`, `parameter_sensitivity.py`, `pnl_decomposer.py` | Convert MT5 output into normalized JSON metrics. |
| **Decision** | `acceptance_check.py` | PASS/FAIL gate. |
| **Logging** | `hypothesis_logger.py`, `docs/HYPOTHESIS_LOG.md` | Append-only research diary. |
| **Orchestration** | `automation/pipeline.py`, `scripts/*.bat` | Entry points. |

## Data Flow per Pipeline Run

```
YAML (~2KB)
  → prompt.md (~5KB)
  → .mq5 skeleton (~10KB)
  → .mq5 completed by AI (~15KB)
  → .ex5 binary (~30KB)
  × 3 periods × {.htm report (~50KB) + .csv log (~20KB-2MB)}
  → 6 parsed JSON files per strategy (~5–500KB each)
  → 1 verdict JSON
  → 1 markdown entry in HYPOTHESIS_LOG.md
```

## Architectural Decisions (and why)

| Decision | Why |
|---|---|
| YAML, not JSON, for spec input | Comments matter for hypothesis documentation. |
| Pipeline pauses at EA generation | Enforces AI-assisted **manual** discipline. The trader must read the AI's logic before deploying. |
| Per-trade CSV log alongside MT5 .htm | MT5 reports don't decompose swap from directional. Our CSV does. |
| `OnTester` returns custom score | Uses Sharpe × swap-penalty so optimization runs (future) reward genuinely directional edges. |
| All analysis outputs JSON | Reusable by future Quant Analyzer integration (F1–F20 features) and by Obsidian ingestion. |
| Bootstrap uses centered resampling under H0 | Robust to fat tails (Kurt > 3 in real returns). t-test is wrong here. |
| Multiple-testing correction by default | Without it, p-values across many hypotheses are meaningless. BH method is standard. |
| `mql5/_template/*.mqh` never modified by AI | Risk management and logging are infrastructure, not strategy. |

## Extension Points

| To add | Touch these files |
|---|---|
| New analysis module | Create `analysis/<name>.py`, register in `pipeline.py` `analyze` command. |
| New acceptance criterion | `config/factory_defaults.yaml`, `analysis/acceptance_check.py` `evaluate()`. |
| New indicator type in spec | `automation/spec_validator.py` (loosen schema), `prompts/ea_generation_master.md` (instruct Claude). |
| New broker / tick source | `data_ingestion/tick_data_manager.py` + `config/symbols_map.yaml`. |
| Future: Obsidian sync | `automation/obsidian_sync.py` (planned) — see `INTEGRATION_QUANT_ANALYZER.md`. |
