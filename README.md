# StrategyFactory

> AI-assisted manual quantitative strategy development pipeline for MetaTrader 5.
> Hypothesis → YAML spec → EA generation (with Claude Code) → automated MT5 backtest with **real ticks** → statistical validation → verdict.

---

## What it does (30 seconds)

1. You write a strategy hypothesis in a YAML spec.
2. StrategyFactory validates the spec and generates a Claude Code prompt.
3. You feed the prompt to Claude Code in VS Code → you get an EA back.
4. StrategyFactory compiles it via `metaeditor.exe`, runs IS/OOS/forward backtests via `terminal64.exe` with **every-tick real-tick modeling**, parses MT5 reports, runs bootstrap p-values, walk-forward, parameter sensitivity, and PnL decomposition.
5. PASS or FAIL verdict, logged automatically to `docs/HYPOTHESIS_LOG.md`.

**Built for one thing:** producing strategies disciplined enough to deploy on real money, not curve-fit dashboards.

---

## Prerequisites

- **Windows VPS** with MetaTrader 5 installed (default path: `C:\Program Files\MetaTrader 5\`)
- **Tick data** subscription / availability for target symbols and historical periods
- **Python 3.11+** with venv
- **Claude Code** in VS Code (used manually to generate the EA from the produced prompt)
- **Internet** for initial setup only — pipeline runs fully offline once configured

---

## Quick Start

```cmd
:: 1. Initialize environment
scripts\setup_env.bat

:: 2. Verify your MT5 paths in config\mt5_paths.yaml

:: 3. Copy the example spec and edit it
copy strategy_specs\_EXAMPLE_asian_mr_fx.yaml strategy_specs\my_first_strategy.yaml

:: 4. Run the pipeline
scripts\run_strategy.bat my_first_strategy.yaml
```

The pipeline pauses when it's time for you to generate the EA via Claude Code. Follow the on-screen instructions.

---

## Folder Map

```
StrategyFactory/
├── CLAUDE.md              ← Master rules. Read first.
├── README.md              ← You are here.
├── requirements.txt
├── .gitignore
│
├── config/                ← MT5 paths, defaults, symbol mappings
│   ├── mt5_paths.yaml
│   ├── factory_defaults.yaml
│   └── symbols_map.yaml
│
├── strategy_specs/        ← INPUT: your YAML hypotheses
│   ├── _TEMPLATE.yaml             ← Copy this to start a new strategy
│   ├── _EXAMPLE_asian_mr_fx.yaml  ← Fully filled example
│   └── archive/{rejected,accepted}/
│
├── mql5/
│   ├── _template/         ← Boilerplate EA + .mqh modules (DO NOT MODIFY)
│   │   ├── BaseEA_Template.mq5
│   │   ├── RiskManager.mqh
│   │   ├── Logger.mqh
│   │   └── PnLDecomposer.mqh
│   ├── generated/         ← AI-completed .mq5 files
│   └── compiled/          ← .ex5 binaries
│
├── data_ingestion/
│   └── tick_data_manager.py   ← Verifies tick availability before backtest
│
├── automation/
│   ├── spec_validator.py       ← Pydantic schema validation
│   ├── ea_generator.py         ← YAML → prompt + skeleton
│   ├── mt5_compiler.py         ← Wraps metaeditor64.exe
│   ├── tester_ini_builder.py   ← Generates tester.ini from spec
│   ├── mt5_tester.py           ← Wraps terminal64.exe /config:.ini
│   ├── pipeline.py             ← Click CLI orchestrator
│   └── hypothesis_logger.py    ← Append-only research diary writer
│
├── analysis/
│   ├── report_parser.py        ← MT5 .htm → JSON
│   ├── metrics_calculator.py   ← Sharpe, Sortino, Calmar, etc.
│   ├── bootstrap_validator.py  ← p-value via 5000+ resamples + BH correction
│   ├── walk_forward.py         ← Rolling-window WFA efficiency
│   ├── parameter_sensitivity.py ← ±20% perturbation robustness
│   ├── pnl_decomposer.py       ← Directional / swap / commission split (WTI guard)
│   └── acceptance_check.py     ← Master PASS/FAIL gate
│
├── backtests/
│   ├── raw_reports/       ← MT5 .htm reports + CSV trade logs
│   ├── parsed_results/    ← Normalized JSON (consumed by Quant Analyzer / Obsidian)
│   └── archive/
│
├── prompts/
│   ├── ea_generation_master.md           ← Master template for Claude Code
│   └── generation_prompts/<id>_prompt.md ← Per-strategy generated prompts
│
├── docs/
│   ├── ARCHITECTURE.md             ← System design rationale
│   ├── WORKFLOW.md                 ← Step-by-step usage
│   ├── MT5_SETUP.md                ← Windows VPS configuration
│   ├── HYPOTHESIS_LOG.md           ← Research diary (Obsidian-ready)
│   ├── PROMPT_LIBRARY.md           ← 8 curated Claude Code prompts
│   └── INTEGRATION_QUANT_ANALYZER.md ← Mapping to F1-F20 + Obsidian plan
│
├── tests/
│   ├── test_spec_validator.py    ← 14 unit tests
│   ├── test_pipeline_e2e.py      ← 11 E2E tests with synthetic fixtures
│   └── fixtures/                 ← Synthetic backtest JSONs (pass/fail/swap-trap)
│
├── scripts/
│   ├── setup_env.bat             ← One-time venv + deps install
│   ├── run_strategy.bat          ← 1-click pipeline runner
│   └── run_full_validation.bat   ← pytest + env health check
│
└── logs/                        ← Pipeline execution logs
```

---

## Workflow Diagram

```
┌──────────────────┐
│ YAML hypothesis  │ ← you write
└────────┬─────────┘
         ▼
   spec_validator
         ▼
   ea_generator → produces prompt + EA skeleton
         ▼
  ★ MANUAL: Claude Code generates EA logic ★
         ▼
   mt5_compiler  → metaeditor.exe /compile
         ▼
   mt5_tester    → terminal64.exe (IS, OOS, forward) — REAL TICK DATA
         ▼
  report_parser → metrics → bootstrap → walk_forward → sensitivity → pnl_decomposer
         ▼
   acceptance_check → PASS or FAIL
         ▼
   hypothesis_logger → docs/HYPOTHESIS_LOG.md  → (future) Obsidian vault
```

---

## Status

| Component                              | Status |
|----------------------------------------|:------:|
| Task 0 — Folder scaffold + CLAUDE.md   | ✅ |
| Task 1 — Spec system + validator       | ✅ |
| Task 2 — MQL5 template + EA generator  | ✅ |
| Task 3 — MT5 automation layer          | ✅ |
| Task 4 — Analysis layer                | ✅ |
| Task 5 — Pipeline orchestrator         | ✅ |
| Task 6 — Documentation                 | ✅ |
| Task 7 — E2E tests + fixtures          | ✅ |
| **Tests passing**                      | **25/25** |

### Verified working in this build
- `python automation/spec_validator.py strategy_specs/_EXAMPLE_asian_mr_fx.yaml` → ✅ valid
- `python automation/ea_generator.py strategy_specs/_EXAMPLE_asian_mr_fx.yaml` → ✅ skeleton + prompt produced
- `python -m pytest tests/` → ✅ 25/25 pass

### Requires manual setup on Windows VPS
- MT5 install path → edit `config/mt5_paths.yaml`
- Tick data download per symbol/period → MT5 GUI (View → Symbols → Bars)
- Claude Code in VS Code (for the manual EA generation step)

---

## Future Roadmap — Obsidian "Second Brain" Integration

Long-term vision: **Obsidian = research substrate, StrategyFactory = production validation, Quant Analyzer = portfolio-level decision**.

Planned module: `automation/obsidian_sync.py`. Two-way sync between StrategyFactory and your Obsidian vault:

- Each PASS/FAIL entry in `HYPOTHESIS_LOG.md` mirrored as a backlinked note in Obsidian (the file already uses Dataview-compatible YAML headers — ready to ingest).
- Each spec produces a note for graph visualization, with backlinks to:
  - the hypothesis log entry
  - the original research paper (PDF + summary in vault)
  - the symbols traded
- Research papers tagged `#strategy_idea` in your vault → auto-pre-fill spec template via planned `automation/idea_intake.py`.
- This closes the loop: **research → hypothesis → validation → diary → research**.

For now you can already symlink `docs/HYPOTHESIS_LOG.md` into your Obsidian vault — it renders correctly with backlinks. See [`docs/INTEGRATION_QUANT_ANALYZER.md`](docs/INTEGRATION_QUANT_ANALYZER.md) § "Future: Obsidian Vault Ingestion" for the full plan.

Other planned items:
- **Quant Analyzer integration** — feed parsed JSONs into existing F16/F18/F19/F20 features for portfolio-level deploy/reject decisions (Meucci N_eff delta).
- **Full optimization loop** — replace sensitivity stub with a real perturbation-grid runner that re-launches MT5 per parameter set.
- **Live deployment monitor** — separate module comparing live trade decomposition vs backtest decomposition daily, alerting on drift.

---

## Documentation

- [`CLAUDE.md`](CLAUDE.md) — Project rules, acceptance criteria, forbidden patterns
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System design rationale
- [`docs/WORKFLOW.md`](docs/WORKFLOW.md) — Step-by-step usage
- [`docs/MT5_SETUP.md`](docs/MT5_SETUP.md) — Windows VPS configuration
- [`docs/HYPOTHESIS_LOG.md`](docs/HYPOTHESIS_LOG.md) — Research diary
- [`docs/PROMPT_LIBRARY.md`](docs/PROMPT_LIBRARY.md) — Curated Claude Code prompts
- [`docs/INTEGRATION_QUANT_ANALYZER.md`](docs/INTEGRATION_QUANT_ANALYZER.md) — JSON contract + Obsidian plan

---

## License

Internal/private. Not for redistribution.
