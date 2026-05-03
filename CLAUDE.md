# CLAUDE.md — StrategyFactory Master Context

> **This file is the brain of the project. Read it FIRST before any code generation.**
> Every prompt sent to Claude Code in this repo must respect the rules below.

---

## 1. Project Mission

**StrategyFactory** is an AI-assisted **manual** strategy development pipeline for a senior quant trader managing $1M live AUM on Darwinex (MT5).

**Goal:** Generate **orthogonal alpha** to reduce portfolio variance.
- Current portfolio: Sharpe 1.31, Skew +2.80, **Kurtosis 21**, Max DD -8.74%
- Target: Kurtosis < 8, Skew ≈ +1.0, Sharpe > 1.7
- Method: Add ONE genuinely independent strategy at a time; reject anything that doesn't materially raise portfolio N_eff (Meucci).

**This is NOT a strategy generator factory.** We test ONE hypothesis per cycle, properly.

---

## 2. Hard Rules (NON-NEGOTIABLE)

1. **Never deploy a strategy without:** IS pass + OOS pass + forward demo 30 days + bootstrap p < 0.01 + parameter sensitivity check + PnL decomposition.
2. **Always decompose PnL** into directional / swap / commissions. If directional < 60% of profit, the strategy is a carry trade in disguise — REJECT (lesson learned: WTI overnight backtest looked Sharpe 7.56, was 125% swap).
3. **Always log the hypothesis**, even if it fails. Failed hypotheses are research data and YouTube content.
4. **Never modify template scaffolding** in `mql5/_template/`. AI generation only fills code between `// === AI GENERATED LOGIC START ===` and `// === END ===` markers.
5. **Never optimize on OOS data.** Optimization is IS-only. OOS is for validation, walk-forward is for robustness.
6. **One spec = one hypothesis = one EA.** No bundling unrelated logic into one EA "to save time."
7. **Multiple testing correction is mandatory.** If you've tested N hypotheses recently, p-values must be Benjamini-Hochberg adjusted.

---

## 3. File Responsibility Map

| Component | Path | Responsibility |
|---|---|---|
| Master context | `CLAUDE.md` | This file. Rules + acceptance criteria. |
| User entry | `README.md` | Human-facing intro and quick start. |
| Strategy spec template | `strategy_specs/_TEMPLATE.yaml` | Schema for hypothesis input. |
| Spec validator | `automation/spec_validator.py` | Pydantic validation of YAML specs. |
| EA template | `mql5/_template/BaseEA_Template.mq5` | Boilerplate EA. AI fills only marked sections. |
| Risk manager | `mql5/_template/RiskManager.mqh` | Position sizing, DD circuit breakers. NEVER edited by AI. |
| Logger | `mql5/_template/Logger.mqh` | CSV trade log with PnL decomposition. |
| PnL decomposer | `mql5/_template/PnLDecomposer.mqh` | Directional vs swap vs commission split. |
| EA generator | `automation/ea_generator.py` | YAML → prompt + skeleton (NO LLM call here). |
| MT5 compiler | `automation/mt5_compiler.py` | Wraps `metaeditor.exe /compile`. |
| MT5 tester | `automation/mt5_tester.py` | Wraps `terminal64.exe /config:tester.ini`. |
| Tester ini builder | `automation/tester_ini_builder.py` | Generates tester.ini from spec. |
| Tick data manager | `data_ingestion/tick_data_manager.py` | Verifies tick data availability before backtest. |
| Pipeline orchestrator | `automation/pipeline.py` | End-to-end CLI: prepare → compile → backtest → analyze → verdict. |
| Report parser | `analysis/report_parser.py` | MT5 .htm/.xml → structured JSON. |
| Metrics calculator | `analysis/metrics_calculator.py` | Sharpe, Sortino, Calmar, etc. |
| Bootstrap validator | `analysis/bootstrap_validator.py` | P-value via 5000+ resamples. |
| Walk-forward | `analysis/walk_forward.py` | WFA efficiency score. |
| Parameter sensitivity | `analysis/parameter_sensitivity.py` | ±20% perturbation robustness. |
| PnL decomposer | `analysis/pnl_decomposer.py` | Aggregate decomposition + WTI-style guard. |
| Acceptance check | `analysis/acceptance_check.py` | PASS/FAIL gate against `acceptance_criteria` in spec. |
| Hypothesis log | `docs/HYPOTHESIS_LOG.md` | Append-only research diary. |

---

## 4. Acceptance Criteria (Defaults)

These are the defaults in `config/factory_defaults.yaml`. Each spec can override.

| Criterion | Default | Rationale |
|---|---|---|
| `is_min_sharpe` | 0.8 | Below this, edge is too weak even before OOS degradation. |
| `oos_min_sharpe` | 0.6 | Typical OOS degradation 25–40%. Below this, no real generalization. |
| `bootstrap_max_pvalue` | 0.01 | Multiple testing aware threshold. Hedge fund standard. |
| `min_trades` | 100 | Statistical significance floor. |
| `max_drawdown_pct` | 15.0 | Beyond this, position-level risk too high. |
| `param_sensitivity_min_retained` | 0.5 | Sharpe must retain ≥ 50% at ±20% parameter shift. |
| `pnl_decomposition.min_directional_pnl_pct` | 60 | At least 60% of net profit must come from price movement, not swap. |
| `wfa_efficiency_min` | 0.5 | OOS Sharpe / IS Sharpe in walk-forward. |
| `target_correlation_with_existing_book` | 0.2 | Daily P&L correlation max with current portfolio. |

---

## 5. Forbidden Patterns

- **Curve-fit indicators:** No "RSI(13.7) on 47-minute bars" — round numbers only, justify in spec.
- **In-sample peeking:** No optimization runs that touch OOS or forward data.
- **Multiple-test cherry-picking:** Don't test 50 variants and report the best one as if it were a single hypothesis.
- **Swap-dependent edges:** Strategies whose backtest profit > 40% from swap accumulation. WTI lesson.
- **Strategies dominated by < 5% of trades:** Fragile to outliers. Run trade-removal robustness check.
- **Lookahead bias:** Using future data, e.g., daily close on intraday entries before close.
- **Survivorship bias in symbol selection:** No "I tried 20 symbols, kept the 3 that worked."
- **Optimization metric = profit:** Always Sharpe or Sortino, never raw profit.

---

## 6. AI-Assisted Manual Discipline

The pipeline INTENTIONALLY pauses at the EA generation step.

```
[YAML spec]
    ↓ spec_validator.py (auto)
[validated spec]
    ↓ ea_generator.py (auto: prepares prompt + skeleton)
[prompt at prompts/generation_prompts/{id}_prompt.md]
    ↓ ★ MANUAL: trader feeds prompt to Claude Code, reviews output ★
[EA placed at mql5/generated/{id}.mq5]
    ↓ mt5_compiler.py (auto)
[compiled .ex5]
    ↓ mt5_tester.py × {IS, OOS, forward} (auto)
[raw reports]
    ↓ report_parser.py + metrics + bootstrap + sensitivity + decomposition (auto)
[JSON results]
    ↓ acceptance_check.py (auto)
[PASS / FAIL verdict]
    ↓ hypothesis_logger.py (auto, with manual confirm on PASS)
[HYPOTHESIS_LOG.md updated]
```

The trader is in the loop at the highest-leverage step (logic review). Everything else is automated.

---

## 7. Output JSON Standard (for Quant Analyzer integration)

Every generated JSON has this header:
```json
{
  "strategy_id": "STR_001_asian_mr_fx",
  "generated_at": "2026-05-02T14:30:00Z",
  "engine_version": "StrategyFactory-1.0",
  "spec_hash": "<sha256 of input YAML>",
  "data": { ... }
}
```

Future integration with the existing Quant Analyzer (F1–F20) reads these JSONs to feed F16 (correlation), F18 (regime matrix), F19 (Meucci N_eff), F20 (PCA orthogonality).

---

## 8. Quick Start Commands

```bash
# Activate venv and verify environment
scripts\setup_env.bat

# Validate a strategy spec
python automation/spec_validator.py strategy_specs/my_strategy.yaml

# Run full pipeline (with manual pause at EA generation)
python automation/pipeline.py full strategy_specs/my_strategy.yaml

# Or step-by-step
python automation/pipeline.py prepare strategy_specs/my_strategy.yaml
# [feed prompt to Claude Code, save output to mql5/generated/]
python automation/pipeline.py compile STR_001_asian_mr_fx
python automation/pipeline.py backtest STR_001_asian_mr_fx --period all
python automation/pipeline.py analyze STR_001_asian_mr_fx
python automation/pipeline.py verdict STR_001_asian_mr_fx
```

---

## 9. Reference Documents

- `docs/ARCHITECTURE.md` — System design rationale
- `docs/WORKFLOW.md` — Step-by-step usage guide
- `docs/MT5_SETUP.md` — Windows VPS + MT5 configuration
- `docs/HYPOTHESIS_LOG.md` — Research diary (append-only, Obsidian-compatible YAML headers)
- `docs/INTEGRATION_QUANT_ANALYZER.md` — Output JSON ↔ existing F1-F20 features ↔ planned Obsidian sync
- `docs/PROMPT_LIBRARY.md` — Curated Claude Code prompts

---

## 10. Build Status (current)

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
| Tests passing                          | 25/25 |

---

## 11. Future: Obsidian Vault Integration (planned)

The trader's long-term knowledge stack is centered on Obsidian as a "second brain" for research papers, trading observations, and learning notes. StrategyFactory is designed to plug into that ecosystem.

When `automation/obsidian_sync.py` is added (planned), it will:

- Mirror every `HYPOTHESIS_LOG.md` entry as a backlinked Obsidian note with Dataview metadata.
- Mirror every `strategy_specs/*.yaml` as an Obsidian note linking to the source paper, the verdict, and the symbols traded.
- Watch the Obsidian vault for `#strategy_idea` tagged notes and pre-fill new YAML specs (`automation/idea_intake.py` planned).

The data contract is already in place: all output JSONs use the standard header (strategy_id / generated_at / engine_version / period / data) that `obsidian_sync.py` will consume. The hypothesis log already uses Obsidian-compatible YAML front-matter.

Until the sync module ships, the trader can already symlink `docs/HYPOTHESIS_LOG.md` and `strategy_specs/` into the Obsidian vault — they render with backlinks correctly today.

---

**Last principle:** If a Claude Code generation seems to violate any rule above, STOP. Re-read CLAUDE.md. The rules exist because each was paid for in real losses or research time.
