# Integration: StrategyFactory ↔ Quant Analyzer (F1–F20)

> Forward-looking spec. Integration is **not implemented yet** in code — this
> document describes the data contract so that when the user's existing Quant
> Analyzer (with features F1–F20) consumes StrategyFactory output, the mapping
> is unambiguous.

## Why this matters

The Quant Analyzer answers **portfolio-level** questions:
- F19 Meucci N_eff: how diversified is my book really?
- F20 PCA: which orthogonal factors drive my portfolio?
- F16 Rolling correlation: are my strategies independent in tail events?
- F18 Regime matrix: which regimes am I covered for?

StrategyFactory answers **strategy-level** questions: is this single new edge real and robust?

When a new strategy passes acceptance in StrategyFactory, the Quant Analyzer must be able to **simulate adding it to the live book** and tell us whether it actually moves N_eff up, kurtosis down, etc.

## Output JSON Contract

Every analysis JSON written by StrategyFactory has a stable header:

```json
{
  "strategy_id": "STR_001_asian_mr_fx",
  "generated_at": "2026-05-02T14:30:00Z",
  "engine_version": "StrategyFactory-1.0",
  "period": "is | oos | forward",
  "...": "..."
}
```

The **canonical artifact** for downstream consumption is the parsed trades file:

```
backtests/parsed_results/<strategy_id>/<period>_parsed.json
```

with schema (see `analysis/report_parser.py`):

```json
{
  "strategy_id": "...",
  "period": "is",
  "summary": { ... },
  "trades": [
    {
      "open_time": "ISO-8601",
      "close_time": "ISO-8601",
      "symbol": "EURUSD",
      "direction": "long | short",
      "lots": 0.05,
      "profit_directional": 12.34,
      "profit_swap": 0.21,
      "profit_commission": -0.50,
      "profit_total": 12.05
    }
  ]
}
```

This trade-level resolution lets the Quant Analyzer compute everything else.

## Mapping to Quant Analyzer Features

| Quant Analyzer Feature | StrategyFactory Source | Pseudocode |
|---|---|---|
| **F16 Rolling Correlation** | Daily P&L from `*_parsed.json` | `pnl_series = trades.groupby(date).profit_total.sum()`; combine with current book series; compute rolling Pearson + tail correlation |
| **F18 Regime Matrix** | trades + symbol + close_time | Cross-tabulate `trades.profit_total` by `symbol × regime_classification`. Regime comes from existing F5 HMM |
| **F19 Meucci N_eff** | New strategy daily P&L appended to existing book matrix | `cov = cov(daily_returns)`; PCA on cov; `N_eff = exp(entropy(eigenvalues_normalized))`. Compare before/after adding new strategy |
| **F20 PCA Orthogonality** | Same as F19 but report eigenvectors and loadings | The new strategy should ideally load on PC4–PCn (not PC1–PC2). If it loads heavily on PC1, it's not orthogonal |
| **F17 Statistical Edge Validator** | Already in StrategyFactory bootstrap | Use `*_bootstrap.json.bootstrap.p_value_sharpe` directly |
| **F1 Drawdown DNA** | trades.profit_total time series | Run F1 classification on the new strategy alone AND on the simulated combined book |
| **F9 Kelly Sizing** | trades summary statistics | Compute Quarter Kelly given win_rate + avg_win_loss_ratio from `*_metrics.json` |

## Suggested Integration Module (for the Quant Analyzer side)

```python
# In your Quant Analyzer codebase
def ingest_strategy_factory(strategy_id: str, sf_root: Path):
    base = sf_root / "backtests" / "parsed_results" / strategy_id
    is_p = json.loads((base / "is_parsed.json").read_text())
    oos_p = json.loads((base / "oos_parsed.json").read_text())
    fwd_p = json.loads((base / "forward_parsed.json").read_text())
    verdict = json.loads((base / "acceptance_verdict.json").read_text())

    # Build daily P&L
    df = pd.DataFrame(is_p["trades"] + oos_p["trades"] + fwd_p["trades"])
    df["close_time"] = pd.to_datetime(df["close_time"])
    daily = df.groupby(df["close_time"].dt.date).profit_total.sum()

    # Inject into your existing book matrix
    book_with_new = pd.concat([self.book_daily_pnl, daily.rename(strategy_id)], axis=1)

    # Recompute F19 / F20 on enlarged book
    n_eff_before = compute_meucci_n_eff(self.book_daily_pnl)
    n_eff_after  = compute_meucci_n_eff(book_with_new)
    delta_n_eff  = n_eff_after - n_eff_before

    return {
        "strategy_id": strategy_id,
        "factory_verdict": verdict["verdict"],
        "n_eff_before": n_eff_before,
        "n_eff_after": n_eff_after,
        "delta_n_eff": delta_n_eff,
        "decision": "DEPLOY" if delta_n_eff > 0.3 and verdict["verdict"] == "PASS" else "REJECT",
    }
```

## Future: Obsidian Vault Ingestion

Planned module: `automation/obsidian_sync.py`.

### What it does
- Watches `docs/HYPOTHESIS_LOG.md` and `strategy_specs/` for changes.
- For each entry, writes/updates a corresponding note in your Obsidian vault under `<vault>/StrategyFactory/<strategy_id>.md`.
- Adds Obsidian-style backlinks: `[[STR_001_asian_mr_fx]]`, `[[Asian Session MR]]`, etc.
- Embeds the verdict JSON as a callout block.
- Links references mentioned in spec `notes:` (research papers, blog posts) as Obsidian properties.
- Generates a Dataview-compatible header on each note for graph visualization.

### Why
Your "second brain" already lives in Obsidian (or will). Strategy research notes should be there too — discoverable, linkable, graphed alongside your other knowledge.

### Skeleton (planned)
```python
def sync_strategy(strategy_id: str, vault_path: Path):
    spec = load_spec(strategy_id)
    verdict = load_verdict(strategy_id)
    decomp = load_decomp(strategy_id)

    note_path = vault_path / "StrategyFactory" / f"{strategy_id}.md"
    note_path.write_text(render_obsidian_note(spec, verdict, decomp))
```

### Configuration (planned)
```yaml
# config/obsidian.yaml
obsidian:
  vault_path: "C:/Users/.../Documents/MyVault"
  subfolder: "StrategyFactory"
  use_dataview: true
  embed_charts: true  # ASCII charts of equity curves
  link_research_papers: true
```

For now, you can already symlink the relevant files into your Obsidian vault:
```cmd
mklink /D C:\Users\<You>\Documents\MyVault\StrategyFactory C:\path\to\StrategyFactory\docs
```
This makes `HYPOTHESIS_LOG.md` directly editable from Obsidian, with backlinks rendering correctly.

## Future: Research Paper Ingestion (Obsidian Bridge)

Workflow design when Obsidian is connected:
1. You read a paper → save PDF + summary note in Obsidian.
2. Add tag `#strategy_idea` and a snippet of the proposed entry/exit logic.
3. A planned `automation/idea_intake.py` script will scan vault for `#strategy_idea` notes and pre-fill a YAML spec template with the rationale + reference link already populated.
4. You complete the spec, run the pipeline, and the verdict gets logged back to Obsidian — closing the research loop.

This is the long-term vision: **Obsidian = research substrate, StrategyFactory = production validation, Quant Analyzer = portfolio-level decision**.
