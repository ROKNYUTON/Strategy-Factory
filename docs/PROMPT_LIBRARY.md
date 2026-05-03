# Prompt Library — Curated Claude Code prompts

Copy-paste these into Claude Code in VS Code as needed.

---

## 1. Generate EA from spec

This is the auto-generated prompt. You'll get the full version in `prompts/generation_prompts/<id>_prompt.md` after running `pipeline.py prepare`. Just open that file and paste its contents.

---

## 2. Review Generated MQL5 for Bugs

```
You are reviewing an MQL5 Expert Advisor generated for StrategyFactory.

Below is the strategy YAML spec, followed by the generated .mq5 source.

Your job: identify any of the following issues:
- Conditions implemented as OR instead of AND (or vice versa)
- Wrong indicator buffer index (using [0] forming bar instead of [1] closed)
- Missing filter rules from spec
- Indicator handle leaks (created in OnInit but not released in OnDeinit)
- Lookahead bias (using future data inadvertently)
- Risk management rule bypassed (e.g., trade opens despite session check)
- Hardcoded magic numbers different from STRATEGY_MAGIC
- Code outside AI markers modified (forbidden)

Format your output as:
- 🔴 BUGS: ... (must fix)
- 🟡 WARNINGS: ... (should consider)
- 🟢 GOOD: ... (well-implemented elements)

DO NOT rewrite the EA. Only report issues.

[paste spec YAML here]

---

[paste .mq5 source here]
```

---

## 3. Suggest Parameter Ranges for Sensitivity Test

```
I'm running a parameter sensitivity test on this strategy.

For each numeric parameter in the YAML below, suggest:
- A reasonable ±20% perturbation range
- Whether the parameter is structurally important (=> stricter sensitivity threshold)
  or peripheral (looser threshold)
- Any cross-parameter dependencies I should test (e.g., "if SL multiplier increases,
  TP multiplier should also increase to keep risk:reward sane")

Output as a markdown table:
| parameter | base_value | perturbed_min | perturbed_max | importance | notes |

[paste spec YAML here]
```

---

## 4. Critique Hypothesis from Quant PM Perspective

```
Act as a senior quantitative portfolio manager reviewing a new strategy proposal.

The trader's existing book has these characteristics:
- $1M AUM live
- 17 strategies, 95% US session, mostly trend-following on indices and metals
- Sharpe 1.31, Skew +2.80, Kurtosis 21
- Goal: reduce variance, raise Meucci N_eff

Critique this proposal honestly. Flag:
- Whether the rationale is mechanistic or vibes-based
- Whether the orthogonality claim holds
- Whether the failure modes are realistic (or sanitized)
- Whether the parameters look curve-fit (suspiciously precise numbers)
- Hidden risks (regime dependency, swap exposure, news sensitivity)
- Better alternatives if you'd structure it differently

Be direct. No filler. Senior PM tone.

[paste spec YAML here]
```

---

## 5. Refine Failed Hypothesis

```
This hypothesis FAILED in StrategyFactory. Here is the spec and the failure verdict.

Help me decide:
A. Iterate on the same idea (parameter tweaks, additional filters)
B. Pivot to a related hypothesis (similar mechanism, different implementation)
C. Abandon the line of investigation entirely

Answer with:
1. Diagnosis: WHY did it fail (root cause from the failed checks)
2. Recommendation: A, B, or C with justification
3. If A or B: 2-3 concrete next experiments to try
4. If C: what category of hypothesis would be more productive given my book

Don't just suggest cosmetic tweaks. Be willing to recommend abandonment.

[paste spec YAML]
[paste verdict JSON]
```

---

## 6. Generate Hypothesis from Research Paper

```
I'm reading this research paper / blog post / observation:
[paste excerpt or link summary]

Translate the core claim into a falsifiable trading hypothesis suitable for
StrategyFactory. Output a draft YAML spec with:
- meta + hypothesis sections fully filled
- universe section with reasonable defaults (suggest symbols + timeframe)
- entry_rules and exit_rules as a starting point (I'll refine)
- expected_failure_modes (realistic)
- orthogonality_target (be specific about which dimension)

Keep it minimal — it's a starting point, not the final spec. Use the template at
strategy_specs/_TEMPLATE.yaml as your structural reference.
```

---

## 7. Sanity-Check My Backtest Results

```
Look at this acceptance verdict JSON. The strategy passed.

Play devil's advocate: what could be wrong DESPITE the PASS?

Consider:
- Were the IS/OOS dates inappropriate for this asset (e.g., gold-rally OOS that flatters MR strategies)?
- Did the strategy avoid known stress periods that should be in OOS?
- Is the trade count high enough that a few outliers wouldn't change the verdict?
- Does the WFA show consistency or did it pass on average while having hidden bad windows?
- Could the directional PnL still be misleading (e.g., slippage modeled too generously)?

Be paranoid. Find reasons NOT to trust the PASS.

[paste verdict JSON]
[paste IS/OOS metrics]
[paste walk-forward JSON]
```

---

## 8. Plan Next Hypothesis Given Portfolio Gaps

```
Given the current state of my book and the strategies I've already tested
(passing AND failing), suggest the next 3 hypothesis directions ranked by:
- Likely impact on portfolio Meucci N_eff
- Effort to develop (low / medium / high)
- Independence from already-tested ideas

Don't repeat directions I've already explored — review the failed log to avoid
the same trap.

[paste docs/HYPOTHESIS_LOG.md]
[brief description of current live book — sectors, time, style]
```

---

## Tip: Always commit prompts you reuse

If you find yourself iterating on a prompt to get good results, save the final
version in this file under a new section. Future you will thank you.
