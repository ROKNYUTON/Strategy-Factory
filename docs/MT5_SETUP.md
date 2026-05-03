# MT5 Setup on Windows VPS

## 1. Install MetaTrader 5

1. Download from https://www.metatrader5.com/en/download
2. Install with defaults. Default path: `C:\Program Files\MetaTrader 5\`
3. Verify both binaries exist:
   ```cmd
   dir "C:\Program Files\MetaTrader 5\terminal64.exe"
   dir "C:\Program Files\MetaTrader 5\metaeditor64.exe"
   ```

## 2. Locate the MT5 Data Folder

Inside MT5: `File → Open Data Folder`. Copy the path (typically under `%APPDATA%\MetaQuotes\Terminal\<HEX>\`).

Subfolders that matter:
- `MQL5\Experts\` — where compiled `.ex5` must live to be visible to MT5.
- `MQL5\Files\` — where `Logger.mqh` writes CSV trade logs.
- `tester\Reports\` — where Strategy Tester writes .htm reports.

## 3. Update `config/mt5_paths.yaml`

```yaml
mt5:
  terminal_exe: "C:/Program Files/MetaTrader 5/terminal64.exe"
  metaeditor_exe: "C:/Program Files/MetaTrader 5/metaeditor64.exe"
  data_folder: "C:/Users/<YourUser>/AppData/Roaming/MetaQuotes/Terminal/<HEX>"
  portable_mode: false
```

If you want fully isolated runs (recommended for VPS): enable `portable_mode: true` AND launch MT5 once with `terminal64.exe /portable` so it creates the portable folder structure inside the install dir.

## 4. Connect to Your Broker

1. Open MT5 → File → Login to Trade Account
2. Enter your Darwinex (or other broker) credentials
3. Connect — wait for green "Connected" status

## 5. Download Tick Data

For each symbol you'll backtest:

1. View → Symbols (Ctrl+U)
2. Right-click in Market Watch → Symbols → find your symbol
3. "Show" the symbol if not already shown
4. Click on the symbol → "Bars" tab → set period (e.g., 2018.01.01 → 2025.12.31)
5. "Request" → wait for download (can take 10–30 min for years of tick data)

Repeat for every symbol × period combo your spec requires. The `tick_data_manager.py` script can verify availability (requires the `MetaTrader5` Python package).

Increase max bars: Tools → Options → Charts → "Max bars in chart" → set to maximum.

## 6. MetaEditor Command-Line Compile

Test your install can compile from CLI:

```cmd
"C:\Program Files\MetaTrader 5\metaeditor64.exe" /compile:"C:\path\to\StrategyFactory\mql5\generated\test.mq5" /log:"C:\path\to\compile.log"
```

Exit code 0 = success. Read the .log file for errors/warnings (UTF-16 LE encoded).

## 7. Strategy Tester via .ini Config

```cmd
"C:\Program Files\MetaTrader 5\terminal64.exe" /config:"C:\path\to\tester.ini"
```

The `.ini` must include `ShutdownTerminal=1` so MT5 closes after the test (otherwise the launcher hangs). StrategyFactory generates this automatically.

## 8. Common Issues & Fixes

| Problem | Cause | Fix |
|---|---|---|
| `cannot compile` | Missing include file | Verify `mql5/_template/*.mqh` exists. The compiler `/inc:` flag points to that folder. |
| `tester hangs forever` | `ShutdownTerminal=1` missing OR runtime infinite loop in EA | Check the generated .ini; check OnTick for accidental `while(true)`. |
| `no tick data` | Data not downloaded for that period | Manual download per Step 5. |
| `symbol unavailable` | Symbol code mismatch | Check broker code in Market Watch, update `config/symbols_map.yaml`. |
| `INIT_FAILED` in journal | Indicator handle returned `INVALID_HANDLE` | Check spec's indicator periods are valid. |
| Reports written but with no trades | Risk gates blocking trades, or session never enters | Check `OnInit` log; reduce `max_daily_loss_pct` constraint or check session time format. |
| MT5 keeps the .ex5 locked | MT5 had it loaded for live testing | Close any chart that has the EA attached, then re-compile. |

## 9. VPS-Specific Tips

- Disable Windows updates during long backtest runs (they can reboot the VPS).
- Use Task Scheduler for unattended pipelines.
- If running multiple MT5 instances, install each in a separate folder and use `portable_mode: true`.
- Keep at least 4 GB RAM free per concurrent backtest (tick-based testing is memory-hungry).

## 10. Verify End-to-End

```cmd
scripts\run_full_validation.bat
```

This runs pytest + verifies MT5 paths + validates the example spec. Green output = ready to develop strategies.
