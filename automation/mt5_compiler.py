"""
StrategyFactory — MT5 Compiler Wrapper
======================================
Wraps `metaeditor64.exe /compile:"file.mq5" /log:"compile.log"` to compile
generated EAs and report errors/warnings back to the pipeline.

Usage:
    python automation/mt5_compiler.py mql5/generated/STR_001_asian_mr_fx.mq5
"""

from __future__ import annotations

import sys
import subprocess
import time
import shutil
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).parent.parent
console = Console()


@dataclass
class CompileResult:
    success: bool
    ex5_path: Path | None = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    compile_time_sec: float = 0.0
    log_text: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "ex5_path": str(self.ex5_path) if self.ex5_path else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "compile_time_sec": self.compile_time_sec,
        }


def load_paths() -> dict:
    cfg = ROOT / "config" / "mt5_paths.yaml"
    with cfg.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_log(log_text: str) -> tuple[List[str], List[str]]:
    """Parse MetaEditor compile log for errors and warnings."""
    errors, warnings = [], []
    for line in log_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # MetaEditor log lines look like:
        #   path(line,col) : error 123: message
        #   path(line,col) : warning 456: message
        if re.search(r":\s*error\s+\d+:", line, re.IGNORECASE):
            errors.append(line)
        elif re.search(r":\s*warning\s+\d+:", line, re.IGNORECASE):
            warnings.append(line)
    return errors, warnings


def compile_ea(mq5_path: Path, timeout: int = 60) -> CompileResult:
    """Compile an .mq5 file via metaeditor64.exe."""
    paths = load_paths()
    metaeditor = Path(paths["mt5"]["metaeditor_exe"])

    if not metaeditor.exists():
        return CompileResult(
            success=False,
            errors=[f"metaeditor not found at {metaeditor}. "
                    f"Update config/mt5_paths.yaml."]
        )
    if not mq5_path.exists():
        return CompileResult(
            success=False,
            errors=[f"Source not found: {mq5_path}"]
        )

    log_path = mq5_path.with_suffix(".log")
    # Remove stale log
    if log_path.exists():
        log_path.unlink()

    cmd = [
        str(metaeditor),
        f"/compile:{str(mq5_path)}",
        f"/log:{str(log_path)}",
        "/inc:" + str(mq5_path.parent.parent / "_template"),
    ]

    console.print(f"[cyan]Compiling:[/cyan] {mq5_path.name}")
    t0 = time.time()
    try:
        # MetaEditor exits 0 on success, non-zero on compile errors
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - t0
    except subprocess.TimeoutExpired:
        return CompileResult(
            success=False,
            errors=[f"Compile timeout after {timeout}s"],
            compile_time_sec=time.time() - t0,
        )

    log_text = log_path.read_text(encoding="utf-16-le", errors="ignore") if log_path.exists() else ""
    # MetaEditor logs are UTF-16 LE
    if not log_text and proc.stdout:
        log_text = proc.stdout

    errors, warnings = parse_log(log_text)

    ex5_path = mq5_path.with_suffix(".ex5")
    if not ex5_path.exists() and not errors:
        # No errors found in log but no .ex5 — treat as failure
        errors.append("No .ex5 produced and no errors parsed. Check log manually.")

    success = ex5_path.exists() and not errors

    # Copy to mql5/compiled/
    final_ex5 = None
    if success:
        compiled_dir = ROOT / "mql5" / "compiled"
        compiled_dir.mkdir(parents=True, exist_ok=True)
        final_ex5 = compiled_dir / ex5_path.name
        shutil.copy2(ex5_path, final_ex5)

    return CompileResult(
        success=success,
        ex5_path=final_ex5,
        errors=errors,
        warnings=warnings,
        compile_time_sec=elapsed,
        log_text=log_text,
    )


def print_result(result: CompileResult, mq5_path: Path) -> None:
    if result.success:
        panel = Panel(
            f"[green]✅ Compiled in {result.compile_time_sec:.2f}s[/green]\n"
            f"Output: {result.ex5_path}",
            title="COMPILE OK"
        )
    else:
        body = "[red]❌ Compilation failed[/red]\n"
        for e in result.errors[:20]:
            body += f"  • {e}\n"
        if len(result.errors) > 20:
            body += f"  ... and {len(result.errors) - 20} more\n"
        panel = Panel(body, title="COMPILE FAILED")

    console.print(panel)

    if result.warnings:
        t = Table(title="Warnings")
        t.add_column("Message")
        for w in result.warnings[:10]:
            t.add_row(w)
        console.print(t)


def main() -> int:
    if len(sys.argv) != 2:
        console.print("[red]Usage: python automation/mt5_compiler.py <file.mq5>[/red]")
        return 2
    path = Path(sys.argv[1])
    res = compile_ea(path)
    print_result(res, path)
    return 0 if res.success else 1


if __name__ == "__main__":
    sys.exit(main())
