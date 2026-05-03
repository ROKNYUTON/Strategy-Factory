"""
StrategyFactory — EA Generator
==============================
Reads a validated YAML spec, produces:
  1. mql5/generated/{strategy_id}.mq5  — template skeleton with metadata filled
  2. prompts/generation_prompts/{strategy_id}_prompt.md — full prompt for Claude Code

The trader then manually feeds the prompt to Claude Code in VS Code,
reviews the output, and saves it back to mql5/generated/{strategy_id}.mq5.

Usage:
    python automation/ea_generator.py strategy_specs/my_strategy.yaml
"""

from __future__ import annotations

import sys
import hashlib
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from automation.spec_validator import validate_spec  # noqa: E402

console = Console()


def magic_from_strategy_id(strategy_id: str) -> int:
    """Deterministic magic number from strategy_id (8-digit positive int)."""
    h = hashlib.sha256(strategy_id.encode("utf-8")).hexdigest()
    # Keep within int range and reasonable length
    return int(h[:8], 16) % 99999999 + 10000000


def load_template() -> str:
    template_path = ROOT / "mql5" / "_template" / "BaseEA_Template.mq5"
    return template_path.read_text(encoding="utf-8")


def load_master_prompt() -> str:
    p = ROOT / "prompts" / "ea_generation_master.md"
    return p.read_text(encoding="utf-8")


def generate(spec_path: Path) -> dict:
    """Return paths to generated files."""
    console.print(f"[cyan]Loading spec:[/cyan] {spec_path}")
    spec = validate_spec(spec_path)
    spec_yaml_text = spec_path.read_text(encoding="utf-8")

    strategy_id    = spec.meta.strategy_id
    strategy_name  = strategy_id.replace("_", " ").title()
    strategy_magic = magic_from_strategy_id(strategy_id)

    # 1. Generate skeleton .mq5
    template = load_template()
    skeleton = (template
                .replace("{{STRATEGY_ID}}", strategy_id)
                .replace("{{STRATEGY_NAME}}", strategy_name)
                .replace("{{STRATEGY_MAGIC}}", str(strategy_magic)))

    out_mq5 = ROOT / "mql5" / "generated" / f"{strategy_id}.mq5"
    out_mq5.parent.mkdir(parents=True, exist_ok=True)
    out_mq5.write_text(skeleton, encoding="utf-8")

    # 2. Generate Claude Code prompt
    master_prompt = load_master_prompt()
    prompt = (master_prompt
              .replace("{{STRATEGY_ID}}", strategy_id)
              .replace("{{SPEC_YAML_CONTENT}}", spec_yaml_text))

    out_prompt = ROOT / "prompts" / "generation_prompts" / f"{strategy_id}_prompt.md"
    out_prompt.parent.mkdir(parents=True, exist_ok=True)
    out_prompt.write_text(prompt, encoding="utf-8")

    return {
        "strategy_id": strategy_id,
        "magic": strategy_magic,
        "skeleton_mq5": out_mq5,
        "prompt_md": out_prompt,
    }


def main() -> int:
    if len(sys.argv) != 2:
        console.print("[red]Usage: python automation/ea_generator.py <spec.yaml>[/red]")
        return 2

    spec_path = Path(sys.argv[1])
    if not spec_path.exists():
        console.print(f"[red]Spec not found: {spec_path}[/red]")
        return 2

    try:
        result = generate(spec_path)
    except Exception as e:
        console.print(Panel(f"[red]{e}[/red]", title="❌ GENERATION FAILED"))
        return 1

    console.print(Panel(
        f"[green]✅ EA skeleton + Claude Code prompt generated.[/green]\n\n"
        f"[bold]strategy_id:[/bold] {result['strategy_id']}\n"
        f"[bold]magic:[/bold]       {result['magic']}\n"
        f"[bold]skeleton:[/bold]    {result['skeleton_mq5']}\n"
        f"[bold]prompt:[/bold]      {result['prompt_md']}\n\n"
        f"[yellow]NEXT STEPS (manual):[/yellow]\n"
        f"  1. Open the prompt file:\n"
        f"     {result['prompt_md']}\n"
        f"  2. Copy its content into Claude Code in VS Code.\n"
        f"  3. Review the generated MQL5 — the output must MODIFY ONLY the\n"
        f"     regions between AI markers in:\n"
        f"     {result['skeleton_mq5']}\n"
        f"  4. Save the AI-completed file at the same path.\n"
        f"  5. Resume pipeline:  python automation/pipeline.py compile {result['strategy_id']}\n",
        title="EA GENERATOR — DONE"
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
