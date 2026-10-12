"""Rich UI helpers for CLI commands.

Provides consistent, polished terminal output across all CLI commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


console = Console()


def print_header(title: str, subtitle: str = "") -> None:
    """Print a styled command header."""
    console.print()
    console.rule(f"[bold magenta]{title}[/bold magenta]", style="bright_blue")
    if subtitle:
        console.print(f"  {subtitle}")
    console.print()


def print_success(message: str) -> None:
    console.print(f"  [green]✓[/green] {message}")


def print_error(message: str) -> None:
    console.print(f"  [red]✗[/red] {message}")


def print_warning(message: str) -> None:
    console.print(f"  [yellow]![/yellow] {message}")


def print_info(key: str, value: Any) -> None:
    console.print(f"  [dim]{key}:[/dim] [bold]{value}[/bold]")


def print_kv_block(items: dict[str, Any]) -> None:
    """Print a block of key-value pairs."""
    max_key = max(len(k) for k in items) if items else 0
    for key, val in items.items():
        console.print(f"  [cyan]{key:<{max_key}}[/cyan]  {val}")


def print_footer() -> None:
    console.rule(style="bright_blue")
    console.print()


def print_path(label: str, path: Path | str) -> None:
    console.print(f"  [dim]{label}:[/dim] [bold blue]{path}[/bold blue]")


def make_table(
    title: str,
    columns: list[tuple[str, dict[str, Any]]],
    rows: list[list[str | Text]],
    **kwargs: Any,
) -> Table:
    """Create a standard styled table."""
    table = Table(
        title=title,
        show_lines=False,
        border_style="dim",
        title_style="bold",
        pad_edge=True,
        **kwargs,
    )
    for col_name, col_kwargs in columns:
        table.add_column(col_name, **col_kwargs)
    for row in rows:
        table.add_row(*[str(c) if not isinstance(c, Text) else c for c in row])
    return table


def make_config_tree(data: dict[str, Any], label: str = "Configuration") -> Tree:
    """Render a nested dict as a Rich tree."""
    tree = Tree(f"[bold]{label}[/bold]")
    _add_to_tree(tree, data)
    return tree


def _add_to_tree(parent: Tree, data: dict | list | Any) -> None:
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, (dict, list)):
                branch = parent.add(f"[cyan]{key}[/cyan]")
                _add_to_tree(branch, val)
            else:
                parent.add(f"[cyan]{key}:[/cyan] {val}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                branch = parent.add(f"[dim][{i}][/dim]")
                _add_to_tree(branch, item)
            else:
                parent.add(str(item))
    else:
        parent.add(str(data))
