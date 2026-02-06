"""Output formatting helpers."""
import json
import click
from rich.console import Console
from rich.table import Table

console = Console()


def print_json(data):
    """Print raw JSON to stdout."""
    click.echo(json.dumps(data, indent=2, default=str))


def print_kv(pairs: dict, title: str | None = None):
    """Print key-value pairs as a rich table."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    if title:
        console.print(f"\n[bold]{title}[/bold]")
    for k, v in pairs.items():
        table.add_row(str(k), str(v))
    console.print(table)


def print_list_table(rows: list[dict], columns: list[str], title: str | None = None):
    """Print a list of dicts as a rich table."""
    table = Table(title=title)
    for col in columns:
        table.add_column(col, style="cyan" if col.endswith("id") or col.endswith("ID") else None)
    for row in rows:
        table.add_row(*[str(row.get(c, "")) for c in columns])
    console.print(table)
