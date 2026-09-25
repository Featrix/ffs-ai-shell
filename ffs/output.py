"""Output formatting helpers."""
import json
import logging
from contextlib import contextmanager

import click
from rich.console import Console
from rich.table import Table

console = Console()


class _ConsoleLogHandler(logging.Handler):
    """Write a log record straight to the console, as plain text.

    markup=False because a log line is arbitrary text — a path or an error
    containing "[" would otherwise be eaten as rich markup, or raise.
    """

    def emit(self, record):
        console.print(record.getMessage(), style="dim", markup=False, highlight=False)


@contextmanager
def sdk_progress(enabled: bool = True):
    """Surface featrixsphere's own progress logging for the duration of a call.

    The SDK reports long-running work (a publish copies the session to the
    backplane and uploads it to cloud storage — routinely 7-25 minutes) through
    logging.info. ffs configures no logging at all, so those lines went nowhere
    and the command sat silent until it finished, which is indistinguishable
    from a hang.
    """
    if not enabled:
        yield
        return

    logger = logging.getLogger("featrixsphere")
    handler = _ConsoleLogHandler()
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


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
