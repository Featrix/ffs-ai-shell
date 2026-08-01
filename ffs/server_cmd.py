"""ffs server subcommands."""
import click

from ffs.click_ext import DYMGroup
from ffs.client import pass_client, ClientState
from ffs.output import print_json, print_kv, console


@click.group(cls=DYMGroup)
def server():
    """Server operations."""
    pass


@server.command()
@pass_client
def health(state: ClientState):
    """Check API server health."""
    result = state.client.health_check()
    if state.output_json:
        print_json(result)
    else:
        status = result.get("status", "unknown")
        version = result.get("version", "?")
        color = "green" if status == "healthy" else "red"
        console.print(f"[{color}]{status}[/{color}]  v{version}")
        nodes = result.get("nodes", {})
        for name, info in sorted(nodes.items()):
            nstatus = info.get("status", "?")
            nversion = info.get("version", "?")
            nc = "green" if nstatus == "healthy" else "red"
            console.print(f"  [{nc}]{name}[/{nc}]  v{nversion}")
