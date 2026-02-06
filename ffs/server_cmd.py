"""ffs server subcommands."""
import click

from ffs.client import pass_client, ClientState
from ffs.output import print_json, print_kv, console


@click.group()
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
        print_kv(result, title="Server Health")
