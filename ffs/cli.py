import json
import os
import platform
import socket
import webbrowser
from pathlib import Path
from urllib.parse import urlencode

import click

from ffs.client import pass_client, ClientState
from ffs.output import print_json, print_kv, console
from ffs import model_cmd
from ffs import server_cmd

FEATRIX_CONFIG = Path.home() / ".featrix"
FEATRIX_UI = "https://featrix-ui.lovable.app"


@click.group()
@click.option("--server", envvar="FFS_SERVER", default="https://sphere-api.featrix.com", help="API server URL")
@click.option("--cluster", envvar="FFS_CLUSTER", default=None, help="Compute cluster name")
@click.option("--json", "output_json", is_flag=True, help="Output raw JSON")
@click.option("--quiet", is_flag=True, help="Minimal output")
@click.pass_context
def main(ctx, server, cluster, output_json, quiet):
    """The Featrix Foundation Shell."""
    ctx.ensure_object(dict)
    ctx.obj = ClientState(server=server, cluster=cluster, output_json=output_json, quiet=quiet)


@main.command()
@click.option("--api-key", default=None, help="Featrix API key (skips browser flow)")
@click.pass_context
def login(ctx, api_key):
    """Authenticate with Featrix and save credentials to ~/.featrix.

    Opens the Featrix UI to create an API key if none is provided.
    """
    state = ctx.obj

    if not api_key:
        # Build the deep-link URL with hostname label
        hostname = socket.gethostname()
        user = os.getenv("USER") or os.getenv("USERNAME") or "unknown"
        label = f"{user}@{hostname}"
        params = urlencode({"create": "true", "label": label})
        url = f"{FEATRIX_UI}/api-keys?{params}"

        console.print(f"\nOpening Featrix to create an API key for [bold]{label}[/bold]...\n")
        console.print(f"  {url}\n")

        # Try to open browser; fine if it fails (SSH, headless, etc.)
        try:
            webbrowser.open(url)
        except Exception:
            pass

        console.print("Copy the API key from the browser and paste it here.\n")
        api_key = click.prompt("API key", hide_input=True)

    # Read existing config if present
    config = {}
    if FEATRIX_CONFIG.exists():
        try:
            content = FEATRIX_CONFIG.read_text().strip()
            if content.startswith("{"):
                config = json.loads(content)
        except (json.JSONDecodeError, OSError):
            pass

    config["api_key"] = api_key

    if state.server != "https://sphere-api.featrix.com":
        config["base_url"] = state.server

    FEATRIX_CONFIG.write_text(json.dumps(config, indent=2) + "\n")
    FEATRIX_CONFIG.chmod(0o600)

    # Verify the key works
    try:
        from featrixsphere.api import FeatrixSphere
        fs = FeatrixSphere(api_key=api_key, base_url=state.server)
        fs.health_check()
        console.print(f"[green]Logged in.[/green] Credentials saved to ~/.featrix")
    except Exception as e:
        console.print(f"[yellow]Credentials saved to ~/.featrix[/yellow], but verification failed: {e}")


@main.command()
@pass_client
def whoami(state: ClientState):
    """Show current user, org, and API connection info."""
    identity = state.client.whoami()

    key_source = "~/.featrix"
    if os.getenv("FEATRIX_API_KEY"):
        key_source = "FEATRIX_API_KEY env var"

    identity["server"] = state.server
    if state.cluster:
        identity["cluster"] = state.cluster
    identity["api_key_source"] = key_source

    if state.output_json:
        print_json(identity)
    else:
        print_kv(identity, title="Featrix Identity")


main.add_command(model_cmd.model, "foundation")
main.add_command(server_cmd.server)
