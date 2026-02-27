import json
import os
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path
from urllib.parse import urlencode

import click

from ffs.client import pass_client, ClientState, find_featrix_config, load_config_from
from ffs.output import print_json, print_kv, console
from ffs import model_cmd
from ffs import server_cmd
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
@click.option("--global", "save_global", is_flag=True, help="Save to ~/.featrix instead of ./.featrix")
@click.pass_context
def login(ctx, api_key, save_global):
    """Authenticate with Featrix and save credentials.

    Saves to .featrix in the current directory (project-local) by default.
    Use --global to save to ~/.featrix instead.
    """
    state = ctx.obj

    if not api_key:
        hostname = socket.gethostname()
        user = os.getenv("USER") or os.getenv("USERNAME") or "unknown"
        label = f"{user}@{hostname}"
        params = urlencode({"create": "true", "label": label})
        url = f"{FEATRIX_UI}/api-keys?{params}"

        console.print(f"\nOpening Featrix to create an API key for [bold]{label}[/bold]...\n")
        console.print(f"  {url}\n")

        try:
            webbrowser.open(url)
        except Exception:
            pass

        console.print("Copy the API key from the browser and paste it here.\n")
        api_key = click.prompt("API key", hide_input=True)

    config_path = Path.home() / ".featrix" if save_global else Path.cwd() / ".featrix"

    # Read existing config if present
    config = {}
    if config_path.exists():
        try:
            config = load_config_from(config_path)
        except (json.JSONDecodeError, OSError):
            pass

    config["api_key"] = api_key

    if state.server != "https://sphere-api.featrix.com":
        config["base_url"] = state.server

    config_path.write_text(json.dumps(config, indent=2) + "\n")
    config_path.chmod(0o600)

    # Verify the key works
    try:
        from featrixsphere.api import FeatrixSphere
        fs = FeatrixSphere(api_key=api_key, base_url=state.server)
        fs.health_check()
        console.print(f"[green]Logged in.[/green] Credentials saved to {config_path}")
    except Exception as e:
        console.print(f"[yellow]Credentials saved to {config_path}[/yellow], but verification failed: {e}")


@main.command()
@pass_client
def whoami(state: ClientState):
    """Show current user, org, and API connection info."""
    identity = state.client.whoami()

    if os.getenv("FEATRIX_API_KEY"):
        key_source = "FEATRIX_API_KEY env var"
    else:
        key_source = state.config_source

    identity["server"] = state.server
    if state.cluster:
        identity["cluster"] = state.cluster
    identity["api_key_source"] = key_source

    if state.output_json:
        print_json(identity)
    else:
        print_kv(identity, title="Featrix Identity")


@main.command()
def upgrade():
    """Upgrade featrix-shell and featrixsphere to latest."""
    for pkg in ("featrix-shell", "featrixsphere"):
        console.print(f"Upgrading [bold]{pkg}[/bold]...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", pkg],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            console.print(f"  [green]done[/green]")
        else:
            console.print(f"  [red]failed[/red]: {result.stderr.strip()}")


main.add_command(model_cmd.model, "foundation")
main.add_command(server_cmd.server)


def cli():
    """Entry point that catches exceptions cleanly."""
    try:
        main(standalone_mode=False)
    except click.ClickException as e:
        console.print(f"[red]Error:[/red] {e.format_message()}")
        sys.exit(e.exit_code)
    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
