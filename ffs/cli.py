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
from ffs.click_ext import DYMGroup
from ffs.output import print_json, print_kv, console
from ffs import model_cmd
from ffs import predictor_cmd
from ffs import predict_cmd
from ffs import server_cmd
from ffs import train_cmd
from ffs import jobs_cmd
from ffs import network_cmd
from ffs import endpoint_cmd
from ffs import events_cmd
FEATRIX_UI = "https://featrix-ui.lovable.app"


def _get_version_string():
    from ffs import __version__ as ffs_ver
    try:
        from importlib.metadata import version
        sphere_ver = version("featrixsphere")
    except Exception:
        sphere_ver = "unknown"
    return f"ffs {ffs_ver}  (featrixsphere {sphere_ver})"


def _completions_installed():
    """Check if shell tab completion appears to be set up."""
    home = Path.home()
    for rc in (".bashrc", ".bash_profile", ".zshrc", ".profile"):
        rc_path = home / rc
        if rc_path.is_file():
            try:
                content = rc_path.read_text()
                if "ffs completions" in content or "_ffs_completion" in content:
                    return True
            except OSError:
                pass
    return False


@click.group(cls=DYMGroup, invoke_without_command=True)
@click.option("--server", envvar="FFS_SERVER", default="https://sphere-api.featrix.com", hidden=True, help="API server URL")
@click.option("--cluster", envvar="FFS_CLUSTER", default=None, hidden=True, help="Compute cluster name")
@click.option("--json", "output_json", is_flag=True, help="Output raw JSON")
@click.option("--quiet", is_flag=True, help="Minimal output")
@click.version_option(
    version=_get_version_string(),
    message="%(version)s",
)
@click.pass_context
def main(ctx, server, cluster, output_json, quiet):
    """The Featrix Foundation Shell."""
    ctx.ensure_object(dict)
    ctx.obj = ClientState(server=server, cluster=cluster, output_json=output_json, quiet=quiet)

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        if not _completions_installed():
            console.print()
            console.print("[bold cyan]Tip:[/bold cyan] Enable tab completion for ffs:")
            console.print("  [green]echo 'eval \"$(ffs completions)\"' >> ~/.bashrc[/green]")
            console.print("  [dim]For zsh: replace .bashrc with .zshrc[/dim]")


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


def _pkg_version(pkg):
    import importlib
    importlib.invalidate_caches()
    from importlib.metadata import version, PackageNotFoundError
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


@main.command()
@click.option("--break-system-packages", is_flag=True, hidden=True, help="Pass --break-system-packages to pip")
def upgrade(break_system_packages):
    """Upgrade featrix-shell and featrixsphere to latest."""
    for pkg in ("featrix-shell", "featrixsphere"):
        before = _pkg_version(pkg)
        console.print(f"Upgrading [bold]{pkg}[/bold] ({before or 'not installed'})...")
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", pkg]
        if break_system_packages:
            cmd.append("--break-system-packages")
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            after = _pkg_version(pkg)
            if before != after:
                console.print(f"  [green]done[/green]: {before or 'not installed'} -> {after}")
            else:
                console.print(f"  [green]done[/green]: already {after}")
        else:
            console.print(f"  [red]failed[/red]: {result.stderr.strip()}")


def _bash_supports_nosort():
    """Check whether the system bash is new enough (>=4.4) for click's stock
    completion script, which relies on `complete -o nosort`."""
    import re
    import shutil
    import subprocess

    bash_exe = shutil.which("bash")
    if bash_exe is None:
        return False
    try:
        output = subprocess.run(
            [bash_exe, "--norc", "-c", 'echo "${BASH_VERSION}"'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    match = re.search(r"^(\d+)\.(\d+)\.\d+", output.stdout.decode())
    if not match:
        return False
    return (int(match.group(1)), int(match.group(2))) >= (4, 4)


# macOS ships bash 3.2 (Apple froze it there over the GPLv3 relicensing) and never
# updates it, so `complete -o nosort` (added in 4.4) and `compopt` (added in 4.0)
# aren't available. Click's stock bash template uses both, which makes `complete`
# fail outright and silently skips registering completion. This mirrors that
# template with `-o nosort` dropped and `compopt` calls guarded.
_BASH_LEGACY_COMPLETION = """\
_ffs_completion() {
    local IFS=$'\\n'
    local response

    response=$(env COMP_WORDS="${COMP_WORDS[*]}" COMP_CWORD=$COMP_CWORD _FFS_COMPLETE=bash_complete $1)

    for completion in $response; do
        IFS=',' read type value <<< "$completion"

        if [[ $type == 'dir' ]]; then
            COMPREPLY=()
            type compopt &>/dev/null && compopt -o dirnames
        elif [[ $type == 'file' ]]; then
            COMPREPLY=()
            type compopt &>/dev/null && compopt -o default
        elif [[ $type == 'plain' ]]; then
            COMPREPLY+=($value)
        fi
    done

    return 0
}

_ffs_completion_setup() {
    complete -F _ffs_completion ffs
}

_ffs_completion_setup;
"""


@main.command()
@click.option("--shell", "shell", default="bash", type=click.Choice(["bash", "zsh", "fish"]), help="Shell type")
def completions(shell):
    """Print shell completion script.

    Activate with:  eval "$(ffs completions)"
    """
    if shell == "bash" and not _bash_supports_nosort():
        click.echo(_BASH_LEGACY_COMPLETION)
        return

    from click.shell_completion import get_completion_class
    comp_cls = get_completion_class(shell)
    comp = comp_cls(main, {}, "ffs", "_FFS_COMPLETE")
    click.echo(comp.source())


@main.command("agent-help")
def agent_help():
    """Print the agent reference guide for ffs."""
    from importlib.resources import files
    guide = files("ffs").joinpath("agent_guide.txt").read_text()
    click.echo(guide)


main.add_command(model_cmd.model, "models")
main.add_command(model_cmd.model, "foundation")  # legacy alias
main.add_command(predictor_cmd.predictor)
main.add_command(predict_cmd.predict)
main.add_command(train_cmd.train)
main.add_command(server_cmd.server)
main.add_command(jobs_cmd.jobs)
main.add_command(network_cmd.network)
main.add_command(endpoint_cmd.endpoint)
main.add_command(events_cmd.events)


def _emit_error(message, status=None):
    """Print an error either as JSON (if --json was requested) or as rich text."""
    if "--json" in sys.argv:
        payload = {"error": {"message": message}}
        if status is not None:
            payload["error"]["status"] = status
        click.echo(json.dumps(payload))
    else:
        console.print(f"[red]Error:[/red] {message}")


def cli():
    """Entry point that catches exceptions cleanly."""
    try:
        main(standalone_mode=False)
    except click.ClickException as e:
        _emit_error(e.format_message())
        sys.exit(e.exit_code)
    except SystemExit:
        raise
    except Exception as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        _emit_error(str(e), status=status)
        sys.exit(1)
