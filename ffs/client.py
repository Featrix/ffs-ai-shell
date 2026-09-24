import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import click
from featrixsphere.api import FeatrixSphere


def _is_git_tracked(filepath: Path) -> bool:
    """Check if a file is tracked by git."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(filepath)],
            cwd=str(filepath.parent),
            capture_output=True,
        )
        return result.returncode == 0
    except (OSError, FileNotFoundError):
        return False


def shell_cwd_path() -> Path | None:
    """The shell's idea of the working directory, for when the process's is unusable.

    A remounted volume or a replaced directory leaves the process holding a
    stale handle: `os.getcwd()` fails while the path itself still resolves fine.
    $PWD is what the shell believes, and re-resolving it is exactly what
    `cd "$PWD"` does by hand — which is what unstuck the reported case.

    Only consulted once the real cwd has already failed, and only if it still
    points at a directory.
    """
    pwd = os.environ.get("PWD")
    if not pwd:
        return None
    try:
        candidate = Path(pwd)
        return candidate if candidate.is_dir() else None
    except OSError:
        return None


def safe_cwd() -> Path | None:
    """The current working directory, or None if there isn't a usable one.

    A shell can outlive its working directory — an unmounted volume, a deleted
    or rebuilt checkout, a directory replaced so the old inode goes stale. In
    that state `os.getcwd()` raises FileNotFoundError, which as an error message
    names no file and explains nothing ("[Errno 2] No such file or directory").
    Callers use this to degrade deliberately instead of dying on that errno.
    """
    try:
        return Path.cwd()
    except OSError:
        return shell_cwd_path()


def repair_cwd() -> Path | None:
    """Re-enter the working directory if the process's handle to it went stale.

    Returns the directory re-entered, or None if nothing needed repairing (or it
    couldn't be). Called once at startup so everything downstream — a relative
    --data path, the pip subprocess in `upgrade`, project-local .featrix
    discovery — sees a working directory rather than each coping separately.

    This matters beyond avoiding a crash: without it, config discovery would
    skip a project-local .featrix that is sitting right there and silently fall
    back to ~/.featrix, i.e. quietly use a different org's credentials.
    """
    try:
        Path.cwd()
        return None
    except OSError:
        pass

    recovered = shell_cwd_path()
    if recovered is None:
        return None
    try:
        os.chdir(recovered)
    except OSError:
        return None
    return recovered


CWD_GONE_HINT = (
    "The current directory no longer exists — it may have been deleted or "
    "unmounted. cd to a directory that exists (or `cd \"$PWD\"` to re-resolve it)."
)


def find_featrix_config() -> tuple[Path | None, str]:
    """Walk from cwd up to / looking for .featrix, fall back to ~/.featrix.

    Returns (path, source) where source describes where it was found.
    Raises click.ClickException if a .featrix file is tracked by git.
    """
    cwd = safe_cwd()
    home = Path.home()

    # No cwd to walk up from: a project-local .featrix cannot exist in a
    # directory that doesn't, so go straight to the global config rather than
    # failing every authenticated command.
    if cwd is None:
        fallback = home / ".featrix"
        if fallback.is_file():
            return fallback, "~/.featrix"
        return None, "not found"

    # Walk up from cwd
    for d in [cwd, *cwd.parents]:
        candidate = d / ".featrix"
        if candidate.is_file():
            if d != home and _is_git_tracked(candidate):
                raise click.ClickException(
                    f"\n\n"
                    f"  DANGER: {candidate} is tracked by git!\n\n"
                    f"  Your API key will be pushed to the remote repository.\n"
                    f"  Fix this now:\n\n"
                    f"    git rm --cached {candidate}\n"
                    f"    echo .featrix >> .gitignore\n"
                    f"    git commit -m 'Remove .featrix from tracking'\n\n"
                    f"  Then rotate your API key at https://featrix-ui.lovable.app/api-keys\n"
                )
            if d == home:
                return candidate, "~/.featrix"
            return candidate, str(candidate)
        # Stop at home or root
        if d == home or d == d.parent:
            break

    # Fall back to ~/.featrix
    fallback = home / ".featrix"
    if fallback.is_file():
        return fallback, "~/.featrix"

    return None, "not found"


def load_config_from(path: Path) -> dict:
    """Read a .featrix config file (JSON or key=value)."""
    content = path.read_text().strip()
    if content.startswith("{"):
        return json.loads(content)
    config = {}
    for line in content.splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            value = value.strip().strip("'\"")
            config[key.strip().lower()] = value
    return config


@dataclass
class ClientState:
    server: str
    cluster: str | None
    output_json: bool
    quiet: bool
    _client: FeatrixSphere | None = None
    _config_path: Path | None = field(default=None, repr=False)
    _config_source: str = field(default="", repr=False)

    @property
    def config_source(self) -> str:
        if not self._config_source:
            self._config_path, self._config_source = find_featrix_config()
        return self._config_source

    @property
    def client(self) -> FeatrixSphere:
        if self._client is None:
            kwargs = dict(base_url=self.server, compute_cluster=self.cluster)

            # If env var is set, let FeatrixSphere pick it up naturally.
            # Otherwise, read from the nearest .featrix file.
            if not os.getenv("FEATRIX_API_KEY"):
                if not self._config_path:
                    self._config_path, self._config_source = find_featrix_config()
                if self._config_path:
                    config = load_config_from(self._config_path)
                    api_key = config.get("api_key") or config.get("featrix_api_key")
                    if api_key:
                        kwargs["api_key"] = api_key

            self._client = FeatrixSphere(**kwargs)
        return self._client


pass_client = click.make_pass_decorator(ClientState)
