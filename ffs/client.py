import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import click
from featrixsphere.api import FeatrixSphere


def find_featrix_config() -> tuple[Path | None, str]:
    """Walk from cwd up to / looking for .featrix, fall back to ~/.featrix.

    Returns (path, source) where source describes where it was found.
    """
    cwd = Path.cwd()
    home = Path.home()

    # Walk up from cwd
    for d in [cwd, *cwd.parents]:
        candidate = d / ".featrix"
        if candidate.is_file():
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
