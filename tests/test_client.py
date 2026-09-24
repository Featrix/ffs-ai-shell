"""Tests for ffs/client.py."""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest

from ffs.client import (
    ClientState,
    _is_git_tracked,
    find_featrix_config,
    load_config_from,
)


class TestLoadConfig:
    def test_json_format(self, tmp_path):
        cfg = tmp_path / ".featrix"
        cfg.write_text(json.dumps({"api_key": "sk_live_abc", "base_url": "https://example.com"}))
        result = load_config_from(cfg)
        assert result["api_key"] == "sk_live_abc"
        assert result["base_url"] == "https://example.com"

    def test_keyvalue_format(self, tmp_path):
        cfg = tmp_path / ".featrix"
        cfg.write_text("api_key=sk_live_xyz\nbase_url=https://example.com\n# comment\n")
        result = load_config_from(cfg)
        assert result["api_key"] == "sk_live_xyz"
        assert result["base_url"] == "https://example.com"

    def test_keyvalue_strips_quotes(self, tmp_path):
        cfg = tmp_path / ".featrix"
        cfg.write_text("api_key='sk_live_quoted'\n")
        result = load_config_from(cfg)
        assert result["api_key"] == "sk_live_quoted"


class TestFindFeatrixConfig:
    def test_finds_in_cwd(self, tmp_path):
        cfg = tmp_path / ".featrix"
        cfg.write_text('{"api_key": "test"}')
        with patch("ffs.client.Path.cwd", return_value=tmp_path):
            path, source = find_featrix_config()
        assert path == cfg

    def test_returns_none_when_not_found(self, tmp_path):
        with patch("ffs.client.Path.cwd", return_value=tmp_path), \
             patch("ffs.client.Path.home", return_value=tmp_path):
            path, source = find_featrix_config()
        assert path is None
        assert source == "not found"


class TestClientState:
    def test_client_created_lazily(self, mock_sphere):
        state = ClientState(server="https://test.com", cluster=None, output_json=False, quiet=False)
        with patch.dict(os.environ, {"FEATRIX_API_KEY": "sk_test"}):
            client = state.client
        assert client is not None
        assert state._client is client

    def test_client_reads_config_when_no_env(self, mock_sphere, tmp_path):
        cfg = tmp_path / ".featrix"
        cfg.write_text(json.dumps({"api_key": "sk_from_file"}))
        state = ClientState(server="https://test.com", cluster=None, output_json=False, quiet=False)
        with patch.dict(os.environ, {}, clear=False):
            # Remove FEATRIX_API_KEY if set
            os.environ.pop("FEATRIX_API_KEY", None)
            with patch("ffs.client.find_featrix_config", return_value=(cfg, str(cfg))):
                _ = state.client

    def test_config_source_lazy(self):
        state = ClientState(server="https://test.com", cluster=None, output_json=False, quiet=False)
        with patch("ffs.client.find_featrix_config", return_value=(None, "not found")):
            assert state.config_source == "not found"


class TestGitTrackedConfigGuard:
    """A .featrix committed to git pushes the user's API key to the remote.

    find_featrix_config() refuses to hand back a tracked config rather than
    quietly using it, so these tests cover the refusal and — just as important —
    that it doesn't fire on the ordinary untracked case.
    """

    def test_refuses_a_tracked_config(self, tmp_path, monkeypatch):
        config = tmp_path / ".featrix"
        config.write_text('{"api_key": "fx_secret"}')
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "elsewhere"))
        with patch("ffs.client._is_git_tracked", return_value=True):
            with pytest.raises(click.ClickException) as exc:
                find_featrix_config()
        message = exc.value.format_message()
        assert "DANGER" in message
        assert "git rm --cached" in message
        assert "rotate your api key" in message.lower()

    def test_allows_an_untracked_config(self, tmp_path, monkeypatch):
        config = tmp_path / ".featrix"
        config.write_text('{"api_key": "fx_secret"}')
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "elsewhere"))
        with patch("ffs.client._is_git_tracked", return_value=False):
            found, _ = find_featrix_config()
        assert found == config

    def test_does_not_check_the_home_config(self, tmp_path, monkeypatch):
        """~/.featrix is the documented global location; it isn't in a repo."""
        home = tmp_path / "home"
        home.mkdir()
        (home / ".featrix").write_text('{"api_key": "fx_secret"}')
        monkeypatch.chdir(home)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        with patch("ffs.client._is_git_tracked", return_value=True) as tracked:
            found, source = find_featrix_config()
        assert found == home / ".featrix"
        assert source == "~/.featrix"
        tracked.assert_not_called()

    def test_falls_back_to_the_home_config_from_an_unrelated_directory(
        self, tmp_path, monkeypatch
    ):
        home = tmp_path / "home"
        home.mkdir()
        (home / ".featrix").write_text('{"api_key": "fx_secret"}')
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        found, source = find_featrix_config()
        assert found == home / ".featrix"
        assert source == "~/.featrix"


class TestIsGitTracked:
    def test_tracked_file(self, tmp_path):
        with patch("ffs.client.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0)
            assert _is_git_tracked(tmp_path / ".featrix") is True

    def test_untracked_file(self, tmp_path):
        with patch("ffs.client.subprocess.run") as run:
            run.return_value = MagicMock(returncode=1)
            assert _is_git_tracked(tmp_path / ".featrix") is False

    def test_treats_a_missing_git_binary_as_untracked(self, tmp_path):
        """No git means no repo to leak into — and must not crash every command."""
        with patch("ffs.client.subprocess.run", side_effect=FileNotFoundError):
            assert _is_git_tracked(tmp_path / ".featrix") is False
