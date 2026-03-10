"""Tests for ffs/client.py."""
import json
import os
from pathlib import Path
from unittest.mock import patch

from ffs.client import load_config_from, find_featrix_config, ClientState


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
