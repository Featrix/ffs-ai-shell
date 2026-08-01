"""Tests for ffs/cli.py - main CLI and top-level commands."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from ffs.cli import main, cli


class TestMainGroup:
    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "foundation" in result.output
        assert "predictor" in result.output
        assert "predict" in result.output

    def test_version_not_required(self, runner):
        # Just verify the group works without subcommands
        result = runner.invoke(main, [])
        # Should show help or usage
        assert result.exit_code == 0


class TestWhoami:
    def test_whoami(self, runner, mock_sphere, env):
        mock_sphere.whoami.return_value = {"user": "testuser", "org": "testorg"}
        result = runner.invoke(main, ["whoami"], env=env)
        assert result.exit_code == 0
        assert "testuser" in result.output

    def test_whoami_json(self, runner, mock_sphere, env):
        mock_sphere.whoami.return_value = {"user": "testuser", "org": "testorg"}
        result = runner.invoke(main, ["--json", "whoami"], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["user"] == "testuser"
        assert "server" in data
        assert "api_key_source" in data

    def test_whoami_shows_cluster(self, runner, mock_sphere, env):
        mock_sphere.whoami.return_value = {"user": "testuser"}
        result = runner.invoke(main, ["--json", "--cluster", "gpu-1", "whoami"], env=env)
        data = json.loads(result.output)
        assert data["cluster"] == "gpu-1"


class TestLogin:
    def test_login_with_api_key(self, runner):
        with runner.isolated_filesystem():
            with patch("featrixsphere.api.FeatrixSphere") as MockFS:
                MockFS.return_value.health_check.return_value = {"status": "ok"}
                result = runner.invoke(main, ["login", "--api-key", "sk_test_123"])
                assert result.exit_code == 0
                assert "Logged in" in result.output
                config = json.loads(Path(".featrix").read_text())
                assert config["api_key"] == "sk_test_123"

    def test_login_verification_failure(self, runner):
        with runner.isolated_filesystem():
            with patch("featrixsphere.api.FeatrixSphere") as MockFS:
                MockFS.return_value.health_check.side_effect = Exception("Connection refused")
                result = runner.invoke(main, ["login", "--api-key", "sk_bad_key"])
                assert result.exit_code == 0
                assert "verification failed" in result.output
                # Key is still saved even on verification failure
                assert Path(".featrix").exists()

    def test_login_global(self, runner, tmp_path):
        with patch("featrixsphere.api.FeatrixSphere") as MockFS:
            MockFS.return_value.health_check.return_value = {"status": "ok"}
            with patch("ffs.cli.Path.home", return_value=tmp_path):
                result = runner.invoke(main, ["login", "--api-key", "sk_test_456", "--global"])
                assert result.exit_code == 0
                assert (tmp_path / ".featrix").exists()


class TestCliErrorHandling:
    """The `cli()` entry point (not `main` itself) is what --json-formats
    errors, since it wraps the whole run in a try/except."""

    def test_emits_plain_error_by_default(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.argv", ["ffs", "whoami"])
        monkeypatch.setattr("ffs.cli.main", MagicMock(side_effect=RuntimeError("boom")))
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 1
        assert "boom" in capsys.readouterr().out

    def test_emits_json_error_in_json_mode(self, capsys, monkeypatch):
        monkeypatch.setattr("sys.argv", ["ffs", "--json", "whoami"])
        monkeypatch.setattr("ffs.cli.main", MagicMock(side_effect=RuntimeError("boom")))
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 1
        data = json.loads(capsys.readouterr().out)
        assert data["error"]["message"] == "boom"

    def test_json_error_includes_http_status_when_present(self, capsys, monkeypatch):
        err = RuntimeError("404 Client Error: Not Found for url: https://sphere-api.featrix.com/x")
        err.response = MagicMock(status_code=404)
        monkeypatch.setattr("sys.argv", ["ffs", "--json", "predict", "sid", "{}"])
        monkeypatch.setattr("ffs.cli.main", MagicMock(side_effect=err))
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code == 1
        data = json.loads(capsys.readouterr().out)
        assert data["error"]["status"] == 404


class TestDidYouMean:
    """Typo'd subcommands should suggest the closest real command."""

    def test_suggests_close_match_at_top_level(self, runner):
        result = runner.invoke(main, ["networks"])
        assert result.exit_code == 2
        assert "No such command 'networks'" in result.output
        assert "Did you mean 'network'?" in result.output

    def test_suggests_close_match_in_subgroup(self, runner):
        result = runner.invoke(main, ["network", "lst"])
        assert result.exit_code == 2
        assert "No such command 'lst'" in result.output
        assert "Did you mean 'list'?" in result.output

    def test_no_suggestion_when_nothing_close(self, runner):
        result = runner.invoke(main, ["zzzzzzzz"])
        assert result.exit_code == 2
        assert "No such command 'zzzzzzzz'" in result.output
        assert "Did you mean" not in result.output


class TestUpgrade:
    def test_upgrade_success(self, runner):
        with patch("ffs.cli.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            result = runner.invoke(main, ["upgrade"])
            assert result.exit_code == 0
            assert mock_run.call_count == 2

    def test_upgrade_failure(self, runner):
        with patch("ffs.cli.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="pip error")
            result = runner.invoke(main, ["upgrade"])
            assert result.exit_code == 0  # upgrade doesn't fail the CLI
            assert "failed" in result.output

    def test_upgrade_prints_before_and_after_versions(self, runner):
        with patch("ffs.cli.subprocess.run") as mock_run, \
             patch("ffs.cli._pkg_version") as mock_pkg_version:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            mock_pkg_version.side_effect = ["0.5.12", "0.5.13", "2.0.11488", "2.0.11810"]
            result = runner.invoke(main, ["upgrade"])
            assert result.exit_code == 0
            assert "0.5.12" in result.output and "0.5.13" in result.output
            assert "2.0.11488" in result.output and "2.0.11810" in result.output
