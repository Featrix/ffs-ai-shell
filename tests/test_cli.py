"""Tests for ffs/cli.py - main CLI and top-level commands."""
import json
import os
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


class TestCompletions:
    """`ffs completions` — the script users eval into their shell.

    macOS ships bash 3.2 (frozen over the GPLv3 relicensing), which has neither
    `complete -o nosort` (bash 4.4) nor `compopt` (bash 4.0). Click's stock bash
    template uses both, so on a stock Mac `complete` fails and completion never
    registers — hence the legacy template these tests pin.
    """

    def test_bash_is_the_default(self, runner):
        result = runner.invoke(main, ["completions"])
        assert result.exit_code == 0, result.output
        assert "_ffs_completion" in result.output

    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
    def test_every_supported_shell_emits_a_script(self, runner, shell):
        result = runner.invoke(main, ["completions", "--shell", shell])
        assert result.exit_code == 0, result.output
        assert result.output.strip()
        assert "_FFS_COMPLETE" in result.output

    def test_rejects_an_unsupported_shell(self, runner):
        result = runner.invoke(main, ["completions", "--shell", "tcsh"])
        assert result.exit_code == 2

    def test_legacy_template_on_old_bash_avoids_nosort_and_guards_compopt(self, runner):
        with patch("ffs.cli._bash_supports_nosort", return_value=False):
            result = runner.invoke(main, ["completions", "--shell", "bash"])
        assert result.exit_code == 0, result.output
        assert "nosort" not in result.output
        # compopt must be probed before use, not called blind.
        assert "type compopt &>/dev/null && compopt" in result.output
        assert "complete -F _ffs_completion ffs" in result.output

    def test_modern_bash_gets_clicks_own_template(self, runner):
        with patch("ffs.cli._bash_supports_nosort", return_value=True):
            result = runner.invoke(main, ["completions", "--shell", "bash"])
        assert result.exit_code == 0, result.output
        assert "nosort" in result.output

    def test_zsh_and_fish_are_unaffected_by_the_bash_version(self, runner):
        """The old-bash workaround must not leak into the other shells' scripts.

        `complete -F` is a bashism; emitting it for zsh or fish would break the
        eval outright.
        """
        with patch("ffs.cli._bash_supports_nosort", return_value=False):
            for shell in ("zsh", "fish"):
                result = runner.invoke(main, ["completions", "--shell", shell])
                assert result.exit_code == 0, result.output
                assert "complete -F _ffs_completion ffs" not in result.output

    def test_output_is_evalable_shell_and_not_rich_markup(self, runner):
        """The output gets eval'd; rich's markup tags would be syntax errors."""
        result = runner.invoke(main, ["completions"])
        assert "[/" not in result.output
        assert "[bold" not in result.output


class TestBashSupportsNosort:
    def test_detects_a_modern_bash(self):
        from ffs.cli import _bash_supports_nosort
        with patch("shutil.which", return_value="/bin/bash"), \
             patch("subprocess.run") as run:
            run.return_value = MagicMock(stdout=b"5.2.15(1)-release\n")
            assert _bash_supports_nosort() is True

    def test_detects_the_bash_that_ships_with_macos(self):
        from ffs.cli import _bash_supports_nosort
        with patch("shutil.which", return_value="/bin/bash"), \
             patch("subprocess.run") as run:
            run.return_value = MagicMock(stdout=b"3.2.57(1)-release\n")
            assert _bash_supports_nosort() is False

    def test_treats_a_missing_bash_as_unsupported(self):
        from ffs.cli import _bash_supports_nosort
        with patch("shutil.which", return_value=None):
            assert _bash_supports_nosort() is False

    def test_treats_unparseable_output_as_unsupported(self):
        from ffs.cli import _bash_supports_nosort
        with patch("shutil.which", return_value="/bin/bash"), \
             patch("subprocess.run") as run:
            run.return_value = MagicMock(stdout=b"who knows\n")
            assert _bash_supports_nosort() is False


class TestAgentHelp:
    def test_prints_the_guide(self, runner):
        result = runner.invoke(main, ["agent-help"])
        assert result.exit_code == 0, result.output
        assert len(result.output) > 500

    def test_guide_ships_with_the_package(self):
        """agent_guide.txt is declared as package data; a missing file breaks this."""
        from importlib.resources import files
        assert files("ffs").joinpath("agent_guide.txt").is_file()

    def test_guide_mentions_the_json_flag_agents_depend_on(self, runner):
        result = runner.invoke(main, ["agent-help"])
        assert "--json" in result.output


class TestLoginBrowserFlow:
    """`login` with no --api-key opens the UI and prompts for the pasted key."""

    def test_opens_the_browser_and_saves_the_pasted_key(self, runner):
        with runner.isolated_filesystem():
            with patch("ffs.cli.webbrowser.open") as browser, \
                 patch("featrixsphere.api.FeatrixSphere") as MockFS:
                MockFS.return_value.health_check.return_value = {"status": "healthy"}
                result = runner.invoke(main, ["login"], input="fx_pasted_key\n")
            assert result.exit_code == 0, result.output
            browser.assert_called_once()
            assert json.loads(Path(".featrix").read_text())["api_key"] == "fx_pasted_key"

    def test_labels_the_key_with_user_and_host(self, runner):
        """The label is what the user sees in the UI's key list, so it must identify this machine."""
        with runner.isolated_filesystem():
            with patch("ffs.cli.webbrowser.open"), \
                 patch("ffs.cli.socket.gethostname", return_value="devbox"), \
                 patch("featrixsphere.api.FeatrixSphere"), \
                 patch.dict(os.environ, {"USER": "mitch"}):
                result = runner.invoke(main, ["login"], input="fx_k\n")
            assert "mitch@devbox" in result.output

    def test_survives_a_browser_that_cannot_open(self, runner):
        """Headless boxes can't open a browser; the URL is printed for copying."""
        with runner.isolated_filesystem():
            with patch("ffs.cli.webbrowser.open", side_effect=OSError("no display")), \
                 patch("featrixsphere.api.FeatrixSphere"):
                result = runner.invoke(main, ["login"], input="fx_k\n")
            assert result.exit_code == 0, result.output
            assert "api-keys" in result.output
            assert Path(".featrix").exists()

    def test_credentials_are_written_owner_only(self, runner):
        """The file holds an API key, so it must not be group- or world-readable."""
        with runner.isolated_filesystem():
            with patch("featrixsphere.api.FeatrixSphere"):
                runner.invoke(main, ["login", "--api-key", "fx_k"])
            assert Path(".featrix").stat().st_mode & 0o777 == 0o600

    def test_preserves_other_keys_in_an_existing_config(self, runner):
        with runner.isolated_filesystem():
            Path(".featrix").write_text(json.dumps({"org": "acme", "api_key": "fx_old"}))
            with patch("featrixsphere.api.FeatrixSphere"):
                runner.invoke(main, ["login", "--api-key", "fx_new"])
            config = json.loads(Path(".featrix").read_text())
            assert config["api_key"] == "fx_new"
            assert config["org"] == "acme"

    def test_tolerates_a_corrupt_existing_config(self, runner):
        with runner.isolated_filesystem():
            Path(".featrix").write_text("{not json")
            with patch("featrixsphere.api.FeatrixSphere"):
                result = runner.invoke(main, ["login", "--api-key", "fx_new"])
            assert result.exit_code == 0, result.output
            assert json.loads(Path(".featrix").read_text())["api_key"] == "fx_new"

    def test_records_a_non_default_server(self, runner):
        with runner.isolated_filesystem():
            with patch("featrixsphere.api.FeatrixSphere"):
                runner.invoke(
                    main, ["--server", "https://sphere.internal", "login", "--api-key", "fx_k"]
                )
            assert json.loads(Path(".featrix").read_text())["base_url"] == "https://sphere.internal"

    def test_omits_base_url_for_the_default_server(self, runner):
        """Writing the default would pin the config to today's hostname."""
        with runner.isolated_filesystem():
            with patch("featrixsphere.api.FeatrixSphere"):
                runner.invoke(main, ["login", "--api-key", "fx_k"])
            assert "base_url" not in json.loads(Path(".featrix").read_text())


class TestCompletionsInstalledHint:
    """Bare `ffs` nudges people to install completion — but only once it isn't."""

    def test_hint_shown_when_no_rc_file_mentions_ffs(self, runner, tmp_path):
        with patch("ffs.cli.Path.home", return_value=tmp_path):
            result = runner.invoke(main, [])
        assert result.exit_code == 0
        assert "Enable tab completion" in result.output

    def test_hint_hidden_once_completion_is_set_up(self, runner, tmp_path):
        (tmp_path / ".zshrc").write_text('eval "$(ffs completions)"\n')
        with patch("ffs.cli.Path.home", return_value=tmp_path):
            result = runner.invoke(main, [])
        assert "Enable tab completion" not in result.output

    def test_detects_the_generated_function_name_too(self, runner, tmp_path):
        """Someone may have pasted the script itself rather than the eval line."""
        (tmp_path / ".bashrc").write_text("_ffs_completion_setup;\n")
        with patch("ffs.cli.Path.home", return_value=tmp_path):
            result = runner.invoke(main, [])
        assert "Enable tab completion" not in result.output

    def test_an_unreadable_rc_file_does_not_crash_the_cli(self, runner, tmp_path):
        rc = tmp_path / ".bashrc"
        rc.write_text("whatever")
        rc.chmod(0o000)
        try:
            with patch("ffs.cli.Path.home", return_value=tmp_path):
                result = runner.invoke(main, [])
            assert result.exit_code == 0, result.output
        finally:
            rc.chmod(0o644)


class TestVersionString:
    def test_version_reports_both_packages(self, runner):
        """Bug reports hinge on knowing the ffs *and* featrixsphere versions."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "ffs" in result.output
        assert "featrixsphere" in result.output

    def test_version_string_survives_a_missing_featrixsphere_dist(self):
        from ffs.cli import _get_version_string
        with patch("importlib.metadata.version", side_effect=Exception("gone")):
            assert "unknown" in _get_version_string()


class TestDidYouMeanMultipleMatches:
    """With several near-misses the helper lists them instead of guessing one."""

    def test_lists_several_candidates(self, runner):
        # `predict`, `predictor` and `models predict` are all close to "predic".
        result = runner.invoke(main, ["predic"])
        assert result.exit_code == 2
        assert "Did you mean one of these?" in result.output
        assert "predict" in result.output
        assert "predictor" in result.output

    def test_single_match_uses_the_singular_phrasing(self, runner):
        result = runner.invoke(main, ["wohami"])
        assert result.exit_code == 2
        assert "Did you mean 'whoami'?" in result.output
        assert "one of these" not in result.output
