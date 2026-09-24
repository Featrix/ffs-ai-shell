"""Tests for ffs running in a working directory that no longer exists.

A shell outlives its working directory more often than you'd think: an
unmounted volume, a deleted or rebuilt checkout, a directory replaced so the
old inode goes stale. In that state `os.getcwd()` raises FileNotFoundError.

Reported symptom — `ffs upgrade` from a vanished directory on a detached volume:

    Upgrading featrix-shell (0.5.16)...
      failed: Traceback (most recent call last):
      ...
        if sys.path[0] in ("", os.getcwd()):
      FileNotFoundError: [Errno 2] No such file or directory

That traceback is pip's, not ffs's: pip reads the cwd in its own __main__
before doing anything, so inheriting a dead one kills it on startup. Nothing
about upgrading a package needs the cwd.

`dead_cwd` below deletes a real directory out from under the process rather
than patching getcwd, so these tests exercise the actual OS condition.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ffs.cli import main
from ffs.client import (
    CWD_GONE_HINT,
    find_featrix_config,
    repair_cwd,
    safe_cwd,
    shell_cwd_path,
)

REPO_ROOT = Path(__file__).parent.parent


def _child_env(**extra):
    """Environment for running ffs as a subprocess in a broken directory.

    pytest-cov's subprocess hook reads the cwd when it starts up, so with
    coverage enabled it fails exactly the way this file is about and prints its
    own FileNotFoundError into the output under test. These tests only need the
    child to *start*, not to be measured, so the COV_CORE_* handoff is dropped.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("COV_CORE_")}
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.update(extra)
    return env


def _run_ffs(*args, env=None):
    return subprocess.run(
        [sys.executable, "-c",
         "import sys; from ffs.cli import cli; sys.exit(cli())", *args],
        env=env or _child_env(), capture_output=True, text=True, timeout=300,
    )


@pytest.fixture
def _restore_cwd():
    """Always land back in a real directory, even if the test left none.

    Taken before any test can break the cwd, and tolerant of the case where the
    original is itself gone — otherwise one failure here cascades into every
    later test that touches a relative path.
    """
    original = os.getcwd()
    try:
        yield original
    finally:
        try:
            os.chdir(original)
        except OSError:
            os.chdir(tempfile.gettempdir())


@pytest.fixture
def dead_cwd(_restore_cwd, monkeypatch):
    """The unrecoverable case: the directory is gone and $PWD agrees.

    A detached volume. There is nothing to step back into, so ffs has to
    degrade rather than repair.
    """
    doomed = tempfile.mkdtemp(prefix="ffs-dead-cwd-")
    os.chdir(doomed)
    os.rmdir(doomed)
    # Without this the shell's $PWD would still point at the real directory
    # pytest was launched from, which is the *recoverable* case below.
    monkeypatch.setenv("PWD", doomed)
    assert safe_cwd() is None, "fixture failed to produce an unrecoverable cwd"
    yield Path(doomed)


@pytest.fixture
def stale_cwd(_restore_cwd, monkeypatch):
    """The reported case: the handle is stale but the path still resolves.

    A remounted volume, or a directory replaced at the same path — `pwd` prints
    it happily and `cd "$PWD"` fixes everything, which is what the bug report
    showed.
    """
    base = tempfile.mkdtemp(prefix="ffs-stale-cwd-")
    live = Path(base) / "project"
    live.mkdir()
    os.chdir(live)
    os.rmdir(live)
    live.mkdir()  # same path, different inode
    monkeypatch.setenv("PWD", str(live))
    with pytest.raises(OSError):
        Path.cwd()  # the process's own handle is dead
    assert live.is_dir()  # ...but the path is fine
    yield live


class TestSafeCwd:
    def test_returns_the_directory_when_it_exists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert safe_cwd() == Path(os.getcwd())

    def test_returns_none_when_the_directory_is_gone(self, dead_cwd):
        assert safe_cwd() is None

    def test_path_cwd_really_does_raise_in_that_state(self, dead_cwd):
        """Guard the premise: if this stops raising, these tests prove nothing."""
        with pytest.raises(OSError):
            Path.cwd()


class TestUpgradeWithDeadCwd:
    """The reported bug."""

    def test_runs_pip_in_a_directory_that_exists(self, runner, dead_cwd):
        with patch("ffs.cli.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stderr="")
            result = runner.invoke(main, ["upgrade"])
        assert result.exit_code == 0, result.output
        assert run.call_count == 2
        for call in run.call_args_list:
            given = call.kwargs.get("cwd")
            assert given is not None, "pip was left to inherit the dead cwd"
            assert Path(given).is_dir(), f"pip's cwd {given!r} does not exist either"

    def test_explains_why_rather_than_failing_silently(self, runner, dead_cwd):
        with patch("ffs.cli.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stderr="")
            result = runner.invoke(main, ["upgrade"])
        assert "no longer exists" in result.output

    def test_still_reports_the_upgrade_it_performed(self, runner, dead_cwd):
        """The note is context, not a substitute for doing the work."""
        with patch("ffs.cli.subprocess.run") as run, \
             patch("ffs.cli._pkg_version") as pkg_version:
            run.return_value = MagicMock(returncode=0, stderr="")
            pkg_version.side_effect = ["0.5.16", "0.5.17", "2.0.1", "2.0.2"]
            result = runner.invoke(main, ["upgrade"])
        assert result.exit_code == 0, result.output
        assert "0.5.16" in result.output and "0.5.17" in result.output

    def test_inherits_the_cwd_when_it_is_healthy(self, runner, tmp_path, monkeypatch):
        """No behaviour change in the normal case: cwd=None means inherit."""
        monkeypatch.chdir(tmp_path)
        with patch("ffs.cli.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stderr="")
            result = runner.invoke(main, ["upgrade"])
        assert result.exit_code == 0, result.output
        for call in run.call_args_list:
            assert call.kwargs.get("cwd") is None
        assert "no longer exists" not in result.output


class TestUpgradeSubprocessForReal:
    """No mock on subprocess: the bug was in what the child process saw.

    Mocking subprocess.run proves ffs passes a cwd; only actually spawning pip
    proves the cwd it passes is one pip can start in.
    """

    def test_pip_starts_instead_of_dying_on_getcwd(self, dead_cwd):
        result = _run_ffs("upgrade")
        combined = result.stdout + result.stderr
        assert result.returncode == 0, combined
        # `upgrade` prints "  failed: <pip stderr>" when pip doesn't run.
        assert "failed:" not in combined, combined
        # The specific line pip died on in the report.
        assert "os.getcwd()" not in combined, combined


class TestConfigDiscoveryWithDeadCwd:
    """Every authenticated command goes through find_featrix_config()."""

    def test_falls_back_to_the_global_config(self, dead_cwd, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        (home / ".featrix").write_text('{"api_key": "fx_global"}')
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        found, source = find_featrix_config()
        assert found == home / ".featrix"
        assert source == "~/.featrix"

    def test_reports_not_found_when_there_is_no_global_config(
        self, dead_cwd, tmp_path, monkeypatch
    ):
        home = tmp_path / "empty-home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        found, source = find_featrix_config()
        assert found is None
        assert source == "not found"

    def test_does_not_raise(self, dead_cwd, tmp_path, monkeypatch):
        """The old behaviour: Path.cwd() raised and took the command with it."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        find_featrix_config()

    def test_whoami_works_from_a_dead_directory(
        self, runner, mock_sphere, dead_cwd, tmp_path, monkeypatch
    ):
        home = tmp_path / "home"
        home.mkdir()
        (home / ".featrix").write_text('{"api_key": "fx_global"}')
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        mock_sphere.whoami.return_value = {"user": "testuser"}
        result = runner.invoke(main, ["--json", "whoami"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["api_key_source"] == "~/.featrix"


class TestLoginWithDeadCwd:
    def test_refuses_with_an_explanation_and_a_way_out(self, runner, dead_cwd):
        result = runner.invoke(main, ["login", "--api-key", "fx_k"])
        assert result.exit_code != 0
        assert "no longer exists" in result.output
        assert "ffs login --global" in result.output

    def test_global_login_still_works(self, runner, dead_cwd, tmp_path, monkeypatch):
        """--global writes to ~, which has nothing to do with the cwd."""
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr("ffs.cli.Path.home", staticmethod(lambda: home))
        with patch("featrixsphere.api.FeatrixSphere") as MockFS:
            MockFS.return_value.health_check.return_value = {"status": "healthy"}
            result = runner.invoke(main, ["login", "--api-key", "fx_k", "--global"])
        assert result.exit_code == 0, result.output
        assert json.loads((home / ".featrix").read_text())["api_key"] == "fx_k"


class TestErrorsNameTheRealCause:
    """Whatever still fails on a dead cwd should say why.

    A relative path can't be resolved from a directory that doesn't exist, and
    "[Errno 2] No such file or directory" with no filename in it explains
    nothing — which is exactly how the reported bug read.
    """

    def test_oserror_gets_the_hint_appended(self, dead_cwd, monkeypatch, capsys):
        from ffs.cli import cli

        monkeypatch.setattr("sys.argv", ["ffs", "whoami"])
        monkeypatch.setattr(
            "ffs.cli.main",
            MagicMock(side_effect=FileNotFoundError(2, "No such file or directory")),
        )
        with pytest.raises(SystemExit):
            cli()
        assert "no longer exists" in capsys.readouterr().out

    def test_json_mode_carries_the_hint_too(self, dead_cwd, monkeypatch, capsys):
        from ffs.cli import cli

        monkeypatch.setattr("sys.argv", ["ffs", "--json", "whoami"])
        monkeypatch.setattr(
            "ffs.cli.main",
            MagicMock(side_effect=FileNotFoundError(2, "No such file or directory")),
        )
        with pytest.raises(SystemExit):
            cli()
        payload = json.loads(capsys.readouterr().out)
        assert "no longer exists" in payload["error"]["message"]

    def test_no_hint_when_the_cwd_is_fine(self, tmp_path, monkeypatch, capsys):
        """Don't blame the cwd for an unrelated missing file."""
        from ffs.cli import cli

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["ffs", "predict", "sid", "{}"])
        monkeypatch.setattr(
            "ffs.cli.main",
            MagicMock(side_effect=FileNotFoundError(2, "No such file", "/tmp/nope.csv")),
        )
        with pytest.raises(SystemExit):
            cli()
        assert "no longer exists" not in capsys.readouterr().out

    def test_non_oserror_failures_are_untouched(self, dead_cwd, monkeypatch, capsys):
        from ffs.cli import cli

        monkeypatch.setattr("sys.argv", ["ffs", "whoami"])
        monkeypatch.setattr("ffs.cli.main", MagicMock(side_effect=RuntimeError("boom")))
        with pytest.raises(SystemExit):
            cli()
        out = capsys.readouterr().out
        assert "boom" in out
        assert "no longer exists" not in out


def test_the_hint_tells_the_user_what_to_do():
    """`cd "$PWD"` re-resolves a stale inode, which is what unstuck the report."""
    assert "cd" in CWD_GONE_HINT
    assert "$PWD" in CWD_GONE_HINT


class TestStaleCwdIsRepaired:
    """The reported scenario: recoverable, so recover rather than degrade.

    Degrading here would be worse than crashing. A project-local .featrix sits
    in that very directory; skipping it and falling back to ~/.featrix means
    quietly running against a different org's credentials.
    """

    def test_shell_cwd_path_finds_the_live_path(self, stale_cwd):
        assert shell_cwd_path() == stale_cwd

    def test_safe_cwd_recovers_instead_of_giving_up(self, stale_cwd):
        assert safe_cwd() == stale_cwd

    def test_repair_cwd_steps_back_into_the_directory(self, stale_cwd):
        assert repair_cwd() == stale_cwd
        # resolve(): on macOS /var is a symlink to /private/var, so getcwd()
        # reports the resolved path while $PWD holds the one the shell used.
        assert Path.cwd().resolve() == stale_cwd.resolve()  # no longer raises

    def test_repair_is_a_no_op_when_the_cwd_is_healthy(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert repair_cwd() is None
        assert Path.cwd() == tmp_path

    def test_repair_gives_up_when_pwd_is_also_gone(self, dead_cwd):
        assert repair_cwd() is None

    def test_repair_ignores_a_pwd_that_is_not_a_directory(
        self, stale_cwd, tmp_path, monkeypatch
    ):
        """A stale $PWD pointing at a file must not be chdir'd into."""
        decoy = tmp_path / "a-file"
        decoy.write_text("not a directory")
        monkeypatch.setenv("PWD", str(decoy))
        assert repair_cwd() is None

    def test_repair_ignores_an_unset_pwd(self, dead_cwd, monkeypatch):
        monkeypatch.delenv("PWD", raising=False)
        assert shell_cwd_path() is None
        assert repair_cwd() is None

    def test_the_project_local_config_is_still_found(self, stale_cwd, monkeypatch):
        """The whole point: don't silently switch to another org's credentials."""
        (stale_cwd / ".featrix").write_text('{"api_key": "fx_project_local"}')
        home = stale_cwd.parent / "home"
        home.mkdir()
        (home / ".featrix").write_text('{"api_key": "fx_global"}')
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        with patch("ffs.client._is_git_tracked", return_value=False):
            found, _ = find_featrix_config()
        assert found == stale_cwd / ".featrix"

    def test_the_entry_point_repairs_before_running_a_command(
        self, stale_cwd, monkeypatch, capsys
    ):
        """`cli()` repairs first, so `upgrade` never reaches its fallback path."""
        from ffs.cli import cli

        monkeypatch.setattr("sys.argv", ["ffs", "upgrade"])
        with patch("ffs.cli.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stderr="")
            try:
                cli()
            except SystemExit as exc:
                assert exc.code in (0, None), exc.code
        out = capsys.readouterr().out
        assert "no longer exists" not in out, out
        # Repaired, so pip inherits a working directory like it normally would.
        for call in run.call_args_list:
            assert call.kwargs.get("cwd") is None
        assert Path.cwd().resolve() == stale_cwd.resolve()

    def test_upgrade_works_end_to_end_from_a_stale_directory(self, stale_cwd):
        """Real subprocess, real pip, the reported command."""
        result = _run_ffs("upgrade", env=_child_env(PWD=str(stale_cwd)))
        combined = result.stdout + result.stderr
        assert result.returncode == 0, combined
        assert "failed:" not in combined, combined
        assert "os.getcwd()" not in combined, combined
        # Repaired rather than worked around, so no fallback note.
        assert "no longer exists" not in combined, combined
