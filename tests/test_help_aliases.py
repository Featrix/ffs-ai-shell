"""`ffs help` and `ffs ?` must work — they're what people type.

Reported from a real session, right after upgrading:

    $ ffs ?
    Error: No such command '?'.
    $ ffs help
    Error: No such command 'help'.

    Did you mean 'agent-help'?

click only understands --help, so both bounced. The did-you-mean made it worse
by pointing at `agent-help`, which prints a reference guide written for LLM
agents — not what a person asking for help wants.

No mocks and no network here: these invoke the real command tree.
"""
import pytest
from click.testing import CliRunner

from ffs.cli import main
from ffs.click_ext import HELP_ALIASES, HELP_OPTION_NAMES

# Every group in the tree, so an alias can't work at the root and fail deeper.
GROUP_PATHS = [
    (), ("foundation",), ("models",), ("predictor",), ("endpoint",),
    ("events",), ("jobs",), ("network",), ("server",), ("train",),
]


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.parametrize("alias", sorted(HELP_ALIASES) + HELP_OPTION_NAMES)
def test_alias_prints_help_at_the_root(runner, alias):
    result = runner.invoke(main, [alias])
    assert result.exit_code == 0, f"`ffs {alias}` exited {result.exit_code}\n{result.output}"
    assert "Commands:" in result.output, result.output
    assert "foundation" in result.output


@pytest.mark.parametrize("path", GROUP_PATHS, ids=lambda p: " ".join(p) or "(root)")
def test_every_group_accepts_help(runner, path):
    result = runner.invoke(main, [*path, "help"])
    assert result.exit_code == 0, f"`ffs {' '.join(path)} help` failed:\n{result.output}"
    assert result.output.strip()


@pytest.mark.parametrize("path", GROUP_PATHS, ids=lambda p: " ".join(p) or "(root)")
def test_every_group_accepts_question_mark(runner, path):
    result = runner.invoke(main, [*path, "?"])
    assert result.exit_code == 0, f"`ffs {' '.join(path)} ?` failed:\n{result.output}"
    assert result.output.strip()


def test_help_drills_into_a_subcommand(runner):
    """`ffs help foundation predict` == `ffs foundation predict --help`."""
    viahelp = runner.invoke(main, ["help", "foundation", "predict"])
    direct = runner.invoke(main, ["foundation", "predict", "--help"])
    assert viahelp.exit_code == 0, viahelp.output
    assert "--foundation" in viahelp.output
    assert viahelp.output == direct.output


def test_help_drills_into_a_group(runner):
    result = runner.invoke(main, ["help", "predictor"])
    assert result.exit_code == 0
    assert "create" in result.output


def test_help_ignores_words_it_cannot_resolve(runner):
    """`ffs help me predict` should still show something, not raise."""
    result = runner.invoke(main, ["help", "me", "predict"])
    assert result.exit_code == 0
    assert "Commands:" in result.output


def test_help_does_not_route_to_agent_help(runner):
    """agent-help is an LLM reference guide; a person asking for help wants the CLI's."""
    result = runner.invoke(main, ["help"])
    assert "Usage:" in result.output
    # The guide's prose, not the command list.
    assert "FEATRIX_API_KEY environment variable (highest priority)" not in result.output


class TestUnknownCommands:
    def test_still_suggests_a_near_miss(self, runner):
        result = runner.invoke(main, ["networks"])
        assert result.exit_code != 0
        assert "Did you mean 'network'?" in result.output

    def test_always_points_at_help(self, runner):
        """Even with no close match, the error has to leave a way forward."""
        result = runner.invoke(main, ["zzzzzz"])
        assert result.exit_code != 0
        assert "help' to see the available commands" in result.output

    def test_points_at_help_within_a_group(self, runner):
        result = runner.invoke(main, ["foundation", "zzzzzz"])
        assert result.exit_code != 0
        assert "help' to see the available commands" in result.output


@pytest.mark.parametrize("flag", HELP_OPTION_NAMES)
def test_dash_help_flags_work_on_subcommands(runner, flag):
    """-h and -? are registered as help options, so they reach every command."""
    result = runner.invoke(main, ["foundation", "predict", flag])
    assert result.exit_code == 0, f"`ffs foundation predict {flag}` failed:\n{result.output}"
    assert "--foundation" in result.output
