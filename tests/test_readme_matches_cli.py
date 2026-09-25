"""The README is a contract with the customer. Check it against the real CLI.

A customer reported "publish doesn't work". It didn't: the README said

    ffs foundation publish MODEL_ID --org ORG --name NAME

and `--org` had been removed two months earlier (a505f0b), because the SDK's
publish() never accepted an org — it derives one from the API key. Anyone
following the documentation got `Error: No such option: --org`, exit 2.

Nothing caught that. The mocked suite tests the code, not the instructions we
hand people. So this walks every `ffs ...` line in README.md and checks it
against the actual click command tree: the command must exist, and every long
option must be real. No mocks and no network — it introspects the CLI that
ships.
"""
import re
from pathlib import Path

import click
import pytest

from ffs.cli import main
from ffs.click_ext import HELP_ALIASES

README = Path(__file__).parent.parent / "README.md"

# Sections documenting things that deliberately don't exist yet.
SKIP_SECTIONS = {"Vector Database (not yet implemented)"}

# Placeholders the README uses for values, not command names.
PLACEHOLDER = re.compile(r"^[A-Z_]+$|^\{|^'|^\"|^\[|^<|^-")


def readme_command_lines():
    """Every `ffs ...` invocation in a fenced block, with its section heading."""
    lines = README.read_text().splitlines()
    section, in_block, found = "", False, []
    for line in lines:
        if line.startswith("#"):
            section = line.lstrip("#").strip()
        elif line.startswith("```"):
            in_block = not in_block
        elif in_block and line.strip().startswith("ffs "):
            # "<...>" marks the usage synopsis, which names no real command.
            if section not in SKIP_SECTIONS and "<" not in line:
                found.append((section, line.strip()))
    return found


def parse(invocation):
    """(command path, long options) for one README line.

    The README writes optional arguments as `[--epochs N]` and choices as
    `{classifier,regressor}`; both are stripped down to the bare names click
    would see.
    """
    # Drop inline prose after two+ spaces (the README aligns descriptions there)
    invocation = re.split(r"\s{2,}", invocation)[0]
    tokens = invocation.replace("[", " ").replace("]", " ").split()[1:]  # drop "ffs"

    path, options, cmd = [], [], main
    for token in tokens:
        if token.startswith("--"):
            options.append(token.split("=")[0])
            continue
        if options or PLACEHOLDER.match(token):
            continue
        if isinstance(cmd, click.Group) and token in cmd.commands:
            path.append(token)
            cmd = cmd.commands[token]
        elif not path:
            path.append(token)  # unknown top-level name; let the test report it
            break
    return tuple(path), options


def resolve(path):
    """Walk the real command tree, or None if the README names something absent.

    `help`/`?` resolve in DYMGroup.resolve_command rather than being registered
    commands, so they're valid invocations that `cmd.commands` won't contain.
    """
    cmd = main
    for name in path:
        if not isinstance(cmd, click.Group):
            return None
        if name in HELP_ALIASES:
            return cmd  # a help alias ends the invocation
        if name not in cmd.commands:
            return None
        cmd = cmd.commands[name]
    return cmd


README_LINES = readme_command_lines()


def test_readme_documents_some_commands():
    """Guard the guard — a parser that finds nothing would make this vacuous."""
    assert len(README_LINES) > 20, f"only found {len(README_LINES)} ffs lines in README"


@pytest.mark.parametrize(
    "section,invocation", README_LINES, ids=[f"{s}: {i[:60]}" for s, i in README_LINES]
)
def test_documented_command_exists(section, invocation):
    path, _ = parse(invocation)
    assert resolve(path) is not None, (
        f"README [{section}] documents `{invocation}`, but `ffs {' '.join(path)}` "
        f"is not a command."
    )


@pytest.mark.parametrize(
    "section,invocation", README_LINES, ids=[f"{s}: {i[:60]}" for s, i in README_LINES]
)
def test_documented_options_exist(section, invocation):
    path, options = parse(invocation)
    cmd = resolve(path)
    if cmd is None:
        pytest.skip("command itself is missing; reported by the other test")

    real = set()
    for param in cmd.params:
        real.update(o for o in param.opts + param.secondary_opts if o.startswith("--"))
    # Global options live on the root group and may be written after the command.
    for param in main.params:
        real.update(o for o in param.opts if o.startswith("--"))
    real.add("--help")

    missing = sorted(set(options) - real)
    assert not missing, (
        f"README [{section}] documents `{invocation}`, but `ffs {' '.join(path)}` "
        f"has no {', '.join(missing)} — a customer following this gets "
        f"\"Error: No such option\" and a non-zero exit."
    )
