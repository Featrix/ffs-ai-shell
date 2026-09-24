"""Whole-tree tests: every command exists, helps, runs, and stays covered.

Three guards that apply to the CLI as a whole rather than to any one command:

1. `test_every_command_helps` — every command renders --help without raising.
   Cheap, but it catches the import-time and decorator-level mistakes that make
   a command unreachable.

2. `test_smoke_*` — every leaf command is actually *invoked* against spec'd
   mocks and must exit 0. Because the mocks are pinned to the real
   featrixsphere classes (see conftest.spec_mock), this is where a renamed
   method or a dropped field surfaces as a failing command rather than as a
   customer bug report.

3. `test_every_command_is_exercised_by_some_test` — scrapes the rest of the
   suite and fails if a command has no test invoking it. This is what keeps
   coverage from rotting: a new subcommand added without tests fails CI.
"""
import ast
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest

from ffs.cli import main

TESTS_DIR = Path(__file__).parent
CREDIT_CSV = str(TESTS_DIR / "credit-sklearn.csv")
FAKE_API_KEY = "fx_test_fake_key"

# `foundation` is a legacy alias bound to the same object as `models`; walking
# it would double every model command under a second name.
ALIAS_GROUPS = {"foundation": "models"}


def leaf_paths(group=main, prefix=()):
    """Every runnable (non-group) command path, canonical names only."""
    for name, sub in group.commands.items():
        if prefix == () and name in ALIAS_GROUPS:
            continue
        path = prefix + (name,)
        if isinstance(sub, click.Group):
            yield from leaf_paths(sub, path)
        else:
            yield path


def group_paths(group=main, prefix=()):
    """Every command *group* path, including the root."""
    yield prefix
    for name, sub in group.commands.items():
        if prefix == () and name in ALIAS_GROUPS:
            continue
        if isinstance(sub, click.Group):
            yield from group_paths(sub, prefix + (name,))


ALL_LEAVES = sorted(leaf_paths())
ALL_GROUPS = sorted(group_paths())


def as_id(path):
    return " ".join(path) or "(root)"


# ---------------------------------------------------------------------------
# 1. Inventory and help
# ---------------------------------------------------------------------------

# Pinned so a command can't quietly disappear or get renamed without a failing
# test. Update deliberately when adding a command — and add a smoke entry too.
EXPECTED_LEAVES = {
    ("agent-help",),
    ("completions",),
    ("login",),
    ("upgrade",),
    ("whoami",),
    ("endpoint", "create"),
    ("endpoint", "delete"),
    ("endpoint", "regenerate-key"),
    ("endpoint", "revoke-key"),
    ("endpoint", "show"),
    ("endpoint", "stats"),
    ("events", "list"),
    ("events", "show"),
    ("jobs", "cancel-queued"),
    ("jobs", "list"),
    ("models", "cancel"),
    ("models", "card"),
    ("models", "code"),
    ("models", "columns"),
    ("models", "create"),
    ("models", "delete"),
    ("models", "deprecate"),
    ("models", "encode"),
    ("models", "extend"),
    ("models", "jobs"),
    ("models", "list"),
    ("models", "predict"),
    ("models", "publish"),
    ("models", "recent"),
    ("models", "show"),
    ("models", "unpublish"),
    ("models", "wait"),
    ("network", "list"),
    ("network", "predict"),
    ("network", "register"),
    ("network", "show"),
    ("predict",),
    ("predictor", "cancel"),
    ("predictor", "create"),
    ("predictor", "list"),
    ("predictor", "show"),
    ("server", "health"),
    ("train", "model"),
}


def test_command_inventory_is_unchanged():
    """The set of commands ffs exposes is pinned."""
    actual = set(ALL_LEAVES)
    added = sorted(as_id(p) for p in actual - EXPECTED_LEAVES)
    removed = sorted(as_id(p) for p in EXPECTED_LEAVES - actual)
    assert not (added or removed), (
        f"command inventory changed — added: {added or 'none'}, "
        f"removed/renamed: {removed or 'none'}. If this is intentional, update "
        "EXPECTED_LEAVES and add a SMOKE_ARGS entry for anything new."
    )


@pytest.mark.parametrize("path", ALL_LEAVES + ALL_GROUPS, ids=as_id)
def test_every_command_helps(runner, path):
    """--help must render for every command and group."""
    result = runner.invoke(main, list(path) + ["--help"])
    assert result.exit_code == 0, f"`ffs {as_id(path)} --help` failed:\n{result.output}"
    assert result.output.strip(), f"`ffs {as_id(path)} --help` printed nothing"


@pytest.mark.parametrize("path", ALL_LEAVES, ids=as_id)
def test_every_command_has_a_docstring(path):
    """Help text is the only documentation most of these commands have."""
    cmd = main
    for name in path:
        cmd = cmd.commands[name]
    assert cmd.help and cmd.help.strip(), f"`ffs {as_id(path)}` has no help text"


def test_foundation_alias_is_the_same_group_as_models():
    """`foundation` is documented as a legacy alias; keep it pointing at models."""
    assert main.commands["foundation"] is main.commands["models"]


# ---------------------------------------------------------------------------
# 2. Smoke: actually run every leaf command
# ---------------------------------------------------------------------------

# Plausible arguments per command. Confirmation prompts get --yes so the run is
# non-interactive.
SMOKE_ARGS = {
    ("whoami",): [],
    ("completions",): [],
    ("agent-help",): [],
    ("server", "health"): [],
    ("models", "create"): ["--name", "credit", "--data", str(CREDIT_CSV)],
    ("models", "list"): [],
    ("models", "show"): ["fm-abc123"],
    ("models", "columns"): ["fm-abc123"],
    ("models", "card"): ["fm-abc123"],
    ("models", "jobs"): ["fm-abc123"],
    ("models", "wait"): ["fm-abc123"],
    ("models", "recent"): [],
    ("models", "predict"): ["fm-abc123", "class", '{"duration": 12}'],
    ("models", "extend"): ["fm-abc123", "--data", str(CREDIT_CSV)],
    ("models", "encode"): ["fm-abc123", '{"duration": 12}'],
    ("models", "publish"): ["fm-abc123"],
    ("models", "unpublish"): ["fm-abc123"],
    ("models", "deprecate"): [
        "fm-abc123", "--message", "use v2", "--expires", "2030-01-01",
    ],
    ("models", "cancel"): ["fm-abc123", "--yes"],
    ("models", "delete"): ["fm-abc123", "--yes"],
    ("models", "code"): ["fm-abc123"],
    ("predictor", "create"): [
        "fm-abc123", "--target-column", "class", "--type", "classifier",
    ],
    ("predictor", "list"): ["fm-abc123"],
    ("predictor", "show"): ["fm-abc123"],
    ("predictor", "cancel"): ["fm-abc123", "--yes"],
    ("predict",): ["fm-abc123", '{"duration": 12}'],
    ("train", "model"): [
        "fm-abc123", "--target-column", "class", "--type", "classifier",
        "--data", str(CREDIT_CSV),
    ],
    ("jobs", "list"): [],
    ("jobs", "cancel-queued"): ["--yes"],
    ("network", "list"): [],
    ("network", "show"): ["carrier-qualification"],
    ("network", "predict"): ["carrier-qualification", '{"company_name": "Acme"}'],
    ("network", "register"): ["carrier-qualification", "--spec-file", "SPEC_FILE"],
    ("endpoint", "create"): ["fm-abc123", "--name", "prod"],
    ("endpoint", "show"): ["fm-abc123", "ep-123"],
    ("endpoint", "stats"): ["fm-abc123", "ep-123"],
    ("endpoint", "regenerate-key"): ["fm-abc123", "ep-123", "--yes"],
    ("endpoint", "revoke-key"): ["fm-abc123", "ep-123", "--yes"],
    ("endpoint", "delete"): ["fm-abc123", "ep-123", "--yes"],
    ("events", "list"): [],
    ("events", "show"): ["11111111-1111-1111-1111-111111111111"],
    # Covered by dedicated tests instead: `login` writes credentials and opens a
    # browser, `upgrade` shells out to pip. Neither belongs in a bulk sweep.
    ("login",): None,
    ("upgrade",): None,
}


def test_smoke_args_cover_every_command():
    """Every leaf command must have a smoke entry (even an explicit skip)."""
    undeclared = sorted(as_id(p) for p in set(ALL_LEAVES) - set(SMOKE_ARGS))
    assert not undeclared, (
        f"no SMOKE_ARGS entry for: {undeclared}. Add arguments so the command "
        "gets exercised, or map it to None with a reason if it can't be."
    )


SMOKE_CASES = sorted(p for p in ALL_LEAVES if SMOKE_ARGS.get(p) is not None)


@pytest.mark.parametrize("json_mode", [False, True], ids=["human", "json"])
@pytest.mark.parametrize("path", SMOKE_CASES, ids=as_id)
def test_smoke_command_runs(runner, wired_sphere, env, tmp_path, path, json_mode):
    """Run every command against spec'd mocks, in both output modes.

    Exercises the attribute and keyword surface of each command for real: any
    call that the installed featrixsphere wouldn't accept raises here.
    """
    args = list(SMOKE_ARGS[path])
    if "SPEC_FILE" in args:
        spec_file = tmp_path / "spec.json"
        spec_file.write_text(json.dumps({
            "nodes": [{"id": "is_company", "model": "is-company"}],
            "edges": [],
        }))
        args[args.index("SPEC_FILE")] = str(spec_file)

    argv = (["--json"] if json_mode else []) + list(path) + args
    result = runner.invoke(main, argv, env=env)

    assert result.exit_code == 0, (
        f"`ffs {' '.join(argv)}` exited {result.exit_code}\n"
        f"--- output ---\n{result.output}\n"
        f"--- exception ---\n{result.exception!r}"
    )

    # Exit 0 alone is too weak to mean "worked". Iterating an empty or mock
    # sequence prints nothing and still exits 0, which is how a command that
    # reads the wrong attribute off the SDK looks from the outside: successful
    # and silent. Every command here has something to say.
    assert result.output.strip(), (
        f"`ffs {' '.join(argv)}` exited 0 but printed nothing — it probably "
        "produced an empty result rather than doing the work."
    )


@pytest.mark.parametrize("path", SMOKE_CASES, ids=as_id)
def test_smoke_command_json_mode_emits_valid_json(
    runner, wired_sphere, env, tmp_path, path
):
    """--json output must parse, for the commands that emit a JSON document.

    Agents and scripts consume this; malformed or rich-decorated output there is
    a break even when the exit code is 0. Commands that legitimately print
    something else under --json (help text, a code snippet, a progress redraw)
    are exempt rather than asserted loosely.
    """
    not_json_documents = {
        ("completions",),      # shell script
        ("agent-help",),       # prose guide
        ("models", "code"),    # generated source snippet
        ("models", "wait"),    # live progress redraw, then a summary table
    }
    if path in not_json_documents:
        pytest.skip(f"`ffs {as_id(path)}` does not emit a JSON document")

    args = list(SMOKE_ARGS[path])
    if "SPEC_FILE" in args:
        spec_file = tmp_path / "spec.json"
        spec_file.write_text(json.dumps({"nodes": [], "edges": []}))
        args[args.index("SPEC_FILE")] = str(spec_file)

    result = runner.invoke(main, ["--json"] + list(path) + args, env=env)
    assert result.exit_code == 0, result.output
    try:
        json.loads(result.output)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"`ffs --json {as_id(path)}` did not emit parseable JSON: {exc}\n"
            f"--- output ---\n{result.output}"
        )


def test_smoke_login(runner):
    """`login` is excluded from the sweep because it writes credentials."""
    with runner.isolated_filesystem():
        with patch("featrixsphere.api.FeatrixSphere") as MockFS:
            MockFS.return_value.health_check.return_value = {"status": "healthy"}
            result = runner.invoke(main, ["login", "--api-key", FAKE_API_KEY])
    assert result.exit_code == 0, result.output


def test_smoke_upgrade(runner):
    """`upgrade` is excluded from the sweep because it shells out to pip."""
    with patch("ffs.cli.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = runner.invoke(main, ["upgrade"])
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# 3. Coverage invariant: every command needs a test somewhere
# ---------------------------------------------------------------------------


def _resolve_invocation(tokens):
    """Walk `tokens` down the real command tree and return the leaf path hit.

    Anything that isn't a command name at the current level — an option, an ID,
    a JSON blob, a non-literal like CREDIT_CSV — is skipped, so this reads a
    test's argument list the way click does.
    """
    cmd = main
    path = []
    for token in tokens:
        if not isinstance(token, str) or token.startswith("-"):
            continue
        if isinstance(cmd, click.Group) and token in cmd.commands:
            path.append(ALIAS_GROUPS.get(token, token) if not path else token)
            cmd = cmd.commands[token]
            if not isinstance(cmd, click.Group):
                return tuple(path)
    return None


def _invocations_in(path):
    """Every `.invoke(main, [...])` argument list in a test file."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "invoke"):
            continue
        if len(node.args) < 2 or not isinstance(node.args[1], ast.List):
            continue
        yield [
            el.value if isinstance(el, ast.Constant) else None
            for el in node.args[1].elts
        ]


# test_live_api.py drives the CLI as a subprocess and is skipped unless FFS_LIVE
# is set, so it cannot stand in for a command's coverage in an ordinary run.
# Excluded by name rather than relying on its different calling convention, so
# moving a command's only test there fails this check instead of passing quietly.
UNCOUNTED_TEST_FILES = {"test_live_api.py"}


def _commands_exercised_by_tests():
    covered = {}
    for test_file in sorted(TESTS_DIR.glob("test_*.py")):
        if test_file.name in UNCOUNTED_TEST_FILES:
            continue
        for tokens in _invocations_in(test_file):
            leaf = _resolve_invocation(tokens)
            if leaf:
                covered.setdefault(leaf, set()).add(test_file.name)
    return covered


def test_invocation_scraper_works():
    """Guard the guard: if the scraper finds nothing, the coverage test is a no-op."""
    covered = _commands_exercised_by_tests()
    assert len(covered) > 20, (
        f"scraper resolved only {len(covered)} commands — it has probably "
        "stopped understanding how tests invoke the CLI, which would make "
        "test_every_command_is_exercised_by_some_test vacuous."
    )
    # A path only reachable through the legacy alias must land on its canonical name.
    assert ("models", "show") in covered


def test_every_command_is_exercised_by_some_test():
    """No command may ship without a test that runs it."""
    covered = _commands_exercised_by_tests()
    uncovered = sorted(as_id(p) for p in set(ALL_LEAVES) - set(covered))
    assert not uncovered, (
        "these commands are never invoked by any test: "
        + ", ".join(uncovered)
    )
