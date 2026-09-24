"""Tests for ffs/endpoint_cmd.py — named API endpoints for predictors.

These commands hand out and revoke API keys, so the tests care about more than
exit codes: that a freshly minted key is actually shown (it can't be retrieved
later), that destructive commands refuse to run unconfirmed, and that the
endpoint/session IDs go to the SDK in the right order — both are opaque
strings, so swapping them wouldn't raise, it would act on the wrong endpoint.
"""
import json
import re

import pytest

from ffs.cli import main


def _row(output, label):
    """The value printed beside `label` in a print_kv table."""
    match = re.search(rf"^\s*{re.escape(label)}\s{{2,}}(\S+)", output, re.MULTILINE)
    assert match, f"no {label!r} row in:\n{output}"
    return match.group(1)


@pytest.fixture
def wired(mock_sphere, mock_fm, mock_predictor, mock_endpoint):
    """A client wired for the endpoint commands."""
    mock_fm.list_predictors.return_value = [mock_predictor]
    mock_predictor.create_api_endpoint.return_value = mock_endpoint
    mock_sphere.foundational_model.return_value = mock_fm
    mock_sphere.api_endpoint.return_value = mock_endpoint
    return mock_sphere


class TestEndpointCreate:
    def test_create(self, runner, wired, mock_endpoint, env):
        result = runner.invoke(
            main, ["endpoint", "create", "fm-abc123", "--name", "prod"], env=env
        )
        assert result.exit_code == 0, result.output
        assert "ep-123" in result.output
        assert "prod" in result.output

    def test_create_passes_name_and_description(
        self, runner, wired, mock_predictor, env
    ):
        result = runner.invoke(
            main,
            [
                "endpoint", "create", "fm-abc123",
                "--name", "prod",
                "--description", "production traffic",
                "--api-key", "fx_preset_key",
            ],
            env=env,
        )
        assert result.exit_code == 0, result.output
        _, kwargs = mock_predictor.create_api_endpoint.call_args
        assert kwargs["name"] == "prod"
        assert kwargs["description"] == "production traffic"
        assert kwargs["api_key"] == "fx_preset_key"

    def test_create_shows_the_generated_key_and_warns_it_is_once_only(
        self, runner, wired, env
    ):
        """The key is unretrievable afterwards, so it must be printed on create."""
        result = runner.invoke(
            main, ["endpoint", "create", "fm-abc123", "--name", "prod"], env=env
        )
        assert "sk_endpoint_secret" in result.output
        assert "won't be shown again" in result.output

    def test_create_requires_name(self, runner, wired, env):
        result = runner.invoke(main, ["endpoint", "create", "fm-abc123"], env=env)
        assert result.exit_code == 2
        assert "--name" in result.output

    def test_create_errors_when_session_has_no_predictor(
        self, runner, wired, mock_fm, env
    ):
        mock_fm.list_predictors.return_value = []
        result = runner.invoke(
            main, ["endpoint", "create", "fm-abc123", "--name", "prod"], env=env
        )
        assert result.exit_code != 0
        assert "No predictor found" in result.output

    def test_create_json(self, runner, wired, env):
        result = runner.invoke(
            main, ["--json", "endpoint", "create", "fm-abc123", "--name", "prod"],
            env=env,
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["id"] == "ep-123"


class TestEndpointShow:
    def test_show(self, runner, wired, env):
        result = runner.invoke(main, ["endpoint", "show", "fm-abc123", "ep-123"], env=env)
        assert result.exit_code == 0, result.output
        assert "ep-123" in result.output
        assert "42" in result.output  # usage count

    def test_show_does_not_print_the_api_key(self, runner, wired, env):
        """`show` reports only whether a key exists — it never echoes the secret."""
        result = runner.invoke(main, ["endpoint", "show", "fm-abc123", "ep-123"], env=env)
        assert "sk_endpoint_secret" not in result.output
        assert _row(result.output, "Has API Key") == "yes"

    def test_show_reports_missing_key_as_no(self, runner, wired, mock_endpoint, env):
        mock_endpoint.api_key = None
        result = runner.invoke(main, ["endpoint", "show", "fm-abc123", "ep-123"], env=env)
        assert result.exit_code == 0, result.output
        assert _row(result.output, "Has API Key") == "no"

    def test_show_passes_endpoint_id_before_session_id(self, runner, wired, env):
        """Both args are opaque strings; a swap would silently read the wrong row."""
        runner.invoke(main, ["endpoint", "show", "fm-abc123", "ep-123"], env=env)
        args, _ = wired.api_endpoint.call_args
        assert args == ("ep-123", "fm-abc123")

    def test_show_json(self, runner, wired, env):
        result = runner.invoke(
            main, ["--json", "endpoint", "show", "fm-abc123", "ep-123"], env=env
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["id"] == "ep-123"


class TestEndpointStats:
    def test_stats(self, runner, wired, env):
        result = runner.invoke(main, ["endpoint", "stats", "fm-abc123", "ep-123"], env=env)
        assert result.exit_code == 0, result.output
        assert "42" in result.output

    def test_stats_json(self, runner, wired, env):
        result = runner.invoke(
            main, ["--json", "endpoint", "stats", "fm-abc123", "ep-123"], env=env
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["total_calls"] == 42


class TestEndpointRegenerateKey:
    def test_regenerate_prints_the_new_key(self, runner, wired, env):
        result = runner.invoke(
            main, ["endpoint", "regenerate-key", "fm-abc123", "ep-123", "--yes"], env=env
        )
        assert result.exit_code == 0, result.output
        assert "sk_new_rotated_key" in result.output
        assert "won't be shown again" in result.output

    def test_regenerate_requires_confirmation(self, runner, wired, mock_endpoint, env):
        """Rotating a key breaks every caller using the old one."""
        result = runner.invoke(
            main, ["endpoint", "regenerate-key", "fm-abc123", "ep-123"],
            input="n\n", env=env,
        )
        assert result.exit_code != 0
        mock_endpoint.regenerate_api_key.assert_not_called()

    def test_regenerate_proceeds_on_yes_at_the_prompt(self, runner, wired, mock_endpoint, env):
        result = runner.invoke(
            main, ["endpoint", "regenerate-key", "fm-abc123", "ep-123"],
            input="y\n", env=env,
        )
        assert result.exit_code == 0, result.output
        mock_endpoint.regenerate_api_key.assert_called_once()

    def test_regenerate_json(self, runner, wired, env):
        result = runner.invoke(
            main,
            ["--json", "endpoint", "regenerate-key", "fm-abc123", "ep-123", "--yes"],
            env=env,
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["api_key"] == "sk_new_rotated_key"
        assert data["endpoint_id"] == "ep-123"


class TestEndpointRevokeKey:
    def test_revoke(self, runner, wired, mock_endpoint, env):
        result = runner.invoke(
            main, ["endpoint", "revoke-key", "fm-abc123", "ep-123", "--yes"], env=env
        )
        assert result.exit_code == 0, result.output
        mock_endpoint.revoke_api_key.assert_called_once()
        assert "revoked" in result.output

    def test_revoke_requires_confirmation(self, runner, wired, mock_endpoint, env):
        """Revoking leaves the endpoint callable by anyone, so never do it silently."""
        result = runner.invoke(
            main, ["endpoint", "revoke-key", "fm-abc123", "ep-123"], input="n\n", env=env
        )
        assert result.exit_code != 0
        mock_endpoint.revoke_api_key.assert_not_called()

    def test_revoke_json_reports_key_as_null(self, runner, wired, env):
        result = runner.invoke(
            main, ["--json", "endpoint", "revoke-key", "fm-abc123", "ep-123", "--yes"],
            env=env,
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["api_key"] is None


class TestEndpointDelete:
    def test_delete(self, runner, wired, mock_endpoint, env):
        result = runner.invoke(
            main, ["endpoint", "delete", "fm-abc123", "ep-123", "--yes"], env=env
        )
        assert result.exit_code == 0, result.output
        mock_endpoint.delete.assert_called_once()
        assert "Deleted" in result.output

    def test_delete_requires_confirmation(self, runner, wired, mock_endpoint, env):
        result = runner.invoke(
            main, ["endpoint", "delete", "fm-abc123", "ep-123"], input="n\n", env=env
        )
        assert result.exit_code != 0
        mock_endpoint.delete.assert_not_called()

    def test_delete_json(self, runner, wired, env):
        result = runner.invoke(
            main, ["--json", "endpoint", "delete", "fm-abc123", "ep-123", "--yes"],
            env=env,
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["deleted"] is True


class TestEndpointErrorSurfacing:
    """A failing SDK call must not read as success."""

    def test_api_error_is_reported(self, runner, wired, mock_sphere, env):
        mock_sphere.api_endpoint.side_effect = RuntimeError("404 Not Found")
        result = runner.invoke(main, ["endpoint", "show", "fm-abc123", "nope"], env=env)
        assert result.exit_code != 0
