"""Tests for ffs/events_cmd.py."""
import json
from unittest.mock import MagicMock

import pytest

from ffs.cli import main


def _mock_group(**overrides):
    g = MagicMock()
    g.event_group_id = "11111111-1111-1111-1111-111111111111"
    g.event_count = 537
    g.event_count_at_last_train = 500
    g.extend_count = 2
    g.auto_retrain_enabled = True
    g.last_session_id = "sess-abc"
    g.last_trained_at = "2026-07-01T00:00:00+00:00"
    g.consecutive_failures = 0
    g.last_failed_at = None
    g.created_at = "2026-06-01T00:00:00+00:00"
    g.updated_at = "2026-07-01T00:00:00+00:00"
    for k, v in overrides.items():
        setattr(g, k, v)
    g.to_dict.return_value = {
        "event_group_id": g.event_group_id,
        "event_count": g.event_count,
        "event_count_at_last_train": g.event_count_at_last_train,
        "extend_count": g.extend_count,
        "auto_retrain_enabled": g.auto_retrain_enabled,
        "last_session_id": g.last_session_id,
        "last_trained_at": g.last_trained_at,
        "consecutive_failures": g.consecutive_failures,
        "last_failed_at": g.last_failed_at,
        "created_at": g.created_at,
        "updated_at": g.updated_at,
    }
    return g


class TestEventsList:
    def test_empty_list(self, runner, mock_sphere, env):
        mock_sphere.list_event_groups.return_value = []
        result = runner.invoke(main, ["events", "list"], env=env)
        assert result.exit_code == 0
        assert "No event groups found" in result.output

    def test_lists_groups(self, runner, mock_sphere, env):
        mock_sphere.list_event_groups.return_value = [_mock_group()]
        result = runner.invoke(main, ["events", "list"], env=env)
        assert result.exit_code == 0
        # Rich's default 80-col table truncates the full 36-char UUID (same
        # behavior as every other print_list_table ID column in this CLI) --
        # check the preserved prefix rather than the full value.
        assert "11111111" in result.output
        assert "sess-abc" in result.output
        mock_sphere.list_event_groups.assert_called_once_with(limit=50, offset=0)

    def test_json_output(self, runner, mock_sphere, env):
        mock_sphere.list_event_groups.return_value = [_mock_group()]
        result = runner.invoke(main, ["--json", "events", "list"], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["event_group_id"] == "11111111-1111-1111-1111-111111111111"

    def test_limit_and_offset_passed_through(self, runner, mock_sphere, env):
        mock_sphere.list_event_groups.return_value = []
        result = runner.invoke(main, ["events", "list", "--limit", "10", "--offset", "20"], env=env)
        assert result.exit_code == 0
        mock_sphere.list_event_groups.assert_called_once_with(limit=10, offset=20)


class TestEventsShow:
    def test_shows_group_led_by_event_count(self, runner, mock_sphere, env):
        mock_sphere.event_group.return_value = _mock_group()
        result = runner.invoke(main, ["events", "show", "11111111-1111-1111-1111-111111111111"], env=env)
        assert result.exit_code == 0
        assert "537" in result.output
        assert "sess-abc" in result.output
        mock_sphere.event_group.assert_called_once_with("11111111-1111-1111-1111-111111111111")

    def test_json_output(self, runner, mock_sphere, env):
        mock_sphere.event_group.return_value = _mock_group()
        result = runner.invoke(main, ["--json", "events", "show", "11111111-1111-1111-1111-111111111111"], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["event_count"] == 537

    def test_missing_live_count_shown_as_dash(self, runner, mock_sphere, env):
        mock_sphere.event_group.return_value = _mock_group(event_count=None)
        result = runner.invoke(main, ["events", "show", "11111111-1111-1111-1111-111111111111"], env=env)
        assert result.exit_code == 0
        assert "—" in result.output

    def test_not_found_propagates_error(self, runner, mock_sphere, env):
        mock_sphere.event_group.side_effect = Exception("Event group not found")
        result = runner.invoke(main, ["events", "show", "does-not-exist"], env=env)
        assert result.exit_code != 0
