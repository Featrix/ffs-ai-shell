"""Tests for ffs/jobs_cmd.py."""
import json
from unittest.mock import MagicMock

from ffs.cli import main


def _pending_fm(id_, name, status):
    fm = MagicMock()
    fm.id = id_
    fm.name = name
    fm.status = status
    fm.dimensions = 128
    fm.epochs = 10
    return fm


class TestJobsList:
    def test_lists_pending(self, runner, mock_sphere, env):
        mock_sphere.list_pending_jobs.return_value = [
            _pending_fm("fm-1", "credit-model", "running"),
        ]
        result = runner.invoke(main, ["jobs", "list"], env=env)
        assert result.exit_code == 0
        assert "fm-1" in result.output
        assert "running" in result.output

    def test_empty(self, runner, mock_sphere, env):
        mock_sphere.list_pending_jobs.return_value = []
        result = runner.invoke(main, ["jobs", "list"], env=env)
        assert result.exit_code == 0
        assert "No pending jobs" in result.output

    def test_json_output(self, runner, mock_sphere, env):
        mock_sphere.list_pending_jobs.return_value = [
            _pending_fm("fm-1", "credit-model", "ready"),
        ]
        result = runner.invoke(main, ["--json", "jobs", "list"], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == [{
            "id": "fm-1", "name": "credit-model", "status": "ready",
            "dimensions": 128, "epochs": 10,
        }]

    def test_passes_prefix(self, runner, mock_sphere, env):
        mock_sphere.list_pending_jobs.return_value = []
        runner.invoke(main, ["jobs", "list", "--prefix", "credit"], env=env)
        mock_sphere.list_pending_jobs.assert_called_once_with(name_prefix="credit")


class TestJobsCancelQueued:
    def test_cancels_single_model_when_ready(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.status = "ready"
        mock_fm.cancel.return_value = {"cancelled": True, "status": "cancelled"}
        result = runner.invoke(main, ["jobs", "cancel-queued", "fm-abc123", "--yes"], env=env)
        assert result.exit_code == 0
        assert "Cancelled 1 queued job" in result.output
        mock_fm.cancel.assert_called_once_with(reason=None)

    def test_skips_single_model_when_running(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.status = "running"
        result = runner.invoke(main, ["jobs", "cancel-queued", "fm-abc123", "--yes"], env=env)
        assert result.exit_code == 0
        assert "No queued jobs" in result.output
        mock_fm.cancel.assert_not_called()

    def test_org_wide_only_cancels_ready(self, runner, mock_sphere, env):
        ready_fm = _pending_fm("fm-1", "a", "ready")
        running_fm = _pending_fm("fm-2", "b", "running")
        ready_fm.cancel.return_value = {"cancelled": True}
        mock_sphere.list_pending_jobs.return_value = [ready_fm, running_fm]
        result = runner.invoke(main, ["jobs", "cancel-queued", "--yes"], env=env)
        assert result.exit_code == 0
        ready_fm.cancel.assert_called_once_with(reason=None)
        running_fm.cancel.assert_not_called()
        assert "fm-1" in result.output
        assert "fm-2" not in result.output

    def test_requires_confirmation(self, runner, mock_sphere, env):
        ready_fm = _pending_fm("fm-1", "a", "ready")
        mock_sphere.list_pending_jobs.return_value = [ready_fm]
        result = runner.invoke(main, ["jobs", "cancel-queued"], env=env, input="n\n")
        assert result.exit_code != 0
        ready_fm.cancel.assert_not_called()

    def test_json_output(self, runner, mock_sphere, env):
        ready_fm = _pending_fm("fm-1", "a", "ready")
        ready_fm.cancel.return_value = {"cancelled": True, "status": "cancelled"}
        mock_sphere.list_pending_jobs.return_value = [ready_fm]
        result = runner.invoke(main, ["--json", "jobs", "cancel-queued", "--yes"], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["cancelled"][0]["id"] == "fm-1"

    def test_reason_passed_through(self, runner, mock_sphere, env):
        ready_fm = _pending_fm("fm-1", "a", "ready")
        mock_sphere.list_pending_jobs.return_value = [ready_fm]
        runner.invoke(main, ["jobs", "cancel-queued", "--yes", "--reason", "cleanup"], env=env)
        ready_fm.cancel.assert_called_once_with(reason="cleanup")
