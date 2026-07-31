"""Tests for ffs/model_cmd.py - foundation commands."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from ffs.cli import main
from ffs.model_cmd import _format_duration, _job_status_lines

CREDIT_CSV = str(Path(__file__).parent / "credit-sklearn.csv")


class TestFormatDuration:
    def test_seconds(self):
        assert _format_duration(45) == "45s"

    def test_minutes(self):
        assert _format_duration(125) == "2m05s"

    def test_hours(self):
        assert _format_duration(3725) == "1h02m"

    def test_zero(self):
        assert _format_duration(0) == "0s"


class TestJobStatusLines:
    def test_empty(self):
        assert _job_status_lines({"job_plan": [], "jobs": {}}) == []

    def test_pending_job(self):
        data = {
            "job_plan": [{"job_type": "train", "job_id": None}],
            "jobs": {},
        }
        lines = _job_status_lines(data)
        assert len(lines) == 1
        assert "pending" in lines[0]

    def test_done_job(self):
        data = {
            "job_plan": [{"job_type": "train", "job_id": "j1"}],
            "jobs": {"j1": {"status": "done", "progress": 100}},
        }
        lines = _job_status_lines(data)
        assert len(lines) == 1
        assert "done" in lines[0]
        assert "train" in lines[0]

    def test_running_job_with_progress(self):
        data = {
            "job_plan": [{"job_type": "embed", "job_id": "j1"}],
            "jobs": {"j1": {"status": "running", "progress": 42, "queue": "gpu-1"}},
        }
        lines = _job_status_lines(data)
        assert "running 42%" in lines[0]
        assert "gpu-1" in lines[0]

    def test_running_job_no_progress(self):
        data = {
            "job_plan": [{"job_type": "embed", "job_id": "j1"}],
            "jobs": {"j1": {"status": "running", "progress": 0}},
        }
        lines = _job_status_lines(data)
        assert "running" in lines[0]
        assert "%" not in lines[0]


class TestFoundationCreate:
    def test_creates_model(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.create_foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "create", "--name", "test", "--data", CREDIT_CSV], env=env)
        assert result.exit_code == 0
        assert "fm-abc123" in result.output
        mock_sphere.create_foundational_model.assert_called_once()

    def test_creates_model_json(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.create_foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["--json", "foundation", "create", "--name", "test", "--data", CREDIT_CSV], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["model_id"] == "fm-abc123"

    def test_passes_data_file_param(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.create_foundational_model.return_value = mock_fm
        runner.invoke(main, ["foundation", "create", "--name", "test", "--data", CREDIT_CSV], env=env)
        kwargs = mock_sphere.create_foundational_model.call_args.kwargs
        assert "data_file" in kwargs
        assert kwargs["data_file"] == CREDIT_CSV

    def test_ignore_columns(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.create_foundational_model.return_value = mock_fm
        runner.invoke(main, [
            "foundation", "create", "--name", "test", "--data", CREDIT_CSV,
            "--ignore-columns", "age,duration"
        ], env=env)
        kwargs = mock_sphere.create_foundational_model.call_args.kwargs
        assert kwargs["ignore_columns"] == ["age", "duration"]


class TestFoundationList:
    def test_lists_models(self, runner, mock_sphere, env):
        mock_sphere.list_sessions.return_value = ["session-1", "session-2"]
        result = runner.invoke(main, ["foundation", "list"], env=env)
        assert result.exit_code == 0
        assert "session-1" in result.output
        assert "session-2" in result.output

    def test_empty_list(self, runner, mock_sphere, env):
        mock_sphere.list_sessions.return_value = []
        result = runner.invoke(main, ["foundation", "list"], env=env)
        assert result.exit_code == 0
        assert "No models found" in result.output

    def test_json_output(self, runner, mock_sphere, env):
        mock_sphere.list_sessions.return_value = ["s1"]
        result = runner.invoke(main, ["--json", "foundation", "list"], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == ["s1"]


class TestFoundationShow:
    def test_shows_model(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "show", "fm-abc123"], env=env)
        assert result.exit_code == 0
        assert "fm-abc123" in result.output

    def test_json_output(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["--json", "foundation", "show", "fm-abc123"], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["model_id"] == "fm-abc123"
        assert data["status"] == "done"


class TestFoundationColumns:
    def test_shows_columns(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "columns", "fm-abc123"], env=env)
        assert result.exit_code == 0
        assert "checking_status" in result.output

    def test_json_output(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["--json", "foundation", "columns", "fm-abc123"], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "class" in data


class TestFoundationCard:
    def test_shows_card(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "card", "fm-abc123"], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["name"] == "credit-model"


class TestFoundationWait:
    def test_already_done(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.status = "done"
        result = runner.invoke(main, ["foundation", "wait", "fm-abc123"], env=env)
        assert result.exit_code == 0
        assert "Training complete" in result.output

    def test_already_done_quiet(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.status = "done"
        result = runner.invoke(main, ["--quiet", "foundation", "wait", "fm-abc123"], env=env)
        assert result.exit_code == 0
        assert "Training complete" in result.output
        # In quiet mode, kv details are suppressed
        assert "Dimensions" not in result.output

    def test_error_status(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.status = "error"
        mock_fm.refresh.return_value = {
            "job_plan": [],
            "jobs": {"j1": {"status": "error", "job_type": "train", "error": "OOM"}},
        }
        result = runner.invoke(main, ["foundation", "wait", "fm-abc123"], env=env)
        assert result.exit_code == 1

    def test_timeout(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.status = "training"
        with patch("ffs.model_cmd.time") as mock_time:
            mock_time.time.side_effect = [0, 3700]
            result = runner.invoke(main, ["foundation", "wait", "fm-abc123", "--timeout", "3600"], env=env)
        assert result.exit_code == 1

    def test_es_done_but_predictor_still_training(self, runner, mock_sphere, mock_fm, mock_predictor, env):
        """fm.status can hit "done" (ES/session lifecycle) while an attached
        SP predictor is still training — wait must not report success yet."""
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.status = "done"
        training_predictor = MagicMock()
        training_predictor.id = "pred-xyz789"
        training_predictor.target_column = "class"
        training_predictor.status = "training"
        with patch("ffs.model_cmd.time") as mock_time:
            mock_time.time.side_effect = [0, 5, 3700]
            mock_fm.list_predictors.side_effect = [[training_predictor], [training_predictor]]
            result = runner.invoke(main, ["foundation", "wait", "fm-abc123", "--timeout", "3600"], env=env)
        assert result.exit_code == 1
        assert "Training complete" not in result.output

    def test_predictor_finishes_after_es(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.status = "done"
        done_predictor = MagicMock()
        done_predictor.id = "pred-xyz789"
        done_predictor.target_column = "class"
        done_predictor.status = "done"
        mock_fm.list_predictors.return_value = [done_predictor]
        result = runner.invoke(main, ["foundation", "wait", "fm-abc123"], env=env)
        assert result.exit_code == 0
        assert "Training complete" in result.output


class TestFoundationExtend:
    def test_extends_model(self, runner, mock_sphere, mock_fm, env):
        new_fm = MagicMock()
        new_fm.id = "fm-extended-456"
        new_fm.status = "training"
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.extend.return_value = new_fm
        result = runner.invoke(main, ["foundation", "extend", "fm-abc123", "--data", CREDIT_CSV], env=env)
        assert result.exit_code == 0
        assert "fm-extended-456" in result.output

    def test_json_output(self, runner, mock_sphere, mock_fm, env):
        new_fm = MagicMock()
        new_fm.id = "fm-ext-789"
        new_fm.status = "training"
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.extend.return_value = new_fm
        result = runner.invoke(main, ["--json", "foundation", "extend", "fm-abc123", "--data", CREDIT_CSV], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["model_id"] == "fm-ext-789"


class TestFoundationEncode:
    def test_encodes_record(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.encode.return_value = [0.1, 0.2, 0.3]
        result = runner.invoke(main, ["foundation", "encode", "fm-abc123", '{"age": 35}'], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == [0.1, 0.2, 0.3]


class TestFoundationPublish:
    def test_publishes(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.publish.return_value = {"published_path": "org/model"}
        result = runner.invoke(main, ["foundation", "publish", "fm-abc123"], env=env)
        assert result.exit_code == 0
        assert "Published" in result.output
        mock_fm.publish.assert_called_once_with(name=None, max_wait_time=600, poll_interval=5)

    def test_json_output(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.publish.return_value = {"published_path": "org/model"}
        result = runner.invoke(main, ["--json", "foundation", "publish", "fm-abc123", "--name", "biz-key-v11"], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["published_path"] == "org/model"
        mock_fm.publish.assert_called_once_with(name="biz-key-v11", max_wait_time=600, poll_interval=5)


class TestFoundationUnpublish:
    def test_unpublishes(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.unpublish.return_value = {}
        result = runner.invoke(main, ["foundation", "unpublish", "fm-abc123"], env=env)
        assert result.exit_code == 0
        assert "Unpublished" in result.output


class TestFoundationDeprecate:
    def test_deprecates(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.deprecate.return_value = {}
        result = runner.invoke(main, [
            "foundation", "deprecate", "fm-abc123",
            "--message", "Old model", "--expires", "2025-12-31"
        ], env=env)
        assert result.exit_code == 0
        assert "Deprecated" in result.output
        assert "2025-12-31" in result.output


class TestFoundationCancel:
    def test_cancels_with_yes(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.cancel.return_value = {"cancelled": True, "status": "cancelled"}
        result = runner.invoke(main, ["foundation", "cancel", "fm-abc123", "--yes"], env=env)
        assert result.exit_code == 0
        assert "Cancelled" in result.output
        mock_fm.cancel.assert_called_once_with(reason=None)

    def test_cancels_with_reason(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.cancel.return_value = {"cancelled": True}
        result = runner.invoke(main, [
            "foundation", "cancel", "fm-abc123", "--yes", "--reason", "duplicate run",
        ], env=env)
        assert result.exit_code == 0
        assert "duplicate run" in result.output
        mock_fm.cancel.assert_called_once_with(reason="duplicate run")

    def test_requires_confirmation(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "cancel", "fm-abc123"], env=env, input="n\n")
        assert result.exit_code != 0
        mock_fm.cancel.assert_not_called()

    def test_json_output(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.cancel.return_value = {"cancelled": True, "status": "cancelled"}
        result = runner.invoke(main, ["--json", "foundation", "cancel", "fm-abc123", "--yes"], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["cancelled"] is True


class TestFoundationDelete:
    def test_deletes_with_yes(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.delete.return_value = {}
        result = runner.invoke(main, ["foundation", "delete", "fm-abc123", "--yes"], env=env)
        assert result.exit_code == 0
        assert "deletion" in result.output.lower()

    def test_json_output(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.delete.return_value = {"deleted": True}
        result = runner.invoke(main, ["--json", "foundation", "delete", "fm-abc123", "--yes"], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["deleted"] is True
