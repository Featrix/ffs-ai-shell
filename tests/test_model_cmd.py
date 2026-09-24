"""Tests for ffs/model_cmd.py - foundation commands."""
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from featrixsphere.api import PredictionResult

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
        # The server reports job progress as a 0-1 fraction.
        data = {
            "job_plan": [{"job_type": "embed", "job_id": "j1"}],
            "jobs": {"j1": {"status": "running", "progress": 0.42, "queue": "gpu-1"}},
        }
        lines = _job_status_lines(data)
        assert "running 42.0%" in lines[0]
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


def _session(session_id, name, status="done", dimensions=128, epochs=10, final_loss=0.042):
    """A FoundationalModel summary as list_sessions() returns it."""
    s = MagicMock()
    s.id, s.name, s.status = session_id, name, status
    s.dimensions, s.epochs, s.final_loss = dimensions, epochs, final_loss
    return s


class TestFoundationList:
    def test_lists_models(self, runner, mock_sphere, env):
        mock_sphere.list_sessions.return_value = [
            _session("session-1", "credit-model"),
            _session("session-2", "churn-model", status="running", final_loss=None),
        ]
        result = runner.invoke(main, ["foundation", "list"], env=env)
        assert result.exit_code == 0, result.output
        assert "session-1" in result.output
        assert "session-2" in result.output
        assert "0.0420" in result.output

    def test_empty_list(self, runner, mock_sphere, env):
        mock_sphere.list_sessions.return_value = []
        result = runner.invoke(main, ["foundation", "list"], env=env)
        assert result.exit_code == 0
        assert "No models found" in result.output

    def test_json_output(self, runner, mock_sphere, env):
        mock_sphere.list_sessions.return_value = [_session("s1", "credit-model")]
        result = runner.invoke(main, ["--json", "foundation", "list"], env=env)
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data == [{"id": "s1", "name": "credit-model", "status": "done",
                         "dimensions": 128, "epochs": 10, "final_loss": 0.042}]


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


def _predictor(target_column="class", status="done", pred_id="pred-xyz789"):
    p = MagicMock()
    p.id = pred_id
    p.target_column = target_column
    p.status = status
    return p


class TestFoundationPredict:
    def test_uses_a_trained_predictor(self, runner, mock_sphere, mock_fm, mock_prediction, env):
        trained = _predictor()
        trained.predict.return_value = mock_prediction
        mock_fm.list_predictors.return_value = [trained]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, [
            "foundation", "predict", "fm-abc123", "class", '{"duration": 12}'
        ], env=env)
        assert result.exit_code == 0, result.output
        assert "good" in result.output
        mock_fm.foundation_predict.assert_not_called()

    def test_falls_back_to_the_foundation_model(self, runner, mock_sphere, mock_fm, env):
        """With no trained SP, the column must be predicted by the foundation.

        The single-predictor endpoint ignores target_column and serves whatever
        SP the session has, so reaching for it here would answer for a
        different column — this asserts the foundation path is used instead.
        """
        mock_fm.list_predictors.return_value = []
        mock_fm.foundation_predict.return_value = [PredictionResult(
            predicted_class="good", confidence=0.81,
        )]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, [
            "foundation", "predict", "fm-abc123", "class", '{"duration": 12}'
        ], env=env)
        assert result.exit_code == 0, result.output
        assert "good" in result.output
        mock_fm.foundation_predict.assert_called_once_with("class", [{"duration": 12}])

    def test_refuses_a_predictor_that_is_still_training(self, runner, mock_sphere, mock_fm, env):
        trained = _predictor(status="ready")
        mock_fm.list_predictors.return_value = [trained]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, [
            "foundation", "predict", "fm-abc123", "class", '{"duration": 12}'
        ], env=env)
        assert result.exit_code != 0
        assert "still training" in result.output
        assert "--foundation" in result.output
        trained.predict.assert_not_called()

    def test_foundation_flag_bypasses_an_unservable_predictor(
        self, runner, mock_sphere, mock_fm, env
    ):
        trained = _predictor(status="aborted")
        mock_fm.list_predictors.return_value = [trained]
        mock_fm.foundation_predict.return_value = [PredictionResult(predicted_class="good")]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, [
            "foundation", "predict", "fm-abc123", "class", '{"duration": 12}', "--foundation"
        ], env=env)
        assert result.exit_code == 0, result.output
        trained.predict.assert_not_called()

    def test_unknown_predictor_id_is_an_error(self, runner, mock_sphere, mock_fm, env):
        """Answering from the foundation model would be a different model's answer."""
        mock_fm.list_predictors.return_value = [_predictor()]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, [
            "foundation", "predict", "fm-abc123", "class", '{"duration": 12}',
            "--predictor-id", "pred-nope",
        ], env=env)
        assert result.exit_code != 0
        assert "pred-nope" in result.output
        mock_fm.foundation_predict.assert_not_called()

    def test_null_result_is_an_error(self, runner, mock_sphere, mock_fm, env):
        trained = _predictor()
        trained.predict.return_value = PredictionResult(query_record={"duration": 12})
        mock_fm.list_predictors.return_value = [trained]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, [
            "foundation", "predict", "fm-abc123", "class", '{"duration": 12}'
        ], env=env)
        assert result.exit_code != 0
        assert "No prediction for 'class'" in result.output

    def test_batch_from_file(self, runner, mock_sphere, mock_fm, env):
        trained = _predictor()
        trained.batch_predict.return_value = [
            PredictionResult(predicted_class="good", confidence=0.9),
            PredictionResult(predicted_class="bad", confidence=0.7),
        ]
        mock_fm.list_predictors.return_value = [trained]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, [
            "foundation", "predict", "fm-abc123", "class", "--file", CREDIT_CSV
        ], env=env)
        assert result.exit_code == 0, result.output
        assert "2 predictions" in result.output

    def test_json_output(self, runner, mock_sphere, mock_fm, mock_prediction, env):
        trained = _predictor()
        trained.predict.return_value = mock_prediction
        mock_fm.list_predictors.return_value = [trained]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, [
            "--json", "foundation", "predict", "fm-abc123", "class", '{"duration": 12}'
        ], env=env)
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["predicted_class"] == "good"

    def test_explain_needs_a_trained_predictor(self, runner, mock_sphere, mock_fm, env):
        mock_fm.list_predictors.return_value = []
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, [
            "foundation", "predict", "fm-abc123", "class", '{"duration": 12}', "--explain"
        ], env=env)
        assert result.exit_code != 0
        assert "--explain" in result.output
        mock_fm.foundation_predict.assert_not_called()


class TestFoundationJobs:
    """`foundation jobs` — a one-shot snapshot of a session's training jobs."""

    def test_shows_a_planned_job_with_live_status(self, runner, mock_sphere, mock_fm, env):
        mock_fm.refresh.return_value = {
            "job_plan": [{"job_type": "train_es", "job_id": "j1"}],
            "jobs": {"j1": {"status": "running", "progress": 0.4, "queue": "gpu"}},
        }
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "jobs", "fm-abc123"], env=env)
        assert result.exit_code == 0, result.output
        assert "train_es" in result.output
        assert "running" in result.output

    def test_reports_a_session_with_no_jobs_scheduled_yet(self, runner, mock_sphere, mock_fm, env):
        """An empty plan is normal right after upload — say so rather than nothing."""
        mock_fm.status = "uploading"
        mock_fm.refresh.return_value = {"session": {"status": "uploading"}}
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "jobs", "fm-abc123"], env=env)
        assert result.exit_code == 0, result.output
        assert "no jobs scheduled yet" in result.output

    def test_surfaces_a_job_error(self, runner, mock_sphere, mock_fm, env):
        mock_fm.refresh.return_value = {
            "job_plan": [{"job_type": "train_es", "job_id": "j1"}],
            "jobs": {"j1": {"status": "failed", "error": "OOM on worker"}},
        }
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "jobs", "fm-abc123"], env=env)
        assert result.exit_code == 0, result.output
        assert "OOM on worker" in result.output

    def test_includes_a_live_job_missing_from_the_plan(self, runner, mock_sphere, mock_fm, env):
        """job_plan is persisted state; the jobs block is live and can diverge."""
        mock_fm.refresh.return_value = {
            "job_plan": [],
            "jobs": {"j9": {"job_type": "train_single_predictor", "status": "running"}},
        }
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "jobs", "fm-abc123"], env=env)
        assert result.exit_code == 0, result.output
        assert "train_single_predictor" in result.output

    def test_json_lists_planned_and_unplanned_jobs(self, runner, mock_sphere, mock_fm, env):
        mock_fm.refresh.return_value = {
            "job_plan": [{"job_type": "train_es", "job_id": "j1"}],
            "jobs": {
                "j1": {"status": "done", "progress": 1.0},
                "j2": {"job_type": "train_single_predictor", "status": "running"},
            },
        }
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["--json", "foundation", "jobs", "fm-abc123"], env=env)
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["model_id"] == "fm-abc123"
        types = {j["job_type"] for j in data["jobs"]}
        assert types == {"train_es", "train_single_predictor"}

    def test_json_pairs_plan_entries_with_their_live_record(self, runner, mock_sphere, mock_fm, env):
        mock_fm.refresh.return_value = {
            "job_plan": [{"job_type": "train_es", "job_id": "j1"}],
            "jobs": {"j1": {"status": "running", "progress": 0.5, "queue": "gpu"}},
        }
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["--json", "foundation", "jobs", "fm-abc123"], env=env)
        job = json.loads(result.output)["jobs"][0]
        assert job["job_id"] == "j1"
        assert job["status"] == "running"
        assert job["queue"] == "gpu"

    def test_json_keeps_a_planned_job_with_no_live_record(self, runner, mock_sphere, mock_fm, env):
        """A queued job exists in the plan before it has a Redis record."""
        mock_fm.refresh.return_value = {
            "job_plan": [{"job_type": "train_es", "job_id": "j1"}], "jobs": {},
        }
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["--json", "foundation", "jobs", "fm-abc123"], env=env)
        job = json.loads(result.output)["jobs"][0]
        assert job["job_type"] == "train_es"
        assert job["status"] is None


class TestFoundationRecent:
    """`foundation recent` — most-interesting-first listing of sessions."""

    def test_lists_sessions(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.list_sessions.return_value = [mock_fm]
        result = runner.invoke(main, ["foundation", "recent"], env=env)
        assert result.exit_code == 0, result.output
        assert "fm-abc123" in result.output

    def test_reports_an_empty_account(self, runner, mock_sphere, env):
        mock_sphere.list_sessions.return_value = []
        result = runner.invoke(main, ["foundation", "recent"], env=env)
        assert result.exit_code == 0, result.output
        assert "No sessions found" in result.output

    def test_sorts_active_sessions_ahead_of_finished_ones(self, runner, mock_sphere, env):
        """The point of `recent` over `list` is that in-flight work comes first."""
        done = _session("fm-done", "finished-model", status="done")
        running = _session("fm-running", "live-model", status="running")
        mock_sphere.list_sessions.return_value = [done, running]
        result = runner.invoke(main, ["--json", "foundation", "recent"], env=env)
        assert result.exit_code == 0, result.output
        assert [s["id"] for s in json.loads(result.output)] == ["fm-running", "fm-done"]

    def test_limit_truncates_the_list(self, runner, mock_sphere, env):
        mock_sphere.list_sessions.return_value = [
            _session(f"fm-{i}", f"model-{i}", status="done") for i in range(10)
        ]
        result = runner.invoke(main, ["--json", "foundation", "recent", "--limit", "3"], env=env)
        assert result.exit_code == 0, result.output
        assert len(json.loads(result.output)) == 3

    def test_limit_applies_after_sorting(self, runner, mock_sphere, env):
        """Truncating first would hide the active sessions `recent` exists to show."""
        sessions = [_session(f"fm-done-{i}", f"model-{i}", status="done") for i in range(5)]
        sessions.append(_session("fm-running", "live-model", status="running"))
        mock_sphere.list_sessions.return_value = sessions
        result = runner.invoke(main, ["--json", "foundation", "recent", "--limit", "1"], env=env)
        assert [s["id"] for s in json.loads(result.output)] == ["fm-running"]

    def test_tolerates_a_session_with_no_status(self, runner, mock_sphere, env):
        """Sorting on a null status must not raise."""
        mock_sphere.list_sessions.return_value = [_session("fm-x", None, status=None)]
        result = runner.invoke(main, ["foundation", "recent"], env=env)
        assert result.exit_code == 0, result.output


class TestFoundationCode:
    """`foundation code` — copy-pasteable snippets for a specific model."""

    def test_python_is_the_default(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "code", "fm-abc123"], env=env)
        assert result.exit_code == 0, result.output
        assert "from featrixsphere import FeatrixSphere" in result.output
        assert "fm-abc123" in result.output

    def test_typescript(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "code", "fm-abc123", "--typescript"], env=env)
        assert result.exit_code == 0, result.output
        assert "await fetch(" in result.output
        assert "X-API-Key" in result.output

    def test_includes_the_models_real_columns(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "code", "fm-abc123"], env=env)
        assert "checking_status" in result.output

    def test_shows_a_predict_call_when_a_predictor_exists(
        self, runner, mock_sphere, mock_fm, mock_predictor, env
    ):
        mock_fm.list_predictors.return_value = [mock_predictor]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "code", "fm-abc123"], env=env)
        assert "fm.predict(" in result.output
        assert "class" in result.output

    def test_shows_encode_when_there_is_no_predictor(self, runner, mock_sphere, mock_fm, env):
        mock_fm.list_predictors.return_value = []
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "code", "fm-abc123"], env=env)
        assert "fm.encode(" in result.output

    def test_typescript_references_the_predictor_id(
        self, runner, mock_sphere, mock_fm, mock_predictor, env
    ):
        mock_fm.list_predictors.return_value = [mock_predictor]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "code", "fm-abc123", "--typescript"], env=env)
        assert "pred-xyz789" in result.output

    def test_uses_the_configured_server_url(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(
            main, ["--server", "https://sphere.internal", "foundation", "code", "fm-abc123"],
            env=env,
        )
        assert "https://sphere.internal" in result.output

    def test_still_emits_a_snippet_when_columns_cannot_be_read(
        self, runner, mock_sphere, mock_fm, env
    ):
        """A snippet with no columns still beats an error for this command."""
        mock_fm.get_columns.side_effect = RuntimeError("session expired")
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "code", "fm-abc123"], env=env)
        assert result.exit_code == 0, result.output
        assert "no columns found" in result.output


class TestFoundationShowAll:
    """`foundation show` with no ID walks every session."""

    def test_shows_every_session(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.list_sessions.return_value = [
            _session("fm-1", "one"), _session("fm-2", "two"),
        ]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "show"], env=env)
        assert result.exit_code == 0, result.output
        assert mock_sphere.foundational_model.call_count == 2

    def test_reports_an_empty_account(self, runner, mock_sphere, env):
        mock_sphere.list_sessions.return_value = []
        result = runner.invoke(main, ["foundation", "show"], env=env)
        assert result.exit_code == 0, result.output
        assert "No models found" in result.output

    def test_json_returns_a_list(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.list_sessions.return_value = [_session("fm-1", "one")]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["--json", "foundation", "show"], env=env)
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert isinstance(data, list) and data[0]["model_id"] == "fm-abc123"


class TestFoundationCardOutputModes:
    def test_url_flag_does_not_call_the_api(self, runner, mock_sphere, env):
        """--url just composes a URL; hitting the API would be wasted work."""
        result = runner.invoke(main, ["foundation", "card", "fm-abc123", "--url"], env=env)
        assert result.exit_code == 0, result.output
        assert "/compute/session/fm-abc123/model_card" in result.output
        mock_sphere.foundational_model.assert_not_called()

    def test_url_flag_honours_a_custom_server(self, runner, mock_sphere, env):
        result = runner.invoke(
            main,
            ["--server", "https://sphere.internal", "foundation", "card", "fm-abc123", "--url"],
            env=env,
        )
        assert "https://sphere.internal/compute/session/fm-abc123/model_card" in result.output

    def test_save_writes_renderable_html(self, runner, mock_sphere, mock_fm, env, tmp_path):
        mock_sphere.foundational_model.return_value = mock_fm
        out = tmp_path / "card.html"
        result = runner.invoke(
            main, ["foundation", "card", "fm-abc123", "--save", str(out)], env=env
        )
        assert result.exit_code == 0, result.output
        html = out.read_text()
        assert "<!DOCTYPE html>" in html
        # The renderer comes from the CDN, per the model-card repo's guidance.
        assert "bits.featrix.com/js/featrix-modelcard/model-card.js" in html
        assert "credit-model" in html

    def test_saved_html_embeds_the_card_as_valid_json(
        self, runner, mock_sphere, mock_fm, env, tmp_path
    ):
        """The card is interpolated into a <script> block; broken JSON breaks the page."""
        # A datetime in the card is why print/interpolation uses default=str.
        mock_fm.get_model_card.return_value = {
            "name": "x", "when": datetime(2026, 1, 1, tzinfo=timezone.utc),
        }
        mock_sphere.foundational_model.return_value = mock_fm
        out = tmp_path / "card.html"
        runner.invoke(main, ["foundation", "card", "fm-abc123", "--save", str(out)], env=env)
        # json.dumps here is single-line, so the blob is the rest of that line.
        blob = out.read_text().split("var data = ", 1)[1].split("\n", 1)[0].rstrip(";")
        assert json.loads(blob)["name"] == "x"

    def test_open_launches_a_browser(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        with patch("webbrowser.open") as browser:
            result = runner.invoke(
                main, ["foundation", "card", "fm-abc123", "--open"], env=env
            )
        assert result.exit_code == 0, result.output
        browser.assert_called_once()
        assert browser.call_args[0][0].startswith("file://")


class TestFoundationWaitPredictorFailure:
    """A predictor can die *after* the embedding space finishes.

    The session still reports "done", so `wait` would otherwise announce success
    for a model that can't serve anything.
    """

    def test_reports_a_dead_predictor_and_fails(
        self, runner, mock_sphere, mock_fm, env
    ):
        mock_fm.status = "done"
        mock_fm.list_predictors.return_value = [_predictor(status="aborted")]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "wait", "fm-abc123"], env=env)
        assert result.exit_code == 1
        assert "Predictor training failed" in result.output
        assert "aborted" in result.output

    def test_does_not_announce_training_complete(self, runner, mock_sphere, mock_fm, env):
        mock_fm.status = "done"
        mock_fm.list_predictors.return_value = [_predictor(status="error")]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "wait", "fm-abc123"], env=env)
        assert "Training complete" not in result.output

    def test_points_at_the_jobs_command_for_detail(self, runner, mock_sphere, mock_fm, env):
        mock_fm.status = "done"
        mock_fm.list_predictors.return_value = [_predictor(status="aborted")]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "wait", "fm-abc123"], env=env)
        assert "jobs fm-abc123" in result.output

    def test_a_healthy_predictor_still_reports_success(
        self, runner, mock_sphere, mock_fm, env
    ):
        mock_fm.status = "done"
        mock_fm.list_predictors.return_value = [_predictor(status="done")]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "wait", "fm-abc123"], env=env)
        assert result.exit_code == 0, result.output
        assert "Training complete" in result.output


class TestFoundationPredictBatch:
    def test_batch_via_a_trained_predictor(
        self, runner, mock_sphere, mock_fm, mock_prediction, env, tmp_path
    ):
        data = tmp_path / "rows.csv"
        data.write_text("duration\n12\n24\n")
        trained = _predictor()
        trained.batch_predict.return_value = [mock_prediction, mock_prediction]
        mock_fm.list_predictors.return_value = [trained]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, [
            "foundation", "predict", "fm-abc123", "class", "--file", str(data)
        ], env=env)
        assert result.exit_code == 0, result.output
        assert "2 predictions" in result.output
        assert "trained predictor" in result.output

    def test_batch_via_the_foundation_model(
        self, runner, mock_sphere, mock_fm, mock_prediction, env, tmp_path
    ):
        data = tmp_path / "rows.csv"
        data.write_text("duration\n12\n24\n")
        mock_fm.list_predictors.return_value = []
        mock_fm.foundation_predict.return_value = [mock_prediction, mock_prediction]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, [
            "foundation", "predict", "fm-abc123", "class", "--file", str(data)
        ], env=env)
        assert result.exit_code == 0, result.output
        assert "foundation" in result.output
        # Every row goes in one call, not one call per row.
        mock_fm.foundation_predict.assert_called_once()
        args, _ = mock_fm.foundation_predict.call_args
        assert args[0] == "class" and len(args[1]) == 2

    def test_batch_json_is_a_list(
        self, runner, mock_sphere, mock_fm, mock_prediction, env, tmp_path
    ):
        data = tmp_path / "rows.csv"
        data.write_text("duration\n12\n")
        trained = _predictor()
        trained.batch_predict.return_value = [mock_prediction]
        mock_fm.list_predictors.return_value = [trained]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, [
            "--json", "foundation", "predict", "fm-abc123", "class", "--file", str(data)
        ], env=env)
        assert result.exit_code == 0, result.output
        assert isinstance(json.loads(result.output), list)

    def test_rejects_an_unsupported_file_format(self, runner, mock_sphere, mock_fm, env, tmp_path):
        data = tmp_path / "rows.txt"
        data.write_text("duration\n12\n")
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, [
            "foundation", "predict", "fm-abc123", "class", "--file", str(data)
        ], env=env)
        assert result.exit_code != 0
        assert "Unsupported file format" in result.output

    def test_reads_json_input(
        self, runner, mock_sphere, mock_fm, mock_prediction, env, tmp_path
    ):
        data = tmp_path / "rows.json"
        data.write_text(json.dumps([{"duration": 12}, {"duration": 24}]))
        trained = _predictor()
        trained.batch_predict.return_value = [mock_prediction, mock_prediction]
        mock_fm.list_predictors.return_value = [trained]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, [
            "foundation", "predict", "fm-abc123", "class", "--file", str(data)
        ], env=env)
        assert result.exit_code == 0, result.output
        assert "2 predictions" in result.output

    def test_requires_a_record_or_a_file(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, ["foundation", "predict", "fm-abc123", "class"], env=env)
        assert result.exit_code != 0
        assert "Provide a JSON record or --file" in result.output

    def test_foundation_and_predictor_id_are_mutually_exclusive(
        self, runner, mock_sphere, mock_fm, env
    ):
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, [
            "foundation", "predict", "fm-abc123", "class", '{"duration": 12}',
            "--foundation", "--predictor-id", "pred-1",
        ], env=env)
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_explain_prints_feature_importance(
        self, runner, mock_sphere, mock_fm, mock_prediction, env
    ):
        mock_prediction.feature_importance = {"duration": 0.6, "checking_status": 0.4}
        trained = _predictor()
        trained.predict.return_value = mock_prediction
        mock_fm.list_predictors.return_value = [trained]
        mock_sphere.foundational_model.return_value = mock_fm
        result = runner.invoke(main, [
            "foundation", "predict", "fm-abc123", "class", '{"duration": 12}', "--explain"
        ], env=env)
        assert result.exit_code == 0, result.output
        assert "duration" in result.output
        assert "0.6" in result.output
        _, kwargs = trained.predict.call_args
        assert kwargs["feature_importance"] is True


class TestFormatJobLine:
    """The per-job status lines `foundation jobs` and `wait` print.

    This is what a user watches for minutes at a time while a model trains, so
    the timings and the running/queued distinction are the product here.
    """

    @staticmethod
    def _line(job, now=None):
        from ffs.model_cmd import _format_job_line
        now = now or datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        return _format_job_line(job, "train_es", now)

    def test_done_job_shows_its_duration(self):
        line = self._line({
            "status": "done",
            "created_at": "2026-01-01T11:00:00+00:00",
            "finished_at": "2026-01-01T11:30:00+00:00",
        })
        assert "done" in line
        assert "30m00s" in line

    def test_running_job_shows_percent_and_age(self):
        line = self._line({
            "status": "running", "progress": 0.42,
            "created_at": "2026-01-01T11:55:00+00:00",
        })
        assert "42.0%" in line
        assert "5m00s" in line

    def test_progress_given_as_a_percentage_is_not_doubled(self):
        """The server sends 0-1 in some paths and 0-100 in others."""
        assert "75.0%" in self._line({"status": "running", "progress": 75})

    def test_running_without_progress_still_reports_running(self):
        line = self._line({"status": "running", "created_at": "2026-01-01T11:59:00+00:00"})
        assert "running" in line
        assert "%" not in line

    def test_queue_is_shown_for_unfinished_jobs(self):
        assert "[gpu]" in self._line({"status": "running", "queue": "gpu"})

    def test_queue_is_omitted_once_the_job_is_done(self):
        """Which queue ran it stops being actionable the moment it finishes."""
        assert "[gpu]" not in self._line({"status": "done", "queue": "gpu"})

    def test_queued_job_shows_its_raw_status(self):
        assert "queued" in self._line({"status": "queued", "queue": "cpu"})

    def test_unparseable_timestamps_are_skipped_not_fatal(self):
        line = self._line({"status": "running", "created_at": "not-a-date"})
        assert "running" in line

    def test_missing_status_renders_a_placeholder(self):
        assert "?" in self._line({})

    def test_accepts_datetime_objects_as_well_as_strings(self):
        line = self._line({
            "status": "done",
            "created_at": datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc),
            "finished_at": datetime(2026, 1, 1, 11, 2, tzinfo=timezone.utc),
        })
        assert "2m00s" in line
