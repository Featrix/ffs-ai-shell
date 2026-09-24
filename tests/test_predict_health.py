"""Tests for ffs/predict_health.py and the guards it puts on prediction output.

The bug this module exists to stop: a predictor whose training job is queued,
running or dead has no servable model, and the server answers a prediction
request for it with a well-formed 200 whose every field is null. Without a
guard, `ffs predict` prints an empty table and `ffs --json predict` prints an
object of nulls — both of which read as "the model has no opinion" rather than
"there is no model yet".

So the unit tests below pin the status vocabulary, and the integration tests at
the bottom assert the CLI actually fails loudly on an empty answer.
"""
import dataclasses
import json
from unittest.mock import create_autospec

import pytest
from featrixsphere.api import PredictionResult

from ffs.cli import cli, main
from ffs.predict_health import (
    FAILED_STATUSES,
    PENDING_STATUSES,
    TERMINAL_STATUSES,
    display_status,
    format_training_status,
    has_prediction,
    not_ready_reason,
    require_predictions,
    session_diagnosis,
    training_status_detail,
    unusable_prediction_error,
)


class TestStatusVocabulary:
    def test_ready_is_a_pending_status_not_a_servable_one(self):
        """"ready" is the queued state — the single most misleading name here."""
        assert "ready" in PENDING_STATUSES
        assert "ready" not in TERMINAL_STATUSES

    def test_pending_and_failed_do_not_overlap(self):
        assert not PENDING_STATUSES & FAILED_STATUSES

    def test_terminal_covers_success_and_failure(self):
        assert {"done", "completed"} <= TERMINAL_STATUSES
        assert FAILED_STATUSES <= TERMINAL_STATUSES


class TestDisplayStatus:
    @pytest.mark.parametrize(
        "raw,shown",
        [("ready", "queued"), ("completed", "done"), ("running", "running"),
         ("aborted", "aborted")],
    )
    def test_relabels_misleading_statuses(self, raw, shown):
        assert display_status(raw) == shown

    @pytest.mark.parametrize("value", [None, "", 0, {}, 3])
    def test_non_strings_render_as_a_dash(self, value):
        assert display_status(value) == "—"


class TestNotReadyReason:
    @pytest.mark.parametrize("status", sorted(PENDING_STATUSES))
    def test_pending_statuses_are_not_ready(self, status, mock_predictor):
        mock_predictor.status = status
        reason = not_ready_reason(mock_predictor)
        assert reason and "still training" in reason

    @pytest.mark.parametrize("status", sorted(FAILED_STATUSES))
    def test_failed_statuses_are_not_ready(self, status, mock_predictor):
        mock_predictor.status = status
        reason = not_ready_reason(mock_predictor)
        assert reason and "training job" in reason

    @pytest.mark.parametrize("status", ["done", "completed"])
    def test_finished_predictors_are_ready(self, status, mock_predictor):
        mock_predictor.status = status
        assert not_ready_reason(mock_predictor) is None

    def test_a_missing_status_is_treated_as_maybe_ready(self, mock_predictor):
        """list_predictors() drops status once the job's Redis record ages out.

        That happens long before the model stops being servable, so an unset
        status must not be read as "not ready" — otherwise every prediction
        against an older, perfectly good session would be refused.
        """
        mock_predictor.status = None
        assert not_ready_reason(mock_predictor) is None

    def test_ready_is_reported_as_queued(self, mock_predictor):
        mock_predictor.status = "ready"
        assert "queued" in not_ready_reason(mock_predictor)


class TestHasPrediction:
    @pytest.mark.parametrize(
        "field,value",
        [("prediction", 3.5), ("predicted_class", "good"), ("probability", 0.9),
         ("confidence", 0.8), ("probabilities", {"good": 0.9})],
    )
    def test_any_populated_field_counts(self, field, value):
        assert has_prediction({field: value}) is True

    def test_all_null_is_not_a_prediction(self):
        assert has_prediction({
            "prediction": None, "predicted_class": None, "probability": None,
            "confidence": None, "probabilities": None,
        }) is False

    def test_empty_containers_are_not_predictions(self):
        assert has_prediction({"probabilities": {}}) is False
        assert has_prediction({"predicted_labels": []}) is False

    def test_a_zero_confidence_still_counts_as_an_answer(self):
        """0.0 is a real number the model produced; only null means "nothing"."""
        assert has_prediction({"confidence": 0.0}) is True

    def test_works_on_objects_as_well_as_dicts(self, mock_prediction):
        assert has_prediction(mock_prediction) is True

    def test_an_empty_object_is_not_a_prediction(self, mock_prediction):
        for field in ("prediction", "predicted_class", "probability",
                      "confidence", "probabilities"):
            setattr(mock_prediction, field, None)
        assert has_prediction(mock_prediction) is False


class TestTrainingStatusDetail:
    def test_reads_the_dict_the_sdk_hides_from_to_dict(self, mock_prediction):
        mock_prediction.training_status = {"status": "running", "message": "training"}
        assert training_status_detail(mock_prediction)["status"] == "running"

    @pytest.mark.parametrize("value", [None, "running", 42, []])
    def test_non_dicts_are_ignored(self, value, mock_prediction):
        mock_prediction.training_status = value
        assert training_status_detail(mock_prediction) is None


class TestFormatTrainingStatus:
    def test_uses_the_servers_message(self):
        assert "still warming up" in format_training_status(
            {"message": "still warming up"}
        )

    def test_falls_back_to_the_status_when_there_is_no_message(self):
        line = format_training_status({"status": "running"})
        assert "running" in line

    def test_includes_epoch_progress(self):
        line = format_training_status(
            {"message": "training", "current_epoch": 3, "total_epochs": 10}
        )
        assert "epoch 3/10" in line

    def test_includes_percent_complete(self):
        line = format_training_status({"message": "training", "progress_percent": 42})
        assert "42% complete" in line

    def test_omits_progress_when_total_epochs_is_unknown(self):
        line = format_training_status(
            {"message": "training", "current_epoch": 3, "total_epochs": 0}
        )
        assert "epoch" not in line


class TestSessionDiagnosis:
    def test_names_the_training_job_and_its_status(self, mock_sphere, mock_fm):
        state = _state(mock_sphere)
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.refresh.return_value = {
            "job_plan": [
                {"job_type": "train_single_predictor", "job_id": "j1",
                 "spec": {"target_column": "class"}},
            ],
            "jobs": {"j1": {"status": "aborted", "error": "worker died"}},
            "session": {"status": "done"},
        }
        lines = session_diagnosis(state, "fm-abc123")
        joined = "\n".join(lines)
        assert "train_single_predictor (class)" in joined
        assert "aborted" in joined
        assert "worker died" in joined
        assert "Session status: done" in joined

    def test_ignores_jobs_of_other_types(self, mock_sphere, mock_fm):
        state = _state(mock_sphere)
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.refresh.return_value = {
            "job_plan": [{"job_type": "train_es", "job_id": "j1"}],
            "jobs": {"j1": {"status": "done"}},
        }
        assert not any("train_es" in line for line in session_diagnosis(state, "x"))

    def test_finds_live_jobs_missing_from_the_plan(self, mock_sphere, mock_fm):
        """job_plan is persisted; the jobs block is a live view that can diverge."""
        state = _state(mock_sphere)
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.refresh.return_value = {
            "job_plan": [],
            "jobs": {"j9": {"job_type": "train_single_predictor",
                            "status": "running", "target_column": "class"}},
        }
        assert any("train_single_predictor" in line
                   for line in session_diagnosis(state, "x"))

    def test_a_failed_lookup_degrades_to_no_diagnosis(self, mock_sphere):
        """Diagnosis is a courtesy; it must never mask the original error."""
        state = _state(mock_sphere)
        mock_sphere.foundational_model.side_effect = RuntimeError("500")
        assert session_diagnosis(state, "fm-abc123") == []


class TestRequirePredictions:
    def test_passes_through_a_real_prediction(self, mock_sphere, mock_prediction):
        state = _state(mock_sphere)
        require_predictions(state, "fm-abc123", [mock_prediction])

    def test_rejects_an_empty_result_list(self, mock_sphere):
        state = _state(mock_sphere)
        with pytest.raises(Exception) as exc:
            require_predictions(state, "fm-abc123", [])
        assert "No prediction" in str(exc.value)

    def test_rejects_an_all_null_result(self, mock_sphere, mock_prediction):
        state = _state(mock_sphere)
        for field in ("prediction", "predicted_class", "probability",
                      "confidence", "probabilities"):
            setattr(mock_prediction, field, None)
        with pytest.raises(Exception) as exc:
            require_predictions(state, "fm-abc123", [mock_prediction])
        assert "No prediction" in str(exc.value)

    def test_accepts_a_batch_where_only_some_rows_answered(
        self, mock_sphere, mock_prediction
    ):
        state = _state(mock_sphere)
        require_predictions(state, "fm-abc123", [_blank_result(), mock_prediction])

    def test_includes_the_servers_explanation_when_present(self, mock_sphere):
        state = _state(mock_sphere)
        blank = _blank_result()
        blank.training_status = {"message": "predictor is still training",
                                 "current_epoch": 2, "total_epochs": 8}
        with pytest.raises(Exception) as exc:
            require_predictions(state, "fm-abc123", [blank])
        message = str(exc.value)
        assert "still training" in message
        assert "epoch 2/8" in message

    def test_names_the_target_column_and_the_next_step(self, mock_sphere):
        state = _state(mock_sphere)
        with pytest.raises(Exception) as exc:
            require_predictions(
                state, "fm-abc123", [_blank_result()], target_column="class"
            )
        message = str(exc.value)
        assert "'class'" in message
        assert "ffs foundation jobs fm-abc123" in message


class TestUnusablePredictionError:
    def test_includes_the_reason_and_hint(self, mock_sphere):
        state = _state(mock_sphere)
        err = unusable_prediction_error(
            state, "fm-abc123", target_column="class",
            reason="it is still training", hint="try --foundation",
        )
        message = err.format_message()
        assert "still training" in message
        assert "try --foundation" in message
        assert "ffs foundation jobs fm-abc123" in message


# ---------------------------------------------------------------------------
# Integration: the CLI must not print an empty answer as if it were an answer.
# ---------------------------------------------------------------------------


class TestPredictRefusesEmptyAnswers:
    def test_predict_fails_when_the_predictor_is_still_queued(
        self, runner, wired_sphere, mock_predictor, env
    ):
        mock_predictor.status = "ready"
        result = runner.invoke(main, ["predict", "fm-abc123", '{"duration": 12}'], env=env)
        assert result.exit_code != 0
        assert "queued" in result.output
        mock_predictor.predict.assert_not_called()

    @pytest.mark.parametrize("status", sorted(FAILED_STATUSES))
    def test_predict_fails_when_training_died(
        self, runner, wired_sphere, mock_predictor, env, status
    ):
        mock_predictor.status = status
        result = runner.invoke(main, ["predict", "fm-abc123", '{"duration": 12}'], env=env)
        assert result.exit_code != 0
        assert "No prediction" in result.output

    def test_predict_fails_on_an_all_null_response(
        self, runner, wired_sphere, mock_predictor, env
    ):
        """The server's 200-with-nulls reply must not print as a blank answer."""
        mock_predictor.predict.return_value = _blank_result()
        result = runner.invoke(main, ["predict", "fm-abc123", '{"duration": 12}'], env=env)
        assert result.exit_code != 0
        assert "No prediction" in result.output

    def test_json_mode_does_not_print_an_object_of_nulls(
        self, runner, wired_sphere, mock_predictor, env
    ):
        """Scripts and agents read --json; nulls there look like a real result."""
        mock_predictor.predict.return_value = _blank_result()
        result = runner.invoke(
            main, ["--json", "predict", "fm-abc123", '{"duration": 12}'], env=env
        )
        assert result.exit_code != 0
        assert "No prediction" in result.output
        # The all-null result object must not have been printed as the answer.
        assert '"prediction": null' not in result.output

    def test_the_real_entry_point_reports_the_refusal_as_json(
        self, wired_sphere, mock_predictor, monkeypatch, capsys
    ):
        """`main` alone can't show this: the JSON error envelope is in `cli()`,
        which formats by inspecting sys.argv. Going through the entry point is
        the only way to see what `ffs --json predict ...` actually prints."""
        mock_predictor.predict.return_value = _blank_result()
        argv = ["ffs", "--json", "predict", "fm-abc123", '{"duration": 12}']
        monkeypatch.setattr("sys.argv", argv)
        monkeypatch.setenv("FEATRIX_API_KEY", "fx_test_fake_key")
        with pytest.raises(SystemExit) as exc:
            cli()
        assert exc.value.code != 0
        payload = json.loads(capsys.readouterr().out)
        assert "No prediction" in payload["error"]["message"]

    def test_batch_predict_fails_when_no_row_got_an_answer(
        self, runner, wired_sphere, mock_predictor, env, tmp_path
    ):
        data = tmp_path / "rows.csv"
        data.write_text("duration\n12\n24\n")
        mock_predictor.batch_predict.return_value = [_blank_result(), _blank_result()]
        result = runner.invoke(
            main, ["predict", "fm-abc123", "--file", str(data)], env=env
        )
        assert result.exit_code != 0
        assert "No prediction" in result.output

    def test_a_healthy_prediction_still_works(self, runner, wired_sphere, env):
        """The guard must not block the normal path."""
        result = runner.invoke(main, ["predict", "fm-abc123", '{"duration": 12}'], env=env)
        assert result.exit_code == 0, result.output
        assert "good" in result.output


class TestModelsPredictRefusesEmptyAnswers:
    def test_fails_when_the_trained_predictor_is_not_servable(
        self, runner, wired_sphere, mock_predictor, env
    ):
        mock_predictor.status = "ready"
        result = runner.invoke(
            main, ["models", "predict", "fm-abc123", "class", '{"duration": 12}'],
            env=env,
        )
        assert result.exit_code != 0
        assert "queued" in result.output

    def test_suggests_the_foundation_model_as_a_way_forward(
        self, runner, wired_sphere, mock_predictor, env
    ):
        mock_predictor.status = "ready"
        result = runner.invoke(
            main, ["models", "predict", "fm-abc123", "class", '{"duration": 12}'],
            env=env,
        )
        assert "--foundation" in result.output

    def test_foundation_path_is_also_guarded(
        self, runner, wired_sphere, mock_fm, env
    ):
        mock_fm.foundation_predict.return_value = [_blank_result()]
        result = runner.invoke(
            main,
            ["models", "predict", "fm-abc123", "class", '{"duration": 12}',
             "--foundation"],
            env=env,
        )
        assert result.exit_code != 0
        assert "No prediction" in result.output


# --- helpers ---------------------------------------------------------------


def _state(client):
    """A minimal ClientState whose `.client` is the given mock."""
    from ffs.client import ClientState

    state = ClientState(
        server="https://sphere-api.featrix.com", cluster=None,
        output_json=False, quiet=False,
    )
    state._client = client
    return state


def _blank_result():
    """A fresh PredictionResult-shaped object with nothing predicted in it.

    What the server actually returns for an unservable predictor: a complete,
    well-formed object whose every value is null. Built fresh each call — a
    helper that mutated a shared fixture would make a "some rows answered"
    batch secretly be the same blank row twice.
    """
    r = create_autospec(PredictionResult, instance=True)
    for f in dataclasses.fields(PredictionResult):
        setattr(r, f.name, None)
    r.to_dict.return_value = {
        "prediction": None, "predicted_class": None, "confidence": None,
        "probability": None, "probabilities": None,
    }
    return r
