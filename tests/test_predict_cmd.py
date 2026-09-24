"""Tests for ffs/predict_cmd.py."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import click
import pytest

from featrixsphere.api import PredictionResult

from ffs.cli import main
from ffs.predict_cmd import _read_data_file, _get_predictor

CREDIT_CSV = str(Path(__file__).parent / "credit-sklearn.csv")


def wire_predictor(mock_sphere, mock_fm, mock_predictor):
    """Point `ffs predict` at a predictor the way the real lookup does.

    Not client.predictor(): that returns a Predictor whose .id is the session
    id, which the prediction endpoint 404s on. Verified against the live API —
    see tests/test_live_api.py.
    """
    mock_fm.list_predictors.return_value = [mock_predictor]
    mock_sphere.foundational_model.return_value = mock_fm
    return mock_predictor


class TestReadDataFile:
    def test_reads_csv(self):
        df = _read_data_file(CREDIT_CSV)
        assert len(df) == 1000
        assert "class" in df.columns

    def test_unsupported_format(self):
        with pytest.raises(click.ClickException, match="Unsupported"):
            _read_data_file("data.xyz")


class TestGetPredictor:
    def test_without_target(self):
        """The session's single predictor, resolved via list_predictors().

        Deliberately not client.predictor(): that one's .id is the session id,
        and the prediction endpoint 404s on it. Proven live — see
        tests/test_live_api.py::test_predict_without_target_column.
        """
        state = MagicMock()
        mock_pred = MagicMock()
        state.client.foundational_model.return_value.list_predictors.return_value = [mock_pred]
        assert _get_predictor(state, "model-1") == mock_pred
        state.client.predictor.assert_not_called()

    def test_without_target_when_there_are_none(self):
        state = MagicMock()
        state.client.foundational_model.return_value.list_predictors.return_value = []
        with pytest.raises(click.ClickException, match="No predictor found"):
            _get_predictor(state, "model-1")

    def test_without_target_when_ambiguous(self):
        """Two predictors and no --target-column must not silently pick one."""
        state = MagicMock()
        a, b = MagicMock(target_column="class"), MagicMock(target_column="amount")
        state.client.foundational_model.return_value.list_predictors.return_value = [a, b]
        with pytest.raises(click.ClickException, match="--target-column"):
            _get_predictor(state, "model-1")

    def test_with_target_found(self):
        state = MagicMock()
        mock_pred = MagicMock()
        mock_pred.target_column = "class"
        mock_fm = MagicMock()
        mock_fm.list_predictors.return_value = [mock_pred]
        state.client.foundational_model.return_value = mock_fm
        result = _get_predictor(state, "model-1", target_column="class")
        assert result == mock_pred

    def test_with_target_not_found(self):
        state = MagicMock()
        mock_fm = MagicMock()
        mock_fm.list_predictors.return_value = []
        state.client.foundational_model.return_value = mock_fm
        with pytest.raises(click.ClickException, match="No predictor found"):
            _get_predictor(state, "model-1", target_column="nonexistent")


class TestSinglePredict:
    def test_predict(self, runner, mock_sphere, mock_fm, mock_predictor, mock_prediction, env):
        wire_predictor(mock_sphere, mock_fm, mock_predictor)
        mock_predictor.predict.return_value = mock_prediction
        result = runner.invoke(main, [
            "predict", "model-1", '{"checking_status": "no checking"}'
        ], env=env)
        assert result.exit_code == 0
        assert "good" in result.output

    def test_json_output(self, runner, mock_sphere, mock_fm, mock_predictor, mock_prediction, env):
        wire_predictor(mock_sphere, mock_fm, mock_predictor)
        mock_predictor.predict.return_value = mock_prediction
        result = runner.invoke(main, [
            "--json", "predict", "model-1", '{"checking_status": "no checking"}'
        ], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["predicted_class"] == "good"

    def test_with_explain(self, runner, mock_sphere, mock_fm, mock_predictor, mock_prediction, env):
        mock_prediction.feature_importance = {"checking_status": 0.45, "duration": 0.30}
        wire_predictor(mock_sphere, mock_fm, mock_predictor)
        mock_predictor.predict.return_value = mock_prediction
        result = runner.invoke(main, [
            "predict", "model-1", '{"checking_status": "no checking"}', "--explain"
        ], env=env)
        assert result.exit_code == 0
        assert "checking_status" in result.output

    def test_with_regressor_prediction(self, runner, mock_sphere, mock_fm, mock_predictor, env):
        pred_result = MagicMock()
        pred_result.predicted_class = None
        pred_result.prediction = 42.5
        pred_result.confidence = None
        pred_result.probability = None
        pred_result.probabilities = {}
        pred_result.prediction_uuid = None
        pred_result.feature_importance = None
        pred_result.to_dict.return_value = {"prediction": 42.5}
        wire_predictor(mock_sphere, mock_fm, mock_predictor)
        mock_predictor.predict.return_value = pred_result
        result = runner.invoke(main, [
            "predict", "model-1", '{"age": 35}'
        ], env=env)
        assert result.exit_code == 0
        assert "42.5" in result.output


class TestBatchPredict:
    def test_batch_from_csv(self, runner, mock_sphere, mock_fm, mock_predictor, env):
        batch_r1 = MagicMock()
        batch_r1.predicted_class = "good"
        batch_r1.prediction = None
        batch_r1.confidence = 0.9
        batch_r1.to_dict.return_value = {"predicted_class": "good"}
        batch_r2 = MagicMock()
        batch_r2.predicted_class = "bad"
        batch_r2.prediction = None
        batch_r2.confidence = 0.7
        batch_r2.to_dict.return_value = {"predicted_class": "bad"}
        wire_predictor(mock_sphere, mock_fm, mock_predictor)
        mock_predictor.batch_predict.return_value = [batch_r1, batch_r2]
        result = runner.invoke(main, [
            "predict", "model-1", "--file", CREDIT_CSV
        ], env=env)
        assert result.exit_code == 0
        assert "2 predictions" in result.output

    def test_batch_json_output(self, runner, mock_sphere, mock_fm, mock_predictor, env):
        batch_r = MagicMock()
        batch_r.to_dict.return_value = {"predicted_class": "good", "confidence": 0.9}
        wire_predictor(mock_sphere, mock_fm, mock_predictor)
        mock_predictor.batch_predict.return_value = [batch_r]
        result = runner.invoke(main, [
            "--json", "predict", "model-1", "--file", CREDIT_CSV
        ], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1


class TestPredictErrors:
    def test_no_input(self, runner, mock_sphere, mock_fm, mock_predictor, env):
        result = runner.invoke(main, ["predict", "model-1"], env=env)
        assert result.exit_code != 0


class TestUnservablePredictor:
    """A predictor with no model behind it must error, never print nulls.

    The server answers a prediction against a predictor whose training job is
    queued, running or dead with a 200 whose every result field is null — and
    the SDK drops the server's explanation from to_dict(), so the CLI used to
    print an empty table (or a JSON object of nulls) as if that were an answer.
    """

    def test_queued_predictor_is_refused_up_front(self, runner, mock_sphere, mock_fm, mock_predictor, env):
        mock_predictor.status = "ready"
        wire_predictor(mock_sphere, mock_fm, mock_predictor)
        result = runner.invoke(main, [
            "predict", "model-1", '{"duration": 12}'
        ], env=env)
        assert result.exit_code != 0
        assert "still training" in result.output
        assert "queued" in result.output
        mock_predictor.predict.assert_not_called()

    def test_aborted_predictor_is_refused_up_front(self, runner, mock_sphere, mock_fm, mock_predictor, env):
        mock_predictor.status = "aborted"
        wire_predictor(mock_sphere, mock_fm, mock_predictor)
        result = runner.invoke(main, [
            "predict", "model-1", '{"duration": 12}'
        ], env=env)
        assert result.exit_code != 0
        assert "aborted" in result.output

    def test_null_result_reports_training_progress(self, runner, mock_sphere, mock_fm, mock_predictor, env):
        mock_predictor.predict.return_value = PredictionResult(
            query_record={"duration": 12},
            training_status={
                "status": "training_in_progress_no_checkpoint",
                "message": "Training is in progress.",
                "current_epoch": 5,
                "total_epochs": 50,
            },
        )
        wire_predictor(mock_sphere, mock_fm, mock_predictor)
        result = runner.invoke(main, [
            "predict", "model-1", '{"duration": 12}'
        ], env=env)
        assert result.exit_code != 0
        assert "epoch 5/50" in result.output

    def test_null_result_errors_in_json_mode_too(self, runner, mock_sphere, mock_fm, mock_predictor, env):
        """--json must not hand a script an object of nulls and exit 0."""
        mock_predictor.predict.return_value = PredictionResult(query_record={"duration": 12})
        wire_predictor(mock_sphere, mock_fm, mock_predictor)
        result = runner.invoke(main, [
            "--json", "predict", "model-1", '{"duration": 12}'
        ], env=env)
        assert result.exit_code != 0
        assert "ffs foundation jobs" in result.output

    def test_null_batch_results_are_refused(self, runner, mock_sphere, mock_fm, mock_predictor, env):
        mock_predictor.batch_predict.return_value = [
            PredictionResult(query_record={"duration": 12}),
            PredictionResult(query_record={"duration": 24}),
        ]
        wire_predictor(mock_sphere, mock_fm, mock_predictor)
        result = runner.invoke(main, [
            "predict", "model-1", "--file", CREDIT_CSV
        ], env=env)
        assert result.exit_code != 0
        assert "No prediction" in result.output
