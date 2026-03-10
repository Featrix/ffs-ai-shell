"""Tests for ffs/predict_cmd.py."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import click
import pytest

from ffs.cli import main
from ffs.predict_cmd import _read_data_file, _get_predictor

CREDIT_CSV = str(Path(__file__).parent / "credit-sklearn.csv")


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
        state = MagicMock()
        mock_pred = MagicMock()
        state.client.predictor.return_value = mock_pred
        result = _get_predictor(state, "model-1")
        assert result == mock_pred
        state.client.predictor.assert_called_once_with("model-1")

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
    def test_predict(self, runner, mock_sphere, mock_prediction, env):
        mock_sphere.predictor.return_value.predict.return_value = mock_prediction
        result = runner.invoke(main, [
            "predict", "model-1", '{"checking_status": "no checking"}'
        ], env=env)
        assert result.exit_code == 0
        assert "good" in result.output

    def test_json_output(self, runner, mock_sphere, mock_prediction, env):
        mock_sphere.predictor.return_value.predict.return_value = mock_prediction
        result = runner.invoke(main, [
            "--json", "predict", "model-1", '{"checking_status": "no checking"}'
        ], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["predicted_class"] == "good"

    def test_with_explain(self, runner, mock_sphere, mock_prediction, env):
        mock_prediction.feature_importance = {"checking_status": 0.45, "duration": 0.30}
        mock_sphere.predictor.return_value.predict.return_value = mock_prediction
        result = runner.invoke(main, [
            "predict", "model-1", '{"checking_status": "no checking"}', "--explain"
        ], env=env)
        assert result.exit_code == 0
        assert "checking_status" in result.output

    def test_with_regressor_prediction(self, runner, mock_sphere, env):
        pred_result = MagicMock()
        pred_result.predicted_class = None
        pred_result.prediction = 42.5
        pred_result.confidence = None
        pred_result.probability = None
        pred_result.probabilities = {}
        pred_result.prediction_uuid = None
        pred_result.feature_importance = None
        pred_result.to_dict.return_value = {"prediction": 42.5}
        mock_sphere.predictor.return_value.predict.return_value = pred_result
        result = runner.invoke(main, [
            "predict", "model-1", '{"age": 35}'
        ], env=env)
        assert result.exit_code == 0
        assert "42.5" in result.output


class TestBatchPredict:
    def test_batch_from_csv(self, runner, mock_sphere, env):
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
        mock_sphere.predictor.return_value.batch_predict.return_value = [batch_r1, batch_r2]
        result = runner.invoke(main, [
            "predict", "model-1", "--file", CREDIT_CSV
        ], env=env)
        assert result.exit_code == 0
        assert "2 predictions" in result.output

    def test_batch_json_output(self, runner, mock_sphere, env):
        batch_r = MagicMock()
        batch_r.to_dict.return_value = {"predicted_class": "good", "confidence": 0.9}
        mock_sphere.predictor.return_value.batch_predict.return_value = [batch_r]
        result = runner.invoke(main, [
            "--json", "predict", "model-1", "--file", CREDIT_CSV
        ], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1


class TestPredictErrors:
    def test_no_input(self, runner, mock_sphere, env):
        result = runner.invoke(main, ["predict", "model-1"], env=env)
        assert result.exit_code != 0
