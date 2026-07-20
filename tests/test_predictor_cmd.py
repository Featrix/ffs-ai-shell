"""Tests for ffs/predictor_cmd.py."""
import json

from ffs.cli import main


class TestPredictorCreate:
    def test_create_classifier(self, runner, mock_sphere, mock_fm, mock_predictor, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.create_binary_classifier.return_value = mock_predictor
        result = runner.invoke(main, [
            "predictor", "create", "fm-abc123",
            "--target-column", "class", "--type", "classifier"
        ], env=env)
        assert result.exit_code == 0
        assert "pred-xyz789" in result.output
        mock_fm.create_binary_classifier.assert_called_once()

    def test_create_regressor(self, runner, mock_sphere, mock_fm, mock_predictor, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.create_regressor.return_value = mock_predictor
        result = runner.invoke(main, [
            "predictor", "create", "fm-abc123",
            "--target-column", "credit_amount", "--type", "regressor"
        ], env=env)
        assert result.exit_code == 0
        mock_fm.create_regressor.assert_called_once()

    def test_json_output(self, runner, mock_sphere, mock_fm, mock_predictor, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.create_binary_classifier.return_value = mock_predictor
        result = runner.invoke(main, [
            "--json", "predictor", "create", "fm-abc123",
            "--target-column", "class", "--type", "classifier"
        ], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["predictor_id"] == "pred-xyz789"
        assert data["type"] == "classifier"

    def test_with_labels_file(self, runner, mock_sphere, mock_fm, mock_predictor, env):
        from pathlib import Path
        csv = str(Path(__file__).parent / "credit-sklearn.csv")
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.create_binary_classifier.return_value = mock_predictor
        result = runner.invoke(main, [
            "predictor", "create", "fm-abc123",
            "--target-column", "class", "--type", "classifier",
            "--labels", csv
        ], env=env)
        assert result.exit_code == 0
        kwargs = mock_fm.create_binary_classifier.call_args.kwargs
        assert "labels_file" in kwargs


class TestPredictorList:
    def test_lists_predictors(self, runner, mock_sphere, mock_fm, mock_predictor, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.list_predictors.return_value = [mock_predictor]
        result = runner.invoke(main, ["predictor", "list", "fm-abc123"], env=env)
        assert result.exit_code == 0
        assert "pred-xyz789" in result.output

    def test_empty_list(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.list_predictors.return_value = []
        result = runner.invoke(main, ["predictor", "list", "fm-abc123"], env=env)
        assert result.exit_code == 0
        assert "No predictors found" in result.output

    def test_json_output(self, runner, mock_sphere, mock_fm, mock_predictor, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.list_predictors.return_value = [mock_predictor]
        result = runner.invoke(main, ["--json", "predictor", "list", "fm-abc123"], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["id"] == "pred-xyz789"


class TestPredictorShow:
    """`predictor show` resolves via fm.list_predictors(), not the legacy
    client.predictor() call — that call only reads the live (TTL'd) Redis
    job record and goes null once a done predictor's job record expires."""

    def test_shows_predictor(self, runner, mock_sphere, mock_fm, mock_predictor, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.list_predictors.return_value = [mock_predictor]
        result = runner.invoke(main, ["predictor", "show", "fm-abc123"], env=env)
        assert result.exit_code == 0
        assert "pred-xyz789" in result.output
        assert "0.8500" in result.output  # accuracy

    def test_json_output(self, runner, mock_sphere, mock_fm, mock_predictor, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.list_predictors.return_value = [mock_predictor]
        result = runner.invoke(main, ["--json", "predictor", "show", "fm-abc123"], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["id"] == "pred-xyz789"

    def test_shows_metrics(self, runner, mock_sphere, mock_fm, mock_predictor, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.list_predictors.return_value = [mock_predictor]
        result = runner.invoke(main, ["predictor", "show", "fm-abc123"], env=env)
        assert "0.9200" in result.output  # AUC
        assert "0.8300" in result.output  # F1

    def test_no_predictor_found(self, runner, mock_sphere, mock_fm, env):
        mock_sphere.foundational_model.return_value = mock_fm
        mock_fm.list_predictors.return_value = []
        result = runner.invoke(main, ["predictor", "show", "fm-abc123"], env=env)
        assert result.exit_code != 0
        assert "No predictor found" in result.output
