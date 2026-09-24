"""Tests for ffs/train_cmd.py — `ffs train model`.

`ffs train model` trains a *predictor* on an existing embedding space, always in
a new session. The naming is the trap this file is careful about: the group is
`train`, the subcommand is `model`, and the thing produced is a predictor, so
tests below pin both the plumbing (which SDK call each --type reaches) and the
discoverability of the command itself.
"""
import json
from pathlib import Path

import pytest

from ffs.cli import main

CREDIT_CSV = str(Path(__file__).parent / "credit-sklearn.csv")


@pytest.fixture
def wired(mock_sphere, mock_fm, mock_predictor):
    mock_fm.create_binary_classifier.return_value = mock_predictor
    mock_fm.create_regressor.return_value = mock_predictor
    mock_sphere.foundational_model.return_value = mock_fm
    return mock_sphere


def base_args(*extra):
    return [
        "train", "model", "es-parent-1",
        "--target-column", "class",
        "--data", CREDIT_CSV,
        *extra,
    ]


class TestTrainModel:
    def test_classifier(self, runner, wired, mock_fm, env):
        result = runner.invoke(main, base_args("--type", "classifier"), env=env)
        assert result.exit_code == 0, result.output
        mock_fm.create_binary_classifier.assert_called_once()
        mock_fm.create_regressor.assert_not_called()

    def test_regressor(self, runner, wired, mock_fm, env):
        result = runner.invoke(main, base_args("--type", "regressor"), env=env)
        assert result.exit_code == 0, result.output
        mock_fm.create_regressor.assert_called_once()
        mock_fm.create_binary_classifier.assert_not_called()

    def test_data_file_is_passed_as_labels_file(self, runner, wired, mock_fm, env):
        """--data carries the labels here, unlike `models create` where it's the corpus."""
        runner.invoke(main, base_args("--type", "classifier"), env=env)
        _, kwargs = mock_fm.create_binary_classifier.call_args
        assert kwargs["labels_file"] == CREDIT_CSV
        assert kwargs["target_column"] == "class"

    def test_name_and_epochs_are_passed(self, runner, wired, mock_fm, env):
        runner.invoke(
            main,
            base_args("--type", "classifier", "--name", "churn-v2", "--epochs", "25"),
            env=env,
        )
        _, kwargs = mock_fm.create_binary_classifier.call_args
        assert kwargs["name"] == "churn-v2"
        assert kwargs["epochs"] == 25

    def test_epochs_defaults_to_zero_meaning_auto(self, runner, wired, mock_fm, env):
        runner.invoke(main, base_args("--type", "classifier"), env=env)
        _, kwargs = mock_fm.create_binary_classifier.call_args
        assert kwargs["epochs"] == 0

    def test_resolves_the_parent_embedding_space_by_id(self, runner, wired, env):
        runner.invoke(main, base_args("--type", "classifier"), env=env)
        wired.foundational_model.assert_called_once_with("es-parent-1")

    def test_human_output_distinguishes_the_new_session_from_the_parent(
        self, runner, wired, env
    ):
        """The whole point of this command is that it doesn't touch the parent ES."""
        result = runner.invoke(main, base_args("--type", "classifier"), env=env)
        assert result.exit_code == 0, result.output
        assert "pred-xyz789" in result.output
        assert "es-parent-1" in result.output
        assert "new" in result.output

    def test_human_output_points_at_the_wait_command(self, runner, wired, env):
        result = runner.invoke(main, base_args("--type", "classifier"), env=env)
        assert "wait" in result.output
        # The session to wait on is the predictor's, not the parent ES's.
        assert "wait fm-abc123" in result.output

    def test_json_output(self, runner, wired, env):
        result = runner.invoke(
            main, ["--json"] + base_args("--type", "classifier"), env=env
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["predictor_id"] == "pred-xyz789"
        assert data["session_id"] == "fm-abc123"
        assert data["es_session_id"] == "es-parent-1"
        assert data["type"] == "classifier"


class TestTrainModelValidation:
    @pytest.mark.parametrize("missing", ["--target-column", "--type", "--data"])
    def test_required_options(self, runner, wired, env, missing):
        args = base_args("--type", "classifier")
        # Drop the option and its value.
        i = args.index(missing)
        del args[i : i + 2]
        result = runner.invoke(main, args, env=env)
        assert result.exit_code == 2
        assert missing in result.output

    def test_rejects_an_unknown_type(self, runner, wired, env):
        result = runner.invoke(main, base_args("--type", "clusterer"), env=env)
        assert result.exit_code == 2
        assert "clusterer" in result.output

    def test_rejects_a_missing_data_file(self, runner, wired, env):
        result = runner.invoke(
            main,
            [
                "train", "model", "es-parent-1",
                "--target-column", "class", "--type", "classifier",
                "--data", "/nonexistent/labels.csv",
            ],
            env=env,
        )
        assert result.exit_code == 2

    def test_requires_a_session_id(self, runner, wired, env):
        result = runner.invoke(
            main,
            ["train", "model", "--target-column", "class", "--type", "classifier",
             "--data", CREDIT_CSV],
            env=env,
        )
        assert result.exit_code == 2

    def test_api_failure_is_surfaced(self, runner, wired, mock_fm, env):
        mock_fm.create_binary_classifier.side_effect = RuntimeError("no such column")
        result = runner.invoke(main, base_args("--type", "classifier"), env=env)
        assert result.exit_code != 0


class TestTrainDiscoverability:
    """`train` holds one subcommand whose own help says it trains a *predictor*.

    `ffs train predictor` is the natural thing to type and it fails without a
    suggestion, because difflib puts "predictor" nowhere near "model". These
    tests pin what actually happens today so a fix (an alias, or a better miss
    message) shows up here as a deliberate change rather than a surprise.
    """

    def test_train_group_lists_model(self, runner):
        result = runner.invoke(main, ["train", "--help"])
        assert result.exit_code == 0
        assert "model" in result.output

    def test_train_predictor_is_not_a_command(self, runner):
        result = runner.invoke(main, ["train", "predictor"])
        assert result.exit_code == 2
        assert "No such command 'predictor'" in result.output

    def test_train_predictor_gets_no_suggestion_today(self, runner):
        """Documents a known gap: the near-miss helper can't bridge this one."""
        result = runner.invoke(main, ["train", "predictor"])
        assert "Did you mean" not in result.output

    def test_train_mdoel_typo_is_caught(self, runner):
        """The near-miss helper does work for actual typos of `model`."""
        result = runner.invoke(main, ["train", "mdoel"])
        assert result.exit_code == 2
        assert "Did you mean 'model'?" in result.output
