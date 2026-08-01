"""Tests for ffs/network_cmd.py."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ffs.cli import main
from featrixsphere.api.published_prediction_network import (
    PredictionNetworkResult,
    PredictionNetworkTraceEntry,
)

SPEC = {
    "nodes": [
        {"id": "is_company", "model": "is-company"},
        {"id": "company_type", "model": "company-type"},
    ],
    "edges": [
        {"from": "is_company", "to": "company_type", "when": {"op": "always"}},
    ],
}


@pytest.fixture
def spec_file(tmp_path):
    p = tmp_path / "network.json"
    p.write_text(json.dumps(SPEC))
    return str(p)


@pytest.fixture
def mock_network():
    """A mock PublishedPredictionNetwork."""
    return MagicMock()


@pytest.fixture(autouse=True)
def _wire_org(mock_sphere):
    """Every network_cmd command resolves org via whoami() first."""
    mock_sphere.whoami.return_value = {"org_slug": "alph", "organization_name": "Alph Inc"}


class TestRegister:
    def test_register(self, runner, mock_sphere, mock_network, env, spec_file):
        mock_sphere.published_prediction_network.return_value = mock_network
        mock_network.register.return_value = {"org": "alph", "name": "net1", "version": 1, "nodes": 2}

        result = runner.invoke(main, [
            "network", "register", "net1", "--spec-file", spec_file,
        ], env=env)

        assert result.exit_code == 0
        assert "v1" in result.output
        mock_sphere.published_prediction_network.assert_called_once_with(
            org="alph", name="net1", api_key=mock_sphere.api_key, base_url=mock_sphere.base_url,
        )
        mock_network.register.assert_called_once_with(SPEC)

    def test_register_json_output(self, runner, mock_sphere, mock_network, env, spec_file):
        mock_sphere.published_prediction_network.return_value = mock_network
        mock_network.register.return_value = {"org": "alph", "name": "net1", "version": 3, "nodes": 2}

        result = runner.invoke(main, [
            "--json", "network", "register", "net1", "--spec-file", spec_file,
        ], env=env)

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["version"] == 3

    def test_missing_spec_file_errors(self, runner, mock_sphere, env):
        result = runner.invoke(main, [
            "network", "register", "net1", "--spec-file", "/nonexistent.json",
        ], env=env)
        assert result.exit_code != 0


class TestShow:
    def test_show(self, runner, mock_sphere, mock_network, env):
        mock_sphere.published_prediction_network.return_value = mock_network
        mock_network.get_spec.return_value = {
            "name": "net1", "version": 2, "spec": SPEC,
        }

        result = runner.invoke(main, ["network", "show", "net1"], env=env)

        assert result.exit_code == 0
        assert "is_company" in result.output
        assert "company_type" in result.output

    def test_show_json_output(self, runner, mock_sphere, mock_network, env):
        mock_sphere.published_prediction_network.return_value = mock_network
        row = {"name": "net1", "version": 2, "spec": SPEC}
        mock_network.get_spec.return_value = row

        result = runner.invoke(main, ["--json", "network", "show", "net1"], env=env)

        assert result.exit_code == 0
        assert json.loads(result.output) == row


class TestList:
    def test_list(self, runner, mock_sphere, env):
        mock_sphere.list_prediction_networks.return_value = [
            {"name": "net1", "version": 2, "updated_at": "2026-07-30T00:00:00Z"},
        ]

        result = runner.invoke(main, ["network", "list"], env=env)

        assert result.exit_code == 0
        assert "net1" in result.output
        mock_sphere.list_prediction_networks.assert_called_once_with("alph")

    def test_list_empty(self, runner, mock_sphere, env):
        mock_sphere.list_prediction_networks.return_value = []
        result = runner.invoke(main, ["network", "list"], env=env)
        assert result.exit_code == 0
        assert "No PredictionNetworks" in result.output

    def test_list_json_output(self, runner, mock_sphere, env):
        mock_sphere.list_prediction_networks.return_value = [
            {"name": "net1", "version": 1, "updated_at": None},
        ]
        result = runner.invoke(main, ["--json", "network", "list"], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]["name"] == "net1"


class TestPredict:
    def _result(self):
        return PredictionNetworkResult(
            final={"company_type": {"label": "Trucking", "confidence": 0.88}},
            trace=[
                PredictionNetworkTraceEntry(
                    node_id="is_company", model="alph/is-company", label="Yes",
                    confidence=0.97, latency_ms=42, skipped=False,
                ),
                PredictionNetworkTraceEntry(
                    node_id="company_type", model="alph/company-type", label="Trucking",
                    confidence=0.88, latency_ms=55, skipped=False,
                ),
                PredictionNetworkTraceEntry(
                    node_id="revenue", skipped=True, skip_reason="company_type != Retailer",
                ),
            ],
        )

    def test_predict_inline_record(self, runner, mock_sphere, mock_network, env):
        mock_sphere.published_prediction_network.return_value = mock_network
        mock_network.predict.return_value = self._result()

        result = runner.invoke(main, [
            "network", "predict", "net1", '{"company_name": "Acme Trucking"}',
        ], env=env)

        assert result.exit_code == 0
        assert "Trucking" in result.output
        assert "revenue" in result.output
        assert "skipped" in result.output
        mock_network.predict.assert_called_once_with({"company_name": "Acme Trucking"})

    def test_predict_from_file(self, runner, mock_sphere, mock_network, env, tmp_path):
        record_file = tmp_path / "row.json"
        record_file.write_text(json.dumps({"company_name": "Acme Trucking"}))
        mock_sphere.published_prediction_network.return_value = mock_network
        mock_network.predict.return_value = self._result()

        result = runner.invoke(main, [
            "network", "predict", "net1", "--file", str(record_file),
        ], env=env)

        assert result.exit_code == 0
        mock_network.predict.assert_called_once_with({"company_name": "Acme Trucking"})

    def test_predict_json_output(self, runner, mock_sphere, mock_network, env):
        mock_sphere.published_prediction_network.return_value = mock_network
        mock_network.predict.return_value = self._result()

        result = runner.invoke(main, [
            "--json", "network", "predict", "net1", '{"company_name": "Acme Trucking"}',
        ], env=env)

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["final"]["company_type"]["label"] == "Trucking"
        skipped = [t for t in data["trace"] if t["skipped"]]
        assert skipped[0]["node_id"] == "revenue"

    def test_predict_no_input_errors(self, runner, mock_sphere, env):
        result = runner.invoke(main, ["network", "predict", "net1"], env=env)
        assert result.exit_code != 0
