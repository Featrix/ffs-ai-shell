"""Shared test fixtures."""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from click.testing import CliRunner

TESTS_DIR = Path(__file__).parent
CREDIT_CSV = TESTS_DIR / "credit-sklearn.csv"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def env():
    """Env vars that skip config file lookup."""
    return {"FEATRIX_API_KEY": "sk_test_fake_key"}


@pytest.fixture
def mock_sphere():
    """Patch FeatrixSphere so the CLI uses a mock client."""
    with patch("ffs.client.FeatrixSphere") as MockClass:
        yield MockClass.return_value


@pytest.fixture
def mock_fm():
    """A mock FoundationalModel."""
    fm = MagicMock()
    fm.id = "fm-abc123"
    fm.name = "credit-model"
    fm.status = "done"
    fm.dimensions = 128
    fm.epochs = 10
    fm.final_loss = 0.042
    fm.compute_cluster = "default"
    fm.get_columns.return_value = ["checking_status", "duration", "credit_history", "class"]
    fm.get_model_card.return_value = {"name": "credit-model", "dimensions": 128, "epochs": 10}
    fm.list_predictors.return_value = []
    fm.refresh.return_value = {"job_plan": [], "jobs": {}}
    return fm


@pytest.fixture
def mock_predictor():
    """A mock Predictor."""
    p = MagicMock()
    p.id = "pred-xyz789"
    p.session_id = "fm-abc123"
    p.target_column = "class"
    p.target_type = "classifier"
    p.status = "done"
    p.accuracy = 0.85
    p.auc = 0.92
    p.f1 = 0.83
    p.to_dict.return_value = {
        "id": "pred-xyz789",
        "session_id": "fm-abc123",
        "target_column": "class",
        "target_type": "classifier",
        "status": "done",
        "accuracy": 0.85,
    }
    return p


@pytest.fixture
def mock_prediction():
    """A mock prediction result."""
    r = MagicMock()
    r.predicted_class = "good"
    r.prediction = None
    r.confidence = 0.92
    r.probability = 0.92
    r.probabilities = {"good": 0.92, "bad": 0.08}
    r.prediction_uuid = "uuid-12345"
    r.feature_importance = None
    r.to_dict.return_value = {
        "predicted_class": "good",
        "confidence": 0.92,
        "probabilities": {"good": 0.92, "bad": 0.08},
    }
    return r
