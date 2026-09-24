"""Shared test fixtures.

The mocks here are *spec'd against the real featrixsphere classes* rather than
being bare MagicMocks. That distinction is the whole point: a bare MagicMock
invents any attribute you touch, so a test asserting `exit_code == 0` passes
even when ffs calls a method featrixsphere renamed or never had. The CLI then
breaks for customers while CI stays green.

`spec_mock()` derives the mock's attribute surface from the installed class, so
if featrixsphere renames a method or a dataclass field, every command that
touches it fails here instead of in the field. See test_api_contract.py for the
coarser, signature-level version of the same guard.
"""
import dataclasses
import inspect
import sys
import pytest
from unittest.mock import MagicMock, create_autospec, patch
from pathlib import Path
from click.testing import CliRunner

# Checked before importing featrixsphere, which uses `X | Y` annotations at class
# scope and so dies with ten unrelated TypeErrors on 3.9 — the kind of failure
# that reads as "the test suite is broken" rather than "wrong interpreter".
# pyproject declares requires-python = ">=3.10"; macOS still ships 3.9 as
# `python3`, which is what `make test` would otherwise pick up.
if sys.version_info < (3, 10):
    pytest.exit(
        f"ffs requires Python 3.10+ (running {sys.version.split()[0]}). "
        "Run the tests with a newer interpreter, e.g. "
        "`make test PYTHON=python3.11` or `python3.11 -m pytest`.",
        returncode=1,
    )

from featrixsphere.api import (
    APIEndpoint,
    EventGroup,
    FeatrixSphere,
    FoundationalModel,
    PredictionNetworkResult,
    PredictionNetworkTraceEntry,
    PredictionResult,
    Predictor,
    PublishedPredictionNetwork,
)

TESTS_DIR = Path(__file__).parent
CREDIT_CSV = TESTS_DIR / "credit-sklearn.csv"

# The SDK rejects keys that don't start with "fx_" in FeatrixSphere.__init__,
# so a test key that looks plausible keeps mocked and unmocked paths agreeing.
FAKE_API_KEY = "fx_test_fake_key"


def spec_mock(cls, **attrs):
    """Build a mock whose attribute surface is pinned to the real `cls`.

    Reading an attribute the real class doesn't have raises AttributeError, and
    calling a method with a kwarg the real signature rejects raises TypeError.

    Dataclass fields declared without a default aren't class attributes, so
    autospec blocks them too; they're restored (defaulting to None) from
    `dataclasses.fields()` — which means a *renamed* field is still caught,
    because only the names the installed class actually declares get restored.

    Properties get the same treatment for the opposite reason: autospec renders
    one as a plain MagicMock, which is callable *and* iterable, so misreading a
    property as a method (`fm.columns()` where the real API is `fm.columns`)
    would quietly yield an empty sequence instead of failing. Replacing them
    with plain values makes that call a TypeError.
    """
    m = create_autospec(cls, instance=True)
    names = set()
    if dataclasses.is_dataclass(cls):
        names |= {f.name for f in dataclasses.fields(cls)}
    names |= {
        n for n in dir(cls)
        if isinstance(inspect.getattr_static(cls, n, None), property)
    }
    for name in names:
        if name not in attrs:
            setattr(m, name, None)
    for key, value in attrs.items():
        setattr(m, key, value)
    return m


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def env():
    """Env vars that skip config file lookup."""
    return {"FEATRIX_API_KEY": FAKE_API_KEY}


@pytest.fixture
def mock_sphere():
    """Patch FeatrixSphere so the CLI uses a spec'd mock client."""
    with patch("ffs.client.FeatrixSphere") as MockClass:
        client = spec_mock(
            FeatrixSphere,
            api_key=FAKE_API_KEY,
            base_url="https://sphere-api.featrix.com",
        )
        MockClass.return_value = client
        yield client


@pytest.fixture
def mock_fm():
    """A mock FoundationalModel, spec'd against the real class."""
    fm = spec_mock(
        FoundationalModel,
        id="fm-abc123",
        name="credit-model",
        status="done",
        dimensions=128,
        epochs=10,
        final_loss=0.042,
        compute_cluster="default",
    )
    fm.get_columns.return_value = ["checking_status", "duration", "credit_history", "class"]
    fm.get_model_card.return_value = {"name": "credit-model", "dimensions": 128, "epochs": 10}
    fm.list_predictors.return_value = []
    fm.refresh.return_value = {"job_plan": [], "jobs": {}}
    # A live HTTP context on the real object. Nothing in ffs reaches through it
    # today, but spec_mock sets every dataclass field to None and predict_health
    # walks session data, so keep it callable rather than None.
    fm._ctx = MagicMock()
    return fm


@pytest.fixture
def mock_predictor():
    """A mock Predictor, spec'd against the real class."""
    p = spec_mock(
        Predictor,
        id="pred-xyz789",
        session_id="fm-abc123",
        target_column="class",
        target_type="classifier",
        status="done",
        accuracy=0.85,
        auc=0.92,
        f1=0.83,
    )
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
    """A mock prediction result, spec'd against the real PredictionResult."""
    r = spec_mock(
        PredictionResult,
        predicted_class="good",
        prediction=None,
        confidence=0.92,
        probability=0.92,
        probabilities={"good": 0.92, "bad": 0.08},
        prediction_uuid="uuid-12345",
        feature_importance=None,
    )
    r.to_dict.return_value = {
        "predicted_class": "good",
        "confidence": 0.92,
        "probabilities": {"good": 0.92, "bad": 0.08},
    }
    return r


@pytest.fixture
def mock_endpoint():
    """A mock APIEndpoint, spec'd against the real class."""
    ep = spec_mock(
        APIEndpoint,
        id="ep-123",
        name="prod",
        predictor_id="pred-xyz789",
        session_id="fm-abc123",
        api_key="sk_endpoint_secret",
        description="production endpoint",
        url="https://sphere-api.featrix.com/e/prod",
        usage_count=42,
        last_used_at="2026-01-01T00:00:00Z",
    )
    ep.to_dict.return_value = {"id": "ep-123", "name": "prod", "usage_count": 42}
    ep.get_usage_stats.return_value = {"total_calls": 42, "last_24h": 3}
    ep.regenerate_api_key.return_value = "sk_new_rotated_key"
    ep.revoke_api_key.return_value = None
    ep.delete.return_value = None
    return ep


@pytest.fixture
def mock_event_group():
    """A mock EventGroup, spec'd against the real class."""
    g = spec_mock(
        EventGroup,
        event_group_id="11111111-1111-1111-1111-111111111111",
        event_count=1234,
        event_count_at_last_train=1000,
        extend_count=2,
        auto_retrain_enabled=True,
        last_session_id="fm-abc123",
        last_trained_at="2026-01-01T00:00:00Z",
        consecutive_failures=0,
        created_at="2025-12-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )
    g.to_dict.return_value = {
        "event_group_id": "11111111-1111-1111-1111-111111111111",
        "event_count": 1234,
    }
    return g


@pytest.fixture
def mock_network():
    """A mock PublishedPredictionNetwork, spec'd against the real class."""
    net = spec_mock(
        PublishedPredictionNetwork,
        name="carrier-qualification",
        org="acme",
        api_key=FAKE_API_KEY,
        base_url="https://sphere-api.featrix.com",
    )
    net.register.return_value = {"version": 3, "nodes": 2}
    net.get_spec.return_value = {
        "name": "carrier-qualification",
        "version": 3,
        "spec": {
            "nodes": [
                {"id": "is_company", "model": "is-company"},
                {"id": "company_type", "model": "company-type"},
            ],
            "edges": [
                {"from": "is_company", "to": "company_type", "when": {"op": "always"}},
            ],
        },
    }
    return net


@pytest.fixture
def mock_network_result():
    """A PredictionNetwork run result.

    Built from the real dataclasses rather than mocks: `network predict --json`
    calls `dataclasses.asdict()` on each trace entry, which raises TypeError on
    anything that isn't a genuine dataclass instance.
    """
    return PredictionNetworkResult(
        final={"label": "carrier", "confidence": 0.91},
        trace=[
            PredictionNetworkTraceEntry(
                node_id="is_company", model="is-company", label="yes",
                confidence=0.98, latency_ms=12.0,
            ),
            PredictionNetworkTraceEntry(
                node_id="company_type", model="company-type", label="carrier",
                confidence=0.91, latency_ms=18.0,
            ),
            PredictionNetworkTraceEntry(
                node_id="skipped_node", model="other", skipped=True,
                skip_reason="upstream said no",
            ),
            PredictionNetworkTraceEntry(
                node_id="broken_node", model="broken", error="model not published",
            ),
        ],
    )


@pytest.fixture
def wired_sphere(
    mock_sphere, mock_fm, mock_predictor, mock_prediction, mock_endpoint,
    mock_event_group, mock_network, mock_network_result,
):
    """A client with every accessor wired to a healthy, spec'd return value.

    Lets a test drive any command without restating the same plumbing, which is
    what makes the whole-tree smoke test in test_command_tree.py practical.
    """
    mock_fm.list_predictors.return_value = [mock_predictor]
    mock_fm.create_binary_classifier.return_value = mock_predictor
    mock_fm.create_regressor.return_value = mock_predictor
    mock_fm.extend.return_value = mock_fm
    mock_fm.encode.return_value = [0.1, 0.2, 0.3]
    mock_fm.publish.return_value = {"published": True, "name": "credit-model"}
    mock_fm.unpublish.return_value = {"published": False}
    mock_fm.deprecate.return_value = {"deprecated": True}
    mock_fm.cancel.return_value = {"cancelled": True}
    mock_fm.delete.return_value = {"deleted": True}
    mock_fm.foundation_predict.return_value = [mock_prediction]

    mock_predictor.predict.return_value = mock_prediction
    mock_predictor.batch_predict.return_value = [mock_prediction, mock_prediction]
    mock_predictor.cancel.return_value = {"cancelled": True}
    mock_predictor.create_api_endpoint.return_value = mock_endpoint

    mock_network.predict.return_value = mock_network_result

    mock_sphere.whoami.return_value = {
        "user": "testuser", "org": "Test Org", "org_slug": "test-org",
    }
    mock_sphere.health_check.return_value = {
        "status": "healthy", "version": "1.2.3",
        "nodes": {"gpu-1": {"status": "healthy", "version": "1.2.3"}},
    }
    mock_sphere.create_foundational_model.return_value = mock_fm
    mock_sphere.foundational_model.return_value = mock_fm
    mock_sphere.predictor.return_value = mock_predictor
    mock_sphere.api_endpoint.return_value = mock_endpoint
    mock_sphere.published_prediction_network.return_value = mock_network
    mock_sphere.event_group.return_value = mock_event_group
    mock_sphere.list_sessions.return_value = [mock_fm]
    mock_sphere.list_pending_jobs.return_value = [mock_fm]
    mock_sphere.list_event_groups.return_value = [mock_event_group]
    mock_sphere.list_prediction_networks.return_value = [
        {"name": "carrier-qualification", "version": 3, "updated_at": "2026-01-01"},
    ]
    return mock_sphere
