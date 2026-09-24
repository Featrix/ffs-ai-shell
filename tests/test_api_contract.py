"""Contract tests: what ffs calls vs. what featrixsphere actually provides.

Every other test in this suite runs against mocks. Mocks agree with whatever
ffs asks of them, so they can only prove ffs is internally consistent — never
that it matches the library it's actually calling. When featrixsphere renames a
method, changes a keyword, or drops a field, mock-based tests stay green and
the break surfaces as a customer bug report.

These tests are the missing half: they assert against the *installed*
featrixsphere, with no mocking at all. A failure here means ffs and the pinned
featrixsphere have drifted and some command is broken in the field.

Nothing here touches the network — it's all introspection.
"""
import ast
import dataclasses
import inspect
from pathlib import Path

import pytest

from featrixsphere.api import (
    APIEndpoint,
    EventGroup,
    FeatrixSphere,
    FoundationalModel,
    PredictionResult,
    Predictor,
    PublishedPredictionNetwork,
)

FFS_DIR = Path(__file__).parent.parent / "ffs"


def _init_assigned_attrs(cls):
    """Names assigned as `self.X = ...` in `cls.__init__`.

    FeatrixSphere isn't a dataclass and sets api_key/base_url in __init__, so
    they're invisible to both dir() and __annotations__ — without this, the
    surface check would report attributes that plainly do exist at runtime.
    """
    try:
        source = inspect.getsource(cls.__init__)
    except (OSError, TypeError):
        # Dataclass-generated __init__ has no retrievable source; fields cover it.
        return set()
    tree = ast.parse(inspect.cleandoc(source))
    names = set()
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                names.add(target.attr)
    return names


def attr_surface(cls):
    """Every name readable on an instance of `cls`.

    Three sources, because no single one is complete: `dir()` misses dataclass
    fields declared without a default and misses plain `self.X` assignments,
    while `__init__` scraping misses everything declared on the class.
    """
    names = set(dir(cls)) | set(getattr(cls, "__annotations__", {}) or {})
    if dataclasses.is_dataclass(cls):
        names |= {f.name for f in dataclasses.fields(cls)}
    names |= _init_assigned_attrs(cls)
    return names


def assert_kwargs_accepted(func, kwargs, label):
    """Assert `func` accepts every name in `kwargs`.

    A **kwargs catch-all absorbs unknown names at runtime rather than raising,
    so it counts as accepting them — the call won't blow up, even if the
    keyword is silently ignored.
    """
    sig = inspect.signature(func)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return
    unknown = sorted(k for k in kwargs if k not in sig.parameters)
    assert not unknown, (
        f"{label} passes keyword(s) {unknown} that "
        f"{func.__qualname__}{sig} does not accept — this command is broken."
    )


# ---------------------------------------------------------------------------
# Attribute surface: every attribute ffs reads must exist on the real class.
# ---------------------------------------------------------------------------

# Kept as explicit lists rather than scraped, so a failure names the command
# that breaks. `test_client_calls_are_declared` below guards against this list
# going stale as new code lands.
CLIENT_ATTRS = {
    "api_endpoint": "endpoint show/stats/regenerate-key/revoke-key/delete",
    "api_key": "network (_get_network)",
    "base_url": "network (_get_network)",
    "create_foundational_model": "models create",
    "event_group": "events show",
    "foundational_model": "most model/predictor/endpoint commands",
    "health_check": "server health",
    "list_event_groups": "events list",
    "list_pending_jobs": "jobs list, jobs cancel-queued",
    "list_prediction_networks": "network list",
    "list_sessions": "models list, models recent",
    "predictor": "predict (no --target-column)",
    "published_prediction_network": "network register/show/predict",
    "whoami": "whoami, network (_get_org)",
}

FM_ATTRS = {
    "cancel": "models cancel, jobs cancel-queued",
    "compute_cluster": "models show",
    "create_binary_classifier": "predictor create, train model",
    "create_regressor": "predictor create, train model",
    "delete": "models delete",
    "deprecate": "models deprecate",
    "dimensions": "models show/list/wait/recent",
    "encode": "models encode",
    "epochs": "models show/list/wait/recent",
    "extend": "models extend",
    "final_loss": "models show/list/wait",
    "foundation_predict": "models predict (no trained predictor / --foundation)",
    "get_columns": "models columns, models code",
    "get_model_card": "models card",
    "id": "nearly every model command",
    "list_predictors": "models show/code/predict, predictor *, endpoint create",
    "name": "models show/list/recent",
    "publish": "models publish",
    "refresh": "models show/jobs/wait, jobs cancel-queued",
    "status": "models show/list/wait/recent, jobs cancel-queued",
    "unpublish": "models unpublish",
}

PREDICTOR_ATTRS = {
    "accuracy": "predictor list/show, models show",
    "auc": "predictor show",
    "batch_predict": "predict --file, models predict --file",
    "cancel": "predictor cancel",
    "create_api_endpoint": "endpoint create",
    "f1": "predictor show",
    "id": "predictor list/show, models code",
    "predict": "predict, models predict",
    "session_id": "predictor show/create, train model",
    "status": "predictor list/show, models wait",
    "target_column": "predictor list/show, models predict, predict",
    "target_type": "predictor list/show, models show/code",
    "to_dict": "predictor list/show",
}

ENDPOINT_ATTRS = {
    "api_key": "endpoint create/show",
    "delete": "endpoint delete",
    "description": "endpoint create/show",
    "get_usage_stats": "endpoint stats",
    "id": "endpoint create/show/delete/regenerate-key/revoke-key",
    "last_used_at": "endpoint show",
    "name": "endpoint create/show",
    "predictor_id": "endpoint show",
    "regenerate_api_key": "endpoint regenerate-key",
    "revoke_api_key": "endpoint revoke-key",
    "session_id": "endpoint show",
    "to_dict": "endpoint create/show",
    "url": "endpoint create/show",
    "usage_count": "endpoint show",
}

EVENT_GROUP_ATTRS = {
    "auto_retrain_enabled": "events list/show",
    "consecutive_failures": "events show",
    "created_at": "events show",
    "event_count": "events show",
    "event_count_at_last_train": "events list/show",
    "event_group_id": "events list/show",
    "extend_count": "events show",
    "last_session_id": "events list/show",
    "last_trained_at": "events list/show",
    "to_dict": "events list/show",
    "updated_at": "events show",
}

NETWORK_ATTRS = {
    "get_spec": "network show",
    "predict": "network predict",
    "register": "network register",
}

PREDICTION_RESULT_ATTRS = {
    "confidence": "predict, models predict",
    "feature_importance": "predict --explain, models predict --explain",
    "predicted_class": "predict, models predict",
    "prediction": "predict, models predict",
    "prediction_uuid": "predict, models predict",
    "probabilities": "predict, models predict",
    "probability": "predict, models predict",
    "to_dict": "predict, models predict",
    # predict_health reads this with getattr(..., None) to explain *why* a
    # prediction came back empty. A rename wouldn't raise — the guard would just
    # silently stop reporting the server's reason — so pin the name here.
    "training_status": "predict / models predict (empty-result diagnosis)",
}


@pytest.mark.parametrize(
    "cls,attrs",
    [
        (FeatrixSphere, CLIENT_ATTRS),
        (FoundationalModel, FM_ATTRS),
        (Predictor, PREDICTOR_ATTRS),
        (APIEndpoint, ENDPOINT_ATTRS),
        (EventGroup, EVENT_GROUP_ATTRS),
        (PublishedPredictionNetwork, NETWORK_ATTRS),
        (PredictionResult, PREDICTION_RESULT_ATTRS),
    ],
    ids=lambda v: v.__name__ if inspect.isclass(v) else "",
)
def test_attributes_ffs_uses_exist(cls, attrs):
    """Every attribute ffs reads off a featrixsphere object must exist."""
    surface = attr_surface(cls)
    missing = {name: used_by for name, used_by in attrs.items() if name not in surface}
    assert not missing, (
        f"{cls.__name__} no longer provides: "
        + "; ".join(f"{n} (breaks: {by})" for n, by in sorted(missing.items()))
    )


def test_foundation_predict_accepts_a_list_of_records():
    """`models predict` without a trained predictor batches through this.

    model_cmd._predict_on_foundation hands it a list and normalises a single
    result back into a list, so both the parameter name and the ability to take
    many records matter.
    """
    sig = inspect.signature(FoundationalModel.foundation_predict)
    params = list(sig.parameters)
    assert params[:3] == ["self", "target_column", "records"], (
        f"FoundationalModel.foundation_predict signature changed to {sig} — "
        "model_cmd._predict_on_foundation calls it positionally."
    )


# ---------------------------------------------------------------------------
# Keyword arguments: every kwarg ffs passes must be accepted.
# ---------------------------------------------------------------------------

KWARG_CALLS = [
    (
        FeatrixSphere.create_foundational_model,
        {"name", "data_file", "ignore_columns", "epochs", "session_name_prefix"},
        "models create",
    ),
    (FeatrixSphere.list_sessions, {"name_prefix"}, "models list"),
    (FeatrixSphere.list_pending_jobs, {"name_prefix"}, "jobs list / cancel-queued"),
    (FeatrixSphere.list_event_groups, {"limit", "offset"}, "events list"),
    (
        FeatrixSphere.published_prediction_network,
        {"org", "name", "api_key", "base_url"},
        "network register/show/predict",
    ),
    (FoundationalModel.extend, {"new_data_file", "epochs"}, "models extend"),
    (FoundationalModel.encode, {"short"}, "models encode"),
    (
        FoundationalModel.publish,
        {"name", "max_wait_time", "poll_interval"},
        "models publish",
    ),
    (
        FoundationalModel.deprecate,
        {"warning_message", "expiration_date"},
        "models deprecate",
    ),
    (FoundationalModel.cancel, {"reason"}, "models cancel / jobs cancel-queued"),
    (
        FoundationalModel.create_binary_classifier,
        {"target_column", "name", "epochs", "labels_file"},
        "predictor create --type classifier / train model",
    ),
    (
        FoundationalModel.create_regressor,
        {"target_column", "name", "epochs", "labels_file"},
        "predictor create --type regressor / train model",
    ),
    (Predictor.predict, {"feature_importance"}, "predict / models predict"),
    (Predictor.cancel, {"reason"}, "predictor cancel"),
    (
        Predictor.create_api_endpoint,
        {"name", "api_key", "description"},
        "endpoint create",
    ),
]


@pytest.mark.parametrize(
    "func,kwargs,command", KWARG_CALLS, ids=[c[2] for c in KWARG_CALLS]
)
def test_kwargs_ffs_passes_are_accepted(func, kwargs, command):
    assert_kwargs_accepted(func, kwargs, f"`ffs {command}`")


def test_positional_api_endpoint_argument_order():
    """`endpoint show` calls api_endpoint(endpoint_id, session_id) positionally.

    Both are opaque ID strings, so swapping them server-side wouldn't raise —
    it would just 404 or, worse, read the wrong endpoint. Pin the order.
    """
    params = list(inspect.signature(FeatrixSphere.api_endpoint).parameters)
    assert params[:3] == ["self", "endpoint_id", "session_id"], (
        f"FeatrixSphere.api_endpoint parameter order changed to {params} — "
        "endpoint_cmd passes (endpoint_id, model_id) positionally."
    )


def test_predictor_lookup_takes_session_id_first():
    """`ffs predict MODEL_ID` passes a session ID as the first argument."""
    params = list(inspect.signature(FeatrixSphere.predictor).parameters)
    assert params[:2] == ["self", "session_id"], (
        f"FeatrixSphere.predictor parameter order changed to {params} — "
        "predict_cmd._get_predictor passes the model/session ID positionally."
    )


# ---------------------------------------------------------------------------
# Staleness guard: keep CLIENT_ATTRS honest as new code lands.
# ---------------------------------------------------------------------------


def _scraped_client_attrs():
    """AST-scrape every `state.client.<attr>` in ffs/*.py.

    `state.client` is a consistent enough idiom across the command modules to
    scrape reliably, which makes this a real check rather than a restatement of
    the hand-written list above.
    """
    found = {}
    for path in sorted(FFS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            inner = node.value
            if (
                isinstance(inner, ast.Attribute)
                and inner.attr == "client"
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "state"
            ):
                found.setdefault(node.attr, set()).add(path.name)
    return found


def test_every_client_call_in_the_source_exists_on_featrixsphere():
    """Catches new `state.client.*` calls that no other test covers yet."""
    surface = attr_surface(FeatrixSphere)
    scraped = _scraped_client_attrs()
    assert scraped, "scraper found no state.client.* calls — the idiom changed"
    missing = {a: sorted(f) for a, f in scraped.items() if a not in surface}
    assert not missing, (
        "ffs calls FeatrixSphere attributes that do not exist: "
        + "; ".join(f"{a} (in {', '.join(f)})" for a, f in sorted(missing.items()))
    )


def test_client_attrs_list_is_not_stale():
    """The curated CLIENT_ATTRS list should cover everything in the source."""
    scraped = _scraped_client_attrs()
    undeclared = {a: sorted(f) for a, f in scraped.items() if a not in CLIENT_ATTRS}
    assert not undeclared, (
        "new state.client.* call(s) not declared in CLIENT_ATTRS — add them so "
        "a future featrixsphere change names the broken command: "
        + "; ".join(f"{a} (in {', '.join(f)})" for a, f in sorted(undeclared.items()))
    )
