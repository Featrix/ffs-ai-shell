"""Live tests: the real CLI, the real Sphere API, no mocks anywhere.

Why this file exists. Every other test in this suite talks to a mock, and a
mock agrees with whatever ffs asks of it. That is fine for checking ffs is
internally consistent and useless for checking it works. Two real breaks got
through the mocked suite while it was fully green:

  * `ffs predict` printed a JSON object of nulls — no error, exit 0 — for a
    predictor whose training job had aborted. A customer hit this.
  * `ffs foundation columns` called `fm.columns()`, but `columns` is a
    property on the real FoundationalModel, so it raised TypeError. Mocks
    are callable, so nothing failed.

Neither is detectable without a real server on the other end. So these tests
train a real (small) embedding space and a real predictor, then run `ffs` as a
subprocess: real argv, real HTTP, real exit codes, real stdout.

    FFS_LIVE=1 pytest tests/test_live_api.py -v

They cost compute and minutes, so they skip unless FFS_LIVE is set. To iterate
without retraining, point them at a model that already exists:

    FFS_LIVE=1 FFS_LIVE_MODEL_ID=<id> pytest tests/test_live_api.py -v

Sessions this file creates are deleted afterwards unless FFS_LIVE_KEEP=1.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
CREDIT_CSV = str(Path(__file__).parent / "credit-sklearn.csv")

# Training the ES and then a predictor on 1000 rows. Generous, because a
# queued job waits on org capacity before it even starts.
# Two separate budgets, because a flat wall-clock cap is wrong here: the first
# run of this file timed out at 45 minutes having spent 20 of them waiting in
# the org's job queue before training even started.
QUEUE_TIMEOUT = 3 * 60 * 60     # no jobs scheduled yet — waiting for a GPU slot
STALL_TIMEOUT = 30 * 60         # jobs exist but nothing has moved
POLL_SECONDS = 10

# Kept for the fixtures that still name them; both now mean "the queue budget".
ES_TIMEOUT = QUEUE_TIMEOUT
PREDICTOR_TIMEOUT = QUEUE_TIMEOUT

TERMINAL_SESSION_STATUSES = frozenset(
    {"done", "error", "failed", "cancelled", "aborted", "abandoned"}
)

# A real row from credit-sklearn.csv with the target column removed.
SAMPLE_RECORD = {
    "checking_status": "<0",
    "duration": 6.0,
    "credit_history": "critical/other existing credit",
    "purpose": "radio/tv",
    "credit_amount": 1169.0,
    "savings_status": "no known savings",
    "employment": ">=7",
    "installment_commitment": 4.0,
    "personal_status": "male single",
    "other_parties": "none",
    "residence_since": 4.0,
    "property_magnitude": "real estate",
    "age": 67.0,
    "other_payment_plans": "none",
    "housing": "own",
    "existing_credits": 2.0,
    "job": "skilled",
    "num_dependents": 1.0,
    "own_telephone": "yes",
    "foreign_worker": "yes",
}

pytestmark = pytest.mark.skipif(
    not os.getenv("FFS_LIVE"),
    reason="live API tests — set FFS_LIVE=1 (and have an API key) to run",
)


def ffs(*args, timeout=300):
    """Run the working copy of the CLI as a real subprocess.

    Deliberately not CliRunner: a subprocess exercises the console entry point,
    real argv parsing and a real exit code, which is what a customer and a CI
    script actually see.
    """
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    return subprocess.run(
        [sys.executable, "-c",
         "import sys; from ffs.cli import cli; sys.exit(cli())", *args],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=timeout,
    )


def ffs_ok(*args, timeout=300):
    """Run the CLI and fail the test with its output if it exits non-zero."""
    proc = ffs(*args, timeout=timeout)
    assert proc.returncode == 0, (
        f"`ffs {' '.join(args)}` exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    return proc


def ffs_json(*args, timeout=300):
    """Run the CLI in --json mode and parse what it prints."""
    proc = ffs_ok("--json", *args, timeout=timeout)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(f"`ffs --json {' '.join(args)}` printed unparseable output: "
                    f"{exc}\n--- stdout ---\n{proc.stdout}")


def wait_for_session(model_id, timeout=QUEUE_TIMEOUT):
    """Poll until the session is terminal; return its final status.

    Not a wall-clock cap. Sitting in the job queue is not a failure — it is the
    normal state of a busy org — so a session with no jobs scheduled gets
    `timeout` to reach a GPU. Once jobs exist, the budget becomes a stall
    detector that resets on any observable movement (session status, job
    status, reported progress) and only fires when nothing has changed for
    STALL_TIMEOUT. Same contract the SDK uses for publish.
    """
    started = time.time()
    last_movement = time.time()
    seen = None

    while True:
        status = ffs_json("foundation", "show", model_id).get("status")
        if status in TERMINAL_SESSION_STATUSES:
            return status

        jobs = ffs_json("foundation", "jobs", model_id).get("jobs") or []
        snapshot = json.dumps([status, jobs], sort_keys=True, default=str)
        if snapshot != seen:
            seen, last_movement = snapshot, time.time()

        if not jobs:
            if time.time() - started > timeout:
                pytest.fail(
                    f"{model_id} never got a job scheduled in {timeout}s "
                    f"(status {status!r}) — the org's queue is saturated"
                )
        elif time.time() - last_movement > STALL_TIMEOUT:
            pytest.fail(
                f"{model_id} made no progress for {STALL_TIMEOUT}s "
                f"(status {status!r})\n{ffs('foundation', 'jobs', model_id).stdout}"
            )
        time.sleep(POLL_SECONDS)


def predictor_status(model_id, target_column):
    """The live status of the predictor for `target_column`, or None."""
    for p in ffs_json("predictor", "list", model_id):
        if p.get("target_column") == target_column:
            return p.get("status")
    return None


@pytest.fixture(scope="session")
def api_key_present():
    proc = ffs("whoami")
    if proc.returncode != 0:
        pytest.skip(f"no working API key: {proc.stdout}{proc.stderr}")


@pytest.fixture(scope="session")
def live_model(api_key_present):
    """A real, trained embedding space — reused from FFS_LIVE_MODEL_ID if given."""
    existing = os.getenv("FFS_LIVE_MODEL_ID")
    if existing:
        # Still wait: a reused id may be mid-training (or queued behind the
        # org's job queue), and every test below assumes a finished ES.
        status = wait_for_session(existing, ES_TIMEOUT)
        assert status == "done", (
            f"{existing} finished as {status!r}:\n"
            + ffs("foundation", "jobs", existing).stdout
        )
        yield existing
        return

    created = ffs_json("foundation", "create",
                       "--name", "ffs-live-test",
                       "--data", CREDIT_CSV, timeout=ES_TIMEOUT)
    model_id = created["model_id"]
    try:
        status = wait_for_session(model_id, ES_TIMEOUT)
        assert status == "done", (
            f"ES training finished as {status!r}, not done:\n"
            f"{ffs('foundation', 'jobs', model_id).stdout}"
        )
        yield model_id
    finally:
        if not os.getenv("FFS_LIVE_KEEP"):
            ffs("foundation", "delete", model_id, "--yes")


@pytest.fixture(scope="session")
def predictor_run(live_model):
    """Train a real predictor, capturing what `ffs predict` did at each stage.

    The customer's report is a before/after: `predict` against a predictor that
    isn't servable yet, and `predict` once it is. Both are captured here, in
    order, so the assertions below don't depend on test execution order.
    """
    target = "class"
    if predictor_status(live_model, target) is None:
        ffs_ok("predictor", "create", live_model,
               "--target-column", target, "--type", "classifier")

    record = json.dumps(SAMPLE_RECORD)

    # Immediately after create: the window the customer was in. Whatever the
    # server says here, the CLI must not present it as a prediction.
    status_right_after_create = predictor_status(live_model, target)
    during = ffs("predict", live_model, record, "--target-column", target)
    during_json = ffs("--json", "predict", live_model, record, "--target-column", target)

    status = wait_for_session(live_model, PREDICTOR_TIMEOUT)
    final_predictor_status = predictor_status(live_model, target)

    after = ffs("predict", live_model, record, "--target-column", target)
    after_json = ffs("--json", "predict", live_model, record, "--target-column", target)

    return {
        "model_id": live_model,
        "target": target,
        "status_right_after_create": status_right_after_create,
        "during": during,
        "during_json": during_json,
        "session_status": status,
        "final_predictor_status": final_predictor_status,
        "after": after,
        "after_json": after_json,
    }


# ---------------------------------------------------------------------------
# The reported bug: a prediction with no model behind it
# ---------------------------------------------------------------------------


class TestUnservablePredictorIsNotSilent:
    def test_predict_during_training_is_never_a_silent_success(self, predictor_run):
        """Either a real prediction or a real error — never nulls with exit 0.

        This is the customer's bug stated as an assertion. Before the fix this
        exited 0 having printed an empty table.
        """
        proc = predictor_run["during"]
        if proc.returncode == 0:
            # A checkpoint already existed — fine, but it must be a real answer.
            assert proc.stdout.strip(), "exited 0 but printed nothing at all"
            assert "Predicted" in proc.stdout, (
                "exited 0 without a prediction — the silent-null bug:\n" + proc.stdout
            )
        else:
            combined = proc.stdout + proc.stderr
            assert "No prediction" in combined, (
                "failed without explaining why:\n" + combined
            )

    def test_json_mode_never_emits_an_all_null_prediction(self, predictor_run):
        """`--json` must not hand a script an object whose every field is null."""
        proc = predictor_run["during_json"]
        if proc.returncode != 0:
            return  # errored out loudly, which is the fix working
        data = json.loads(proc.stdout)
        answered = [data.get(f) for f in
                    ("prediction", "predicted_class", "probability",
                     "confidence", "probabilities")]
        assert any(v not in (None, {}, []) for v in answered), (
            "exit 0 with every prediction field null — exactly what the customer "
            f"reported:\n{json.dumps(data, indent=2)[:800]}"
        )

    def test_an_error_names_the_training_job(self, predictor_run):
        """A failure has to say which job is responsible, not just that it failed."""
        proc = predictor_run["during"]
        if proc.returncode == 0:
            pytest.skip("predictor was already servable; nothing to diagnose")
        combined = proc.stdout + proc.stderr
        assert "ffs foundation jobs" in combined, (
            "error gives the user no next step:\n" + combined
        )


class TestPredictorStatusIsHonest:
    def test_status_right_after_create_is_not_reported_as_servable(self, predictor_run):
        """A just-created predictor must not read as finished.

        The server reports JobStatus.READY here, which is the *queued* state.
        Surfacing that word to a user is what made the customer believe
        training had finished in under a minute.
        """
        status = predictor_run["status_right_after_create"]
        assert status != "done", (
            f"predictor reported {status!r} immediately after create"
        )

    def test_list_does_not_print_the_word_ready(self, live_model):
        """`ready` means queued; the human-readable table must not imply otherwise."""
        proc = ffs_ok("predictor", "list", live_model)
        assert " ready " not in proc.stdout, (
            "predictor list still shows the raw 'ready' status:\n" + proc.stdout
        )

    def test_training_finished_successfully(self, predictor_run):
        """If this fails, the predictor genuinely didn't train — a server problem."""
        assert predictor_run["final_predictor_status"] == "done", (
            f"predictor ended as {predictor_run['final_predictor_status']!r}; "
            f"session {predictor_run['session_status']!r}\n"
            + ffs("foundation", "jobs", predictor_run["model_id"]).stdout
        )


class TestPredictionsWorkOnceTrained:
    def test_predict_returns_a_real_class(self, predictor_run):
        proc = predictor_run["after"]
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "Predicted" in proc.stdout
        assert ("good" in proc.stdout or "bad" in proc.stdout), proc.stdout

    def test_json_predict_is_populated(self, predictor_run):
        proc = predictor_run["after_json"]
        assert proc.returncode == 0, proc.stdout + proc.stderr
        data = json.loads(proc.stdout)
        assert data.get("predicted_class") in ("good", "bad"), data
        assert data.get("confidence") is not None, data
        assert data.get("query_record"), "the input should be echoed back"

    def test_batch_predict_from_a_file(self, predictor_run, tmp_path):
        import pandas as pd
        df = pd.read_csv(CREDIT_CSV).drop(columns=["class"]).head(5)
        path = tmp_path / "batch.csv"
        df.to_csv(path, index=False)
        proc = ffs_ok("predict", predictor_run["model_id"], "--file", str(path),
                      "--target-column", predictor_run["target"])
        assert "5 predictions" in proc.stdout, proc.stdout

    def test_foundation_predict_uses_the_trained_predictor(self, predictor_run):
        proc = ffs_ok("foundation", "predict", predictor_run["model_id"],
                      predictor_run["target"], json.dumps(SAMPLE_RECORD))
        assert "trained predictor" in proc.stdout, proc.stdout

    def test_foundation_predict_with_the_foundation_flag(self, predictor_run):
        """--foundation must reach the probe endpoint, which honours target_column.

        The old code posted to the single-predictor endpoint, which ignores
        target_column entirely and serves whatever SP the session has.
        """
        proc = ffs_ok("foundation", "predict", predictor_run["model_id"],
                      predictor_run["target"], json.dumps(SAMPLE_RECORD),
                      "--foundation")
        assert "foundation" in proc.stdout
        assert ("good" in proc.stdout or "bad" in proc.stdout), proc.stdout

    def test_foundation_predict_on_an_untrained_column(self, predictor_run):
        """Any column can be predicted from the foundation, not just trained ones."""
        record = {k: v for k, v in SAMPLE_RECORD.items() if k != "purpose"}
        proc = ffs_ok("foundation", "predict", predictor_run["model_id"],
                      "purpose", json.dumps(record))
        assert "Predicted" in proc.stdout, proc.stdout

    def test_unknown_predictor_id_errors(self, predictor_run):
        """It must not quietly answer from some other model."""
        proc = ffs("foundation", "predict", predictor_run["model_id"],
                   predictor_run["target"], json.dumps(SAMPLE_RECORD),
                   "--predictor-id", "predictor-does-not-exist")
        assert proc.returncode != 0
        assert "predictor-does-not-exist" in proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# Commands whose breakage a mock physically cannot see
# ---------------------------------------------------------------------------


class TestCommandsAgainstTheRealSDK:
    """Every one of these calls a real SDK method against a real session.

    `test_columns` is the case in point: `FoundationalModel.columns` is a
    property, so calling it raises TypeError against the real object while a
    MagicMock happily returns another mock.
    """

    def test_columns(self, live_model):
        proc = ffs_ok("foundation", "columns", live_model)
        assert "checking_status" in proc.stdout, proc.stdout

    def test_columns_json(self, live_model):
        assert "class" in ffs_json("foundation", "columns", live_model)

    def test_show(self, live_model):
        data = ffs_json("foundation", "show", live_model)
        assert data["model_id"] == live_model
        assert data["status"] == "done"
        assert data["dimensions"], "a trained ES should report its dimensions"

    def test_jobs(self, live_model):
        proc = ffs_ok("foundation", "jobs", live_model)
        assert "train" in proc.stdout, proc.stdout

    def test_card(self, live_model):
        ffs_ok("foundation", "card", live_model)

    def test_code(self, live_model):
        proc = ffs_ok("foundation", "code", live_model)
        assert live_model in proc.stdout

    def test_encode(self, live_model):
        vector = ffs_json("foundation", "encode", live_model,
                          json.dumps(SAMPLE_RECORD))
        assert isinstance(vector, list) and vector, vector

    def test_wait_returns_immediately_when_done(self, predictor_run):
        proc = ffs_ok("foundation", "wait", predictor_run["model_id"])
        assert "Training complete" in proc.stdout, proc.stdout

    def test_predictor_show(self, predictor_run):
        proc = ffs_ok("predictor", "show", predictor_run["model_id"])
        assert predictor_run["target"] in proc.stdout

    def test_predictor_list_json(self, predictor_run):
        rows = ffs_json("predictor", "list", predictor_run["model_id"])
        assert any(r.get("target_column") == predictor_run["target"] for r in rows), rows

    def test_foundation_list_includes_this_model(self, live_model):
        ids = [m.get("id") for m in ffs_json("foundation", "list")]
        assert live_model in ids

    def test_server_health(self, api_key_present):
        data = ffs_json("server", "health")
        assert data.get("status"), data


# ---------------------------------------------------------------------------
# `ffs train model` — a predictor on an existing ES, in a brand-new session
# ---------------------------------------------------------------------------


class TestTrainModelAgainstTheRealAPI:
    """The one predictor-creating path the mocked suite can't prove.

    `train model` differs from `predictor create` in that it always forks a new
    session off the parent ES and passes the data file as `labels_file`. Both of
    those are server-side behaviours, so a mock signing off on the call says
    nothing about whether the parent survived or the labels were actually used.
    """

    @pytest.fixture(scope="class")
    def trained(self, live_model):
        created = ffs_json(
            "train", "model", live_model,
            "--target-column", "class",
            "--type", "classifier",
            "--data", CREDIT_CSV,
            timeout=PREDICTOR_TIMEOUT,
        )
        try:
            yield created
        finally:
            if not os.getenv("FFS_LIVE_KEEP") and created["session_id"] != live_model:
                ffs("foundation", "delete", created["session_id"], "--yes")

    def test_returns_a_predictor_and_a_session(self, trained):
        assert trained["predictor_id"]
        assert trained["session_id"]
        assert trained["type"] == "classifier"
        assert trained["target_column"] == "class"

    def test_creates_a_new_session_rather_than_reusing_the_parent(
        self, trained, live_model
    ):
        """The command's documented contract: it never modifies the parent ES."""
        assert trained["es_session_id"] == live_model
        assert trained["session_id"] != live_model

    def test_the_parent_es_is_left_alone(self, trained, live_model):
        parent = ffs_json("foundation", "show", live_model)
        assert parent["status"] == "done"

    def test_the_new_session_is_real_and_inspectable(self, trained):
        shown = ffs_json("foundation", "show", trained["session_id"])
        assert shown["model_id"] == trained["session_id"]

    def test_the_predictor_trains_to_completion(self, trained):
        status = wait_for_session(trained["session_id"], PREDICTOR_TIMEOUT)
        assert status == "done", (
            f"train model session finished as {status!r}:\n"
            + ffs("foundation", "jobs", trained["session_id"]).stdout
        )

    def test_the_trained_predictor_actually_predicts(self, trained):
        """The end of the line: a real prediction out of a `train model` run."""
        wait_for_session(trained["session_id"], PREDICTOR_TIMEOUT)
        result = ffs_json(
            "predict", trained["session_id"], json.dumps(SAMPLE_RECORD),
            "--target-column", "class",
        )
        assert result.get("predicted_class") is not None or \
            result.get("prediction") is not None, \
            f"train model produced a predictor that predicts nothing: {result}"

    def test_rejects_a_bad_target_column(self, live_model):
        """A column that isn't in the ES must fail loudly, not train on nothing."""
        proc = ffs(
            "train", "model", live_model,
            "--target-column", "no_such_column_xyz",
            "--type", "classifier",
            "--data", CREDIT_CSV,
        )
        assert proc.returncode != 0, (
            "training on a nonexistent target column reported success:\n"
            f"{proc.stdout}"
        )
