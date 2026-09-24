"""Checks that stop a prediction command from answering with nothing.

A predictor's status is really its training job's status, and "ready" there is
the queued state — the job hasn't started, let alone written a model. Ask such
a predictor for a prediction and the server replies with every result field
empty; the SDK parks the server's "still training" explanation on
``PredictionResult.training_status`` and then leaves it out of ``to_dict()``.
So without the checks here, ``ffs predict`` prints an empty table and ``--json``
prints an object of nulls — which reads as "the model has no opinion" rather
than "there is no model yet".
"""
import click

# Job statuses that mean "no servable model behind this predictor *yet*".
# "ready" is the queued state; "paused" is a live job that isn't advancing.
PENDING_STATUSES = frozenset({"ready", "queued", "pending", "training", "running", "paused"})

# ...and the ones that mean "and there never will be". "aborted" is what the
# server's watchdog writes when a training subprocess dies without reporting
# anything itself; downstream jobs of a failed one are marked "abandoned".
FAILED_STATUSES = frozenset({"error", "failed", "cancelled", "aborted", "abandoned"})

TERMINAL_STATUSES = frozenset({"done", "completed"}) | FAILED_STATUSES

# Statuses whose raw names read as better news than they are.
_STATUS_LABELS = {"ready": "queued", "completed": "done"}

# Any one of these carrying a value means the server actually predicted something.
_PREDICTION_FIELDS = ("prediction", "predicted_class", "probability", "confidence", "probabilities")


def display_status(status):
    """Render a raw job status for humans — "ready" means queued, not servable."""
    if not isinstance(status, str) or not status:
        return "—"
    return _STATUS_LABELS.get(status, status)


def not_ready_reason(predictor):
    """Why this predictor can't serve a prediction, or None if it might.

    A missing status counts as "might": list_predictors() leaves it unset once
    the training job's Redis record ages out, which happens long before the
    model it produced stops being servable.
    """
    status = getattr(predictor, "status", None)
    if not isinstance(status, str):
        return None
    if status in FAILED_STATUSES:
        return f"its training job {display_status(status)}"
    if status in PENDING_STATUSES:
        return f"it is still training (job status: {display_status(status)})"
    return None


def has_prediction(result):
    """True if `result` carries an actual predicted value."""
    for name in _PREDICTION_FIELDS:
        value = result.get(name) if isinstance(result, dict) else getattr(result, name, None)
        if value is not None and value != {} and value != []:
            return True
    return False


def training_status_detail(result):
    """The server's "not servable yet" payload for a prediction, or None.

    The SDK hangs this off the result and drops it from to_dict(), so nothing
    downstream sees it unless it's read here.
    """
    detail = getattr(result, "training_status", None)
    return detail if isinstance(detail, dict) else None


def format_training_status(detail):
    """One line describing how far along the predictor's training is."""
    message = detail.get("message") or f"Predictor not servable (status: {detail.get('status')})."
    progress = []
    epoch, total = detail.get("current_epoch"), detail.get("total_epochs")
    if epoch is not None and total:
        progress.append(f"epoch {epoch}/{total}")
    percent = detail.get("progress_percent")
    if percent:
        progress.append(f"{percent}% complete")
    return f"{message} [{', '.join(progress)}]" if progress else message


def _training_jobs(data):
    """The session's train_single_predictor jobs, plan entries merged with live records.

    Same two sources `ffs foundation jobs` reads: job_plan is persisted on the
    session, the jobs block is a live Redis view that can age out from under it.
    """
    if not isinstance(data, dict):
        return []
    jobs = data.get("jobs") if isinstance(data.get("jobs"), dict) else {}
    session = data.get("session") if isinstance(data.get("session"), dict) else {}
    plan = data.get("job_plan") or session.get("job_plan") or []

    found = []
    planned_ids = set()
    for entry in plan if isinstance(plan, list) else []:
        if not isinstance(entry, dict) or entry.get("job_type") != "train_single_predictor":
            continue
        job_id = entry.get("job_id")
        if job_id:
            planned_ids.add(job_id)
        live = jobs.get(job_id) if job_id else None
        live = live if isinstance(live, dict) else {}
        found.append((
            (entry.get("spec") or {}).get("target_column"),
            live.get("status") or entry.get("status"),
            live.get("error"),
        ))
    for job_id, job in jobs.items():
        if job_id in planned_ids or not isinstance(job, dict):
            continue
        if job.get("job_type") != "train_single_predictor":
            continue
        found.append((job.get("target_column"), job.get("status"), job.get("error")))
    return found


def session_diagnosis(state, session_id):
    """Lines naming the training job(s) behind a predictor, and the session status.

    This is what `ffs foundation jobs` prints, repeated at the point of failure
    so a dead prediction names the job that died instead of leaving the caller
    to go find it.
    """
    try:
        fm = state.client.foundational_model(session_id)
        data = fm.refresh()
    except Exception:
        return []

    lines = []
    for target_column, status, error in _training_jobs(data):
        target = f" ({target_column})" if target_column else ""
        lines.append(f"train_single_predictor{target}: {display_status(status)}")
        if error:
            lines.append(f"  error: {error}")

    session = data.get("session") if isinstance(data, dict) else None
    status = session.get("status") if isinstance(session, dict) else None
    status = status or getattr(fm, "status", None)
    if isinstance(status, str) and status:
        lines.append(f"Session status: {status}")
    return lines


def unusable_prediction_error(state, session_id, *, target_column=None, reason=None,
                              detail=None, hint=None):
    """A ClickException saying *why* there's no prediction, not just that there isn't one."""
    target = f" for '{target_column}'" if target_column else ""
    if not reason:
        # The server's own mid-training payload is a better headline than a
        # generic one; the detail line below still spells out how far it got.
        reason = "it is still training" if detail else "the predictor has no servable model"
    lines = [f"No prediction{target}: {reason}."]
    if detail:
        lines.append(f"  {format_training_status(detail)}")
    lines.extend(f"  {line}" for line in session_diagnosis(state, session_id))
    lines.append("")
    lines.append(f"Check training with: ffs foundation jobs {session_id}")
    if hint:
        lines.append(hint)
    return click.ClickException("\n".join(lines))


def require_predictions(state, session_id, results, target_column=None, hint=None):
    """Raise unless something in `results` is an actual prediction.

    An empty `results` counts as nothing predicted — the prediction endpoints
    answer a request against an unservable predictor with a well-formed 200
    whose every field is null, and that must not reach the user as an answer.
    """
    detail = None
    for result in results:
        detail = detail or training_status_detail(result)
        if has_prediction(result):
            return
    raise unusable_prediction_error(
        state, session_id, target_column=target_column, detail=detail, hint=hint,
    )
