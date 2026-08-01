"""ffs model subcommands."""
import json
import time
from datetime import datetime, timezone

import click

from ffs.click_ext import DYMGroup
from ffs.client import pass_client, ClientState
from ffs.output import print_json, print_kv, print_list_table, console


@click.group(cls=DYMGroup)
def model():
    """Manage foundational models."""
    pass


@model.command()
@click.option("--name", required=True, help="Model name")
@click.option("--data", "data_file", required=True, type=click.Path(exists=True), help="CSV/Parquet/JSON file")
@click.option("--epochs", type=int, default=None, help="Training epochs (auto if omitted)")
@click.option("--ignore-columns", default=None, help="Comma-separated columns to ignore")
@pass_client
def create(state: ClientState, name, data_file, epochs, ignore_columns):
    """Create a new foundational model from data."""
    ignore = [c.strip() for c in ignore_columns.split(",")] if ignore_columns else None
    fm = state.client.create_foundational_model(
        name=name,
        data_file=data_file,
        ignore_columns=ignore,
        epochs=epochs,
        session_name_prefix=name,
    )
    if state.output_json:
        print_json({"model_id": fm.id, "status": fm.status})
    else:
        console.print(f"[green]Model created:[/green] {fm.id}")
        console.print(f"Status: {fm.status}")
        console.print(f"\nRun [bold]ffs model wait {fm.id}[/bold] to monitor training.")


@model.command("list")
@click.option("--prefix", default="", help="Filter by name prefix")
@click.option("--active", is_flag=True, help="Show only active (non-done) sessions")
@pass_client
def list_models(state: ClientState, prefix, active):
    """List models."""
    sessions = state.client.list_sessions(name_prefix=prefix)
    if active:
        sessions = [s for s in sessions if s.status not in ("done", None)]
    if state.output_json:
        rows = []
        for s in sessions:
            rows.append({
                "id": s.id,
                "name": s.name,
                "status": s.status,
                "dimensions": s.dimensions,
                "epochs": s.epochs,
                "final_loss": s.final_loss,
            })
        print_json(rows)
    elif not sessions:
        console.print("No models found.")
    else:
        rows = []
        for s in sessions:
            loss = f"{s.final_loss:.4f}" if s.final_loss else "—"
            rows.append({
                "ID": s.id,
                "Name": s.name or "—",
                "Status": s.status or "—",
                "Dims": str(s.dimensions) if s.dimensions else "—",
                "Epochs": str(s.epochs) if s.epochs else "—",
                "Loss": loss,
            })
        print_list_table(rows, ["ID", "Name", "Status", "Dims", "Epochs", "Loss"])


def _show_one_model(state, fm):
    """Show details for a single model, including predictors."""
    # Refresh to get full server data (dimensions, epochs, loss, etc.)
    server_data = fm.refresh()
    session = server_data.get("session", server_data)
    model_info = session.get("model_info", {})
    training_stats = session.get("training_stats", {})

    predictors = []
    try:
        predictors = fm.list_predictors()
    except Exception:
        pass

    # Extract parameter count from model_info if available
    num_params = model_info.get("num_parameters") or model_info.get("n_parameters") or model_info.get("total_parameters")
    num_columns = model_info.get("num_columns")

    if state.output_json:
        data = {
            "model_id": fm.id,
            "name": fm.name,
            "status": fm.status,
            "dimensions": fm.dimensions,
            "epochs": fm.epochs,
            "final_loss": fm.final_loss,
            "compute_cluster": fm.compute_cluster,
            "model_info": model_info or None,
            "training_stats": training_stats or None,
            "predictors": [
                {"id": p.id, "target": p.target_column, "type": p.target_type,
                 "status": p.status, "accuracy": p.accuracy}
                for p in predictors
            ],
        }
        return data

    kv = {
        "Model ID": fm.id,
        "Name": fm.name or "(unnamed)",
        "Status": fm.status,
        "Dimensions": fm.dimensions or "—",
        "Epochs": fm.epochs or "—",
        "Final Loss": f"{fm.final_loss:.4f}" if fm.final_loss else "—",
    }
    if num_params:
        kv["Parameters"] = f"{num_params:,}" if isinstance(num_params, int) else str(num_params)
    if num_columns:
        kv["Columns"] = str(num_columns)
    kv["Cluster"] = fm.compute_cluster or "—"

    print_kv(kv)
    if predictors:
        console.print(f"\n  [bold]Predictors:[/bold]")
        for p in predictors:
            acc = f"  acc={p.accuracy:.3f}" if p.accuracy else ""
            console.print(f"    {p.target_column} ({p.target_type}) — {p.status or '?'}{acc}")
    return None


@model.command()
@click.argument("model_id", required=False)
@pass_client
def show(state: ClientState, model_id):
    """Show model details. Without MODEL_ID, shows all models."""
    if model_id:
        fm = state.client.foundational_model(model_id)
        data = _show_one_model(state, fm)
        if data:
            print_json(data)
        return

    # Show all
    sessions = state.client.list_sessions()
    if not sessions:
        console.print("No models found.")
        return

    if state.output_json:
        out = []
        for s in sessions:
            fm = state.client.foundational_model(s.id)
            out.append(_show_one_model(state, fm))
        print_json(out)
    else:
        for i, s in enumerate(sessions):
            if i > 0:
                console.print()
            fm = state.client.foundational_model(s.id)
            _show_one_model(state, fm)


@model.command()
@click.argument("model_id")
@pass_client
def columns(state: ClientState, model_id):
    """Show columns in the model's embedding space."""
    fm = state.client.foundational_model(model_id)
    cols = fm.get_columns()
    if state.output_json:
        print_json(cols)
    else:
        for col in cols:
            console.print(col)


_MODEL_CARD_HTML = """\
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Model Card</title></head>
<body>
<div id="model-card"></div>
<script src="https://bits.featrix.com/js/featrix-modelcard/model-card.js"></script>
<script>
var data = %s;
document.getElementById('model-card').innerHTML = FeatrixModelCard.renderHTML(data);
FeatrixModelCard.attachEventListeners();
</script>
</body></html>
"""


@model.command()
@click.argument("model_id")
@click.option("--url", "show_url", is_flag=True, help="Print the model card API URL")
@click.option("--open", "open_browser", is_flag=True, help="Render HTML and open in browser")
@click.option("--save", "save_path", default=None, type=click.Path(), help="Save rendered HTML to file")
@pass_client
def card(state: ClientState, model_id, show_url, open_browser, save_path):
    """Show the model card."""
    if show_url:
        url = f"{state.server}/compute/session/{model_id}/model_card"
        click.echo(url)
        return

    fm = state.client.foundational_model(model_id)
    card_data = fm.get_model_card()

    if open_browser or save_path:
        html = _MODEL_CARD_HTML % json.dumps(card_data, default=str)
        if save_path:
            with open(save_path, "w") as f:
                f.write(html)
            console.print(f"[green]Saved:[/green] {save_path}")
        if open_browser:
            import tempfile
            import webbrowser
            tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w")
            tmp.write(html)
            tmp.close()
            webbrowser.open(f"file://{tmp.name}")
            console.print(f"[green]Opened in browser.[/green]")
        return

    print_json(card_data)


def _format_duration(seconds: int) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _format_job_line(j: dict, jtype: str, now) -> str:
    """Format a single job into a status line."""
    status = j.get("status", "?")
    progress = j.get("progress", 0)
    created = j.get("created_at", "")

    age = ""
    if created:
        try:
            created_dt = datetime.fromisoformat(created) if isinstance(created, str) else created
            delta = int((now - created_dt).total_seconds())
            age = f" ({_format_duration(delta)})"
        except (ValueError, TypeError):
            pass

    finished = j.get("finished_at", "")
    duration = ""
    if finished and created:
        try:
            finished_dt = datetime.fromisoformat(finished) if isinstance(finished, str) else finished
            created_dt = datetime.fromisoformat(created) if isinstance(created, str) else created
            dur = int((finished_dt - created_dt).total_seconds())
            duration = f" ({_format_duration(dur)})"
        except (ValueError, TypeError):
            pass

    queue = j.get("queue", "")
    queue_str = f" [{queue}]" if queue and status != "done" else ""

    if status == "done":
        return f"  [green]done[/green]  {jtype}{duration}"
    elif status == "running" and progress:
        pct = progress * 100 if progress <= 1 else progress
        return f"  [yellow]running {pct:.1f}%[/yellow]  {jtype}{queue_str}{age}"
    elif status == "running":
        return f"  [yellow]running[/yellow]  {jtype}{queue_str}{age}"
    else:
        return f"  [dim]{status}[/dim]  {jtype}{queue_str}{age}"


def _job_status_lines(server_data: dict) -> list[str]:
    """Build status lines from server response dict (from fm.refresh())."""
    now = datetime.now(timezone.utc)
    lines = []
    job_plan = server_data.get("job_plan", [])
    jobs = server_data.get("jobs", {})

    # Track which job IDs are covered by the plan
    planned_ids = set()

    for job in job_plan:
        jtype = job.get("job_type", "?")
        jid = job.get("job_id")
        if jid:
            planned_ids.add(jid)
        if jid and jid in jobs:
            lines.append(_format_job_line(jobs[jid], jtype, now))
        else:
            lines.append(f"  [dim]pending[/dim]  {jtype}")

    # Show jobs that exist but aren't in the plan yet
    for jid, j in jobs.items():
        if jid not in planned_ids:
            jtype = j.get("job_type", "?")
            lines.append(_format_job_line(j, jtype, now))

    return lines


@model.command()
@click.argument("model_id")
@pass_client
def jobs(state: ClientState, model_id):
    """Show current job status for a model (one-shot snapshot)."""
    fm = state.client.foundational_model(model_id)
    data = fm.refresh()

    if state.output_json:
        job_plan = data.get("job_plan", [])
        job_map = data.get("jobs", {})
        out = []
        for entry in job_plan:
            jid = entry.get("job_id")
            j = job_map.get(jid, {}) if jid else {}
            out.append({
                "job_type": entry.get("job_type"),
                "job_id": jid,
                "status": j.get("status"),
                "progress": j.get("progress"),
                "queue": j.get("queue"),
                "created_at": j.get("created_at"),
                "finished_at": j.get("finished_at"),
                "error": j.get("error"),
            })
        for jid, j in job_map.items():
            if jid not in {e.get("job_id") for e in job_plan}:
                out.append({
                    "job_type": j.get("job_type"),
                    "job_id": jid,
                    "status": j.get("status"),
                    "progress": j.get("progress"),
                    "queue": j.get("queue"),
                    "created_at": j.get("created_at"),
                    "finished_at": j.get("finished_at"),
                    "error": j.get("error"),
                })
        print_json({"model_id": model_id, "status": fm.status, "jobs": out})
        return

    status_lines = _job_status_lines(data)
    console.print(f"[bold]{model_id}[/bold]  [dim]{fm.status or '?'}[/dim]")
    if status_lines:
        for line in status_lines:
            console.print(line)
    else:
        session = data.get("session", data)
        sess_status = session.get("status", fm.status or "unknown")
        console.print(f"  [dim]Session status: {sess_status} — no jobs scheduled yet[/dim]")

    # Show any errors
    for j in data.get("jobs", {}).values():
        if j.get("error"):
            console.print(f"  [red]Error:[/red] {j.get('job_type', '?')}: {j.get('error')}")


_TERMINAL_PREDICTOR_STATUSES = {"done", "error", "failed", "cancelled"}


def _pending_predictors(fm):
    """Predictors attached to this model that haven't reached a terminal status.

    fm.status reflects the ES/session lifecycle, which can hit "done" while an
    attached SP predictor is still training — checked separately here so `wait`
    doesn't report success before the predictor is actually servable.
    """
    try:
        predictors = fm.list_predictors()
    except Exception:
        return []
    return [p for p in predictors if p.status not in _TERMINAL_PREDICTOR_STATUSES]


@model.command()
@click.argument("model_id")
@click.option("--poll-interval", type=int, default=10, help="Seconds between checks")
@click.option("--timeout", type=int, default=3600, help="Max wait time in seconds")
@pass_client
def wait(state: ClientState, model_id, poll_interval, timeout):
    """Wait for model training to complete."""
    fm = state.client.foundational_model(model_id)
    start = time.time()
    first = True
    prev_lines = 1
    while True:
        data = fm.refresh()
        elapsed = int(time.time() - start)

        pending_predictors = _pending_predictors(fm) if fm.status == "done" else []

        if fm.status == "done" and not pending_predictors:
            console.print(f"\n[green]Training complete.[/green]")
            if not state.quiet:
                print_kv({
                    "Model ID": fm.id,
                    "Status": fm.status,
                    "Dimensions": fm.dimensions or "—",
                    "Epochs": fm.epochs or "—",
                    "Final Loss": fm.final_loss or "—",
                })
            return

        if fm.status in ("error", "failed"):
            console.print(f"\n[red]Training failed.[/red]")
            jobs = data.get("jobs", {})
            for j in jobs.values():
                if j.get("status") in ("error", "failed"):
                    console.print(f"  {j.get('job_type', '?')}: {j.get('error', 'unknown error')}")
            raise SystemExit(1)

        if elapsed > timeout:
            console.print(f"\n[red]Timeout after {timeout}s. Status: {fm.status}[/red]")
            raise SystemExit(1)

        # Build output lines
        job_plan = data.get("job_plan", [])
        status_lines = _job_status_lines(data)
        extra_lines = []

        if not job_plan and not status_lines:
            # No jobs scheduled yet — show what we know
            session = data.get("session", data)
            sess_status = session.get("status", fm.status or "unknown")
            extra_lines.append(f"  [dim]Session status: {sess_status}[/dim]")
            if sess_status in ("new", "uploading", "uploaded", "queued"):
                extra_lines.append(f"  [dim]Waiting for jobs to be scheduled...[/dim]")

        if pending_predictors:
            names = ", ".join(p.target_column or p.id for p in pending_predictors)
            extra_lines.append(f"  [dim]ES done — waiting for predictor(s) to finish: {names}[/dim]")

        # Clear screen and redraw
        total_lines = len(status_lines) + len(extra_lines) + 1
        if not first:
            click.echo(f"\033[{prev_lines}A\033[J", nl=False)
        first = False
        prev_lines = total_lines

        console.print(f"[bold]{model_id}[/bold]  [dim]{fm.status or '?'}[/dim]  ({_format_duration(elapsed)})")
        for line in status_lines:
            console.print(line)
        for line in extra_lines:
            console.print(line)

        time.sleep(poll_interval)



@model.command()
@click.option("--limit", type=int, default=10, help="Max sessions to show")
@pass_client
def recent(state: ClientState, limit):
    """Show recent sessions with job status."""
    sessions = state.client.list_sessions()
    # Sort: active first, then by status
    status_order = {"running": 0, "training": 0, "queued": 1, "new": 1,
                    "uploading": 2, "uploaded": 2, "error": 3, "failed": 3, "done": 4}
    sessions.sort(key=lambda s: status_order.get(s.status or "", 5))
    sessions = sessions[:limit]

    if state.output_json:
        out = []
        for s in sessions:
            out.append({
                "id": s.id, "name": s.name, "status": s.status,
                "dimensions": s.dimensions, "epochs": s.epochs,
            })
        print_json(out)
        return

    if not sessions:
        console.print("No sessions found.")
        return

    for s in sessions:
        name_str = f"  [dim]{s.name}[/dim]" if s.name else ""
        status = s.status or "?"
        if status in ("done",):
            status_fmt = f"[green]{status}[/green]"
        elif status in ("running", "training"):
            status_fmt = f"[yellow]{status}[/yellow]"
        elif status in ("error", "failed"):
            status_fmt = f"[red]{status}[/red]"
        else:
            status_fmt = f"[dim]{status}[/dim]"

        dims = f"  {s.dimensions}d" if s.dimensions else ""
        console.print(f"{status_fmt}  {s.id}{name_str}{dims}")


def _predict_on_es(fm, target_column, record):
    """Predict using the ES's built-in predictor for any column."""
    payload = {
        "query_record": record,
        "target_column": target_column,
    }
    return fm._ctx.post_json(f"/session/{fm.id}/predict", data=payload)


def _find_trained_predictor(fm, target_column, predictor_id=None):
    """Find a trained SP predictor, or None if not found."""
    try:
        predictors = fm.list_predictors()
    except Exception:
        return None
    if predictor_id:
        for p in predictors:
            if p.id == predictor_id:
                return p
        return None
    for p in predictors:
        if p.target_column == target_column:
            return p
    return None


def _format_prediction_data(result):
    """Format prediction result dict or object into display dict."""
    # Handle both raw API dicts and PredictionResult objects
    if isinstance(result, dict):
        data = {}
        pred = result.get("prediction") or result.get("predicted_class")
        if pred is not None:
            data["Predicted"] = str(pred)
        conf = result.get("confidence")
        if conf is not None:
            data["Confidence"] = f"{conf:.4f}"
        prob = result.get("probability")
        if prob is not None:
            data["Probability"] = f"{prob:.4f}"
        probs = result.get("probabilities")
        if probs:
            data["Distribution"] = "  ".join(f"{k}: {v:.3f}" for k, v in probs.items())
        uuid = result.get("prediction_uuid")
        if uuid:
            data["Prediction UUID"] = uuid
        return data

    data = {}
    if result.predicted_class is not None:
        data["Predicted"] = result.predicted_class
    elif hasattr(result, "prediction") and result.prediction is not None:
        data["Predicted"] = str(result.prediction)
    if result.confidence is not None:
        data["Confidence"] = f"{result.confidence:.4f}"
    if hasattr(result, "probability") and result.probability is not None:
        data["Probability"] = f"{result.probability:.4f}"
    if hasattr(result, "probabilities") and result.probabilities:
        data["Distribution"] = "  ".join(f"{k}: {v:.3f}" for k, v in result.probabilities.items())
    if hasattr(result, "prediction_uuid") and result.prediction_uuid:
        data["Prediction UUID"] = result.prediction_uuid
    return data


@model.command("predict")
@click.argument("model_id")
@click.argument("target_column")
@click.argument("record_json", required=False)
@click.option("--predictor-id", default=None, help="Use a specific trained predictor by ID")
@click.option("--file", "data_file", type=click.Path(exists=True),
              help="Batch predict from file (CSV, JSON, Parquet)")
@click.option("--explain", is_flag=True, help="Include feature importance (trained predictors only)")
@pass_client
def model_predict(state: ClientState, model_id, target_column, record_json,
                  predictor_id, data_file, explain):
    """Predict a column using the ES built-in predictor or a trained SP.

    Every column in an ES can be predicted directly. If a trained predictor
    exists for the target column, it will be used automatically.

    \b
    Single:  ffs models predict ES_ID melody_t3 '{"melody_t0": 60, "melody_t1": 62}'
    Batch:   ffs models predict ES_ID churned --file data.csv
    """
    if not record_json and not data_file:
        raise click.ClickException("Provide a JSON record or --file")

    fm = state.client.foundational_model(model_id)

    # Check for a trained SP predictor first; fall back to ES built-in
    trained = _find_trained_predictor(fm, target_column, predictor_id)

    if data_file:
        import pandas as pd
        lower = data_file.lower()
        if lower.endswith(".csv"):
            df = pd.read_csv(data_file)
        elif lower.endswith(".json"):
            df = pd.read_json(data_file)
        elif lower.endswith(".parquet"):
            df = pd.read_parquet(data_file)
        else:
            raise click.ClickException(f"Unsupported file format: {data_file}")

        if trained:
            results = trained.batch_predict(df)
            if state.output_json:
                print_json([r.to_dict() for r in results])
            else:
                console.print(f"[green]{len(results)} predictions[/green] (trained predictor)\n")
                rows = []
                for i, r in enumerate(results):
                    row = {"#": str(i + 1)}
                    d = _format_prediction_data(r)
                    row.update({k: str(v) for k, v in d.items() if k != "Prediction UUID"})
                    rows.append(row)
                if rows:
                    print_list_table(rows, list(rows[0].keys()))
        else:
            records = df.to_dict(orient="records")
            results = []
            for rec in records:
                results.append(_predict_on_es(fm, target_column, rec))
            if state.output_json:
                print_json(results)
            else:
                console.print(f"[green]{len(results)} predictions[/green] (ES built-in)\n")
                rows = []
                for i, r in enumerate(results):
                    row = {"#": str(i + 1)}
                    d = _format_prediction_data(r)
                    row.update({k: str(v) for k, v in d.items() if k != "Prediction UUID"})
                    rows.append(row)
                if rows:
                    print_list_table(rows, list(rows[0].keys()))
        return

    record = json.loads(record_json)

    if trained:
        result = trained.predict(record, feature_importance=explain)
        if state.output_json:
            print_json(result.to_dict())
        else:
            data = _format_prediction_data(result)
            if explain and hasattr(result, "feature_importance") and result.feature_importance:
                data["Feature Importance"] = ""
                print_kv(data, title="Prediction (trained)")
                for col, score in result.feature_importance.items():
                    console.print(f"  {col}: {score:.4f}")
            else:
                print_kv(data, title="Prediction (trained)")
    else:
        result = _predict_on_es(fm, target_column, record)
        if state.output_json:
            print_json(result)
        else:
            data = _format_prediction_data(result)
            print_kv(data, title="Prediction (ES built-in)")


@model.command()
@click.argument("model_id")
@click.option("--data", "data_file", required=True, type=click.Path(exists=True), help="New data file")
@click.option("--epochs", type=int, default=None, help="Additional epochs")
@pass_client
def extend(state: ClientState, model_id, data_file, epochs):
    """Extend a model with new data."""
    fm = state.client.foundational_model(model_id)
    kwargs = {}
    if epochs:
        kwargs["epochs"] = epochs
    new_fm = fm.extend(new_data_file=data_file, **kwargs)
    if state.output_json:
        print_json({"model_id": new_fm.id, "parent_model_id": model_id, "status": new_fm.status})
    else:
        console.print(f"[green]Extended model created:[/green] {new_fm.id}")
        console.print(f"Run [bold]ffs model wait {new_fm.id}[/bold] to monitor.")



@model.command()
@click.argument("model_id")
@click.argument("record_json")
@click.option("--short", is_flag=True, help="Return 3D short embedding for visualization")
@pass_client
def encode(state: ClientState, model_id, record_json, short):
    """Encode a record into the embedding space."""
    record = json.loads(record_json)
    fm = state.client.foundational_model(model_id)
    vectors = fm.encode(record, short=short)
    print_json(vectors)


@model.command()
@click.argument("model_id")
@click.option("--name", default=None, help="Published name (defaults to the model's own name)")
@click.option("--max-wait-time", type=int, default=600, help="Max seconds to wait for publish to complete")
@click.option("--poll-interval", type=int, default=5, help="Seconds between status polls")
@pass_client
def publish(state: ClientState, model_id, name, max_wait_time, poll_interval):
    """Publish a model.

    The org is derived from your API key — there is no --org option.
    """
    fm = state.client.foundational_model(model_id)
    result = fm.publish(name=name, max_wait_time=max_wait_time, poll_interval=poll_interval)
    if state.output_json:
        print_json(result)
    else:
        console.print(f"[green]Published:[/green] {result.get('published_path', model_id)}")


@model.command()
@click.argument("model_id")
@pass_client
def unpublish(state: ClientState, model_id):
    """Unpublish a model."""
    fm = state.client.foundational_model(model_id)
    result = fm.unpublish()
    if state.output_json:
        print_json(result)
    else:
        console.print(f"[green]Unpublished:[/green] {model_id}")


@model.command()
@click.argument("model_id")
@click.option("--message", required=True, help="Deprecation warning message")
@click.option("--expires", required=True, help="Expiration date (ISO format)")
@pass_client
def deprecate(state: ClientState, model_id, message, expires):
    """Deprecate a model with a warning and expiration date."""
    fm = state.client.foundational_model(model_id)
    result = fm.deprecate(warning_message=message, expiration_date=expires)
    if state.output_json:
        print_json(result)
    else:
        console.print(f"[yellow]Deprecated:[/yellow] {model_id}")
        console.print(f"Expires: {expires}")


@model.command()
@click.argument("model_id")
@click.option("--reason", default=None, help="Reason for cancellation (stored for audit)")
@click.confirmation_option(prompt="Are you sure you want to cancel training for this model?")
@pass_client
def cancel(state: ClientState, model_id, reason):
    """Cancel training for a foundation model.

    Cancels whatever job is currently active — queued or running — not just
    queued jobs. If it's already training, cancellation is cooperative (the
    training loop notices at its next checkpoint, not instantly).
    """
    fm = state.client.foundational_model(model_id)
    result = fm.cancel(reason=reason)
    if state.output_json:
        print_json(result)
    else:
        console.print(f"[yellow]Cancelled:[/yellow] {model_id}")
        if reason:
            console.print(f"Reason: {reason}")


@model.command()
@click.argument("model_id")
@click.confirmation_option(prompt="Are you sure you want to delete this model?")
@pass_client
def delete(state: ClientState, model_id):
    """Delete a model."""
    fm = state.client.foundational_model(model_id)
    result = fm.delete()
    if state.output_json:
        print_json(result)
    else:
        console.print(f"[red]Marked for deletion:[/red] {model_id}")


def _python_snippet(session_id, columns, predictors, server):
    """Generate a Python code snippet with real values."""
    sample = {c: f"..." for c in columns[:6]}
    sample_str = json.dumps(sample, indent=4)

    lines = [
        'from featrixsphere import FeatrixSphere',
        '',
        f'featrix = FeatrixSphere(api_key="YOUR_API_KEY", base_url="{server}")',
        f'fm = featrix.foundational_model("{session_id}")',
        '',
        '# Columns in this model:',
        f'# {", ".join(columns)}' if columns else '# (no columns found)',
        '',
        f'record = {sample_str}',
    ]

    if predictors:
        for p in predictors:
            lines.append('')
            lines.append(f'# Predict: {p.target_column} ({p.target_type})')
            lines.append(f'result = fm.predict("{p.target_column}", record)')
            lines.append('print(result.predicted_class, result.confidence)')
    else:
        lines.append('')
        lines.append('# Encode into embedding space')
        lines.append('vectors = fm.encode(record)')
        lines.append('print(vectors)')

    return '\n'.join(lines)


def _typescript_snippet(session_id, columns, predictors, server):
    """Generate a TypeScript/fetch code snippet with real values."""
    sample = {c: "..." for c in columns[:6]}
    sample_str = json.dumps(sample, indent=2)

    lines = [
        f'const API_KEY = "YOUR_API_KEY";',
        f'const BASE_URL = "{server}";',
        f'const SESSION_ID = "{session_id}";',
        '',
        f'const record = {sample_str};',
    ]

    if predictors:
        for p in predictors:
            lines.append('')
            lines.append(f'// Predict: {p.target_column} ({p.target_type})')
            lines.append(f'const response = await fetch(')
            lines.append(f'  `${{BASE_URL}}/session/${{SESSION_ID}}/predict`,')
            lines.append(f'  {{')
            lines.append(f'    method: "POST",')
            lines.append(f'    headers: {{')
            lines.append(f'      "Content-Type": "application/json",')
            lines.append(f'      "X-API-Key": API_KEY,')
            lines.append(f'    }},')
            lines.append(f'    body: JSON.stringify({{')
            lines.append(f'      query_record: record,')
            lines.append(f'      predictor_id: "{p.id}",')
            lines.append(f'    }}),')
            lines.append(f'  }}')
            lines.append(f');')
            lines.append(f'const result = await response.json();')
            lines.append(f'console.log(result.prediction, result.confidence);')
    else:
        lines.append('')
        lines.append('// Encode into embedding space')
        lines.append(f'const response = await fetch(')
        lines.append(f'  `${{BASE_URL}}/compute/session/${{SESSION_ID}}/encode`,')
        lines.append(f'  {{')
        lines.append(f'    method: "POST",')
        lines.append(f'    headers: {{')
        lines.append(f'      "Content-Type": "application/json",')
        lines.append(f'      "X-API-Key": API_KEY,')
        lines.append(f'    }},')
        lines.append(f'    body: JSON.stringify({{ record }}),')
        lines.append(f'  }}')
        lines.append(f');')
        lines.append(f'const vectors = await response.json();')
        lines.append(f'console.log(vectors);')

    return '\n'.join(lines)


@model.command()
@click.argument("model_id")
@click.option("--typescript", "lang", flag_value="typescript", help="Generate TypeScript snippet")
@click.option("--python", "lang", flag_value="python", default=True, help="Generate Python snippet (default)")
@pass_client
def code(state: ClientState, model_id, lang):
    """Generate ready-to-use code for this model.

    \b
    Examples:
      ffs models code SESSION_ID
      ffs models code SESSION_ID --typescript
    """
    fm = state.client.foundational_model(model_id)

    try:
        columns = fm.get_columns()
    except Exception:
        columns = []

    try:
        predictors = fm.list_predictors()
    except Exception:
        predictors = []

    if lang == "typescript":
        snippet = _typescript_snippet(model_id, columns, predictors, state.server)
        console.print(f"\n[bold]TypeScript[/bold] — {model_id}\n")
    else:
        snippet = _python_snippet(model_id, columns, predictors, state.server)
        console.print(f"\n[bold]Python[/bold] — {model_id}\n")

    click.echo(snippet)
    console.print()
