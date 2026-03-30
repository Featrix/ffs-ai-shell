"""ffs model subcommands."""
import json
import time
from datetime import datetime, timezone

import click

from ffs.client import pass_client, ClientState
from ffs.output import print_json, print_kv, print_list_table, console


@click.group()
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


@model.command()
@click.argument("model_id")
@pass_client
def show(state: ClientState, model_id):
    """Show model details."""
    fm = state.client.foundational_model(model_id)
    data = {
        "model_id": fm.id,
        "name": fm.name,
        "status": fm.status,
        "dimensions": fm.dimensions,
        "epochs": fm.epochs,
        "final_loss": fm.final_loss,
        "compute_cluster": fm.compute_cluster,
    }
    if state.output_json:
        print_json(data)
    else:
        print_kv({
            "Model ID": fm.id,
            "Name": fm.name or "(unnamed)",
            "Status": fm.status,
            "Dimensions": fm.dimensions or "—",
            "Epochs": fm.epochs or "—",
            "Final Loss": fm.final_loss or "—",
            "Cluster": fm.compute_cluster or "—",
        })


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

        if fm.status == "done":
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


@model.command("predict")
@click.argument("model_id")
@click.argument("target_column")
@click.argument("record_json", required=False)
@click.option("--model-type", "model_type", default=None,
              type=click.Choice(["sp", "foundation", "foundation+xgboost", "foundation+linear"]),
              help="Force prediction model type (default: auto-route)")
@click.option("--predictor-id", default=None, help="Use a specific predictor by ID")
@click.option("--file", "data_file", type=click.Path(exists=True),
              help="Batch predict from file (CSV, JSON, Parquet)")
@click.option("--explain", is_flag=True, help="Include feature importance")
@pass_client
def model_predict(state: ClientState, model_id, target_column, record_json,
                  model_type, predictor_id, data_file, explain):
    """Predict a column on a foundation model.

    \b
    Auto-routes to SP if trained, otherwise uses foundation+xgboost.

    \b
    Single:  ffs foundation predict MODEL_ID churned '{"age": 35, "income": 50000}'
    Batch:   ffs foundation predict MODEL_ID churned --file data.csv
    Force:   ffs foundation predict MODEL_ID churned '{"age": 35}' --model-type foundation
    """
    if not record_json and not data_file:
        raise click.ClickException("Provide a JSON record or --file")

    fm = state.client.foundational_model(model_id)

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
            raise click.ClickException(f"Unsupported file format (use .csv, .json, or .parquet): {data_file}")

        records = df.to_dict(orient="records")
        results = []
        for rec in records:
            kwargs = {}
            if model_type:
                kwargs["model_type"] = model_type
            if predictor_id:
                kwargs["predictor_id"] = predictor_id
            results.append(fm.predict(target_column, rec, **kwargs))

        if state.output_json:
            print_json([r.to_dict() for r in results])
        else:
            console.print(f"[green]{len(results)} predictions[/green]\n")
            rows = []
            for i, r in enumerate(results):
                row = {"#": str(i + 1)}
                if r.predicted_class is not None:
                    row["Predicted"] = r.predicted_class
                elif hasattr(r, "prediction") and r.prediction is not None:
                    row["Predicted"] = str(r.prediction)
                if r.confidence is not None:
                    row["Confidence"] = f"{r.confidence:.3f}"
                if r.model_type:
                    row["Model"] = r.model_type
                rows.append(row)
            if rows:
                print_list_table(rows, list(rows[0].keys()))
        return

    record = json.loads(record_json)
    kwargs = {}
    if model_type:
        kwargs["model_type"] = model_type
    if predictor_id:
        kwargs["predictor_id"] = predictor_id

    result = fm.predict(target_column, record, **kwargs)

    if state.output_json:
        print_json(result.to_dict())
    else:
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
        if result.model_type:
            data["Model Type"] = result.model_type
        if result.prediction_uuid:
            data["Prediction UUID"] = result.prediction_uuid
        if explain and hasattr(result, "feature_importance") and result.feature_importance:
            data["Feature Importance"] = ""
            print_kv(data, title="Prediction")
            for col, score in result.feature_importance.items():
                console.print(f"  {col}: {score:.4f}")
        else:
            print_kv(data, title="Prediction")


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
@click.option("--org", required=True, help="Organization ID")
@click.option("--name", default=None, help="Published name")
@pass_client
def publish(state: ClientState, model_id, org, name):
    """Publish a model."""
    fm = state.client.foundational_model(model_id)
    result = fm.publish(org_id=org, name=name)
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
