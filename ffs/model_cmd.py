"""ffs model subcommands."""
import json
import time
import click

from ffs.client import pass_client, ClientState
from ffs.output import print_json, print_kv, console


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
        csv_file=data_file,
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
@pass_client
def list_models(state: ClientState, prefix):
    """List models."""
    sessions = state.client.list_sessions(name_prefix=prefix)
    if state.output_json:
        print_json(sessions)
    elif not sessions:
        console.print("No models found.")
    else:
        for sid in sessions:
            console.print(sid)


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


@model.command()
@click.argument("model_id")
@pass_client
def card(state: ClientState, model_id):
    """Show the model card."""
    fm = state.client.foundational_model(model_id)
    print_json(fm.get_model_card())


def _format_duration(seconds: int) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _job_status_lines(server_data: dict) -> list[str]:
    """Build status lines from server response dict (from fm.refresh())."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    lines = []
    job_plan = server_data.get("job_plan", [])
    jobs = server_data.get("jobs", {})
    for job in job_plan:
        jtype = job.get("job_type", "?")
        jid = job.get("job_id")
        if jid and jid in jobs:
            j = jobs[jid]
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
                lines.append(f"  [green]done[/green]  {jtype}{duration}")
            elif status == "running" and progress:
                lines.append(f"  [yellow]running {progress}%[/yellow]  {jtype}{queue_str}{age}")
            elif status == "running":
                lines.append(f"  [yellow]running[/yellow]  {jtype}{queue_str}{age}")
            else:
                lines.append(f"  [dim]{status}[/dim]  {jtype}{queue_str}{age}")
        else:
            lines.append(f"  [dim]pending[/dim]  {jtype}")
    return lines


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

        # Clear screen and redraw
        job_plan = data.get("job_plan", [])
        if not first:
            n_lines = len(job_plan) + 2
            click.echo(f"\033[{n_lines}A\033[J", nl=False)
        first = False

        console.print(f"[bold]Waiting for {model_id}[/bold]  ({_format_duration(elapsed)})")
        for line in _job_status_lines(data):
            console.print(line)

        time.sleep(poll_interval)



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
