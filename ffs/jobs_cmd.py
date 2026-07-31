"""ffs jobs subcommands."""
import click

from ffs.client import pass_client, ClientState
from ffs.output import print_json, print_list_table, console


@click.group()
def jobs():
    """Inspect training jobs across the org."""
    pass


@jobs.command("list")
@click.option("--prefix", default="", help="Filter by name or session ID prefix")
@pass_client
def list_pending(state: ClientState, prefix):
    """List sessions with jobs still queued or running, org-wide.

    Scans every session in the account (via a single call) rather than
    requiring a model ID up front — use --prefix to narrow by name/ID.
    """
    pending = state.client.list_pending_jobs(name_prefix=prefix)

    if state.output_json:
        print_json([
            {
                "id": fm.id,
                "name": fm.name,
                "status": fm.status,
                "dimensions": fm.dimensions,
                "epochs": fm.epochs,
            }
            for fm in pending
        ])
        return

    if not pending:
        console.print("No pending jobs.")
        return

    rows = []
    for fm in pending:
        rows.append({
            "ID": fm.id,
            "Name": fm.name or "—",
            "Status": fm.status or "—",
            "Dims": str(fm.dimensions) if fm.dimensions else "—",
            "Epochs": str(fm.epochs) if fm.epochs else "—",
        })
    print_list_table(rows, ["ID", "Name", "Status", "Dims", "Epochs"])


@jobs.command("cancel-queued")
@click.argument("model_id", required=False)
@click.option("--prefix", default="", help="Filter by name/ID prefix (org-wide sweep only)")
@click.option("--reason", default=None, help="Reason for cancellation (stored for audit)")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt")
@pass_client
def cancel_queued(state: ClientState, model_id, prefix, reason, yes):
    """Cancel jobs that are still queued — leaves running jobs alone.

    Scoped to one model if MODEL_ID is given, otherwise sweeps every queued
    session in the org. Targets sessions whose status is "ready" (every job
    non-terminal, none running yet) — a session already "running" is skipped
    even if given explicitly. There's an inherent gap between listing and
    cancelling: a job can start running in that window, in which case it
    still gets cancelled (cooperatively), same as `foundation cancel`.
    """
    if model_id:
        fm = state.client.foundational_model(model_id)
        fm.refresh()
        targets = [fm] if fm.status == "ready" else []
    else:
        targets = [fm for fm in state.client.list_pending_jobs(name_prefix=prefix) if fm.status == "ready"]

    if not targets:
        if state.output_json:
            print_json({"cancelled": []})
        else:
            console.print("No queued jobs to cancel.")
        return

    if not yes:
        ids = ", ".join(fm.id for fm in targets)
        click.confirm(f"Cancel {len(targets)} queued job(s)? [{ids}]", abort=True)

    results = []
    for fm in targets:
        result = fm.cancel(reason=reason)
        results.append({"id": fm.id, "name": fm.name, "result": result})

    if state.output_json:
        print_json({"cancelled": results})
    else:
        console.print(f"[yellow]Cancelled {len(results)} queued job(s):[/yellow]")
        for r in results:
            console.print(f"  {r['id']}  {r['name'] or ''}")
