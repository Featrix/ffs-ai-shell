"""ffs events subcommands."""
import click

from ffs.click_ext import DYMGroup
from ffs.client import pass_client, ClientState
from ffs.output import print_json, print_kv, print_list_table, console


@click.group(cls=DYMGroup)
def events():
    """Query event groups fed by the featrixevents library.

    Read-only: posting events is featrixevents' job (a separate library
    apps use to write to event streams), not ffs's.
    """
    pass


@events.command("list")
@click.option("--limit", type=int, default=50, help="Max groups to return")
@click.option("--offset", type=int, default=0, help="Pagination offset")
@pass_client
def list_groups(state: ClientState, limit, offset):
    """List event groups registered for your org.

    A group only appears here once it has received at least one event --
    an event_group_id that's never been posted to never shows up.
    """
    groups = state.client.list_event_groups(limit=limit, offset=offset)

    if state.output_json:
        print_json([g.to_dict() for g in groups])
    elif not groups:
        console.print("No event groups found.")
    else:
        rows = []
        for g in groups:
            rows.append({
                "Event Group ID": g.event_group_id,
                "Events (at last train)": g.event_count_at_last_train,
                "Last Trained": g.last_trained_at or "—",
                "Last Session": g.last_session_id or "—",
                "Auto-Retrain": "yes" if g.auto_retrain_enabled else "no",
            })
        print_list_table(rows, ["Event Group ID", "Events (at last train)", "Last Trained", "Last Session", "Auto-Retrain"])


@events.command()
@click.argument("event_group_id")
@pass_client
def show(state: ClientState, event_group_id):
    """Show one event group's detail, led by its live event count."""
    g = state.client.event_group(event_group_id)

    if state.output_json:
        print_json(g.to_dict())
    else:
        data = {
            "Event Count": g.event_count if g.event_count is not None else "—",
            "Event Count (at last train)": g.event_count_at_last_train,
            "Extend Count": g.extend_count,
            "Auto-Retrain Enabled": "yes" if g.auto_retrain_enabled else "no",
            "Last Session ID": g.last_session_id or "—",
            "Last Trained": g.last_trained_at or "—",
            "Consecutive Failures": g.consecutive_failures,
            "Created": g.created_at or "—",
            "Updated": g.updated_at or "—",
        }
        print_kv(data, title=f"Event Group {g.event_group_id}")
