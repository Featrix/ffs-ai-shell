"""ffs network subcommands.

PredictionNetworks — chains of a customer's own published models, where
one node's output (label / confidence / value) gates whether downstream
nodes run. See taco-fixes/docs/internal/plans/2026-07-prediction-networks.md
for the full design.
"""
import json
from dataclasses import asdict

import click

from ffs.client import pass_client, ClientState
from ffs.output import print_json, print_kv, print_list_table, console


def _get_org(state: ClientState) -> str:
    """Resolve the active org from the authenticated API key.

    PredictionNetworks are org-scoped server-side by the API key alone —
    one key belongs to exactly one org (enforced in
    route_prediction_networks.py's _authenticate). There's no
    `ffs network set-org`: switching orgs means switching which
    .featrix/API key is active, same as everything else in ffs.
    """
    identity = state.client.whoami()
    org = identity.get("org_slug")
    if not org:
        raise click.ClickException(
            "Could not resolve org from API key — check `ffs whoami`."
        )
    return org


def _get_network(state: ClientState, name: str):
    org = _get_org(state)
    return state.client.published_prediction_network(
        org=org,
        name=name,
        api_key=state.client.api_key,
        base_url=state.client.base_url,
    )


@click.group()
def network():
    """Register and run PredictionNetworks — chains of your published models."""
    pass


@network.command()
@click.argument("name")
@click.option(
    "--spec-file",
    required=True,
    type=click.Path(exists=True),
    help="JSON file with {nodes, edges}",
)
@pass_client
def register(state: ClientState, name, spec_file):
    """Register or update a PredictionNetwork's spec.

    \b
    Spec file shape:
      {"nodes": [{"id": "is_company", "model": "is-company"}, ...],
       "edges": [{"from": "is_company", "to": "company_type",
                   "when": {"op": "always"}}, ...]}

    \b
    Condition ops: always, ==, !=, >, >=, <, <= (against a field on the
    source node's output — label, confidence, or a named probability).
    """
    with open(spec_file) as f:
        spec = json.load(f)

    net = _get_network(state, name)
    result = net.register(spec)

    if state.output_json:
        print_json(result)
    else:
        console.print(
            f"[green]PredictionNetwork registered:[/green] {name} "
            f"v{result['version']} ({result['nodes']} nodes)"
        )


@network.command()
@click.argument("name")
@pass_client
def show(state: ClientState, name):
    """Show a PredictionNetwork's registered spec."""
    net = _get_network(state, name)
    row = net.get_spec()

    if state.output_json:
        print_json(row)
        return

    spec = row.get("spec", {})
    data = {
        "Name": row.get("name", name),
        "Version": row.get("version", "—"),
        "Nodes": len(spec.get("nodes", [])),
        "Edges": len(spec.get("edges", [])),
    }
    print_kv(data, title="PredictionNetwork")
    for n in spec.get("nodes", []):
        console.print(f"  [cyan]{n['id']}[/cyan] -> {n['model']}")
    for e in spec.get("edges", []):
        when = e.get("when", {"op": "always"})
        cond = (
            "always"
            if when.get("op") == "always"
            else f"{when.get('field')} {when.get('op')} {when.get('value')}"
        )
        console.print(f"    {e['from']} --[{cond}]--> {e['to']}")


@network.command("list")
@pass_client
def list_networks(state: ClientState):
    """List PredictionNetworks registered for your org."""
    org = _get_org(state)
    networks = state.client.list_prediction_networks(org)

    if state.output_json:
        print_json(networks)
    elif not networks:
        console.print("No PredictionNetworks registered.")
    else:
        rows = [
            {
                "Name": n["name"],
                "Version": n["version"],
                "Updated": n.get("updated_at", "—"),
            }
            for n in networks
        ]
        print_list_table(rows, ["Name", "Version", "Updated"])


@network.command()
@click.argument("name")
@click.argument("record_json", required=False)
@click.option(
    "--file",
    "data_file",
    type=click.Path(exists=True),
    help="Predict a single record read from a JSON file",
)
@pass_client
def predict(state: ClientState, name, record_json, data_file):
    """Run a PredictionNetwork against one record.

    v1 executes exactly one record per call — no batch mode yet.

    \b
      ffs network predict carrier-qualification '{"company_name": "Acme Trucking"}'
      ffs network predict carrier-qualification --file row.json
    """
    if not record_json and not data_file:
        raise click.ClickException("Provide a JSON record or --file")

    if data_file:
        with open(data_file) as f:
            record = json.load(f)
    else:
        record = json.loads(record_json)

    net = _get_network(state, name)
    result = net.predict(record)

    if state.output_json:
        print_json({
            "final": result.final,
            "trace": [asdict(t) for t in result.trace],
        })
        return

    print_kv(result.final, title="Final")
    console.print("\n[bold]Trace[/bold]")
    rows = []
    for t in result.trace:
        if t.skipped:
            status, detail = "skipped", t.skip_reason or "—"
        elif t.error:
            status, detail = "error", t.error
        else:
            conf = f"{t.confidence:.3f}" if t.confidence is not None else "—"
            latency = f"{t.latency_ms:.0f}ms" if t.latency_ms is not None else "—"
            status, detail = str(t.label), f"conf={conf}  {latency}"
        rows.append({
            "Node": t.node_id,
            "Model": t.model or "—",
            "Status": status,
            "Detail": detail,
        })
    print_list_table(rows, ["Node", "Model", "Status", "Detail"])
