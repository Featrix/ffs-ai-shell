"""ffs endpoint subcommands."""
import click

from ffs.click_ext import DYMGroup
from ffs.client import pass_client, ClientState
from ffs.output import print_json, print_kv, console


def _get_predictor(state: ClientState, model_id):
    fm = state.client.foundational_model(model_id)
    predictors = fm.list_predictors()
    if not predictors:
        raise click.ClickException(f"No predictor found in session {model_id}")
    return predictors[0]


@click.group(cls=DYMGroup)
def endpoint():
    """Create and manage named API endpoints for predictors."""
    pass


@endpoint.command()
@click.argument("model_id")
@click.option("--name", required=True, help="Endpoint name")
@click.option("--api-key", default=None, help="API key to assign (auto-generated if omitted)")
@click.option("--description", default=None, help="Endpoint description")
@pass_client
def create(state: ClientState, model_id, name, api_key, description):
    """Create a named API endpoint for a predictor's foundation."""
    predictor = _get_predictor(state, model_id)
    ep = predictor.create_api_endpoint(name=name, api_key=api_key, description=description)

    if state.output_json:
        print_json(ep.to_dict())
    else:
        console.print(f"[green]Endpoint created:[/green] {ep.id}")
        console.print(f"Name: {ep.name}")
        if ep.url:
            console.print(f"URL: {ep.url}")
        if ep.api_key:
            console.print(f"[bold cyan]API Key:[/bold cyan] {ep.api_key}")
            console.print("[dim]Save this now — it won't be shown again.[/dim]")


@endpoint.command()
@click.argument("model_id")
@click.argument("endpoint_id")
@pass_client
def show(state: ClientState, model_id, endpoint_id):
    """Show details for an API endpoint."""
    ep = state.client.api_endpoint(endpoint_id, model_id)

    if state.output_json:
        print_json(ep.to_dict())
    else:
        data = {
            "Endpoint ID": ep.id,
            "Name": ep.name,
            "Predictor ID": ep.predictor_id,
            "Session ID": ep.session_id,
            "Has API Key": "yes" if ep.api_key else "no",
            "Usage Count": ep.usage_count,
        }
        if ep.description:
            data["Description"] = ep.description
        if ep.url:
            data["URL"] = ep.url
        if ep.last_used_at:
            data["Last Used"] = ep.last_used_at
        print_kv(data, title="API Endpoint")


@endpoint.command()
@click.argument("model_id")
@click.argument("endpoint_id")
@pass_client
def stats(state: ClientState, model_id, endpoint_id):
    """Show usage statistics for an API endpoint."""
    ep = state.client.api_endpoint(endpoint_id, model_id)
    data = ep.get_usage_stats()

    if state.output_json:
        print_json(data)
    else:
        print_kv(data, title="Endpoint Usage")


@endpoint.command("regenerate-key")
@click.argument("model_id")
@click.argument("endpoint_id")
@click.confirmation_option(prompt="This invalidates the current API key. Continue?")
@pass_client
def regenerate_key(state: ClientState, model_id, endpoint_id):
    """Regenerate an endpoint's API key, invalidating the old one."""
    ep = state.client.api_endpoint(endpoint_id, model_id)
    new_key = ep.regenerate_api_key()

    if state.output_json:
        print_json({"endpoint_id": ep.id, "api_key": new_key})
    else:
        console.print(f"[bold cyan]New API Key:[/bold cyan] {new_key}")
        console.print("[dim]Save this now — it won't be shown again.[/dim]")


@endpoint.command("revoke-key")
@click.argument("model_id")
@click.argument("endpoint_id")
@click.confirmation_option(prompt="This removes the API key and makes the endpoint publicly callable. Continue?")
@pass_client
def revoke_key(state: ClientState, model_id, endpoint_id):
    """Revoke an endpoint's API key, leaving it open to unauthenticated calls."""
    ep = state.client.api_endpoint(endpoint_id, model_id)
    ep.revoke_api_key()

    if state.output_json:
        print_json({"endpoint_id": ep.id, "api_key": None})
    else:
        console.print(f"[yellow]API key revoked:[/yellow] {ep.id}")


@endpoint.command()
@click.argument("model_id")
@click.argument("endpoint_id")
@click.confirmation_option(prompt="Are you sure you want to delete this endpoint?")
@pass_client
def delete(state: ClientState, model_id, endpoint_id):
    """Delete an API endpoint."""
    ep = state.client.api_endpoint(endpoint_id, model_id)
    ep.delete()

    if state.output_json:
        print_json({"endpoint_id": ep.id, "deleted": True})
    else:
        console.print(f"[yellow]Deleted:[/yellow] {ep.id}")
