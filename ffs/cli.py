import click

from ffs.client import pass_client, ClientState
from ffs import model_cmd
from ffs import server_cmd


@click.group()
@click.option("--server", envvar="FFS_SERVER", default="https://sphere-api.featrix.com", help="API server URL")
@click.option("--cluster", envvar="FFS_CLUSTER", default=None, help="Compute cluster name")
@click.option("--json", "output_json", is_flag=True, help="Output raw JSON")
@click.option("--quiet", is_flag=True, help="Minimal output")
@click.pass_context
def main(ctx, server, cluster, output_json, quiet):
    """The Featrix Foundation Shell."""
    ctx.ensure_object(dict)
    ctx.obj = ClientState(server=server, cluster=cluster, output_json=output_json, quiet=quiet)


main.add_command(model_cmd.model)
main.add_command(server_cmd.server)
