import click
from dataclasses import dataclass

from featrixsphere.api import FeatrixSphere


@dataclass
class ClientState:
    server: str
    cluster: str | None
    output_json: bool
    quiet: bool
    _client: FeatrixSphere | None = None

    @property
    def client(self) -> FeatrixSphere:
        if self._client is None:
            self._client = FeatrixSphere(
                base_url=self.server,
                compute_cluster=self.cluster,
            )
        return self._client


pass_client = click.make_pass_decorator(ClientState)
