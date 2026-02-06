import click
from dataclasses import dataclass

from featrixsphere.api import FeatrixSphere
from featrixsphere import FeatrixSphereClient


@dataclass
class ClientState:
    server: str
    cluster: str | None
    output_json: bool
    quiet: bool
    _client: FeatrixSphere | None = None
    _low_client: FeatrixSphereClient | None = None

    @property
    def client(self) -> FeatrixSphere:
        """OO API client."""
        if self._client is None:
            self._client = FeatrixSphere(
                base_url=self.server,
                compute_cluster=self.cluster,
            )
        return self._client

    @property
    def low(self) -> FeatrixSphereClient:
        """Low-level client for operations not on the OO API."""
        if self._low_client is None:
            self._low_client = FeatrixSphereClient(
                base_url=self.server,
                compute_cluster=self.cluster,
            )
        return self._low_client


pass_client = click.make_pass_decorator(ClientState)
