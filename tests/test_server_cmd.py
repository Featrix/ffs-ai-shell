"""Tests for ffs/server_cmd.py."""
import json

from ffs.cli import main


class TestServerHealth:
    def test_health(self, runner, mock_sphere, env):
        mock_sphere.health_check.return_value = {"status": "healthy", "version": "1.0"}
        result = runner.invoke(main, ["server", "health"], env=env)
        assert result.exit_code == 0
        assert "healthy" in result.output

    def test_health_json(self, runner, mock_sphere, env):
        mock_sphere.health_check.return_value = {"status": "healthy"}
        result = runner.invoke(main, ["--json", "server", "health"], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "healthy"
