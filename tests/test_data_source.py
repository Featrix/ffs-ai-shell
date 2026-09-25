"""`ffs foundation create --data` takes a local file OR a URL the server fetches."""
from datetime import datetime, timezone

import pytest

from ffs.cli import main
from ffs.data_source import link_expiry

PRESIGNED = (
    "https://bucket.s3.amazonaws.com/churn.parquet"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260925T010000Z"
    "&X-Amz-Expires=3600&X-Amz-Signature=abc"
)


@pytest.mark.parametrize("url", [
    "s3://bucket/churn.parquet",
    "https://example.com/data/churn.csv",
    PRESIGNED,
])
def test_url_is_passed_to_the_sdk_untouched(runner, wired_sphere, env, url):
    result = runner.invoke(main, ["foundation", "create", "--name", "churn", "--data", url], env=env)
    assert result.exit_code == 0, result.output
    assert wired_sphere.create_foundational_model.call_args.kwargs["data_file"] == url


def test_local_file_still_works(runner, wired_sphere, env, tmp_path):
    f = tmp_path / "churn.csv"
    f.write_text("a,b\n1,2\n")
    result = runner.invoke(main, ["foundation", "create", "--name", "churn", "--data", str(f)], env=env)
    assert result.exit_code == 0, result.output
    assert wired_sphere.create_foundational_model.call_args.kwargs["data_file"] == str(f)


def test_missing_local_file_is_still_a_clear_error(runner, wired_sphere, env):
    result = runner.invoke(main, ["foundation", "create", "--name", "churn", "--data", "nope.csv"], env=env)
    assert result.exit_code != 0
    assert "does not exist" in result.output
    wired_sphere.create_foundational_model.assert_not_called()


def test_http_is_refused_before_calling_the_server(runner, wired_sphere, env):
    result = runner.invoke(
        main, ["foundation", "create", "--name", "churn", "--data", "http://example.com/x.csv"], env=env,
    )
    assert result.exit_code != 0
    assert "https://" in result.output
    wired_sphere.create_foundational_model.assert_not_called()


def test_presigned_link_expiry_is_shown(runner, wired_sphere, env):
    result = runner.invoke(main, ["foundation", "create", "--name", "churn", "--data", PRESIGNED], env=env)
    assert result.exit_code == 0, result.output
    assert "expires 2026-09-25 02:00 UTC" in result.output
    assert "X-Amz-Signature" not in result.output  # never echo the credential


def test_link_expiry_parsing():
    assert link_expiry(PRESIGNED) == datetime(2026, 9, 25, 2, 0, tzinfo=timezone.utc)
    assert link_expiry("https://example.com/x.csv") is None
    assert link_expiry("https://b.s3.amazonaws.com/k?Expires=1790000000&Signature=x") == \
        datetime.fromtimestamp(1790000000, tz=timezone.utc)
