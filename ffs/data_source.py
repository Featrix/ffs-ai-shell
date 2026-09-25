"""`--data` for training: a local file, or a URL the Featrix server fetches."""
import urllib.parse
from datetime import datetime, timedelta, timezone

import click

URL_PREFIXES = ("https://", "s3://")


class DataSource(click.ParamType):
    """A local file that exists, an https:// link (public or presigned), or a
    public s3:// URL. URLs are passed through untouched -- the server fetches
    them, nothing is uploaded -- and a mistyped local path still gets click's
    own "does not exist" error."""

    name = "FILE|URL"

    def convert(self, value, param, ctx):
        lowered = str(value).strip().lower()
        if lowered.startswith(URL_PREFIXES):
            return str(value).strip()
        if lowered.startswith("http://"):
            self.fail("http:// links aren't accepted -- use https://", param, ctx)
        return click.Path(exists=True, dir_okay=False).convert(value, param, ctx)


def is_url(value: str) -> bool:
    return str(value).strip().lower().startswith(URL_PREFIXES)


def link_expiry(url: str) -> datetime | None:
    """When a presigned link stops working (UTC), if it says: AWS SigV4
    (X-Amz-Date + X-Amz-Expires), SigV2 (Expires), or GCS V4 (X-Goog-*)."""
    query = {k.lower(): v for k, v in urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query)}
    for date_key, ttl_key in (("x-amz-date", "x-amz-expires"), ("x-goog-date", "x-goog-expires")):
        if date_key in query and ttl_key in query:
            try:
                signed_at = datetime.strptime(query[date_key], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                return signed_at + timedelta(seconds=int(query[ttl_key]))
            except ValueError:
                return None
    if "expires" in query and "signature" in query:
        try:
            return datetime.fromtimestamp(int(query["expires"]), tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None
    return None
