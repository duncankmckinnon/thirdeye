from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

_ROLLING_RE = re.compile(r"^(\d+)(d|h|m)$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_ERR = "could not parse {input!r}: expected ISO date, 'NN(d|h|m)', 'today', or 'yesterday'"


def parse_when(s: str | None, *, now: datetime | None = None) -> datetime | None:
    """Parse a `--since` / `--until` value into an aware UTC datetime.

    Returns None when ``s`` is None so route handlers can pass missing query
    params straight through without per-call guards.
    """
    if s is None:
        return None
    if not isinstance(s, str) or not s:
        raise ValueError(_ERR.format(input=s))

    now = now if now is not None else datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    else:
        now = now.astimezone(UTC)

    lowered = s.lower()
    if lowered == "today":
        return datetime(now.year, now.month, now.day, tzinfo=UTC)
    if lowered == "yesterday":
        midnight = datetime(now.year, now.month, now.day, tzinfo=UTC)
        return midnight - timedelta(days=1)

    m = _ROLLING_RE.match(s)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit == "d":
            delta = timedelta(days=n)
        elif unit == "h":
            delta = timedelta(hours=n)
        else:
            delta = timedelta(minutes=n)
        return now - delta

    if _ISO_DATE_RE.match(s):
        try:
            d = datetime.fromisoformat(s)
        except ValueError as e:
            raise ValueError(_ERR.format(input=s)) from e
        return d.replace(tzinfo=UTC)

    # ISO datetime: accept Z suffix or explicit offset.
    iso = s
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError as e:
        raise ValueError(_ERR.format(input=s)) from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt
