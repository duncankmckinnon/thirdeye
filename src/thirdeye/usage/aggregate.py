"""Aggregate usage rows into per-day buckets. All bucketing is UTC."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from thirdeye.config import Config
from thirdeye.paths import session_dir as _session_dir
from thirdeye.store import Store
from thirdeye.usage.read import iter_calls


@dataclass(frozen=True)
class TimeBucket:
    day: str
    sessions: int
    events: int
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class UsageReport:
    buckets: list[TimeBucket]
    totals: TimeBucket


def _row_day(ts_str: str) -> str:
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    return datetime.fromisoformat(ts_str).astimezone(UTC).date().isoformat()


def aggregate_by_day(
    config: Config,
    *,
    platform: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> UsageReport:
    """Stream every matching session, read its usage.jsonl, bucket by UTC day.

    The output `buckets` list covers a contiguous date range — every day
    from min(seen) to max(seen) inclusive (or [since.date, until.date] if
    bounds were supplied), with zeros for empty days.
    """
    sums: dict[str, dict[str, object]] = defaultdict(
        lambda: {"events": 0, "input": 0, "output": 0, "sessions": set()}
    )
    store = Store(config)
    for meta in store.list_sessions(platform=platform, since=since, until=until):
        sd = _session_dir(config.root, meta.platform, meta.session_id)
        for row in iter_calls(sd):
            day = _row_day(row.ts)
            b = sums[day]
            b["events"] = int(b["events"]) + 1
            b["input"] = int(b["input"]) + row.input_tokens
            b["output"] = int(b["output"]) + row.output_tokens
            assert isinstance(b["sessions"], set)
            b["sessions"].add(meta.session_id)

    if not sums and since is None and until is None:
        buckets: list[TimeBucket] = []
    else:
        if sums:
            min_day = date.fromisoformat(min(sums.keys()))
            max_day = date.fromisoformat(max(sums.keys()))
        else:
            assert since is not None and until is not None
            min_day, max_day = since.date(), until.date()
        if since is not None:
            min_day = min(min_day, since.date())
        if until is not None:
            max_day = max(max_day, until.date())
        days: list[str] = []
        d = min_day
        while d <= max_day:
            days.append(d.isoformat())
            d += timedelta(days=1)
        buckets = [
            TimeBucket(
                day=day,
                sessions=len(sums[day]["sessions"]) if day in sums else 0,
                events=int(sums[day]["events"]) if day in sums else 0,
                input_tokens=int(sums[day]["input"]) if day in sums else 0,
                output_tokens=int(sums[day]["output"]) if day in sums else 0,
            )
            for day in days
        ]

    total_sessions: set[str] = set()
    if sums:
        for s in sums.values():
            assert isinstance(s["sessions"], set)
            total_sessions |= s["sessions"]
    totals = TimeBucket(
        day="TOTAL",
        sessions=len(total_sessions),
        events=sum(int(s["events"]) for s in sums.values()),
        input_tokens=sum(int(s["input"]) for s in sums.values()),
        output_tokens=sum(int(s["output"]) for s in sums.values()),
    )
    return UsageReport(buckets=buckets, totals=totals)
