from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import pytest

from thirdeye.config import Config
from thirdeye.meta import SessionMeta, write_meta
from thirdeye.paths import meta_path, session_dir
from thirdeye.usage.aggregate import aggregate_by_day
from thirdeye.usage.store import UsageStore
from thirdeye.usage.types import UsageRow


def _make_session(
    root: Path,
    *,
    sid: str,
    platform: str,
    started_at: str,
    last_ts: str | None = None,
) -> Path:
    sd = session_dir(root, platform, sid)
    sd.mkdir(parents=True, exist_ok=True)
    meta = SessionMeta(
        session_id=sid,
        platform=platform,
        cwd="/tmp/proj",
        started_at=started_at,
        ended_at=None,
        status="open",
        event_count=0,
        last_seq=-1,
        last_ts=last_ts or started_at,
    )
    write_meta(meta_path(sd), meta)
    return sd


def _row(
    *,
    sid: str,
    platform: str,
    seq: int,
    ts: str,
    call_id: str | None = None,
    input_tokens: int = 100,
    output_tokens: int = 10,
    cache_read_input_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
    reasoning_output_tokens: int | None = None,
) -> UsageRow:
    return UsageRow(
        session_id=sid,
        seq=seq,
        call_id=call_id if call_id is not None else f"{sid}-{seq}",
        ts=ts,
        platform=platform,
        provider_name="anthropic",
        response_model="claude-opus-4-7",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
    )


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(root=tmp_path)


def test_empty_store_returns_empty_report(config: Config) -> None:
    report = aggregate_by_day(config)
    assert report.buckets == []
    assert report.totals.day == "TOTAL"
    assert report.totals.sessions == 0
    assert report.totals.events == 0
    assert report.totals.input_tokens == 0
    assert report.totals.output_tokens == 0


def test_duplicate_call_id_counted_once(config: Config) -> None:
    """Six duplicate frames of one call (100/10) collapse to a single call.

    This is the guard against dedup silently regressing: the bucket must be
    100 input / 10 output, not 600 / 60.
    """
    sd = _make_session(
        config.root,
        sid="s1",
        platform="claude",
        started_at="2026-05-20T10:00:00.000Z",
    )
    UsageStore(sd).append(
        [
            _row(
                sid="s1",
                platform="claude",
                seq=0,
                ts="2026-05-20T10:00:00.000Z",
                call_id="call-abc",
            )
            for _ in range(6)
        ]
    )

    report = aggregate_by_day(config)
    by_day = {b.day: b for b in report.buckets}
    assert by_day["2026-05-20"].events == 1
    assert by_day["2026-05-20"].input_tokens == 100
    assert by_day["2026-05-20"].output_tokens == 10
    assert report.totals.input_tokens == 100
    assert report.totals.output_tokens == 10


def test_two_sessions_same_day_sum_into_one_bucket(config: Config) -> None:
    sd1 = _make_session(
        config.root, sid="s1", platform="claude", started_at="2026-05-20T10:00:00.000Z"
    )
    UsageStore(sd1).append(
        [_row(sid="s1", platform="claude", seq=0, ts="2026-05-20T10:00:00.000Z")]
    )
    sd2 = _make_session(
        config.root, sid="s2", platform="claude", started_at="2026-05-20T12:00:00.000Z"
    )
    UsageStore(sd2).append(
        [
            _row(
                sid="s2",
                platform="claude",
                seq=0,
                ts="2026-05-20T12:00:00.000Z",
                input_tokens=50,
                output_tokens=5,
            )
        ]
    )

    report = aggregate_by_day(config)
    assert [b.day for b in report.buckets] == ["2026-05-20"]
    bucket = report.buckets[0]
    assert bucket.sessions == 2
    assert bucket.events == 2
    assert bucket.input_tokens == 150
    assert bucket.output_tokens == 15


def test_rows_spanning_three_days_zero_fill(config: Config) -> None:
    sd1 = _make_session(
        config.root, sid="s1", platform="claude", started_at="2026-05-20T10:00:00.000Z"
    )
    UsageStore(sd1).append(
        [_row(sid="s1", platform="claude", seq=0, ts="2026-05-20T10:00:00.000Z")]
    )
    sd2 = _make_session(
        config.root, sid="s2", platform="claude", started_at="2026-05-22T10:00:00.000Z"
    )
    UsageStore(sd2).append(
        [_row(sid="s2", platform="claude", seq=0, ts="2026-05-22T10:00:00.000Z")]
    )

    report = aggregate_by_day(config)
    assert [b.day for b in report.buckets] == ["2026-05-20", "2026-05-21", "2026-05-22"]
    by_day = {b.day: b for b in report.buckets}
    assert by_day["2026-05-21"].events == 0
    assert by_day["2026-05-21"].sessions == 0
    assert by_day["2026-05-21"].input_tokens == 0
    assert by_day["2026-05-21"].output_tokens == 0


def test_since_until_bounds_control_range(config: Config) -> None:
    sd_old = _make_session(
        config.root,
        sid="old",
        platform="claude",
        started_at="2026-05-01T10:00:00.000Z",
        last_ts="2026-05-01T11:00:00.000Z",
    )
    UsageStore(sd_old).append(
        [_row(sid="old", platform="claude", seq=0, ts="2026-05-01T10:00:00.000Z")]
    )
    sd_new = _make_session(
        config.root,
        sid="new",
        platform="claude",
        started_at="2026-05-20T10:00:00.000Z",
        last_ts="2026-05-20T11:00:00.000Z",
    )
    UsageStore(sd_new).append(
        [_row(sid="new", platform="claude", seq=0, ts="2026-05-20T10:00:00.000Z")]
    )

    since = datetime(2026, 5, 15, tzinfo=UTC)
    until = datetime(2026, 5, 25, tzinfo=UTC)
    report = aggregate_by_day(config, since=since, until=until)
    days = {b.day for b in report.buckets}
    assert report.totals.sessions == 1
    assert report.totals.events == 1
    assert "2026-05-20" in days
    assert "2026-05-01" not in days


def test_z_suffixed_timestamp_buckets_to_utc_day(config: Config) -> None:
    sd = _make_session(
        config.root,
        sid="s1",
        platform="claude",
        started_at="2026-05-20T23:00:00.000Z",
        last_ts="2026-05-20T23:30:00.000Z",
    )
    UsageStore(sd).append([_row(sid="s1", platform="claude", seq=0, ts="2026-05-20T23:30:00.000Z")])

    report = aggregate_by_day(config)
    by_day = {b.day: b.events for b in report.buckets}
    assert by_day.get("2026-05-20") == 1


def test_none_cache_attributes_aggregate_without_raising(config: Config) -> None:
    sd = _make_session(
        config.root, sid="s1", platform="claude", started_at="2026-05-20T10:00:00.000Z"
    )
    UsageStore(sd).append(
        [
            _row(
                sid="s1",
                platform="claude",
                seq=0,
                ts="2026-05-20T10:00:00.000Z",
                cache_read_input_tokens=None,
                cache_creation_input_tokens=None,
                reasoning_output_tokens=None,
            )
        ]
    )

    report = aggregate_by_day(config)
    bucket = report.buckets[0]
    assert bucket.input_tokens == 100
    assert bucket.output_tokens == 10
    assert bucket.total_tokens == 110


def test_aggregate_returns_immutable_dataclasses(config: Config) -> None:
    report = aggregate_by_day(config)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.totals.events = 99  # type: ignore[misc]


def test_row_day_helper_handles_z_and_offset() -> None:
    from thirdeye.usage.aggregate import _row_day

    assert _row_day("2026-05-20T10:00:00.000Z") == "2026-05-20"
    assert _row_day("2026-05-21T01:30:00.000+02:00") == "2026-05-20"
