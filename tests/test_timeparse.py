"""Tests for thirdeye.timeparse: parse_when for --since / --until values."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from thirdeye.timeparse import parse_when


class TestParseWhenIsoDate:
    def test_basic_iso_date(self):
        got = parse_when("2026-05-13")
        assert got == datetime(2026, 5, 13, 0, 0, 0, tzinfo=UTC)

    def test_iso_date_is_tz_aware_utc(self):
        got = parse_when("2026-01-01")
        assert got.tzinfo is not None
        assert got.utcoffset() == timedelta(0)

    def test_iso_date_at_year_boundary(self):
        got = parse_when("2026-12-31")
        assert got == datetime(2026, 12, 31, 0, 0, 0, tzinfo=UTC)


class TestParseWhenIsoDatetime:
    def test_z_suffix(self):
        got = parse_when("2026-05-13T15:30:00Z")
        assert got == datetime(2026, 5, 13, 15, 30, 0, tzinfo=UTC)

    def test_explicit_utc_offset(self):
        got = parse_when("2026-05-13T15:30:00+00:00")
        assert got == datetime(2026, 5, 13, 15, 30, 0, tzinfo=UTC)

    def test_negative_offset_normalizes_to_utc(self):
        got = parse_when("2026-05-13T10:30:00-05:00")
        assert got == datetime(2026, 5, 13, 15, 30, 0, tzinfo=UTC)

    def test_positive_offset_normalizes_to_utc(self):
        got = parse_when("2026-05-13T18:30:00+03:00")
        assert got == datetime(2026, 5, 13, 15, 30, 0, tzinfo=UTC)

    def test_naive_datetime_assumed_utc(self):
        got = parse_when("2026-05-13T15:30:00")
        assert got == datetime(2026, 5, 13, 15, 30, 0, tzinfo=UTC)

    def test_result_is_tz_aware_utc(self):
        got = parse_when("2026-05-13T10:30:00-05:00")
        assert got.tzinfo is not None
        assert got.utcoffset() == timedelta(0)


class TestParseWhenRolling:
    @pytest.mark.parametrize(
        "spec,delta",
        [
            ("7d", timedelta(days=7)),
            ("24h", timedelta(hours=24)),
            ("30m", timedelta(minutes=30)),
            ("0m", timedelta(0)),
            ("1d", timedelta(days=1)),
            ("1h", timedelta(hours=1)),
            ("1m", timedelta(minutes=1)),
            ("365d", timedelta(days=365)),
        ],
    )
    def test_rolling(self, spec: str, delta: timedelta):
        now = datetime(2026, 5, 13, 12, 0, 0, tzinfo=UTC)
        got = parse_when(spec, now=now)
        assert got == now - delta

    def test_now_defaults_to_utc(self):
        # 0m should equal "now" — just verify tz-aware UTC.
        got = parse_when("0m")
        assert got.tzinfo is not None
        assert got.utcoffset() == timedelta(0)

    def test_naive_now_assumed_utc(self):
        naive = datetime(2026, 5, 13, 12, 0, 0)
        got = parse_when("1h", now=naive)
        assert got == datetime(2026, 5, 13, 11, 0, 0, tzinfo=UTC)


class TestParseWhenShortcuts:
    @pytest.mark.parametrize("spec", ["today", "Today", "TODAY", "ToDay"])
    def test_today_case_insensitive(self, spec: str):
        now = datetime(2026, 5, 13, 15, 30, 45, tzinfo=UTC)
        got = parse_when(spec, now=now)
        assert got == datetime(2026, 5, 13, 0, 0, 0, tzinfo=UTC)

    @pytest.mark.parametrize("spec", ["yesterday", "Yesterday", "YESTERDAY"])
    def test_yesterday_case_insensitive(self, spec: str):
        now = datetime(2026, 5, 13, 15, 30, 45, tzinfo=UTC)
        got = parse_when(spec, now=now)
        assert got == datetime(2026, 5, 12, 0, 0, 0, tzinfo=UTC)

    def test_yesterday_across_month_boundary(self):
        now = datetime(2026, 6, 1, 10, 0, 0, tzinfo=UTC)
        got = parse_when("yesterday", now=now)
        assert got == datetime(2026, 5, 31, 0, 0, 0, tzinfo=UTC)

    def test_today_with_non_utc_now_uses_utc_calendar_day(self):
        now = datetime(2026, 5, 13, 23, 0, 0, tzinfo=timezone(timedelta(hours=-5)))
        # That's 2026-05-14T04:00:00 UTC.
        got = parse_when("today", now=now)
        assert got == datetime(2026, 5, 14, 0, 0, 0, tzinfo=UTC)


class TestParseWhenErrors:
    @pytest.mark.parametrize(
        "bad",
        ["", "tomorrow", "7", "7days", "2026/05/13", "abc", "5x", "d7", " 7d", "7d "],
    )
    def test_raises_value_error(self, bad: str):
        with pytest.raises(ValueError) as exc:
            parse_when(bad)
        msg = str(exc.value)
        assert repr(bad) in msg or bad in msg
        assert "expected" in msg

    def test_none_returns_none(self):
        assert parse_when(None) is None
