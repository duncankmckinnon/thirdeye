from __future__ import annotations

import json
from pathlib import Path

import pytest

from thirdeye.usage.read import call_totals, iter_calls
from thirdeye.usage.store import UsageStore
from thirdeye.usage.types import UsageRow


@pytest.fixture
def session(tmp_path: Path) -> Path:
    sd = tmp_path / "traces" / "claude" / "abc123"
    sd.mkdir(parents=True)
    return sd


def make_row(call_id: str, seq: int = 0, **overrides) -> UsageRow:
    defaults = dict(
        session_id="abc123",
        seq=seq,
        call_id=call_id,
        ts=f"2026-05-15T00:00:{seq:02d}.000Z",
        platform="claude",
        provider_name="anthropic",
        response_model="claude-opus-4-7",
        input_tokens=100,
        output_tokens=10,
    )
    defaults.update(overrides)
    return UsageRow(**defaults)


def test_collapses_duplicate_call_id(session: Path) -> None:
    store = UsageStore(session)
    store.append([make_row("call-a") for _ in range(6)])
    rows = list(iter_calls(session))
    assert len(rows) == 1
    assert rows[0].call_id == "call-a"


def test_distinct_calls_in_first_appearance_order(session: Path) -> None:
    store = UsageStore(session)
    store.append([make_row("a", 0), make_row("b", 1), make_row("c", 2)])
    rows = list(iter_calls(session))
    assert [r.call_id for r in rows] == ["a", "b", "c"]


def test_first_appearance_order_survives_interleaved_duplicates(session: Path) -> None:
    store = UsageStore(session)
    store.append(
        [
            make_row("a", 0),
            make_row("b", 1),
            make_row("a", 2),  # duplicate of the first call, seen later
            make_row("c", 3),
        ]
    )
    rows = list(iter_calls(session))
    assert [r.call_id for r in rows] == ["a", "b", "c"]


def test_last_wins(session: Path) -> None:
    store = UsageStore(session)
    store.append([make_row("a", 0, output_tokens=10), make_row("a", 1, output_tokens=99)])
    rows = list(iter_calls(session))
    assert len(rows) == 1
    assert rows[0].output_tokens == 99


def test_missing_sidecar_yields_nothing(session: Path) -> None:
    assert list(iter_calls(session)) == []


def test_empty_sidecar_yields_nothing(session: Path) -> None:
    (session / "usage.jsonl").write_text("")
    assert list(iter_calls(session)) == []


def test_malformed_line_skipped_valid_rows_survive(session: Path) -> None:
    (session / "usage.jsonl").write_text(
        json.dumps(make_row("a", 0).to_dict())
        + "\n"
        + "not valid json\n"
        + json.dumps(make_row("b", 1).to_dict())
        + "\n"
    )
    rows = list(iter_calls(session))
    assert [r.call_id for r in rows] == ["a", "b"]


def test_call_totals_sums_deduplicated_rows(session: Path) -> None:
    store = UsageStore(session)
    store.append([make_row("a", input_tokens=100, output_tokens=10) for _ in range(6)])
    assert call_totals(iter_calls(session)) == (100, 10)


def test_call_totals_sums_distinct_rows(session: Path) -> None:
    rows = [
        make_row("a", input_tokens=100, output_tokens=10),
        make_row("b", input_tokens=50, output_tokens=5),
    ]
    assert call_totals(rows) == (150, 15)


def test_optional_attributes_absent_round_trip_as_none(session: Path) -> None:
    store = UsageStore(session)
    store.append([make_row("a")])
    (row,) = list(iter_calls(session))
    assert row.cache_read_input_tokens is None
    assert row.cache_creation_input_tokens is None
    assert row.reasoning_output_tokens is None
