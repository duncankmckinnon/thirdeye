from __future__ import annotations

import json
from pathlib import Path

import pytest

from thirdeye.usage.store import UsageStore
from thirdeye.usage.types import UsageRow


@pytest.fixture
def session(tmp_path: Path) -> Path:
    sd = tmp_path / "traces" / "claude" / "abc123"
    sd.mkdir(parents=True)
    return sd


def make_row(seq: int, **overrides) -> UsageRow:
    defaults = dict(
        session_id="abc123",
        seq=seq,
        call_id=f"call-{seq}",
        ts=f"2026-05-15T00:00:{seq:02d}.000Z",
        platform="claude",
        provider_name="anthropic",
        response_model="claude-opus-4-7",
        input_tokens=100,
        output_tokens=10,
    )
    defaults.update(overrides)
    return UsageRow(**defaults)


def test_append_writes_jsonl(session: Path) -> None:
    store = UsageStore(session)
    store.append([make_row(0), make_row(1)])
    lines = (session / "usage.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["seq"] == 0


def test_append_is_additive(session: Path) -> None:
    store = UsageStore(session)
    store.append([make_row(0)])
    store.append([make_row(1)])
    rows = list(store.iter_rows())
    assert [r.seq for r in rows] == [0, 1]


def test_append_empty_list_no_op(session: Path) -> None:
    store = UsageStore(session)
    store.append([])
    assert not (session / "usage.jsonl").exists()


def test_append_creates_session_dir(tmp_path: Path) -> None:
    """If the session directory doesn't exist yet, append creates it."""
    sd = tmp_path / "traces" / "claude" / "new"
    assert not sd.exists()
    UsageStore(sd).append([make_row(0)])
    assert sd.is_dir()
    assert (sd / "usage.jsonl").exists()


def test_iter_rows_handles_missing_file(session: Path) -> None:
    store = UsageStore(session)
    assert list(store.iter_rows()) == []


def test_iter_rows_skips_malformed_lines(session: Path) -> None:
    (session / "usage.jsonl").write_text(
        json.dumps(make_row(0).to_dict()) + "\n"
        "this is not valid json\n" + json.dumps(make_row(1).to_dict()) + "\n"
        "\n"  # blank line
    )
    rows = list(UsageStore(session).iter_rows())
    assert [r.seq for r in rows] == [0, 1]


def test_iter_rows_returns_duplicates_raw(session: Path) -> None:
    """The sidecar is a faithful raw mirror: iter_rows must NOT deduplicate.

    If dedup ever leaks into the writer or store, this fails — the collapse of
    duplicate call_ids belongs only in read.iter_calls.
    """
    store = UsageStore(session)
    store.append([make_row(0, call_id="dup") for _ in range(6)])
    rows = list(store.iter_rows())
    assert len(rows) == 6
    assert all(r.call_id == "dup" for r in rows)


def test_read_state_missing_returns_empty(session: Path) -> None:
    assert UsageStore(session).read_state() == {}


def test_write_state_round_trip(session: Path) -> None:
    store = UsageStore(session)
    store.write_state(transcript_offset=42, last_seq=7)
    assert store.read_state() == {"transcript_offset": 42, "last_seq": 7}


def test_write_state_merges_not_replaces(session: Path) -> None:
    store = UsageStore(session)
    store.write_state(transcript_offset=1, last_seq=5)
    store.write_state(transcript_offset=100)
    state = store.read_state()
    assert state["transcript_offset"] == 100
    assert state["last_seq"] == 5


def test_write_state_atomic_on_rename_failure(
    session: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If os.replace raises mid-write, the previous state must survive intact."""
    store = UsageStore(session)
    store.write_state(transcript_offset=1)

    import thirdeye.usage.store as store_mod

    def boom(src, dst):
        raise OSError("simulated crash")

    monkeypatch.setattr(store_mod.os, "replace", boom)
    with pytest.raises(OSError):
        store.write_state(transcript_offset=999)
    # Restore so test cleanup works
    monkeypatch.undo()
    assert store.read_state() == {"transcript_offset": 1}


def test_read_state_handles_malformed_json(session: Path) -> None:
    (session / "usage.state.json").write_text("{not json")
    assert UsageStore(session).read_state() == {}
