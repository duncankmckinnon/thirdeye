from __future__ import annotations

import json
from pathlib import Path

from thirdeye.paths import session_dir, usage_jsonl_path, usage_log_path
from thirdeye.usage.index import SCHEMA_VERSION, UsageIndex
from thirdeye.usage.store import UsageStore
from thirdeye.usage.types import UsageRow


def make_row(seq: int, **overrides) -> UsageRow:
    defaults = dict(
        session_id="abc",
        seq=seq,
        call_id=f"call{seq}",
        ts=f"2026-05-15T00:00:{seq:02d}.000Z",
        platform="claude",
        provider_name="anthropic",
        response_model="claude-opus-4-7",
        input_tokens=100,
        output_tokens=10,
    )
    defaults.update(overrides)
    return UsageRow(**defaults)


def _seed_session(home: Path, platform: str, sid: str, rows: list[UsageRow]) -> Path:
    sd = session_dir(home, platform, sid)
    sd.mkdir(parents=True)
    UsageStore(sd).append(rows)
    return sd


def test_connect_creates_schema(tmp_path: Path) -> None:
    idx = UsageIndex(tmp_path)
    conn = idx.connect()
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    assert "usage" in tables and "usage_sync" in tables
    user_version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert user_version == SCHEMA_VERSION


def test_refresh_cold(tmp_path: Path) -> None:
    sid = "abc"
    _seed_session(tmp_path, "claude", sid, [make_row(0), make_row(1), make_row(2)])
    idx = UsageIndex(tmp_path)
    conn = idx.connect()
    assert idx.refresh(conn) == 3
    seqs = [r[0] for r in conn.execute("SELECT seq FROM usage ORDER BY seq").fetchall()]
    assert seqs == [0, 1, 2]


def test_refresh_incremental(tmp_path: Path) -> None:
    sd = _seed_session(tmp_path, "claude", "abc", [make_row(0)])
    idx = UsageIndex(tmp_path)
    conn = idx.connect()
    assert idx.refresh(conn) == 1
    UsageStore(sd).append([make_row(1)])
    # The grown sidecar is re-read whole via iter_calls and every logical call is
    # upserted, so both calls are (re)written; the new row lands in the table.
    assert idx.refresh(conn) == 2
    seqs = [r[0] for r in conn.execute("SELECT seq FROM usage ORDER BY seq").fetchall()]
    assert seqs == [0, 1]


def test_refresh_steady_state_no_changes(tmp_path: Path) -> None:
    _seed_session(tmp_path, "claude", "abc", [make_row(0)])
    idx = UsageIndex(tmp_path)
    conn = idx.connect()
    idx.refresh(conn)
    assert idx.refresh(conn) == 0  # nothing new


def test_refresh_dedups_on_pk(tmp_path: Path) -> None:
    """Wiping usage_sync should not produce duplicate rows."""
    _seed_session(tmp_path, "claude", "abc", [make_row(0)])
    idx = UsageIndex(tmp_path)
    conn = idx.connect()
    idx.refresh(conn)
    conn.execute("DELETE FROM usage_sync")
    conn.commit()
    idx.refresh(conn)
    n = conn.execute("SELECT COUNT(*) FROM usage").fetchone()[0]
    assert n == 1


def test_refresh_handles_shrunk_sidecar(tmp_path: Path) -> None:
    sd = _seed_session(tmp_path, "claude", "abc", [make_row(0), make_row(1)])
    idx = UsageIndex(tmp_path)
    conn = idx.connect()
    idx.refresh(conn)

    # Truncate the sidecar
    usage_jsonl_path(sd).write_text("")
    idx.refresh(conn)
    assert conn.execute("SELECT COUNT(*) FROM usage").fetchone()[0] == 0
    assert "shrank" in usage_log_path(tmp_path).read_text()


def test_refresh_skips_malformed_line(tmp_path: Path) -> None:
    sd = session_dir(tmp_path, "claude", "abc")
    sd.mkdir(parents=True)
    jsonl = usage_jsonl_path(sd)
    jsonl.write_text(
        json.dumps(make_row(0).to_dict()) + "\n"
        "{this is not valid json\n" + json.dumps(make_row(1).to_dict()) + "\n"
    )
    idx = UsageIndex(tmp_path)
    conn = idx.connect()
    assert idx.refresh(conn) == 2
    assert "malformed" in usage_log_path(tmp_path).read_text()


def test_refresh_handles_empty_sessions_root(tmp_path: Path) -> None:
    """No traces dir at all should not raise."""
    idx = UsageIndex(tmp_path)
    conn = idx.connect()
    assert idx.refresh(conn) == 0


def test_refresh_across_multiple_platforms(tmp_path: Path) -> None:
    _seed_session(tmp_path, "claude", "abc", [make_row(0, platform="claude")])
    _seed_session(
        tmp_path,
        "codex",
        "def",
        [
            make_row(
                0,
                session_id="def",
                platform="codex",
                provider_name="openai",
                response_model="gpt-5.5",
            )
        ],
    )
    idx = UsageIndex(tmp_path)
    conn = idx.connect()
    assert idx.refresh(conn) == 2
    platforms = sorted(r[0] for r in conn.execute("SELECT DISTINCT platform FROM usage").fetchall())
    assert platforms == ["claude", "codex"]


def test_refresh_session_targets_one_session(tmp_path: Path) -> None:
    sd_a = _seed_session(tmp_path, "claude", "aaa", [make_row(0, session_id="aaa")])
    _seed_session(tmp_path, "claude", "bbb", [make_row(0, session_id="bbb")])
    idx = UsageIndex(tmp_path)
    conn = idx.connect()
    n = idx.refresh_session(conn, "aaa", sd_a)
    assert n == 1
    rows = conn.execute("SELECT session_id FROM usage").fetchall()
    assert rows == [("aaa",)]
