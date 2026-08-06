from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from thirdeye.commands.usage import usage
from thirdeye.paths import (
    events_path,
    index_path,
    meta_path,
    session_dir,
    usage_db_path,
    usage_jsonl_path,
    usage_log_path,
    usage_state_path,
)
from thirdeye.usage.types import UsageRow


def _seed(home: Path, platform: str, sid: str, rows: list[UsageRow]) -> Path:
    """Write `rows` to a session's usage.jsonl in the faithful sidecar format."""
    sd = session_dir(home, platform, sid)
    sd.mkdir(parents=True, exist_ok=True)
    with usage_jsonl_path(sd).open("w") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict()) + "\n")
    return sd


def _row(**overrides) -> UsageRow:
    base = dict(
        session_id="abc123",
        seq=0,
        call_id="c1",
        ts="2026-05-10T00:00:00Z",
        platform="claude",
        provider_name="anthropic",
        response_model="claude-opus-4-7",
        input_tokens=100,
        output_tokens=10,
    )
    base.update(overrides)
    return UsageRow(**base)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point THIRDEYE_HOME at tmp_path and seed two sessions with usage data."""
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))

    _seed(
        tmp_path,
        "claude",
        "abc123",
        [
            _row(
                seq=0,
                call_id="c1",
                input_tokens=100,
                output_tokens=10,
                cache_read_input_tokens=50,
                cache_creation_input_tokens=5,
            ),
            _row(
                seq=5,
                call_id="c2",
                ts="2026-05-10T00:00:05Z",
                input_tokens=200,
                output_tokens=20,
                cache_read_input_tokens=0,  # reported as none → renders "0"
                # cache_creation absent → renders "-"
            ),
            _row(
                seq=9,
                call_id="c3",
                ts="2026-05-10T00:00:09Z",
                input_tokens=5,
                output_tokens=500,
            ),
            # Duplicate of c1 (identical values): the sidecar is a raw mirror,
            # so the same call_id may appear twice. iter_calls collapses it.
            _row(
                seq=0,
                call_id="c1",
                input_tokens=100,
                output_tokens=10,
                cache_read_input_tokens=50,
                cache_creation_input_tokens=5,
            ),
        ],
    )
    _seed(
        tmp_path,
        "codex",
        "def456",
        [
            _row(
                session_id="def456",
                seq=0,
                call_id="d1",
                ts="2026-05-12T00:00:00Z",
                platform="codex",
                provider_name="openai",
                response_model="gpt-5",
                input_tokens=9582,
                output_tokens=1,
            ),
        ],
    )
    return tmp_path


def _make_db(home: Path, n_rows: int) -> None:
    """Create a minimal usage.db with `n_rows` rows so reset can count/delete."""
    db = usage_db_path(home)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE usage (session_id TEXT, call_id TEXT)")
    conn.executemany(
        "INSERT INTO usage (session_id, call_id) VALUES (?, ?)",
        [("abc123", f"c{i}") for i in range(n_rows)],
    )
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------
# rollup + per-session views
# --------------------------------------------------------------------------


def test_rollup_default(home: Path) -> None:
    result = CliRunner().invoke(usage, [], catch_exceptions=False)
    assert result.exit_code == 0
    assert "abc123" in result.output
    assert "def456" in result.output


def test_rollup_filter_by_platform(home: Path) -> None:
    result = CliRunner().invoke(usage, ["--platform", "codex"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "def456" in result.output
    assert "abc123" not in result.output


def test_rollup_harness_alias(home: Path) -> None:
    result = CliRunner().invoke(usage, ["--harness", "claude"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "abc123" in result.output
    assert "def456" not in result.output


def test_rollup_top_n(home: Path) -> None:
    result = CliRunner().invoke(usage, ["--top", "1"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "def456" in result.output  # 9583 > 835
    assert "abc123" not in result.output


def test_rollup_json_uses_dotted_keys(home: Path) -> None:
    result = CliRunner().invoke(usage, ["--json"], catch_exceptions=False)
    assert result.exit_code == 0
    rows = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    by_sid = {r["session_id"]: r for r in rows}
    assert by_sid["abc123"]["gen_ai.usage.input_tokens"] == 305  # 100 + 200 + 5
    assert by_sid["abc123"]["gen_ai.usage.output_tokens"] == 530  # 10 + 20 + 500
    # No underscored SQL token names on the JSON surface.
    for r in rows:
        assert "input_tokens" not in r
        assert "output_tokens" not in r
        assert "total_tokens" not in r


def test_per_session_view_shows_model_and_cache(home: Path) -> None:
    result = CliRunner().invoke(usage, ["abc"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "claude-opus-4-7" in result.output
    assert "MODEL" in result.output

    # Find the data line for c2: cache_read reported as 0, cache_creation absent.
    # The row order is unspecified here; locate by its distinctive input value.
    lines = [ln for ln in result.output.splitlines() if "claude-opus-4-7" in ln]
    c2_line = next(ln for ln in lines if "200" in ln and "20" in ln)
    # cache_read=0 renders "0"; cache_creation=None renders "-".
    assert "0" in c2_line
    assert "-" in c2_line


def test_absent_cache_distinct_from_zero(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # One session, one call: cache_read reported as 0, cache_creation absent.
    _seed(
        home,
        "claude",
        "zero1",
        [
            _row(
                session_id="zero1",
                call_id="z1",
                cache_read_input_tokens=0,
                # cache_creation absent
            )
        ],
    )
    result = CliRunner().invoke(usage, ["zero1"], catch_exceptions=False)
    assert result.exit_code == 0
    data = next(ln for ln in result.output.splitlines() if "claude-opus-4-7" in ln)
    cells = data.split()
    # Layout: SEQ TS MODEL INPUT OUTPUT CACHE_R CACHE_C TOTAL
    # CACHE_R is the 6th cell (reported 0), CACHE_C the 7th (absent -> "-").
    assert cells[-3] == "0"
    assert cells[-2] == "-"


def test_per_session_json_dotted_keys(home: Path) -> None:
    result = CliRunner().invoke(usage, ["abc", "--json"], catch_exceptions=False)
    assert result.exit_code == 0
    rows = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    r0 = rows[0]
    assert "gen_ai.usage.input_tokens" in r0
    assert "gen_ai.response.model" in r0
    assert "gen_ai.conversation.id" in r0
    # Underscored SQL column names must not appear.
    assert "input_tokens" not in r0
    assert "output_tokens" not in r0
    assert "total_tokens" not in r0
    assert "model" not in r0


def test_duplicate_call_id_reported_once(home: Path) -> None:
    result = CliRunner().invoke(usage, ["abc", "--json"], catch_exceptions=False)
    assert result.exit_code == 0
    rows = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    call_ids = [r["call_id"] for r in rows]
    # c1 is written twice in the sidecar but must collapse to one row.
    assert sorted(call_ids) == ["c1", "c2", "c3"]


@pytest.mark.parametrize(
    "sort,expected",
    [
        ("total", ["c3", "c2", "c1"]),  # 505, 220, 110
        ("input", ["c2", "c1", "c3"]),  # 200, 100, 5
        ("output", ["c3", "c2", "c1"]),  # 500, 20, 10
    ],
)
def test_per_session_sort(home: Path, sort: str, expected: list[str]) -> None:
    result = CliRunner().invoke(usage, ["abc", "--sort", sort, "--json"], catch_exceptions=False)
    assert result.exit_code == 0
    rows = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    assert [r["call_id"] for r in rows] == expected


def test_sort_total_matches_input_plus_output(home: Path) -> None:
    result = CliRunner().invoke(usage, ["abc", "--sort", "total", "--json"], catch_exceptions=False)
    rows = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    totals = [r["gen_ai.usage.input_tokens"] + r["gen_ai.usage.output_tokens"] for r in rows]
    assert totals == sorted(totals, reverse=True)


# --------------------------------------------------------------------------
# reset
# --------------------------------------------------------------------------


def test_reset_requires_yes(home: Path) -> None:
    _make_db(home, 5)
    result = CliRunner().invoke(usage, ["reset"])
    assert result.exit_code != 0
    assert "--yes" in result.output
    # Nothing deleted.
    assert usage_jsonl_path(session_dir(home, "claude", "abc123")).exists()
    assert usage_jsonl_path(session_dir(home, "codex", "def456")).exists()
    assert usage_db_path(home).exists()


def test_reset_yes_removes_sidecars_and_db(home: Path) -> None:
    # Add a state file too.
    sd = session_dir(home, "claude", "abc123")
    usage_state_path(sd).write_text("{}")
    _make_db(home, 5)

    result = CliRunner().invoke(usage, ["reset", "--yes"], catch_exceptions=False)
    assert result.exit_code == 0

    assert not usage_jsonl_path(session_dir(home, "claude", "abc123")).exists()
    assert not usage_jsonl_path(session_dir(home, "codex", "def456")).exists()
    assert not usage_state_path(sd).exists()
    assert not usage_db_path(home).exists()
    # Counts printed.
    assert "5" in result.output  # db rows
    assert "sidecar files" in result.output


def test_reset_yes_preserves_event_store(home: Path) -> None:
    sd = session_dir(home, "claude", "abc123")
    events_path(sd).write_text("event-log")
    index_path(sd).write_text("index")
    meta_path(sd).write_text("meta")

    result = CliRunner().invoke(usage, ["reset", "--yes"], catch_exceptions=False)
    assert result.exit_code == 0

    assert events_path(sd).read_text() == "event-log"
    assert index_path(sd).read_text() == "index"
    assert meta_path(sd).read_text() == "meta"


def test_reset_clean_home_reports_zeros(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    result = CliRunner().invoke(usage, ["reset", "--yes"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "sidecar files:     0" in result.output
    assert "sessions affected: 0" in result.output
    assert "usage.db rows:     0" in result.output


# --------------------------------------------------------------------------
# errors (format-independent, retained coverage)
# --------------------------------------------------------------------------


def test_errors_subcommand_no_log(home: Path) -> None:
    result = CliRunner().invoke(usage, ["errors"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "No usage errors" in result.output


def _write_log(home: Path, entries: list[dict]) -> None:
    log = usage_log_path(home)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _entry(**overrides) -> dict:
    base = {
        "ts": "2026-05-12T00:00:00Z",
        "level": "warn",
        "platform": "claude",
        "session_id": "abc123",
        "phase": "parse_transcript",
        "source_path": "/x",
        "error_class": "FileNotFoundError",
        "message": "gone",
        "traceback": "",
    }
    base.update(overrides)
    return base


def test_errors_filter_by_platform(home: Path) -> None:
    _write_log(
        home,
        [
            _entry(platform="claude", phase="parse_transcript"),
            _entry(platform="codex", phase="parse_rollout", session_id="def456"),
        ],
    )
    result = CliRunner().invoke(
        usage, ["errors", "--platform", "codex", "--json"], catch_exceptions=False
    )
    assert result.exit_code == 0
    lines = [json.loads(ln) for ln in result.output.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0]["platform"] == "codex"


def test_errors_respects_tail_n(home: Path) -> None:
    _write_log(home, [_entry(phase=f"p{i}") for i in range(5)])
    result = CliRunner().invoke(usage, ["errors", "-n", "2", "--json"], catch_exceptions=False)
    assert result.exit_code == 0
    phases = [json.loads(ln)["phase"] for ln in result.output.splitlines() if ln.strip()]
    assert phases == ["p3", "p4"]
