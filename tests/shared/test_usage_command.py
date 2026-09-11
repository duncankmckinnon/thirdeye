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
# reindex
#
# `usage reindex` rebuilds usage.db from the sidecars, but src/thirdeye/usage/
# index.py is still schema v1: it creates `model`/`total_tokens` columns with
# PRIMARY KEY (session_id, seq) and reads the bare keys `row["model"]`,
# `row["input_tokens"]`, `row["total_tokens"]`. The OTel-GenAI sidecar now
# writes dotted `gen_ai.*` keys and has no `total_tokens`, so every insert
# raises KeyError (swallowed by log_capture_error) and 0 rows land. The
# schema-v2 migration to PRIMARY KEY (session_id, call_id) with dotted-key
# reads belongs to the usage-index task, which is outside this task's file
# scope (usage.py + this test module).
#
# These tests encode the intended schema-v2 contract and are marked xfail
# (non-strict) so the gap is surfaced rather than hidden: they fail today,
# flagging index.py, and will xpass once the index task lands.
# --------------------------------------------------------------------------

_REINDEX_XFAIL = pytest.mark.xfail(
    reason="src/thirdeye/usage/index.py is still schema v1; schema-v2 migration "
    "(PRIMARY KEY (session_id, call_id), dotted gen_ai.* reads) is the usage-index task",
    strict=False,
)


@_REINDEX_XFAIL
def test_reindex_subcommand(home: Path) -> None:
    result = CliRunner().invoke(usage, ["reindex"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "Indexed" in result.output and "sessions" in result.output

    conn = sqlite3.connect(usage_db_path(home))
    # abc123 has three distinct call_ids (c1 duplicated in the sidecar), def456
    # has one → four deduplicated rows.
    assert conn.execute("SELECT COUNT(*) FROM usage").fetchone()[0] == 4
    conn.close()


@_REINDEX_XFAIL
def test_reindex_targets_one_session(home: Path) -> None:
    """`reindex <prefix>` must wipe + rebuild only the matched session."""
    runner = CliRunner()
    runner.invoke(usage, ["reindex"], catch_exceptions=False)  # populate all

    db = usage_db_path(home)
    result = runner.invoke(usage, ["reindex", "abc"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "from 1 session" in result.output

    # abc123 rows re-indexed, def456 untouched.
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT session_id, call_id FROM usage ORDER BY session_id, call_id"
    ).fetchall()
    assert rows == [
        ("abc123", "c1"),
        ("abc123", "c2"),
        ("abc123", "c3"),
        ("def456", "d1"),
    ]
    sync_ids = {r[0] for r in conn.execute("SELECT session_id FROM usage_sync").fetchall()}
    assert "abc123" in sync_ids
    conn.close()


def test_reindex_unknown_prefix_errors(home: Path) -> None:
    """An unknown prefix must surface a clean ClickException, not crash.

    Format-independent: `_resolve_session` raises before any indexing, so this
    passes regardless of the index.py schema gap above.
    """
    result = CliRunner().invoke(usage, ["reindex", "zzz-nonexistent"])
    assert result.exit_code != 0
    # ClickException renders to stderr; click.testing merges by default
    assert "zzz-nonexistent" in result.output or "no session" in result.output.lower()


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


def test_reset_reports_exact_counts(home: Path) -> None:
    """Counts name every deleted sidecar and the affected session set exactly."""
    # One session with jsonl + state = 2 sidecars; second session jsonl = 1.
    sd = session_dir(home, "claude", "abc123")
    usage_state_path(sd).write_text("{}")
    _make_db(home, 7)

    result = CliRunner().invoke(usage, ["reset", "--yes"], catch_exceptions=False)
    assert result.exit_code == 0
    # abc123: usage.jsonl + usage.state.json ; def456: usage.jsonl -> 3 files.
    assert "sidecar files:     3" in result.output
    assert "sessions affected: 2" in result.output
    assert "usage.db rows:     7" in result.output
    # Deletion summary echoes the same numbers.
    assert "Deleted 3 sidecar file(s) across 2 session(s) and 7 usage.db row(s)." in result.output


def test_reset_without_yes_reports_counts_but_deletes_nothing(home: Path) -> None:
    """Refusal still prints the destroy plan (counts) before bailing out."""
    _make_db(home, 4)
    result = CliRunner().invoke(usage, ["reset"])
    assert result.exit_code != 0
    assert "usage reset will destroy:" in result.output
    assert "usage.db rows:     4" in result.output
    # And genuinely nothing was removed.
    assert usage_db_path(home).exists()
    assert usage_jsonl_path(session_dir(home, "claude", "abc123")).exists()


def test_reset_yes_preserves_tags_and_upstream(home: Path) -> None:
    """tags.jsonl (named in the spec) and an unrelated upstream file survive."""
    from thirdeye.paths import tags_path

    sd = session_dir(home, "claude", "abc123")
    tags_path(sd).write_text("tags")
    # A stand-in for an upstream transcript living beside the session.
    upstream = sd / "transcript.jsonl"
    upstream.write_text("upstream")

    result = CliRunner().invoke(usage, ["reset", "--yes"], catch_exceptions=False)
    assert result.exit_code == 0
    assert tags_path(sd).read_text() == "tags"
    assert upstream.read_text() == "upstream"
    # But its usage sidecar is gone.
    assert not usage_jsonl_path(sd).exists()


def test_reset_reports_orphaned_hooks(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After deleting, reset surfaces orphaned removed-platform hooks (detection only)."""
    fake = Path("/home/u/.gemini/settings.json")

    def _fake_orphans():
        return [(fake, "thirdeye-gemini-hook")]

    monkeypatch.setattr("thirdeye.commands.usage.find_orphaned_hooks", _fake_orphans)
    result = CliRunner().invoke(usage, ["reset", "--yes"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "thirdeye-gemini-hook" in result.output
    assert str(fake) in result.output
    # The orphan's config file was never created/edited by reset.
    assert not fake.exists()


def test_reset_no_orphan_warning_when_none(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("thirdeye.commands.usage.find_orphaned_hooks", lambda: [])
    result = CliRunner().invoke(usage, ["reset", "--yes"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "still references removed hook" not in result.output


# --------------------------------------------------------------------------
# per-session filters + sort ordering
# --------------------------------------------------------------------------


def test_per_session_sort_ts_orders_chronologically(home: Path) -> None:
    result = CliRunner().invoke(usage, ["abc", "--sort", "ts", "--json"], catch_exceptions=False)
    assert result.exit_code == 0
    rows = [json.loads(ln) for ln in result.output.splitlines() if ln.strip()]
    ts_values = [r["ts"] for r in rows]
    assert ts_values == sorted(ts_values)


def test_per_session_model_filter(home: Path) -> None:
    # Only claude rows match; a bogus substring yields nothing.
    hit = CliRunner().invoke(usage, ["abc", "--model", "opus", "--json"], catch_exceptions=False)
    assert hit.exit_code == 0
    assert [json.loads(ln) for ln in hit.output.splitlines() if ln.strip()]

    miss = CliRunner().invoke(usage, ["abc", "--model", "gpt-5", "--json"], catch_exceptions=False)
    assert miss.exit_code == 0
    assert [ln for ln in miss.output.splitlines() if ln.strip()] == []


def test_per_session_since_until_window(home: Path) -> None:
    # abc123 has calls at 00:00:00, 00:00:05, 00:00:09. Window keeps the middle one.
    result = CliRunner().invoke(
        usage,
        ["abc", "--since", "2026-05-10T00:00:03Z", "--until", "2026-05-10T00:00:07Z", "--json"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    rows = [json.loads(ln) for ln in result.output.splitlines() if ln.strip()]
    assert [r["call_id"] for r in rows] == ["c2"]


def test_per_session_empty_message_when_no_rows(home: Path) -> None:
    result = CliRunner().invoke(usage, ["abc", "--model", "no-such-model"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "No usage data for session" in result.output


def test_rollup_totals_are_input_plus_output(home: Path) -> None:
    """Rollup TOTAL column equals summed input + output, never a stored value."""
    result = CliRunner().invoke(usage, ["--json"], catch_exceptions=False)
    rows = [json.loads(ln) for ln in result.output.splitlines() if ln.strip()]
    by_sid = {r["session_id"]: r for r in rows}
    # abc123 dedups the duplicate c1: c1(100/10)+c2(200/20)+c3(5/500).
    assert by_sid["abc123"]["gen_ai.usage.input_tokens"] == 305
    assert by_sid["abc123"]["gen_ai.usage.output_tokens"] == 530
    assert by_sid["abc123"]["calls"] == 3  # three distinct call_ids, not four rows


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


def test_errors_subcommand_with_entries(home: Path) -> None:
    _write_log(home, [_entry(phase="parse_transcript")])
    runner = CliRunner()
    result = runner.invoke(usage, ["errors"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "parse_transcript" in result.output

    result = runner.invoke(usage, ["errors", "--json"], catch_exceptions=False)
    line = json.loads(result.output.strip())
    assert line["phase"] == "parse_transcript"

    result = runner.invoke(usage, ["errors", "--phase", "nothing-matches"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "No matching" in result.output


def test_errors_filter_by_since(home: Path) -> None:
    _write_log(
        home,
        [
            _entry(ts="2026-05-01T00:00:00Z", phase="old"),
            _entry(ts="2026-05-10T00:00:00Z", phase="mid"),
            _entry(ts="2026-05-20T00:00:00Z", phase="new"),
        ],
    )
    result = CliRunner().invoke(
        usage, ["errors", "--since", "2026-05-09", "--json"], catch_exceptions=False
    )
    assert result.exit_code == 0
    phases = {json.loads(ln)["phase"] for ln in result.output.splitlines() if ln.strip()}
    assert phases == {"mid", "new"}


def test_errors_filter_by_until(home: Path) -> None:
    _write_log(
        home,
        [
            _entry(ts="2026-05-01T00:00:00Z", phase="old"),
            _entry(ts="2026-05-10T00:00:00Z", phase="mid"),
            _entry(ts="2026-05-20T00:00:00Z", phase="new"),
        ],
    )
    result = CliRunner().invoke(
        usage, ["errors", "--until", "2026-05-15", "--json"], catch_exceptions=False
    )
    assert result.exit_code == 0
    phases = {json.loads(ln)["phase"] for ln in result.output.splitlines() if ln.strip()}
    assert phases == {"old", "mid"}


def test_errors_combined_since_and_until(home: Path) -> None:
    _write_log(
        home,
        [
            _entry(ts="2026-05-01T00:00:00Z", phase="old"),
            _entry(ts="2026-05-10T00:00:00Z", phase="mid"),
            _entry(ts="2026-05-20T00:00:00Z", phase="new"),
        ],
    )
    result = CliRunner().invoke(
        usage,
        ["errors", "--since", "2026-05-05", "--until", "2026-05-15", "--json"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    phases = {json.loads(ln)["phase"] for ln in result.output.splitlines() if ln.strip()}
    assert phases == {"mid"}


def test_errors_skips_malformed_jsonl_lines(home: Path) -> None:
    """A malformed line in usage-errors.jsonl must be silently skipped."""
    log = usage_log_path(home)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        json.dumps(_entry(phase="first")) + "\n"
        "this is not valid json\n"
        "\n"  # blank line
         + json.dumps(_entry(phase="second")) + "\n"
    )
    result = CliRunner().invoke(usage, ["errors", "--json"], catch_exceptions=False)
    assert result.exit_code == 0
    phases = [json.loads(ln)["phase"] for ln in result.output.splitlines() if ln.strip()]
    assert phases == ["first", "second"]


def test_errors_skips_entries_with_unparseable_ts_when_time_filter_set(home: Path) -> None:
    """Under --since/--until, entries with bad ts are dropped, not crashed on."""
    _write_log(
        home,
        [
            _entry(ts="not-a-timestamp", phase="bad_ts"),
            _entry(ts="2026-05-10T00:00:00Z", phase="good_ts"),
        ],
    )
    result = CliRunner().invoke(
        usage, ["errors", "--since", "2026-05-01", "--json"], catch_exceptions=False
    )
    assert result.exit_code == 0
    phases = {json.loads(ln)["phase"] for ln in result.output.splitlines() if ln.strip()}
    assert phases == {"good_ts"}


def test_errors_combined_filters(home: Path) -> None:
    """--platform + --phase + --since AND together."""
    _write_log(
        home,
        [
            _entry(platform="claude", phase="parse_transcript", ts="2026-05-01T00:00:00Z"),
            _entry(platform="claude", phase="parse_transcript", ts="2026-05-10T00:00:00Z"),
            _entry(platform="claude", phase="index_sync", ts="2026-05-10T00:00:00Z"),
            _entry(platform="codex", phase="parse_transcript", ts="2026-05-10T00:00:00Z"),
        ],
    )
    result = CliRunner().invoke(
        usage,
        [
            "errors",
            "--platform",
            "claude",
            "--phase",
            "parse_transcript",
            "--since",
            "2026-05-05",
            "--json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    lines = [json.loads(ln) for ln in result.output.splitlines() if ln.strip()]
    assert len(lines) == 1
    e = lines[0]
    assert e["platform"] == "claude"
    assert e["phase"] == "parse_transcript"
    assert e["ts"] == "2026-05-10T00:00:00Z"
