"""Behavioral tests for the Copilot CLI transcript reader."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from thirdeye.platforms.copilot.constants import SOURCE_SCHEMA_VERSION
from thirdeye.platforms.copilot.identity import resolve_sources
from thirdeye.platforms.copilot.transcript import discover_transcripts, read_transcript
from thirdeye.platforms.copilot.types import SourcePaths, SourceRecord, SourceSlice

FIXTURES = Path(__file__).parent / "fixtures" / "copilot"
V1_CASES = FIXTURES / "v1-cases"
CLI_FIXTURE = FIXTURES / "cli-1.0.83"
NATIVE_SESSION_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"


def _session_paths(home: Path) -> SourcePaths:
    return resolve_sources(home)


def _write_session(
    home: Path,
    native_id: str,
    *,
    events: str | bytes | None = None,
    workspace: str | None = None,
) -> Path:
    session_dir = home / "session-state" / native_id
    session_dir.mkdir(parents=True, exist_ok=True)
    if events is not None:
        (session_dir / "events.jsonl").write_bytes(
            events if isinstance(events, bytes) else events.encode("utf-8")
        )
    if workspace is not None:
        (session_dir / "workspace.yaml").write_text(workspace, encoding="utf-8")
    return session_dir


def _transcript_records(slice_: SourceSlice) -> list[SourceRecord]:
    return [record for record in slice_["records"] if record["source_kind"] == "transcript"]


def _metadata_records(slice_: SourceSlice) -> list[SourceRecord]:
    return [record for record in slice_["records"] if record["source_kind"] == "metadata"]


def _diagnostic_codes(slice_: SourceSlice) -> set[str]:
    return {item["code"] for item in slice_["diagnostics"]}


def _drain_transcript(
    paths: SourcePaths,
    native_id: str,
    *,
    max_records: int = 1000,
    max_bytes: int = 4_194_304,
) -> tuple[list[SourceRecord], list[dict[str, Any]], SourceSlice]:
    cursor: dict[str, Any] = {}
    records: list[SourceRecord] = []
    diagnostics: list[dict[str, Any]] = []
    last: SourceSlice | None = None
    while True:
        last = read_transcript(
            paths,
            native_id,
            cursor,
            max_records=max_records,
            max_bytes=max_bytes,
        )
        records.extend(last["records"])
        diagnostics.extend(last["diagnostics"])
        cursor = last["next_cursor"]
        if last["exhausted"]:
            break
    assert last is not None
    return records, diagnostics, last


# --- discovery ---


def test_discover_transcripts_finds_sessions_with_events(tmp_path: Path):
    home = tmp_path / "copilot"
    _write_session(home, "session-b", events='{"id":"1"}\n')
    _write_session(home, "session-a", events='{"id":"2"}\n')
    (home / "session-state" / "no-events").mkdir(parents=True)

    assert discover_transcripts(_session_paths(home)) == ["session-a", "session-b"]


def test_discover_transcripts_ignores_unsafe_or_outside_entries(tmp_path: Path):
    home = tmp_path / "copilot"
    root = home / "session-state"
    root.mkdir(parents=True)
    _write_session(home, "valid", events='{"id":"1"}\n')
    (root / "bad/id").mkdir(parents=True)
    (root / "bad/id" / "events.jsonl").write_text('{"id":"x"}\n', encoding="utf-8")
    (root / "..").resolve()  # ensure traversal candidate exists on POSIX

    assert discover_transcripts(_session_paths(home)) == ["valid"]


def test_discover_transcripts_returns_empty_when_root_missing(tmp_path: Path):
    home = tmp_path / "missing-home"
    assert discover_transcripts(_session_paths(home)) == []


def test_read_transcript_rejects_traversal_native_id(tmp_path: Path):
    home = tmp_path / "copilot"
    _write_session(home, "valid", events='{"id":"1"}\n')
    paths = _session_paths(home)
    with pytest.raises(ValueError, match="path separator"):
        read_transcript(paths, "../valid", {})


# --- happy path and metadata ---


def test_read_transcript_preserves_unknown_fields_and_native_event_id(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    fixture = (V1_CASES / "unknown-event-fields.jsonl").read_text(encoding="utf-8")
    _write_session(home, native, events=fixture, workspace="cwd: /fixture/workspace\n")
    paths = _session_paths(home)

    slice_ = read_transcript(paths, native, {})
    transcript = _transcript_records(slice_)[0]
    metadata = _metadata_records(slice_)[0]

    assert transcript["source_id"] == f"{paths['source_key']}/{native}/unknown-1"
    assert transcript["ts"] == "2026-09-10T17:08:24.000Z"
    assert transcript["payload"]["schema_version"] == SOURCE_SCHEMA_VERSION
    assert transcript["payload"]["top_level_future"] == "retain"
    assert transcript["payload"]["data"]["future_field"]["nested"] == [1, True, {"opaque": "retain"}]
    assert transcript["locator"]["native_event_id"] == "unknown-1"
    assert metadata["payload"]["data"]["cwd"] == "/fixture/workspace"
    assert slice_["cwd"] == "/fixture/workspace"


def test_read_transcript_uses_workspace_path_fallback(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    _write_session(
        home,
        native,
        events='{"id":"1","type":"user.message"}\n',
        workspace="workspacePath: /from/workspacePath\n",
    )
    slice_ = read_transcript(_session_paths(home), native, {})
    assert slice_["cwd"] == "/from/workspacePath"


def test_read_transcript_missing_event_id_uses_lower_confidence_locator(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    fixture = (V1_CASES / "missing-event-id.jsonl").read_text(encoding="utf-8")
    _write_session(home, native, events=fixture)
    paths = _session_paths(home)

    record = _transcript_records(read_transcript(paths, native, {}))[0]
    assert record["source_id"].startswith(f"{paths['source_key']}/{native}/")
    assert "native_event_id" not in record["locator"]
    assert record["locator"]["identity_confidence"] == "lower"
    assert "content_digest" in record["locator"]
    assert record["locator"]["byte_offset"] == 0


# --- malformed and invalid complete lines ---


def test_read_transcript_invalid_utf8_complete_line_is_retained(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    bad = b"\xffnot-utf8\n"
    _write_session(home, native, events=bad)
    slice_ = read_transcript(_session_paths(home), native, {})

    record = _transcript_records(slice_)[0]
    assert record["payload"]["malformed"] == "invalid_utf8"
    assert "raw_bytes_base64" in record["payload"]
    assert "transcript_invalid_utf8" in _diagnostic_codes(slice_)
    assert slice_["exhausted"] is True


def test_read_transcript_invalid_json_complete_line_is_retained(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    _write_session(home, native, events="not json at all\n")
    slice_ = read_transcript(_session_paths(home), native, {})

    record = _transcript_records(slice_)[0]
    assert record["payload"]["malformed"] == "invalid_json"
    assert record["payload"]["raw_line"] == "not json at all"
    assert "transcript_invalid_json" in _diagnostic_codes(slice_)


def test_read_transcript_non_object_json_complete_line_is_retained(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    _write_session(home, native, events='["array"]\n')
    slice_ = read_transcript(_session_paths(home), native, {})

    record = _transcript_records(slice_)[0]
    assert record["payload"]["malformed"] == "non_object_json"
    assert record["payload"]["raw_value"] == ["array"]
    assert "transcript_non_object" in _diagnostic_codes(slice_)


def test_read_transcript_invalid_timestamp_emits_diagnostic_but_keeps_record(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    _write_session(home, native, events='{"id":"1","timestamp":"not-a-date"}\n')
    slice_ = read_transcript(_session_paths(home), native, {})

    record = _transcript_records(slice_)[0]
    assert record["ts"] is None
    assert record["payload"]["timestamp"] == "not-a-date"
    assert "transcript_invalid_timestamp" in _diagnostic_codes(slice_)


# --- partial trailing lines ---


def test_read_transcript_trailing_json_fixture_treats_unterminated_line_as_malformed(tmp_path: Path):
    """The v1 trailing-json fixture ends with a newline; it is a complete physical line."""

    home = tmp_path / "copilot"
    native = "session-a"
    fixture = (V1_CASES / "trailing-json.jsonl").read_bytes()
    _write_session(home, native, events=fixture)
    paths = _session_paths(home)

    slice_ = read_transcript(paths, native, {})
    records = _transcript_records(slice_)
    assert len(records) == 2
    assert records[0]["payload"]["id"] == "complete-1"
    assert records[1]["payload"]["malformed"] == "invalid_json"
    assert "transcript_invalid_json" in _diagnostic_codes(slice_)
    assert slice_["exhausted"] is True


def test_read_transcript_defers_physical_line_without_newline(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    complete = b'{"id":"complete-1","type":"known"}\n'
    partial = b'{"id":"incomplete-2","type":"unterminated"'
    _write_session(home, native, events=complete + partial)
    paths = _session_paths(home)

    first = read_transcript(paths, native, {})
    assert len(_transcript_records(first)) == 1
    assert first["next_cursor"]["byte_offset"] == len(complete)
    assert first["exhausted"] is False


def test_read_transcript_replacement_replays_deferred_partial_line(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    complete = b'{"id":"complete-1","type":"known"}\n'
    partial = b'{"id":"incomplete-2","type":"unterminated"'
    finished = complete + b'{"id":"incomplete-2","type":"now-complete"}\n'
    path = _write_session(home, native, events=complete + partial)
    paths = _session_paths(home)

    first = read_transcript(paths, native, {})
    event_path = path / "events.jsonl"
    event_path.unlink()
    event_path.write_bytes(finished)

    second = read_transcript(paths, native, first["next_cursor"])
    assert "transcript_replaced" in _diagnostic_codes(second)
    ids = [record["payload"].get("id") for record in _transcript_records(second)]
    assert ids == ["complete-1", "incomplete-2"]
    assert second["exhausted"] is True


def test_read_transcript_defers_trailing_incomplete_utf8(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    raw = bytes.fromhex((V1_CASES / "trailing-utf8.hex").read_text(encoding="utf-8").strip())
    _write_session(home, native, events=raw)
    paths = _session_paths(home)

    first = read_transcript(paths, native, {})
    assert len(_transcript_records(first)) == 1
    assert first["exhausted"] is False
    assert first["next_cursor"]["byte_offset"] == raw.index(b"\n") + 1


def test_read_transcript_completes_deferred_utf8_after_replacement(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    raw = bytes.fromhex((V1_CASES / "trailing-utf8.hex").read_text(encoding="utf-8").strip())
    path = _write_session(home, native, events=raw)
    paths = _session_paths(home)

    first = read_transcript(paths, native, {})
    assert len(_transcript_records(first)) == 1
    assert first["exhausted"] is False

    completed = raw + b"\xac\n"
    replacement = path / "events.jsonl"
    replacement.unlink()
    replacement.write_bytes(completed)

    second = read_transcript(paths, native, first["next_cursor"])
    assert "transcript_replaced" in _diagnostic_codes(second)
    transcript = _transcript_records(second)
    assert len(transcript) == 2
    assert transcript[1]["payload"]["malformed"] == "invalid_json"
    assert "transcript_invalid_json" in _diagnostic_codes(second)


# --- bounds and snapshot-end behavior ---


def test_read_transcript_respects_max_records_and_max_bytes(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    lines = [json.dumps({"id": f"event-{index}", "payload": "x" * 40}) for index in range(4)]
    _write_session(home, native, events="\n".join(lines) + "\n")
    paths = _session_paths(home)

    by_count = read_transcript(paths, native, {}, max_records=2)
    assert len(_transcript_records(by_count)) == 2
    assert by_count["exhausted"] is False

    by_bytes = read_transcript(paths, native, {}, max_records=100, max_bytes=120)
    assert len(_transcript_records(by_bytes)) == 1
    assert by_bytes["exhausted"] is False


def test_read_transcript_accepts_one_oversize_complete_record(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    big = json.dumps({"id": "big", "payload": "x" * 256})
    _write_session(home, native, events=f"{big}\n")
    slice_ = read_transcript(_session_paths(home), native, {}, max_bytes=32)

    assert len(_transcript_records(slice_)) == 1
    assert "transcript_record_oversize" in _diagnostic_codes(slice_)
    assert slice_["exhausted"] is True


def test_read_transcript_snapshot_end_drains_initial_file_without_later_appends(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    path = _write_session(
        home,
        native,
        events='{"id":"first"}\n{"id":"second"}\n',
    )
    paths = _session_paths(home)

    first = read_transcript(paths, native, {}, max_records=1)
    assert [record["payload"]["id"] for record in _transcript_records(first)] == ["first"]
    snapshot_end = first["next_cursor"]["snapshot_end"]

    (path / "events.jsonl").open("a", encoding="utf-8").write('{"id":"third"}\n')

    second = read_transcript(paths, native, first["next_cursor"], max_records=10)
    assert second["next_cursor"]["snapshot_end"] == snapshot_end
    assert [record["payload"]["id"] for record in _transcript_records(second)] == ["second"]
    assert second["exhausted"] is True

    third = read_transcript(paths, native, second["next_cursor"])
    assert [record["payload"]["id"] for record in _transcript_records(third)] == ["third"]
    assert third["exhausted"] is True


def test_read_transcript_opens_new_snapshot_for_appended_partial_line(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    path = _write_session(home, native, events='{"id":"first"}\n')
    paths = _session_paths(home)
    first = read_transcript(paths, native, {})
    assert first["exhausted"] is True

    with (path / "events.jsonl").open("ab") as stream:
        stream.write(b'{"id":"partial-without-newline"')

    second = read_transcript(paths, native, first["next_cursor"])
    assert _transcript_records(second) == []
    assert second["next_cursor"]["byte_offset"] == first["next_cursor"]["byte_offset"]
    assert second["exhausted"] is False


def test_read_transcript_appended_newline_completes_deferred_partial_line(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    path = _write_session(home, native, events='{"id":"first"}\n')
    paths = _session_paths(home)
    first = read_transcript(paths, native, {})
    assert first["exhausted"] is True

    with (path / "events.jsonl").open("ab") as stream:
        stream.write(b'{"id":"second","type":"appended"')

    second = read_transcript(paths, native, first["next_cursor"])
    assert _transcript_records(second) == []
    assert second["exhausted"] is False

    with (path / "events.jsonl").open("ab") as stream:
        stream.write(b'}\n')

    third = read_transcript(paths, native, second["next_cursor"])
    assert [record["payload"]["id"] for record in _transcript_records(third)] == ["second"]
    assert third["exhausted"] is True


def test_read_transcript_reads_appended_complete_line_in_fresh_snapshot(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    path = _write_session(home, native, events='{"id":"first"}\n')
    paths = _session_paths(home)
    first = read_transcript(paths, native, {})
    assert first["exhausted"] is True

    with (path / "events.jsonl").open("ab") as stream:
        stream.write(b'{"id":"second"}\n')

    second = read_transcript(paths, native, first["next_cursor"])
    assert [record["payload"]["id"] for record in _transcript_records(second)] == ["second"]
    assert second["exhausted"] is True


# --- replacement, truncation, and cursor recovery ---


def test_read_transcript_replays_after_file_replacement(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    path = _write_session(home, native, events='{"id":"old"}\n')
    paths = _session_paths(home)

    first = read_transcript(paths, native, {})
    assert first["exhausted"] is True

    replacement = path / "events.jsonl"
    replacement.unlink()
    replacement.write_text('{"id":"new"}\n', encoding="utf-8")

    second = read_transcript(paths, native, first["next_cursor"])
    assert "transcript_replaced" in _diagnostic_codes(second)
    assert [record["payload"]["id"] for record in _transcript_records(second)] == ["new"]
    assert second["next_cursor"]["byte_offset"] == len('{"id":"new"}\n')


def test_read_transcript_replays_after_truncation(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    path = _write_session(
        home,
        native,
        events='{"id":"first"}\n{"id":"second"}\n',
    )
    paths = _session_paths(home)
    first = read_transcript(paths, native, {}, max_records=1)
    assert first["next_cursor"]["byte_offset"] > 0

    event_path = path / "events.jsonl"
    event_path.write_text('{"id":"shorter"}\n', encoding="utf-8")

    second = read_transcript(paths, native, first["next_cursor"])
    assert "transcript_replaced" in _diagnostic_codes(second)
    assert [record["payload"]["id"] for record in _transcript_records(second)] == ["shorter"]


def test_read_transcript_invalid_cursor_replays_from_zero(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    _write_session(home, native, events='{"id":"one"}\n')
    paths = _session_paths(home)

    slice_ = read_transcript(paths, native, {"byte_offset": -5})
    assert "transcript_cursor_invalid" in _diagnostic_codes(slice_)
    assert [record["payload"]["id"] for record in _transcript_records(slice_)] == ["one"]


def test_read_transcript_repeatable_source_ids_after_replacement(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    line = '{"id":"stable","type":"user.message"}\n'
    path = _write_session(home, native, events=line)
    paths = _session_paths(home)

    first_id = _transcript_records(read_transcript(paths, native, {}))[0]["source_id"]
    (path / "events.jsonl").write_text(line, encoding="utf-8")
    second_id = _transcript_records(read_transcript(paths, native, {}))[0]["source_id"]
    assert first_id == second_id


def test_read_transcript_inplace_rewrite_before_continuity_window_replays(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    original = b'{"id":"orig"}\n'
    rewritten = b'{"id":"edit"}\n'
    assert len(original) == len(rewritten)
    padding = b"".join(json.dumps({"id": f"pad-{index:03d}", "body": "x" * 64}).encode() + b"\n" for index in range(80))
    path = _write_session(home, native, events=original + padding)
    paths = _session_paths(home)

    first = read_transcript(paths, native, {})
    assert first["next_cursor"]["byte_offset"] > 4096
    assert [record["payload"]["id"] for record in _transcript_records(first)[:1]] == ["orig"]

    with (path / "events.jsonl").open("r+b") as stream:
        stream.seek(0)
        stream.write(rewritten)

    second = read_transcript(paths, native, first["next_cursor"])
    assert "transcript_replaced" in _diagnostic_codes(second)
    assert [record["payload"]["id"] for record in _transcript_records(second)[:1]] == ["edit"]


# --- unavailable source and workspace errors ---


def test_read_transcript_missing_events_is_not_exhausted(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    (home / "session-state" / native).mkdir(parents=True)
    slice_ = read_transcript(_session_paths(home), native, {})

    assert slice_["records"] == []
    assert slice_["exhausted"] is False
    assert "transcript_unavailable" in _diagnostic_codes(slice_)


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


def test_read_transcript_rejects_events_symlink_outside_session(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    session_dir = _write_session(home, native)
    outside = tmp_path / "outside-events.jsonl"
    outside.write_text('{"id":"stolen"}\n', encoding="utf-8")
    _symlink_or_skip(session_dir / "events.jsonl", outside)

    paths = _session_paths(home)
    slice_ = read_transcript(paths, native, {})
    assert _transcript_records(slice_) == []
    assert "stolen" not in json.dumps(slice_["records"])
    assert slice_["exhausted"] is False
    assert "transcript_path_escaped" in _diagnostic_codes(slice_)
    assert discover_transcripts(paths) == []


def test_read_transcript_rejects_workspace_symlink_outside_session(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    session_dir = _write_session(home, native, events='{"id":"1"}\n')
    outside = tmp_path / "outside-workspace.yaml"
    outside.write_text("cwd: /stolen/workspace\n", encoding="utf-8")
    _symlink_or_skip(session_dir / "workspace.yaml", outside)

    slice_ = read_transcript(_session_paths(home), native, {})
    assert [record["payload"]["id"] for record in _transcript_records(slice_)] == ["1"]
    assert _metadata_records(slice_) == []
    assert slice_["cwd"] is None
    assert "/stolen/workspace" not in json.dumps(slice_["records"])
    assert "workspace_path_escaped" in _diagnostic_codes(slice_)


def test_read_transcript_invalid_workspace_emits_diagnostic_without_blocking_events(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    _write_session(
        home,
        native,
        events='{"id":"1"}\n',
        workspace="!!: not: valid: yaml: [\n",
    )
    slice_ = read_transcript(_session_paths(home), native, {})

    assert len(_transcript_records(slice_)) == 1
    assert "workspace_metadata_invalid" in _diagnostic_codes(slice_)
    assert slice_["cwd"] is None


def test_read_transcript_rejects_negative_limits(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    _write_session(home, native, events='{"id":"1"}\n')
    paths = _session_paths(home)
    with pytest.raises(ValueError, match="max_records and max_bytes"):
        read_transcript(paths, native, {}, max_records=-1)


# --- observed cli fixture via injected home layout ---


def test_read_transcript_cli_fixture_preserves_seventy_six_events(tmp_path: Path):
    home = tmp_path / "copilot"
    native = NATIVE_SESSION_ID
    session_dir = _write_session(home, native)
    shutil.copy(CLI_FIXTURE / "events.jsonl", session_dir / "events.jsonl")
    (session_dir / "workspace.yaml").write_text(
        "cwd: /sanitized/workspace\nrepo: probe\n",
        encoding="utf-8",
    )
    paths = _session_paths(home)

    records, diagnostics, last = _drain_transcript(paths, native)
    transcript = [record for record in records if record["source_kind"] == "transcript"]
    metadata = [record for record in records if record["source_kind"] == "metadata"]

    assert len(transcript) == 76
    assert len(metadata) == 1
    assert last["cwd"] == "/sanitized/workspace"
    assert diagnostics == [] or all(item["code"] != "transcript_invalid_json" for item in diagnostics)

    user_messages = [record for record in transcript if record["payload"].get("type") == "user.message"]
    assert len(user_messages) == 3
    assert all(record["payload"]["schema_version"] == SOURCE_SCHEMA_VERSION for record in transcript)


def test_read_transcript_cursor_advances_by_byte_offsets(tmp_path: Path):
    home = tmp_path / "copilot"
    native = "session-a"
    first_line = '{"id":"first"}\n'
    second_line = '{"id":"second"}\n'
    _write_session(home, native, events=first_line + second_line)
    paths = _session_paths(home)

    first = read_transcript(paths, native, {}, max_records=1)
    assert first["next_cursor"]["byte_offset"] == len(first_line)
    assert "continuity_start" in first["next_cursor"]
    assert "continuity_digest" in first["next_cursor"]

    second = read_transcript(paths, native, first["next_cursor"], max_records=1)
    assert second["next_cursor"]["byte_offset"] == len(first_line) + len(second_line)


def test_read_transcript_module_has_no_forbidden_imports():
    source = Path(__import__("thirdeye.platforms.copilot.transcript", fromlist=["__file__"]).__file__)
    imports = [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.startswith(("import ", "from "))
    ]
    joined = "\n".join(imports)
    for forbidden in (
        "thirdeye.store",
        "sqlite3",
        "hook_payload",
        "platforms.copilot.archive",
        "platforms.copilot.database",
    ):
        assert forbidden not in joined
