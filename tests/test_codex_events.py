from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from thirdeye.config import Config
from thirdeye.platforms.codex.events import capture_events_codex
from thirdeye.store import Store

FIXTURE = Path(__file__).parent / "fixtures" / "usage" / "codex_rollout.jsonl"
EXPECTED = Path(__file__).parent / "fixtures" / "usage" / "codex_rollout.expected.json"

FIXTURE_SID = "019fb579-cdda-7a03-86df-65c87b6c4ae2"


def _config(tmp_path: Path) -> Config:
    return Config(root=tmp_path)


def _place(tmp_path: Path, *, src: Path = FIXTURE, name: str = "rollout.jsonl") -> Path:
    """Copy a rollout source into tmp_path and return its path."""
    dest = tmp_path / name
    dest.write_bytes(src.read_bytes())
    return dest


def _events(config: Config, session_id: str = FIXTURE_SID) -> list[dict]:
    return list(Store(config).reader(session_id).iter_events())


def _run(tmp_path: Path, rollout: Path, *, offset: int = 0, sid: str = FIXTURE_SID):
    return capture_events_codex(
        config=_config(tmp_path),
        session_id=sid,
        cwd="/proj/codex",
        rollout_path=str(rollout),
        offset=offset,
    )


# -- core mapping --------------------------------------------------------------


def test_fixture_yields_the_core_event_vocabulary(tmp_path: Path) -> None:
    rollout = _place(tmp_path)
    appended, new_offset = _run(tmp_path, rollout)

    types = {e["t"] for e in _events(_config(tmp_path))}
    for expected in {
        "session_start",
        "user_message",
        "tool_call",
        "tool_result",
        "assistant_message",
    }:
        assert expected in types, f"{expected} missing from {types}"
    assert appended > 0
    assert new_offset == rollout.stat().st_size


def test_tool_call_and_result_counts_match_expected(tmp_path: Path) -> None:
    expected = json.loads(EXPECTED.read_text())
    rollout = _place(tmp_path)
    _run(tmp_path, rollout)

    counts = Counter(e["t"] for e in _events(_config(tmp_path)))
    assert counts["tool_call"] == expected["expected_tool_call_events"]
    assert counts["tool_result"] == expected["expected_tool_result_events"]


def test_custom_tool_call_frames_produce_tool_call_events(tmp_path: Path) -> None:
    """Explicit: the custom_tool_call family (which now outnumbers function_call)
    must map to tool_call/tool_result, or a version shift silently drops tools."""
    # A rollout containing ONLY custom_tool_call frames (no function_call).
    lines = [
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "call_id": "call_custom_1",
                    "name": "exec",
                    "input": "{}",
                },
            }
        ),
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": "call_custom_1",
                    "output": "done",
                },
            }
        ),
    ]
    rollout = tmp_path / "custom.jsonl"
    rollout.write_text("\n".join(lines) + "\n")
    _run(tmp_path, rollout, sid="customsession")

    counts = Counter(e["t"] for e in _events(_config(tmp_path), "customsession"))
    assert counts["tool_call"] == 1
    assert counts["tool_result"] == 1


def test_token_count_frames_produce_no_events(tmp_path: Path) -> None:
    rollout = _place(tmp_path)
    _run(tmp_path, rollout)

    # The fixture has 81 token_count frames; none may become an event, and no
    # event type should be one that only a token_count could have produced.
    events = _events(_config(tmp_path))
    # Total events must equal the non-token, mapped frame count.
    expected = json.loads(EXPECTED.read_text())["frame_census"]
    mapped = (
        expected["session_meta/-"]
        + expected["turn_context/-"]
        + expected["event_msg/user_message"]
        + expected["event_msg/agent_message"]
        + expected["response_item/function_call"]
        + expected["response_item/function_call_output"]
        + expected["response_item/custom_tool_call"]
        + expected["response_item/custom_tool_call_output"]
    )
    assert len(events) == mapped
    # And none of the events carry token_count-only payload markers.
    for e in events:
        assert "total_token_usage" not in e["data"]
        assert "last_token_usage" not in e["data"]


def test_notification_mapping(tmp_path: Path) -> None:
    """turn_context, task_started, task_complete all map to notification;
    turn_aborted maps to error."""
    lines = [
        {"type": "turn_context", "payload": {"model": "gpt-5.6", "cwd": "/x"}},
        {"type": "event_msg", "payload": {"type": "task_started"}},
        {"type": "event_msg", "payload": {"type": "task_complete"}},
    ]
    rollout = tmp_path / "notif.jsonl"
    rollout.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    _run(tmp_path, rollout, sid="notifsession")

    counts = Counter(e["t"] for e in _events(_config(tmp_path), "notifsession"))
    assert counts["notification"] == 3


def test_turn_aborted_produces_exactly_one_error_event(tmp_path: Path) -> None:
    lines = [
        {"type": "session_meta", "payload": {"id": "abortsession"}},
        {"type": "event_msg", "payload": {"type": "turn_aborted", "reason": "user"}},
    ]
    rollout = tmp_path / "abort.jsonl"
    rollout.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    _run(tmp_path, rollout, sid="abortsession")

    counts = Counter(e["t"] for e in _events(_config(tmp_path), "abortsession"))
    assert counts["error"] == 1


def test_unknown_frame_type_produces_no_event_and_no_error(tmp_path: Path) -> None:
    lines = [
        {"type": "session_meta", "payload": {"id": "unknownsession"}},
        {"type": "brand_new_frame_type_v9", "payload": {"type": "who_knows", "x": 1}},
        {"type": "response_item", "payload": {"type": "some_future_item"}},
        {"type": "event_msg", "payload": {"type": "some_future_event"}},
    ]
    rollout = tmp_path / "unknown.jsonl"
    rollout.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    appended, _ = _run(tmp_path, rollout, sid="unknownsession")

    # Only the session_meta maps; the three unknowns are silently skipped.
    assert appended == 1
    counts = Counter(e["t"] for e in _events(_config(tmp_path), "unknownsession"))
    assert counts == Counter({"session_start": 1})
    # And no error was logged.
    errlog = tmp_path / "logs" / "usage-errors.jsonl"
    assert not errlog.exists()


# -- offset / data invariants --------------------------------------------------


def test_every_event_carries_strictly_increasing_rollout_offset(tmp_path: Path) -> None:
    rollout = _place(tmp_path)
    _run(tmp_path, rollout)

    offsets = [e["data"]["rollout_offset"] for e in _events(_config(tmp_path))]
    assert all(isinstance(o, int) for o in offsets)
    assert offsets == sorted(offsets)
    assert len(set(offsets)) == len(offsets)


def test_tool_call_and_result_share_call_id(tmp_path: Path) -> None:
    rollout = _place(tmp_path)
    _run(tmp_path, rollout)

    events = _events(_config(tmp_path))
    calls = {e["data"]["call_id"] for e in events if e["t"] == "tool_call"}
    results = {e["data"]["call_id"] for e in events if e["t"] == "tool_result"}
    assert calls, "expected at least one tool_call with a call_id"
    # Every tool_call in the fixture is paired with a matching tool_result.
    assert calls == results


def test_rollout_offset_matches_source_byte_position(tmp_path: Path) -> None:
    """The recorded offset must be the real byte position of the source frame,
    so post-hoc dedup/repair can seek back to it."""
    rollout = _place(tmp_path)
    _run(tmp_path, rollout)

    raw = rollout.read_bytes()
    for e in _events(_config(tmp_path)):
        off = e["data"]["rollout_offset"]
        # The line at `off` must parse and carry the same call_id (when present).
        end = raw.index(b"\n", off)
        frame = json.loads(raw[off:end].decode())
        payload = frame.get("payload") or {}
        if "call_id" in e["data"]:
            assert payload.get("call_id") == e["data"]["call_id"]


# -- tailing / idempotency -----------------------------------------------------


def test_second_call_from_returned_offset_appends_nothing(tmp_path: Path) -> None:
    rollout = _place(tmp_path)
    appended1, offset1 = _run(tmp_path, rollout)
    assert appended1 > 0

    appended2, offset2 = _run(tmp_path, rollout, offset=offset1)
    assert appended2 == 0
    assert offset2 == offset1
    # Store still holds only the first pass's events.
    assert len(_events(_config(tmp_path))) == appended1


def test_replay_from_same_offset_duplicates_events(tmp_path: Path) -> None:
    """Pins the documented crash-window tradeoff: events.alog has no dedup, so
    replaying the same range DOES duplicate. This is intentional, not a bug."""
    rollout = _place(tmp_path)
    appended1, _ = _run(tmp_path, rollout)
    appended2, _ = _run(tmp_path, rollout)  # same offset=0 again

    assert appended1 == appended2
    assert len(_events(_config(tmp_path))) == appended1 + appended2


def test_incremental_tail_captures_only_new_frames(tmp_path: Path) -> None:
    """Write half a rollout, capture, append the rest, capture from the bookmark;
    the second pass sees only the appended frames."""
    all_lines = FIXTURE.read_text().splitlines(keepends=True)
    half = len(all_lines) // 2
    rollout = tmp_path / "grow.jsonl"
    rollout.write_text("".join(all_lines[:half]))

    appended1, offset1 = _run(tmp_path, rollout)
    rollout.write_text("".join(all_lines))  # append the rest
    appended2, offset2 = _run(tmp_path, rollout, offset=offset1)

    assert offset2 == rollout.stat().st_size
    # No double counting: total equals the full-file single-pass count.
    full = tmp_path / "full.jsonl"
    full.write_text("".join(all_lines))
    fresh = Path(tmp_path / "fresh")
    fresh.mkdir()
    single, _ = capture_events_codex(
        config=Config(root=fresh),
        session_id=FIXTURE_SID,
        cwd="/proj/codex",
        rollout_path=str(full),
        offset=0,
    )
    assert appended1 + appended2 == single


# -- fail-soft -----------------------------------------------------------------


def test_malformed_line_is_skipped_and_neighbours_still_captured(tmp_path: Path) -> None:
    rollout = tmp_path / "malformed.jsonl"
    rollout.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "malfsession"}})
        + "\n"
        + "this is not json at all\n"
        + json.dumps({"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}})
        + "\n"
    )
    appended, new_offset = _run(tmp_path, rollout, sid="malfsession")

    assert appended == 2  # session_start + user_message; the bad line skipped
    counts = Counter(e["t"] for e in _events(_config(tmp_path), "malfsession"))
    assert counts == Counter({"session_start": 1, "user_message": 1})
    assert new_offset == rollout.stat().st_size


def test_missing_rollout_file_returns_zero_and_logs(tmp_path: Path) -> None:
    missing = tmp_path / "nope.jsonl"
    appended, new_offset = capture_events_codex(
        config=_config(tmp_path),
        session_id="missingsession",
        cwd="/proj/codex",
        rollout_path=str(missing),
        offset=42,
    )
    assert appended == 0
    assert new_offset == 42  # offset returned unchanged so nothing is skipped
    errlog = tmp_path / "logs" / "usage-errors.jsonl"
    assert errlog.exists()
    entry = json.loads(errlog.read_text().strip())
    assert entry["platform"] == "codex"


def test_truncated_final_line_excluded_from_offset(tmp_path: Path) -> None:
    """A rollout being written concurrently: the trailing partial line is not
    yielded and the bookmark stops before it."""
    good = json.dumps({"type": "session_meta", "payload": {"id": "truncsession"}}) + "\n"
    rollout = tmp_path / "trunc.jsonl"
    rollout.write_bytes(good.encode() + b'{"type":"event_msg","payload":{"type":"user_')

    appended, new_offset = _run(tmp_path, rollout, sid="truncsession")
    assert appended == 1
    assert new_offset == len(good.encode())
