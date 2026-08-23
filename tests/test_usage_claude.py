from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from thirdeye.otel_export import _ts_to_ns
from thirdeye.paths import (
    session_dir,
    usage_jsonl_path,
    usage_log_path,
    usage_state_path,
)
from thirdeye.platforms.claude.usage import (
    _extract_row,
    _map_content_block,
    capture_usage_claude,
    extract_calls_from_transcript,
    parse_new_usage_rows_claude,
    persist_usage_rows_claude,
)

FIXTURE = Path(__file__).parent / "fixtures" / "usage" / "claude_transcript.jsonl"
EXPECTED_JSON = Path(__file__).parent / "fixtures" / "usage" / "claude_transcript.expected.json"


@pytest.fixture(scope="session")
def expected() -> dict:
    return json.loads(EXPECTED_JSON.read_text())


def _capture_rows(tmp_path: Path, session_id: str = "abc123", seq: int = 5) -> list[dict]:
    """Run a capture over the shipped fixture and return the parsed sidecar rows."""
    capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id=session_id,
        transcript_path=str(FIXTURE),
        triggering_seq=seq,
    )
    jsonl = usage_jsonl_path(session_dir(tmp_path, "claude", session_id))
    return [json.loads(line) for line in jsonl.read_text().strip().splitlines()]


def test_row_count_equals_assistant_minus_synthetic(tmp_path: Path, expected: dict) -> None:
    """One row per source frame: every assistant frame except the synthetic placeholder.

    This is the row count, NOT the de-duplicated call count. Collapsing the
    repeated-message.id frames is usage/read.py's job, so the writer emits every
    frame it sees.
    """
    rows = _capture_rows(tmp_path)
    assert len(rows) == expected["assistant_frames"] - expected["synthetic_frames"]


def test_distinct_call_id_equals_expected_calls(tmp_path: Path, expected: dict) -> None:
    rows = _capture_rows(tmp_path)
    distinct = {r["call_id"] for r in rows}
    assert len(distinct) == expected["expected_calls"]


def test_repeated_message_id_frames_share_identical_usage(tmp_path: Path, expected: dict) -> None:
    rows = _capture_rows(tmp_path)
    sample = expected["sample_call"]
    repeated = [r for r in rows if r["call_id"] == expected["repeated_message_id"]]

    assert len(repeated) == expected["repeated_message_id_frame_count"]
    # Every frame of the repeated call carries the same id and token values.
    assert all(r["call_id"] == expected["repeated_message_id"] for r in repeated)
    first = repeated[0]
    assert all(
        r["gen_ai.usage.input_tokens"] == first["gen_ai.usage.input_tokens"] for r in repeated
    )
    assert all(
        r["gen_ai.usage.output_tokens"] == first["gen_ai.usage.output_tokens"] for r in repeated
    )
    assert first["gen_ai.usage.input_tokens"] == sample["expected_inclusive_input_tokens"]
    assert first["gen_ai.usage.output_tokens"] == sample["output_tokens"]
    assert first["gen_ai.usage.cache_read.input_tokens"] == sample["cache_read_input_tokens"]
    assert (
        first["gen_ai.usage.cache_creation.input_tokens"] == sample["cache_creation_input_tokens"]
    )


def test_no_synthetic_model_row(tmp_path: Path) -> None:
    rows = _capture_rows(tmp_path)
    assert all(r["gen_ai.response.model"] != "<synthetic>" for r in rows)


def test_input_tokens_are_cache_inclusive(tmp_path: Path) -> None:
    """gen_ai.usage.input_tokens must be >= reported cache reads + cache creation."""
    rows = _capture_rows(tmp_path)
    for r in rows:
        cache_read = r.get("gen_ai.usage.cache_read.input_tokens") or 0
        cache_crea = r.get("gen_ai.usage.cache_creation.input_tokens") or 0
        assert r["gen_ai.usage.input_tokens"] >= cache_read + cache_crea


def test_reasoning_output_tokens_always_none(tmp_path: Path) -> None:
    """Anthropic does not break out thinking tokens, so the key is always omitted."""
    rows = _capture_rows(tmp_path)
    assert all("gen_ai.usage.reasoning.output_tokens" not in r for r in rows)


def test_provider_and_operation_metadata(tmp_path: Path) -> None:
    rows = _capture_rows(tmp_path)
    assert all(r["gen_ai.provider.name"] == "anthropic" for r in rows)
    assert all(r["gen_ai.operation.name"] == "chat" for r in rows)
    assert all(r["platform"] == "claude" for r in rows)
    assert all(r["seq"] == 5 for r in rows)


# --- absent vs zero -------------------------------------------------------


def test_absent_cache_read_yields_none() -> None:
    frame = {
        "type": "assistant",
        "timestamp": "2026-07-19T00:00:00Z",
        "message": {
            "id": "msg_absent",
            "model": "claude-sonnet-5",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    }
    row = _extract_row(frame, "sid", 1)
    assert row is not None
    assert row.cache_read_input_tokens is None
    assert row.cache_creation_input_tokens is None
    # input stays raw when no cache is reported.
    assert row.input_tokens == 10


def test_explicit_zero_cache_read_yields_zero() -> None:
    frame = {
        "type": "assistant",
        "timestamp": "2026-07-19T00:00:00Z",
        "message": {
            "id": "msg_zero",
            "model": "claude-sonnet-5",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 0,
            },
        },
    }
    row = _extract_row(frame, "sid", 1)
    assert row is not None
    assert row.cache_read_input_tokens == 0


# --- call_id fallback chain ----------------------------------------------


def _frame(message: dict, **top: object) -> dict:
    base = {"type": "assistant", "timestamp": "2026-07-19T00:00:00Z", "message": message}
    base.update(top)
    return base


def test_call_id_prefers_message_id() -> None:
    frame = _frame(
        {"id": "msg_x", "model": "m", "usage": {"input_tokens": 1, "output_tokens": 1}},
        requestId="req_x",
        uuid="uuid_x",
    )
    row = _extract_row(frame, "sid", 1)
    assert row is not None and row.call_id == "msg_x"


def test_call_id_falls_back_to_request_id() -> None:
    frame = _frame(
        {"model": "m", "usage": {"input_tokens": 1, "output_tokens": 1}},
        requestId="req_x",
        uuid="uuid_x",
    )
    row = _extract_row(frame, "sid", 1)
    assert row is not None and row.call_id == "req_x"


def test_call_id_falls_back_to_uuid() -> None:
    frame = _frame(
        {"model": "m", "usage": {"input_tokens": 1, "output_tokens": 1}},
        uuid="uuid_x",
    )
    row = _extract_row(frame, "sid", 1)
    assert row is not None and row.call_id == "uuid_x"


def test_call_id_all_missing_returns_none() -> None:
    frame = _frame({"model": "m", "usage": {"input_tokens": 1, "output_tokens": 1}})
    assert _extract_row(frame, "sid", 1) is None


# --- offset / incremental behaviour --------------------------------------


def test_capture_appends_only_new_rows_on_growth(tmp_path: Path) -> None:
    transcript = tmp_path / "grow.jsonl"
    frame_a = json.dumps(
        {
            "type": "assistant",
            "timestamp": "2026-07-19T00:00:00Z",
            "requestId": "req_a",
            "message": {
                "id": "msg_a",
                "model": "c",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        }
    )
    transcript.write_text(frame_a + "\n")

    rows1 = capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path=str(transcript),
        triggering_seq=1,
    )
    assert rows1 == 1

    frame_b = json.dumps(
        {
            "type": "assistant",
            "timestamp": "2026-07-19T00:01:00Z",
            "requestId": "req_b",
            "message": {
                "id": "msg_b",
                "model": "c",
                "usage": {"input_tokens": 20, "output_tokens": 7},
            },
        }
    )
    with transcript.open("a") as f:
        f.write(frame_b + "\n")

    rows2 = capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path=str(transcript),
        triggering_seq=2,
    )
    assert rows2 == 1

    sd = session_dir(tmp_path, "claude", "abc")
    lines = usage_jsonl_path(sd).read_text().strip().splitlines()
    assert len(lines) == 2
    second = json.loads(lines[1])
    assert second["seq"] == 2 and second["call_id"] == "msg_b"


def test_capture_is_incremental(tmp_path: Path) -> None:
    capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path=str(FIXTURE),
        triggering_seq=1,
    )
    sd = session_dir(tmp_path, "claude", "abc")
    state = json.loads(usage_state_path(sd).read_text())
    initial_offset = state["transcript_offset"]
    assert initial_offset > 0

    rows = capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path=str(FIXTURE),
        triggering_seq=2,
    )
    assert rows == 0
    state2 = json.loads(usage_state_path(sd).read_text())
    assert state2["transcript_offset"] == initial_offset


# --- error handling / logging --------------------------------------------


def test_capture_missing_transcript_logs_error(tmp_path: Path) -> None:
    rows = capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path="/nonexistent/path.jsonl",
        triggering_seq=1,
    )
    assert rows == 0
    log = usage_log_path(tmp_path)
    assert log.exists() and "open_source" in log.read_text()


def test_capture_with_no_transcript_path(tmp_path: Path) -> None:
    rows = capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path=None,
        triggering_seq=1,
    )
    assert rows == 0
    sd = session_dir(tmp_path, "claude", "abc")
    assert not usage_jsonl_path(sd).exists()


def test_first_capture_emits_no_unverified_warning(tmp_path: Path) -> None:
    """The old 'shape is unverified' warning was deleted; first capture logs nothing."""
    capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="fresh",
        transcript_path=str(FIXTURE),
        triggering_seq=1,
    )
    log = usage_log_path(tmp_path)
    contents = log.read_text() if log.exists() else ""
    assert "unverified" not in contents


# --- rejected-frame branches (spec steps 1 & 5) --------------------------


def test_non_dict_frame_yields_none() -> None:
    assert _extract_row("not a dict", "sid", 1) is None  # type: ignore[arg-type]


def test_non_dict_message_yields_none() -> None:
    frame = {"type": "assistant", "message": "not a dict"}
    assert _extract_row(frame, "sid", 1) is None


def test_non_assistant_frame_yields_none() -> None:
    """Only type=="assistant" frames are candidates; a user frame is dropped."""
    frame = {
        "type": "user",
        "message": {"id": "msg_u", "usage": {"input_tokens": 5, "output_tokens": 2}},
    }
    assert _extract_row(frame, "sid", 1) is None


def test_empty_usage_dict_yields_none() -> None:
    """An assistant frame whose message.usage is an empty dict carries no usage."""
    frame = _frame({"id": "msg_e", "model": "m", "usage": {}})
    assert _extract_row(frame, "sid", 1) is None


def test_missing_usage_key_yields_none() -> None:
    frame = _frame({"id": "msg_n", "model": "m"})
    assert _extract_row(frame, "sid", 1) is None


def test_both_token_fields_absent_yields_none() -> None:
    """Spec step 5: with neither input_tokens nor output_tokens reported, drop it.

    Cache-only usage (no primary token counts) is not a recordable call.
    """
    frame = _frame(
        {"id": "msg_c", "model": "m", "usage": {"cache_read_input_tokens": 100}},
    )
    assert _extract_row(frame, "sid", 1) is None


def test_output_absent_but_input_present_yields_row_with_zero_output() -> None:
    """Only one primary field present is still a call; the missing one reads as 0."""
    frame = _frame({"id": "msg_o", "model": "m", "usage": {"input_tokens": 12}})
    row = _extract_row(frame, "sid", 1)
    assert row is not None
    assert row.input_tokens == 12
    assert row.output_tokens == 0


def test_synthetic_model_frame_yields_none() -> None:
    frame = _frame(
        {"id": "msg_s", "model": "<synthetic>", "usage": {"input_tokens": 0, "output_tokens": 0}},
    )
    assert _extract_row(frame, "sid", 1) is None


def test_capture_skips_corrupt_jsonl_lines(tmp_path: Path) -> None:
    """A malformed line between valid frames is skipped, not fatal."""
    transcript = tmp_path / "corrupt.jsonl"
    good = json.dumps(
        {
            "type": "assistant",
            "timestamp": "2026-07-19T00:00:00Z",
            "requestId": "req_g",
            "message": {
                "id": "msg_g",
                "model": "c",
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        }
    )
    transcript.write_text(good + "\nnot json at all\n" + good + "\n")
    rows = capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        transcript_path=str(transcript),
        triggering_seq=1,
    )
    assert rows == 2


# --- parse/persist split: lets the caller aggregate usage onto the
# triggering event's own span before it's built (see otel_export.py's module
# docstring and stop() in claude/hooks.py) -------------------------------


def test_parse_is_pure_and_does_not_persist(tmp_path: Path) -> None:
    """Calling parse twice with no persist in between must yield the same
    rows and offset both times — proves it never advances state itself.
    """
    first = parse_new_usage_rows_claude(
        thirdeye_home=tmp_path, session_id="abc", transcript_path=str(FIXTURE)
    )
    second = parse_new_usage_rows_claude(
        thirdeye_home=tmp_path, session_id="abc", transcript_path=str(FIXTURE)
    )
    assert first is not None and second is not None
    assert first == second
    sd = session_dir(tmp_path, "claude", "abc")
    assert not usage_jsonl_path(sd).exists()
    assert not usage_state_path(sd).exists()


def test_parse_no_transcript_path_returns_empty_none_offset(tmp_path: Path) -> None:
    parsed = parse_new_usage_rows_claude(
        thirdeye_home=tmp_path, session_id="abc", transcript_path=None
    )
    assert parsed == ([], None)


def test_parse_missing_file_logs_and_returns_empty_none_offset(tmp_path: Path) -> None:
    parsed = parse_new_usage_rows_claude(
        thirdeye_home=tmp_path, session_id="abc", transcript_path="/nonexistent/path.jsonl"
    )
    assert parsed == ([], None)
    log = usage_log_path(tmp_path)
    assert log.exists() and "open_source" in log.read_text()


def test_persist_stamps_seq_onto_stored_rows(tmp_path: Path) -> None:
    parsed = parse_new_usage_rows_claude(
        thirdeye_home=tmp_path, session_id="abc", transcript_path=str(FIXTURE)
    )
    assert parsed is not None
    rows, new_offset = parsed
    assert new_offset is not None
    assert all(row.seq == 0 for row in rows)  # placeholder, pre-persist

    n = persist_usage_rows_claude(
        thirdeye_home=tmp_path,
        session_id="abc",
        rows=rows,
        new_offset=new_offset,
        triggering_seq=42,
    )
    assert n == len(rows)
    sd = session_dir(tmp_path, "claude", "abc")
    stored = [json.loads(line) for line in usage_jsonl_path(sd).read_text().strip().splitlines()]
    assert all(row["seq"] == 42 for row in stored)


def test_persist_advances_offset_and_state(tmp_path: Path) -> None:
    parsed = parse_new_usage_rows_claude(
        thirdeye_home=tmp_path, session_id="abc", transcript_path=str(FIXTURE)
    )
    assert parsed is not None
    rows, new_offset = parsed
    assert new_offset is not None

    persist_usage_rows_claude(
        thirdeye_home=tmp_path, session_id="abc", rows=rows, new_offset=new_offset, triggering_seq=1
    )
    sd = session_dir(tmp_path, "claude", "abc")
    state = json.loads(usage_state_path(sd).read_text())
    assert state["transcript_offset"] == new_offset
    assert state["last_seq"] == 1

    # A second parse with nothing new must see the advanced offset and find
    # no further rows.
    second = parse_new_usage_rows_claude(
        thirdeye_home=tmp_path, session_id="abc", transcript_path=str(FIXTURE)
    )
    assert second == ([], new_offset)


def test_parse_then_persist_matches_capture_usage_claude(tmp_path: Path) -> None:
    """The split path and the combined compat wrapper must agree exactly on
    what ends up in the sidecar, for the same fixture and triggering_seq.
    """
    combined_home = tmp_path / "combined"
    split_home = tmp_path / "split"

    capture_usage_claude(
        thirdeye_home=combined_home,
        session_id="abc",
        transcript_path=str(FIXTURE),
        triggering_seq=7,
    )
    parsed = parse_new_usage_rows_claude(
        thirdeye_home=split_home, session_id="abc", transcript_path=str(FIXTURE)
    )
    assert parsed is not None
    rows, new_offset = parsed
    persist_usage_rows_claude(
        thirdeye_home=split_home,
        session_id="abc",
        rows=rows,
        new_offset=new_offset,
        triggering_seq=7,
    )

    combined_rows = usage_jsonl_path(session_dir(combined_home, "claude", "abc")).read_text()
    split_rows = usage_jsonl_path(session_dir(split_home, "claude", "abc")).read_text()
    assert combined_rows == split_rows


# --- LLM call span duration derivation -----------------------------------


def _write_transcript(path: Path, *frames: object) -> None:
    path.write_text("".join(json.dumps(frame) + "\n" for frame in frames))


def _assistant_frame(
    call_id: str,
    timestamp: object,
    content: list[dict] | None = None,
) -> dict:
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "message": {
            "id": call_id,
            "model": "claude-sonnet-5",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "content": content or [],
        },
    }


def test_call_duration_spans_preceding_frame_to_last_group_frame(tmp_path: Path) -> None:
    transcript = tmp_path / "multi-frame.jsonl"
    dispatch_ts = "2026-08-22T10:00:00.000Z"
    group_timestamps = [
        "2026-08-22T10:00:01.000Z",
        "2026-08-22T10:00:02.000Z",
        "2026-08-22T10:00:03.000Z",
        "2026-08-22T10:00:04.000Z",
    ]
    _write_transcript(
        transcript,
        {"type": "user", "timestamp": dispatch_ts, "message": {"content": "go"}},
        *[
            _assistant_frame("msg_multi", timestamp, [{"type": "text", "text": f"part-{index}"}])
            for index, timestamp in enumerate(group_timestamps)
        ],
    )

    calls, new_offset = extract_calls_from_transcript(str(transcript), 0)

    assert len(calls) == 1
    assert calls[0]["start_ts"] == dispatch_ts
    assert calls[0]["end_ts"] == group_timestamps[-1]
    assert len(calls[0]["output_messages"][0]["parts"]) == 4
    assert new_offset == transcript.stat().st_size


def test_single_frame_call_uses_preceding_tool_result_as_start(tmp_path: Path) -> None:
    transcript = tmp_path / "single-frame.jsonl"
    dispatch_ts = "2026-08-22T10:00:05.000Z"
    response_ts = "2026-08-22T10:00:08.000Z"
    _write_transcript(
        transcript,
        {
            "type": "user",
            "timestamp": dispatch_ts,
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "done"}]
            },
        },
        _assistant_frame("msg_single", response_ts, [{"type": "text", "text": "finished"}]),
    )

    calls, _ = extract_calls_from_transcript(str(transcript), 0)

    assert len(calls) == 1
    assert calls[0]["start_ts"] == dispatch_ts
    assert calls[0]["end_ts"] == response_ts
    assert calls[0]["start_ts"] != calls[0]["end_ts"]
    assert calls[0]["input_messages"][0]["role"] == "tool"


def test_first_group_without_preceding_frame_is_zero_width(tmp_path: Path) -> None:
    transcript = tmp_path / "first-group.jsonl"
    response_ts = "2026-08-22T10:00:08.000Z"
    _write_transcript(transcript, _assistant_frame("msg_first", response_ts))

    calls, _ = extract_calls_from_transcript(str(transcript), 0)

    assert len(calls) == 1
    assert calls[0]["start_ts"] == response_ts
    assert calls[0]["end_ts"] == response_ts


def test_initial_prev_ts_starts_first_group(tmp_path: Path) -> None:
    transcript = tmp_path / "seeded-group.jsonl"
    dispatch_ts = "2026-08-22T10:00:00.000Z"
    response_ts = "2026-08-22T10:00:08.000Z"
    preceding_frame = {
        "type": "user",
        "timestamp": dispatch_ts,
        "message": {"content": "behind the cursor"},
    }
    _write_transcript(
        transcript,
        preceding_frame,
        _assistant_frame("msg_seeded", response_ts),
    )
    offset = len((json.dumps(preceding_frame) + "\n").encode())

    calls, _ = extract_calls_from_transcript(str(transcript), offset, initial_prev_ts=dispatch_ts)

    assert len(calls) == 1
    assert calls[0]["start_ts"] == dispatch_ts
    assert calls[0]["end_ts"] == response_ts


def test_parallel_tool_use_blocks_remain_one_call(tmp_path: Path) -> None:
    transcript = tmp_path / "parallel-tools.jsonl"
    _write_transcript(
        transcript,
        {"type": "user", "timestamp": "2026-08-22T10:00:00.000Z", "message": {}},
        _assistant_frame(
            "msg_tools",
            "2026-08-22T10:00:01.000Z",
            [{"type": "tool_use", "id": "tool-1", "name": "Read", "input": {}}],
        ),
        _assistant_frame(
            "msg_tools",
            "2026-08-22T10:00:02.000Z",
            [{"type": "tool_use", "id": "tool-2", "name": "Bash", "input": {}}],
        ),
    )

    calls, _ = extract_calls_from_transcript(str(transcript), 0)

    assert len(calls) == 1
    parts = calls[0]["output_messages"][0]["parts"]
    assert [part["id"] for part in parts] == ["tool-1", "tool-2"]


def test_multiple_calls_each_use_their_own_dispatch_and_last_group_frame(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "multiple-calls.jsonl"
    first_dispatch_ts = "2026-08-22T10:00:00.000Z"
    first_end_ts = "2026-08-22T10:00:02.000Z"
    second_dispatch_ts = "2026-08-22T10:00:03.000Z"
    second_end_ts = "2026-08-22T10:00:05.000Z"
    _write_transcript(
        transcript,
        {"type": "user", "timestamp": first_dispatch_ts, "message": {"content": "go"}},
        _assistant_frame("msg_first", "2026-08-22T10:00:01.000Z"),
        _assistant_frame("msg_first", first_end_ts),
        {
            "type": "user",
            "timestamp": second_dispatch_ts,
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "ok"}]
            },
        },
        _assistant_frame("msg_second", "2026-08-22T10:00:04.000Z"),
        _assistant_frame("msg_second", second_end_ts),
    )

    calls, _ = extract_calls_from_transcript(str(transcript), 0)

    assert [(call["start_ts"], call["end_ts"]) for call in calls] == [
        (first_dispatch_ts, first_end_ts),
        (second_dispatch_ts, second_end_ts),
    ]


def test_valid_timestamp_on_any_preceding_frame_is_used_as_dispatch(tmp_path: Path) -> None:
    transcript = tmp_path / "progress-dispatch.jsonl"
    older_user_ts = "2026-08-22T10:00:00.000Z"
    progress_ts = "2026-08-22T10:00:01.000Z"
    response_ts = "2026-08-22T10:00:02.000Z"
    _write_transcript(
        transcript,
        {"type": "user", "timestamp": older_user_ts, "message": {"content": "go"}},
        {"type": "progress", "timestamp": progress_ts},
        _assistant_frame("msg_after_progress", response_ts),
    )

    calls, _ = extract_calls_from_transcript(str(transcript), 0)

    assert len(calls) == 1
    assert calls[0]["start_ts"] == progress_ts
    assert calls[0]["end_ts"] == response_ts


@pytest.mark.parametrize(
    "bad_ts",
    [
        12345,
        "not-a-timestamp",
        "",
        None,
        {"at": "now"},
        # `datetime.fromisoformat` takes all of these, but a date without a
        # time of day is not a timestamp and must not become a span bound.
        "2026-08-22",
        "2026-08-22T10",
        "20260822T100000",
    ],
)
def test_malformed_timestamp_does_not_replace_previous_frame_timestamp(
    tmp_path: Path, bad_ts: object
) -> None:
    transcript = tmp_path / "malformed-timestamp.jsonl"
    valid_dispatch_ts = "2026-08-22T10:00:00.000Z"
    response_ts = "2026-08-22T10:00:02.000Z"
    _write_transcript(
        transcript,
        {"type": "user", "timestamp": valid_dispatch_ts, "message": {"content": "go"}},
        {"type": "progress", "timestamp": bad_ts},
        _assistant_frame("msg_after_bad_ts", response_ts),
    )

    calls, _ = extract_calls_from_transcript(str(transcript), 0)

    assert len(calls) == 1
    assert calls[0]["start_ts"] == valid_dispatch_ts
    assert calls[0]["end_ts"] == response_ts


@pytest.mark.parametrize("bad_seed", ["whenever", "2026-08-22", "2026-08-22T10"])
def test_malformed_initial_prev_ts_is_ignored(tmp_path: Path, bad_seed: str) -> None:
    transcript = tmp_path / "bad-seed.jsonl"
    response_ts = "2026-08-22T10:00:02.000Z"
    _write_transcript(transcript, _assistant_frame("msg_bad_seed", response_ts))

    calls, _ = extract_calls_from_transcript(str(transcript), 0, initial_prev_ts=bad_seed)

    assert len(calls) == 1
    assert calls[0]["start_ts"] == response_ts
    assert calls[0]["end_ts"] == response_ts


def test_group_without_any_timestamp_emits_empty_bounds(tmp_path: Path) -> None:
    transcript = tmp_path / "no-timestamps.jsonl"
    _write_transcript(transcript, _assistant_frame("msg_no_ts", None))

    calls, _ = extract_calls_from_transcript(str(transcript), 0)

    assert len(calls) == 1
    assert calls[0]["start_ts"] == ""
    assert calls[0]["end_ts"] == ""


def test_untimestamped_first_frame_takes_end_from_later_group_frame(tmp_path: Path) -> None:
    transcript = tmp_path / "late-timestamp.jsonl"
    later_ts = "2026-08-22T10:00:03.000Z"
    _write_transcript(
        transcript,
        _assistant_frame("msg_late", None, [{"type": "text", "text": "first"}]),
        _assistant_frame("msg_late", later_ts, [{"type": "text", "text": "second"}]),
    )

    calls, _ = extract_calls_from_transcript(str(transcript), 0)

    assert len(calls) == 1
    assert calls[0]["start_ts"] == later_ts
    assert calls[0]["end_ts"] == later_ts


def test_missing_group_frame_timestamp_keeps_last_valid_end(tmp_path: Path) -> None:
    transcript = tmp_path / "missing-group-timestamp.jsonl"
    first_ts = "2026-08-22T10:00:01.000Z"
    _write_transcript(
        transcript,
        _assistant_frame("msg_missing", first_ts, [{"type": "text", "text": "first"}]),
        _assistant_frame("msg_missing", None, [{"type": "text", "text": "second"}]),
    )

    calls, _ = extract_calls_from_transcript(str(transcript), 0)

    assert len(calls) == 1
    assert calls[0]["start_ts"] == first_ts
    assert calls[0]["end_ts"] == first_ts


def test_clock_anomaly_collapses_call_to_end_timestamp(tmp_path: Path) -> None:
    transcript = tmp_path / "clock-anomaly.jsonl"
    response_ts = "2026-08-22T10:00:01.000Z"
    _write_transcript(transcript, _assistant_frame("msg_clock", response_ts))

    calls, _ = extract_calls_from_transcript(
        str(transcript), 0, initial_prev_ts="2026-08-22T10:00:02.000Z"
    )

    assert len(calls) == 1
    assert calls[0]["start_ts"] == response_ts
    assert calls[0]["end_ts"] == response_ts


def test_clock_anomaly_with_offsets_collapses_call_to_end_timestamp(tmp_path: Path) -> None:
    """ISO offsets must be compared chronologically, not lexicographically."""
    transcript = tmp_path / "clock-anomaly-offsets.jsonl"
    # 10:30 at UTC+01:00 is 09:30 UTC, before the 10:00 UTC dispatch.
    response_ts = "2026-08-22T10:30:00+01:00"
    _write_transcript(transcript, _assistant_frame("msg_clock_offsets", response_ts))

    calls, _ = extract_calls_from_transcript(
        str(transcript), 0, initial_prev_ts="2026-08-22T10:00:00+00:00"
    )

    assert len(calls) == 1
    assert calls[0]["start_ts"] == response_ts
    assert calls[0]["end_ts"] == response_ts


def test_naive_timestamps_are_pinned_to_utc(tmp_path: Path) -> None:
    transcript = tmp_path / "naive-timestamps.jsonl"
    _write_transcript(
        transcript,
        {
            "type": "user",
            "timestamp": "2026-08-22T10:00:00.000",
            "message": {"content": "go"},
        },
        _assistant_frame("msg_naive", "2026-08-22T10:00:02.000"),
    )

    calls, _ = extract_calls_from_transcript(str(transcript), 0)

    assert len(calls) == 1
    assert calls[0]["start_ts"] == "2026-08-22T10:00:00.000+00:00"
    assert calls[0]["end_ts"] == "2026-08-22T10:00:02.000+00:00"


def test_mixed_naive_and_offset_bounds_stay_ordered_after_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A naive bound must mean the same instant here and in the exporter.

    `_ts_to_ns` reads a naive timestamp in the worker's local timezone, so in
    a zone behind UTC a naive start that this module accepted as earlier than
    an offset-aware end used to export as later than it — a backwards span.
    """
    transcript = tmp_path / "mixed-timestamps.jsonl"
    _write_transcript(
        transcript,
        {
            "type": "user",
            "timestamp": "2026-08-22T10:00:00.000",
            "message": {"content": "go"},
        },
        _assistant_frame("msg_mixed", "2026-08-22T12:00:00.000+00:00"),
    )

    calls, _ = extract_calls_from_transcript(str(transcript), 0)

    assert len(calls) == 1
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    try:
        assert _ts_to_ns(calls[0]["start_ts"]) < _ts_to_ns(calls[0]["end_ts"])
    finally:
        monkeypatch.undo()
        time.tzset()


class TestMapContentBlock:
    def test_text(self):
        assert _map_content_block({"type": "text", "text": "hi"}) == {
            "type": "text",
            "content": "hi",
        }

    def test_empty_text_is_dropped(self):
        assert _map_content_block({"type": "text", "text": ""}) is None

    def test_thinking_becomes_reasoning(self):
        assert _map_content_block({"type": "thinking", "thinking": "hmm"}) == {
            "type": "reasoning",
            "content": "hmm",
        }

    def test_tool_use(self):
        block = {"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {"command": "ls"}}
        assert _map_content_block(block) == {
            "type": "tool_call",
            "id": "tu_1",
            "name": "Bash",
            "arguments": {"command": "ls"},
        }

    def test_tool_result(self):
        block = {"type": "tool_result", "tool_use_id": "tu_1", "content": "file1\nfile2"}
        assert _map_content_block(block) == {
            "type": "tool_call_response",
            "id": "tu_1",
            "response": "file1\nfile2",
        }

    def test_unknown_type_is_ignored(self):
        assert _map_content_block({"type": "redacted_thinking", "data": "..."}) is None

    def test_non_dict_is_ignored(self):
        assert _map_content_block("not a block") is None
