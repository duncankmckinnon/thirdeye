from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    extract_new_calls_claude,
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


# --- extract_new_calls_claude: the per-call content (not just tokens) that
# lets Logfire render reasoning/input/output like a real instrumented trace
# (see otel_export.py's module docstring) -----------------------------------


def _user_frame(content) -> str:
    return json.dumps({"type": "user", "message": {"role": "user", "content": content}})


def _assistant_frame(
    msg_id: str, content: list, *, model: str = "claude-sonnet-5", input_tokens=10, output_tokens=5
) -> str:
    return json.dumps(
        {
            "type": "assistant",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "message": {
                "id": msg_id,
                "model": model,
                "content": content,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            },
        }
    )


class TestMapContentBlock:
    def test_text(self):
        assert _map_content_block({"type": "text", "text": "hi"}) == {"type": "text", "content": "hi"}

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


class TestExtractNewCallsClaude:
    def test_no_transcript_path_yields_nothing(self):
        calls, new_offset = extract_new_calls_claude(transcript_path=None, offset=0)
        assert calls == []
        assert new_offset == 0

    def test_missing_file_yields_nothing(self):
        calls, new_offset = extract_new_calls_claude(
            transcript_path="/nonexistent/path.jsonl", offset=0
        )
        assert calls == []
        assert new_offset == 0

    def test_simple_call_has_input_and_output(self, tmp_path: Path):
        transcript = tmp_path / "t.jsonl"
        transcript.write_text(
            _user_frame("hello")
            + "\n"
            + _assistant_frame("msg_1", [{"type": "text", "text": "hi there"}])
            + "\n"
        )
        calls, new_offset = extract_new_calls_claude(transcript_path=str(transcript), offset=0)
        assert len(calls) == 1
        call = calls[0]
        assert call["call_id"] == "msg_1"
        assert call["seq"] == 0  # placeholder, stamped by the caller
        data = call["data"]
        assert data["gen_ai.input.messages"] == [
            {"role": "user", "parts": [{"type": "text", "content": "hello"}]}
        ]
        assert data["gen_ai.output.messages"] == [
            {"role": "assistant", "parts": [{"type": "text", "content": "hi there"}]}
        ]
        assert data["gen_ai.usage.input_tokens"] == 10
        assert data["gen_ai.usage.output_tokens"] == 5
        assert data["gen_ai.response.model"] == "claude-sonnet-5"
        assert new_offset == len(transcript.read_bytes())

    def test_tool_result_becomes_input_to_the_next_call(self, tmp_path: Path):
        transcript = tmp_path / "t.jsonl"
        transcript.write_text(
            _user_frame("run ls")
            + "\n"
            + _assistant_frame(
                "msg_1", [{"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {"command": "ls"}}]
            )
            + "\n"
            + _user_frame(
                [{"type": "tool_result", "tool_use_id": "tu_1", "content": "file1"}]
            )
            + "\n"
            + _assistant_frame("msg_2", [{"type": "text", "text": "found file1"}])
            + "\n"
        )
        calls, _new_offset = extract_new_calls_claude(transcript_path=str(transcript), offset=0)
        assert len(calls) == 2
        first, second = calls
        assert first["call_id"] == "msg_1"
        assert first["data"]["gen_ai.input.messages"][0]["role"] == "user"
        assert second["call_id"] == "msg_2"
        assert second["data"]["gen_ai.input.messages"] == [
            {
                "role": "tool",
                "parts": [{"type": "tool_call_response", "id": "tu_1", "response": "file1"}],
            }
        ]

    def test_consecutive_calls_with_no_intervening_user_frame_has_no_input(
        self, tmp_path: Path
    ):
        transcript = tmp_path / "t.jsonl"
        transcript.write_text(
            _user_frame("go")
            + "\n"
            + _assistant_frame("msg_1", [{"type": "thinking", "thinking": "first"}])
            + "\n"
            + _assistant_frame("msg_2", [{"type": "text", "text": "second"}])
            + "\n"
        )
        calls, _new_offset = extract_new_calls_claude(transcript_path=str(transcript), offset=0)
        assert len(calls) == 2
        assert calls[0]["call_id"] == "msg_1"
        assert "gen_ai.input.messages" in calls[0]["data"]
        assert calls[1]["call_id"] == "msg_2"
        assert "gen_ai.input.messages" not in calls[1]["data"]

    def test_frames_sharing_one_message_id_merge_into_one_call(self, tmp_path: Path):
        """Claude Code logs each content block of one API response as its own
        JSONL line — several consecutive frames sharing one message.id must
        merge into a single call, not become several.
        """
        transcript = tmp_path / "t.jsonl"
        transcript.write_text(
            _user_frame("go")
            + "\n"
            + _assistant_frame("msg_1", [{"type": "thinking", "thinking": "planning"}])
            + "\n"
            + _assistant_frame(
                "msg_1", [{"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {}}]
            )
            + "\n"
        )
        calls, _new_offset = extract_new_calls_claude(transcript_path=str(transcript), offset=0)
        assert len(calls) == 1
        parts = calls[0]["data"]["gen_ai.output.messages"][0]["parts"]
        assert [p["type"] for p in parts] == ["reasoning", "tool_call"]

    def test_synthetic_model_is_skipped(self, tmp_path: Path):
        transcript = tmp_path / "t.jsonl"
        transcript.write_text(
            _assistant_frame("msg_1", [{"type": "text", "text": "hi"}], model="<synthetic>") + "\n"
        )
        calls, _new_offset = extract_new_calls_claude(transcript_path=str(transcript), offset=0)
        assert calls == []

    def test_offset_makes_a_second_call_incremental(self, tmp_path: Path):
        transcript = tmp_path / "t.jsonl"
        transcript.write_text(
            _user_frame("hello")
            + "\n"
            + _assistant_frame("msg_1", [{"type": "text", "text": "hi"}])
            + "\n"
        )
        first_calls, offset_after = extract_new_calls_claude(
            transcript_path=str(transcript), offset=0
        )
        assert len(first_calls) == 1

        with transcript.open("a") as f:
            f.write(_assistant_frame("msg_2", [{"type": "text", "text": "again"}]) + "\n")
        second_calls, _offset = extract_new_calls_claude(
            transcript_path=str(transcript), offset=offset_after
        )
        assert len(second_calls) == 1
        assert second_calls[0]["call_id"] == "msg_2"

    def test_no_output_content_omits_output_messages_key(self, tmp_path: Path):
        """A call whose only content block doesn't map to anything visible
        (e.g. an unrecognized block type) must not claim empty output."""
        transcript = tmp_path / "t.jsonl"
        transcript.write_text(
            _assistant_frame("msg_1", [{"type": "redacted_thinking", "data": "..."}]) + "\n"
        )
        calls, _new_offset = extract_new_calls_claude(transcript_path=str(transcript), offset=0)
        assert len(calls) == 1
        assert "gen_ai.output.messages" not in calls[0]["data"]

