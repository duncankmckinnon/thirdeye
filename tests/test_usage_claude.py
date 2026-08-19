from __future__ import annotations

import json
from pathlib import Path

import pytest

from thirdeye.config import Config, LogfireSettings
from thirdeye.paths import (
    session_dir,
    usage_jsonl_path,
    usage_log_path,
    usage_state_path,
)
from thirdeye.platforms.claude import usage as usage_module
from thirdeye.platforms.claude.usage import _extract_row, capture_usage_claude

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


def test_config_and_cwd_export_new_rows_to_logfire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New rows must reach `otel_export.export_usage_rows`, otherwise token
    usage is captured to the sidecar sqlite store but never mirrored to
    Logfire — `stop()` in claude/hooks.py must pass `config` and `cwd` through.
    """
    calls = []
    monkeypatch.setattr(
        usage_module,
        "export_usage_rows",
        lambda config, sd, session_id, platform, cwd, rows: calls.append(
            (config, sd, session_id, platform, cwd, rows)
        ),
    )
    config = Config(root=tmp_path, logfire=LogfireSettings(enabled=True, token="t"))
    rows = capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc123",
        transcript_path=str(FIXTURE),
        triggering_seq=5,
        config=config,
        cwd="/proj",
    )
    assert len(calls) == 1
    got_config, sd, session_id, platform, cwd, exported_rows = calls[0]
    assert got_config is config
    assert sd == session_dir(tmp_path, "claude", "abc123")
    assert session_id == "abc123"
    assert platform == "claude"
    assert cwd == "/proj"
    assert len(exported_rows) == rows


def test_no_config_passes_none_through_to_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the caller omits `config`/`cwd` (as every existing test call site
    does), `capture_usage_claude` still calls `export_usage_rows` — it relies
    entirely on `export_usage_rows`'s own `config is None` guard (tested in
    test_otel_export.py) to no-op, rather than duplicating that guard here.
    """
    calls = []
    monkeypatch.setattr(
        usage_module, "export_usage_rows", lambda *a, **k: calls.append((a, k))
    )
    capture_usage_claude(
        thirdeye_home=tmp_path,
        session_id="abc123",
        transcript_path=str(FIXTURE),
        triggering_seq=5,
    )
    assert len(calls) == 1
    args, _kwargs = calls[0]
    assert args[0] is None  # config
    assert args[4] is None  # cwd
