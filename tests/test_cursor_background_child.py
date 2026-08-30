from __future__ import annotations

import json
from pathlib import Path

from thirdeye.config import Config, LogfireSettings
from thirdeye.platforms.cursor import hook
from thirdeye.span_ids import tool_span_id
from thirdeye.store import Store

from tests.test_cursor_hook import _capture_detached_jobs, _cursor_payload, _invoke, _job


def _write_transcript(root: Path, parent_sid: str, child_sid: str, *, ended: bool) -> Path:
    path = root / parent_sid / "subagents" / f"{child_sid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "role": "user",
            "message": {"content": [{"type": "text", "text": "Review the PR"}]},
        },
        {
            "role": "assistant",
            "message": {"content": [{"type": "text", "text": "Looks good."}]},
        },
    ]
    if ended:
        records.append({"type": "turn_ended", "status": "success"})
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def test_background_child_turn_ended_exports_parented_child(tmp_path: Path, monkeypatch):
    jobs = _capture_detached_jobs(tmp_path, monkeypatch)
    transcripts = tmp_path / "agent-transcripts"
    monkeypatch.setenv("THIRDEYE_CURSOR_TRANSCRIPT_ROOTS", str(transcripts))
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.ide_children.transcript_roots",
        lambda: [transcripts],
    )
    child_sid = "be6660e0-c9cf-41dd-8f96-51a035046beb"
    _write_transcript(transcripts, "session-1", child_sid, ended=True)

    _invoke(monkeypatch, _cursor_payload("beforeSubmitPrompt", prompt="delegate"))
    _invoke(
        monkeypatch,
        _cursor_payload("preToolUse", tool_name="Task", tool_use_id="call-bg", tool_input={"prompt": "Review the PR"}),
    )
    _invoke(
        monkeypatch,
        _cursor_payload(
            "subagentStart",
            subagent_id="call-bg",
            tool_call_id="call-bg",
            task="Review the PR",
        ),
    )
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    Config(root=tmp_path).write_logfire_settings(LogfireSettings(enabled=True, token="test-token"))
    store = Store(Config.load())
    store.append_event(
        session_id=child_sid,
        platform="cursor",
        cwd="/repo",
        t="tool_call",
        data={
            "generation_id": "child-gen",
            "tool_name": "read_file",
            "tool_use_id": "child-read",
            "cursor_tool_family": "read_file",
        },
    )
    store.append_event(
        session_id=child_sid,
        platform="cursor",
        cwd="/repo",
        t="tool_result",
        data={
            "generation_id": "child-gen",
            "tool_name": "read_file",
            "tool_use_id": "child-read",
            "cursor_tool_family": "read_file",
            "tool_output": "ok",
        },
    )

    _invoke(
        monkeypatch,
        {
            "conversation_id": child_sid,
            "generation_id": "child-gen",
            "cwd": "/repo",
            "hook_event_name": "afterAgentThought",
            "text": "done",
        },
    )

    child_jobs = [_job(path) for path in jobs if _job(path)["kind"] == "subagent_turn"]
    assert len(child_jobs) == 1
    job = child_jobs[0]
    assert int(job["parent_span_id"]) == tool_span_id("cursor", "session-1", "call-bg")
    tools = job["turn"]["llm_calls"][0]["tool_calls"]
    assert [tool["tool_call_id"] for tool in tools] == ["child-read"]


def test_background_child_without_turn_ended_does_not_export(tmp_path: Path, monkeypatch):
    jobs = _capture_detached_jobs(tmp_path, monkeypatch)
    transcripts = tmp_path / "agent-transcripts"
    monkeypatch.setenv("THIRDEYE_CURSOR_TRANSCRIPT_ROOTS", str(transcripts))
    monkeypatch.setattr(
        "thirdeye.platforms.cursor.ide_children.transcript_roots",
        lambda: [transcripts],
    )
    child_sid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    _write_transcript(transcripts, "session-1", child_sid, ended=False)
    _invoke(
        monkeypatch,
        _cursor_payload("preToolUse", tool_name="Task", tool_use_id="call-open"),
    )
    _invoke(
        monkeypatch,
        _cursor_payload(
            "subagentStart", subagent_id="call-open", tool_call_id="call-open", task="still running"
        ),
    )
    _invoke(
        monkeypatch,
        {
            "conversation_id": child_sid,
            "generation_id": "child-gen",
            "cwd": "/repo",
            "hook_event_name": "afterAgentThought",
            "text": "working",
        },
    )
    assert [_job(path)["kind"] for path in jobs if _job(path)["kind"] == "subagent_turn"] == []
