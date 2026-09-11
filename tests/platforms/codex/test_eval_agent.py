from __future__ import annotations

import json
from pathlib import Path

from thirdeye.eval.agents.codex import CodexAdapter


def test_name():
    assert CodexAdapter().name == "codex"


def test_build_command_uses_exec_subcommand_and_read_only_sandbox():
    cmd = CodexAdapter().build_command("evaluate this", Path("/"))
    assert cmd[0] == "codex"
    assert cmd[1] == "exec"
    assert "--sandbox" in cmd
    assert "read-only" in cmd
    assert "--json" in cmd
    assert "evaluate this" in cmd


def test_parse_output_extracts_last_assistant_message():
    raw = "\n".join(
        [
            json.dumps({"type": "session_started", "session_id": "x"}),
            json.dumps({"type": "message", "role": "assistant", "content": "first reply"}),
            json.dumps({"type": "tool_call", "name": "Read"}),
            json.dumps({"type": "message", "role": "assistant", "content": "final reply"}),
        ]
    )
    text, cost = CodexAdapter().parse_output(raw)
    assert text == "final reply"
    assert cost == {}


def test_parse_output_ignores_user_messages():
    raw = "\n".join(
        [
            json.dumps({"type": "message", "role": "user", "content": "ignored"}),
            json.dumps({"type": "message", "role": "assistant", "content": "kept"}),
        ]
    )
    text, _ = CodexAdapter().parse_output(raw)
    assert text == "kept"


def test_parse_output_skips_malformed_lines():
    raw = "\n".join(
        [
            "not json",
            json.dumps({"type": "message", "role": "assistant", "content": "ok"}),
            "{also broken",
        ]
    )
    text, _ = CodexAdapter().parse_output(raw)
    assert text == "ok"


def test_parse_output_falls_back_when_no_assistant_message():
    raw = json.dumps({"type": "session_started"})
    text, cost = CodexAdapter().parse_output(raw)
    # Falls back to raw text
    assert text == raw
    assert cost == {}


def test_parse_output_empty_input():
    text, cost = CodexAdapter().parse_output("")
    assert text == ""
    assert cost == {}


def test_parse_output_non_string_content_ignored():
    raw = "\n".join(
        [
            json.dumps(
                {"type": "message", "role": "assistant", "content": ["array", "not string"]}
            ),
            json.dumps({"type": "message", "role": "assistant", "content": "fallback"}),
        ]
    )
    text, _ = CodexAdapter().parse_output(raw)
    assert text == "fallback"
