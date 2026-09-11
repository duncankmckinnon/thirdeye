from __future__ import annotations

import json
from pathlib import Path

from thirdeye.eval.agents.base import OutputFormat
from thirdeye.eval.agents.claude import ClaudeAdapter


def test_name():
    assert ClaudeAdapter().name == "claude"


def test_config_is_json_output():
    a = ClaudeAdapter()
    assert a.config.output_format == OutputFormat.JSON
    assert a.config.command == "claude"


def test_build_command_substitutes_prompt():
    a = ClaudeAdapter()
    cmd = a.build_command("evaluate this", Path("/tmp"))
    assert cmd[0] == "claude"
    assert cmd[1] == "-p"
    assert cmd[2] == "evaluate this"
    assert "--output-format" in cmd
    assert "json" in cmd
    assert "--allowedTools" in cmd


def test_allowed_tools_are_read_only():
    """The allowedTools allowlist must not include Edit/Write/Bash(rm/git push)."""
    cmd = ClaudeAdapter().build_command("x", Path("/"))
    idx = cmd.index("--allowedTools")
    tools = cmd[idx + 1]
    for expected in ["Bash(thirdeye *)", "Bash(sqlite3 *)", "Read"]:
        assert expected in tools, f"missing tool: {expected}"
    for forbidden in ["Edit", "Write", "Bash(rm"]:
        assert forbidden not in tools, f"forbidden tool present: {forbidden}"


def test_parse_output_extracts_text_and_cost():
    a = ClaudeAdapter()
    raw = json.dumps(
        {
            "result": "the agent's reply",
            "cost_usd": {"input_tokens": 1000, "output_tokens": 50, "usd": 0.012},
        }
    )
    text, cost = a.parse_output(raw)
    assert text == "the agent's reply"
    assert cost == {"input_tokens": 1000, "output_tokens": 50, "usd": 0.012}


def test_parse_output_falls_back_on_malformed_json():
    a = ClaudeAdapter()
    text, cost = a.parse_output("not json at all")
    assert text == "not json at all"
    assert cost == {}


def test_parse_output_missing_result_key():
    a = ClaudeAdapter()
    raw = json.dumps({"cost_usd": {"usd": 0.01}})
    text, cost = a.parse_output(raw)
    assert text == raw
    assert cost == {"usd": 0.01}


def test_to_config_round_trippable():
    a = ClaudeAdapter()
    d = a.to_config()
    assert d["command"] == "claude"
    assert d["output_format"] == "json"
