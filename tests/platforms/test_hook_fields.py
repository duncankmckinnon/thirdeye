from __future__ import annotations

from thirdeye.platforms.hook_fields import hook_span_attributes


def test_shared_fields_are_renamed_into_span_vocabulary():
    assert hook_span_attributes(
        {
            "tool_name": "Bash",
            "tool_use_id": "tu_1",
            "tool_input": {"command": "ls"},
            "tool_response": {"stdout": ""},
            "agent_id": "a1",
            "agent_type": "Explore",
            "permission_mode": "acceptEdits",
            "effort": "high",
            "model": "gpt-5",
            "duration_ms": 12,
            "is_interrupt": True,
            "mcp_server": "logfire",
            "reason": "needs approval",
        }
    ) == {
        "gen_ai.tool.name": "Bash",
        "gen_ai.tool.call.id": "tu_1",
        "gen_ai.tool.call.arguments": {"command": "ls"},
        "gen_ai.tool.call.result": {"stdout": ""},
        "thirdeye.subagent.id": "a1",
        "thirdeye.subagent.type": "Explore",
        "thirdeye.permission_mode": "acceptEdits",
        "thirdeye.effort": "high",
        "thirdeye.hook.model": "gpt-5",
        "thirdeye.tool.duration_ms": 12,
        "thirdeye.tool.interrupted": True,
        "thirdeye.tool.mcp_server": "logfire",
        "thirdeye.permission.reason": "needs approval",
    }


def test_plumbing_and_values_carried_elsewhere_are_dropped():
    assert (
        hook_span_attributes(
            {
                "hook_event_name": "PostToolUse",
                "prompt_id": "p1",
                "turn_id": "t1",
                "scratchpad_dir": "/tmp/x",
                "stop_hook_active": False,
                "error": "Exit code 1",
            }
        )
        == {}
    )


def test_unknown_fields_are_namespaced_not_dropped():
    assert hook_span_attributes({"new_field": 1}) == {"thirdeye.hook.new_field": 1}
