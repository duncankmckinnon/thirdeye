from __future__ import annotations

import json
from pathlib import Path

import pytest

from thirdeye.eval.agents.base import (
    AgentAdapter,
    AgentConfig,
    ConfigAdapter,
    OutputFormat,
)

# --- AgentConfig ---


def test_config_defaults():
    c = AgentConfig(command="x", args=["{prompt}"])
    assert c.output_format == OutputFormat.TEXT
    assert c.json_result_key == "result"


def test_config_string_output_format_coerces():
    c = AgentConfig(command="x", args=["{prompt}"], output_format="json")
    assert c.output_format == OutputFormat.JSON


def test_config_rejects_empty_command():
    with pytest.raises(ValueError, match="command"):
        AgentConfig(command="", args=["{prompt}"])


def test_config_rejects_empty_args():
    with pytest.raises(ValueError, match="args"):
        AgentConfig(command="x", args=[])


def test_config_rejects_missing_prompt_placeholder():
    with pytest.raises(ValueError, match="prompt"):
        AgentConfig(command="x", args=["--no-prompt"])


def test_config_to_dict_text_omits_json_keys():
    c = AgentConfig(command="x", args=["{prompt}"])
    d = c.to_dict()
    assert "json_result_key" not in d
    assert d["output_format"] == "text"


def test_config_to_dict_json_includes_keys():
    c = AgentConfig(
        command="x",
        args=["{prompt}"],
        output_format="json",
        json_result_key="foo",
        json_cost_key="bar",
    )
    d = c.to_dict()
    assert d["json_result_key"] == "foo"
    assert d["json_cost_key"] == "bar"


def test_config_from_dict_uses_default_command_when_missing():
    c = AgentConfig.from_dict({"args": ["{prompt}"]}, default_command="x")
    assert c.command == "x"


# --- AgentAdapter (via ConfigAdapter) ---


def test_build_command_substitutes_prompt():
    c = ConfigAdapter(name="x", config=AgentConfig(command="x", args=["-p", "{prompt}"]))
    cmd = c.build_command("hello", Path("/tmp"))
    assert cmd == ["x", "-p", "hello"]


def test_build_command_with_extra_flags():
    c = ConfigAdapter(name="x", config=AgentConfig(command="x", args=["-p", "{prompt}", "--flag"]))
    cmd = c.build_command("h", Path("/"))
    assert cmd == ["x", "-p", "h", "--flag"]


def test_parse_output_text_strips_whitespace():
    c = ConfigAdapter(name="x", config=AgentConfig(command="x", args=["{prompt}"]))
    text, cost = c.parse_output("  hello  \n")
    assert text == "hello"
    assert cost == {}


def test_parse_output_json_extracts_result_and_cost():
    c = ConfigAdapter(
        name="x",
        config=AgentConfig(
            command="x",
            args=["{prompt}"],
            output_format="json",
            json_result_key="result",
            json_cost_key="cost_usd",
        ),
    )
    raw = json.dumps({"result": "the answer", "cost_usd": {"usd": 0.01}})
    text, cost = c.parse_output(raw)
    assert text == "the answer"
    assert cost == {"usd": 0.01}


def test_parse_output_json_falls_back_on_decode_error():
    c = ConfigAdapter(
        name="x",
        config=AgentConfig(
            command="x",
            args=["{prompt}"],
            output_format="json",
        ),
    )
    text, cost = c.parse_output("not json")
    assert text == "not json"
    assert cost == {}


def test_parse_output_json_non_dict_cost_becomes_empty():
    c = ConfigAdapter(
        name="x",
        config=AgentConfig(
            command="x",
            args=["{prompt}"],
            output_format="json",
        ),
    )
    raw = json.dumps({"result": "ok", "cost_usd": 42})  # cost not a dict
    text, cost = c.parse_output(raw)
    assert text == "ok"
    assert cost == {}


# --- ConfigAdapter.from_config ---


def test_config_adapter_from_yaml_entry():
    a = ConfigAdapter.from_config(
        "myagent",
        {
            "command": "myagent",
            "args": ["-p", "{prompt}"],
            "output_format": "json",
        },
    )
    assert a.name == "myagent"
    assert a.config.command == "myagent"
    assert a.config.output_format == OutputFormat.JSON


# --- Additional coverage ---


def test_adapter_to_config_round_trips_via_config():
    a = ConfigAdapter(
        name="x",
        config=AgentConfig(
            command="x",
            args=["-p", "{prompt}"],
            output_format="json",
            json_result_key="r",
            json_cost_key="c",
        ),
    )
    d = a.to_config()
    assert d == {
        "command": "x",
        "args": ["-p", "{prompt}"],
        "output_format": "json",
        "json_result_key": "r",
        "json_cost_key": "c",
    }


def test_from_config_round_trip_preserves_fields():
    entry = {
        "command": "agentbin",
        "args": ["-p", "{prompt}", "--flag"],
        "output_format": "json",
        "json_result_key": "out",
        "json_cost_key": "spend",
    }
    a = ConfigAdapter.from_config("agentbin", entry)
    assert a.to_config() == entry


def test_from_config_defaults_when_command_explicitly_empty():
    # entry has empty string command — `or default_command` kicks in
    a = ConfigAdapter.from_config("fallback", {"command": "", "args": ["{prompt}"]})
    assert a.config.command == "fallback"


def test_from_dict_missing_args_uses_default_prompt_list():
    c = AgentConfig.from_dict({"command": "x"})
    assert c.args == ["{prompt}"]


def test_invalid_output_format_string_raises():
    with pytest.raises(ValueError):
        AgentConfig(command="x", args=["{prompt}"], output_format="bogus")


def test_build_command_with_prompt_containing_spaces():
    c = ConfigAdapter(name="x", config=AgentConfig(command="x", args=["-p", "{prompt}"]))
    cmd = c.build_command("hello world with spaces", Path("/tmp"))
    # prompt stays a single argv element (no shell-splitting)
    assert cmd == ["x", "-p", "hello world with spaces"]


def test_config_rejects_prompt_as_substring_only():
    # validation requires {prompt} as a standalone list element, not embedded
    with pytest.raises(ValueError, match="prompt"):
        AgentConfig(command="x", args=["--input={prompt}"])


def test_parse_output_json_missing_result_key_returns_raw():
    c = ConfigAdapter(
        name="x",
        config=AgentConfig(
            command="x",
            args=["{prompt}"],
            output_format="json",
            json_result_key="missing",
        ),
    )
    raw = json.dumps({"other": "value"})
    text, cost = c.parse_output(raw)
    assert text == raw
    assert cost == {}


def test_output_format_enum_string_values():
    # StrEnum subclasses str — values can be compared loosely
    assert OutputFormat.TEXT == "text"
    assert OutputFormat.JSON == "json"


def test_agent_adapter_is_abstract_base_marker():
    # ABC import path — confirm exposed
    from thirdeye.eval.agents.base import AgentAdapter as AA

    assert AA is AgentAdapter
