from __future__ import annotations

from thirdeye.env_capture import capture_env, env_to_tag


def test_capture_env_exact_match():
    assert capture_env(
        ["WB_PLAN"], environ={"WB_PLAN": "p", "OTHER": "x"}
    ) == {"WB_PLAN": "p"}


def test_capture_env_prefix_glob():
    assert capture_env(
        ["WB_*"],
        environ={"WB_PLAN": "p", "WB_WAVE": "2", "OTHER": "x"},
    ) == {"WB_PLAN": "p", "WB_WAVE": "2"}


def test_capture_env_whitespace_and_empties():
    assert capture_env(
        [" WB_*  ", "", "OTHER"],
        environ={"WB_PLAN": "p", "OTHER": "x", "NOPE": "n"},
    ) == {"OTHER": "x", "WB_PLAN": "p"}


def test_capture_env_bare_star_rejected():
    assert capture_env(["*"], environ={"WB_PLAN": "p"}) == {}


def test_capture_env_empty_patterns():
    assert capture_env([], environ={"WB_PLAN": "p"}) == {}


def test_capture_env_default_environ(monkeypatch):
    monkeypatch.setenv("WB_PLAN", "p")
    result = capture_env(["WB_*"])
    assert result.get("WB_PLAN") == "p"


def test_env_to_tag_plan():
    assert env_to_tag("WB_PLAN", "session-trace-metadata") == "plan-session-trace-metadata"


def test_env_to_tag_task_doubled():
    assert env_to_tag("WB_TASK", "task-3") == "task-task-3"


def test_env_to_tag_step_with_hash():
    assert env_to_tag("WB_STEP", "test#1") == "step-test#1"


def test_env_to_tag_wave_numeric():
    assert env_to_tag("WB_WAVE", "2") == "wave-2"


def test_env_to_tag_collapses_spaces_and_punct():
    assert env_to_tag("WB_PLAN", "Plan With Spaces!") == "plan-plan-with-spaces"


def test_env_to_tag_otel_attrs_returns_none():
    assert (
        env_to_tag(
            "OTEL_RESOURCE_ATTRIBUTES",
            "service.name=wb,wb.plan=foo",
        )
        is None
    )


def test_env_to_tag_empty_name():
    assert env_to_tag("", "x") is None


def test_env_to_tag_key_collapses_to_empty():
    assert env_to_tag("WB_", "v") is None


def test_env_to_tag_value_sanitizes_to_empty():
    assert env_to_tag("WB_X", "===") is None
