from __future__ import annotations

from thirdeye.config import Config
from thirdeye.store import Store
from thirdeye.turns import filter_turns, session_turns

from tests.shared.copilot_projection_fixtures import (
    TURN_ONE_ID,
    TURN_TWO_ID,
    seed_two_main_interaction_projection,
)


def test_claude_turns_are_bounded_by_user_and_assistant_messages(tmp_path):
    store = Store(Config(root=tmp_path))
    with store.open_session("claude-session", platform="claude", cwd="/project") as writer:
        writer.append("user_message", {"prompt": "first"})
        writer.append("tool_call", {"tool_use_id": "one"})
        writer.append("assistant_message", {"text": "done"})
        writer.append("user_message", {"prompt": "second"})
    meta = store.get_meta("claude-session")

    turns = session_turns(meta, store)

    assert [turn["turn_id"] for turn in turns] == ["0", "3"]
    assert [event["t"] for event in turns[0]["events"]] == [
        "user_message",
        "tool_call",
        "assistant_message",
    ]
    assert turns[1]["events"][0]["data"]["prompt"] == "second"


def test_codex_turns_use_explicit_ids_and_successive_agent_turn_boundaries(tmp_path):
    store = Store(Config(root=tmp_path))
    with store.open_session("codex-session", platform="codex", cwd="/project") as writer:
        writer.append("agent_turn", {"turn-id": "turn-a"})
        writer.append("tool_call", {"call_id": "call-a"})
        writer.append("agent_turn", {"turn-id": "turn-b"})
        writer.append("assistant_message", {"text": "second"})
    meta = store.get_meta("codex-session")

    turns = session_turns(meta, store)

    assert [turn["turn_id"] for turn in turns] == ["turn-a", "turn-b"]
    assert [event["seq"] for event in turns[0]["events"]] == [0, 1]
    assert filter_turns([meta], store, "turn-b") == [turns[1]]


def test_turn_query_searches_every_session_and_ands_terms_within_turn(tmp_path):
    store = Store(Config(root=tmp_path))
    metas = []
    for sid, tool, result in (
        ("first-session", "apply_patch", "Logfire updated"),
        ("second-session", "Read", "Logfire inspected"),
    ):
        with store.open_session(sid, platform="claude", cwd="/project") as writer:
            writer.append("user_message", {"prompt": "dataset work"})
            writer.append("tool_call", {"tool_name": tool})
            writer.append("tool_result", {"content": result})
            writer.append("assistant_message", {"text": "done"})
        metas.append(store.get_meta(sid))

    matches = filter_turns(metas, store, query="apply_patch, logfire")

    assert [turn["session_id"] for turn in matches] == ["first-session"]


def test_copilot_session_turns_use_projected_main_interactions_not_child_slices(tmp_path):
    config = Config(root=tmp_path / "thirdeye")
    store = Store(config)
    stored_id = seed_two_main_interaction_projection(config, tmp_path)
    meta = store.get_meta(stored_id)

    turns = session_turns(meta, store)

    assert [turn["turn_id"] for turn in turns] == [TURN_ONE_ID, TURN_TWO_ID]
    assert len(turns[0]["events"]) == 3
    assert len(turns[1]["events"]) == 2
    assert "alpha.txt" in turns[0]["events"][0]["data"]["source_record"]["payload"]["data"]["content"]
    assert "final sum" in turns[1]["events"][0]["data"]["source_record"]["payload"]["data"]["content"]
    assert filter_turns([meta], store, query="alpha.txt") == [turns[0]]
    assert filter_turns([meta], store, query="final sum") == [turns[1]]
    assert filter_turns([meta], store, query="alpha.txt,final sum") == []


def test_turn_query_terms_cannot_match_across_different_turns(tmp_path):
    store = Store(Config(root=tmp_path))
    with store.open_session("session", platform="claude", cwd="/project") as writer:
        writer.append("user_message", {"prompt": "alpha"})
        writer.append("assistant_message", {"text": "done"})
        writer.append("user_message", {"prompt": "beta"})
        writer.append("assistant_message", {"text": "done"})
    meta = store.get_meta("session")

    assert filter_turns([meta], store, query="alpha,beta") == []
