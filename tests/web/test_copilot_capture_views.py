"""Copilot web views: raw archive evidence and projected turn/usage surfaces."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from tests.shared.copilot_projection_fixtures import (
    TURN_ONE_ID,
    TURN_THREE_ID,
    TURN_TWO_ID,
    USAGE_MODEL_ONE,
    USAGE_MODEL_TWO,
    USAGE_TOKENS_ONE,
    USAGE_TOKENS_TWO,
    seed_observed_six_call_projection,
    seed_two_main_interaction_projection,
)
from thirdeye.config import LogfireSettings
from thirdeye.platforms.copilot.archive import commit_batch
from thirdeye.platforms.copilot.identity import resolve_sources, stored_session_id
from thirdeye.platforms.copilot.types import SourceBatch, SourceRecord
from thirdeye.turns import session_turns

pytest.importorskip("starlette")

NATIVE_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"
SOURCE_TS = "2026-09-10T17:08:44.557Z"
USAGE_MODEL = "gpt-4.1-copilot-sentinel"
USAGE_INPUT_TOKENS = 424242


def _capture_synthetic_batch(web_config, tmp_path: Path) -> str:
    """Archive labeled synthetic child evidence; no turn projection is involved."""
    paths = resolve_sources(tmp_path / "copilot-home")
    records: list[SourceRecord] = [
        {
            "source_id": f"{paths['source_key']}/{NATIVE_ID}/child-message",
            "source_kind": "transcript",
            "native_session_id": NATIVE_ID,
            "ts": SOURCE_TS,
            "observed_at": "2026-09-10T17:08:45.000Z",
            "payload": {
                "type": "user.message",
                "id": "child-message",
                "timestamp": SOURCE_TS,
                "agentId": "child-agent-id",
                "data": {"content": "Read alpha.txt and beta.txt as the explore child."},
            },
            "locator": {
                "file": "events.jsonl",
                "file_generation": "fixture-gen",
                "byte_offset": 512,
            },
        },
        {
            "source_id": f"hook/{NATIVE_ID}/child-stop",
            "source_kind": "hook",
            "native_session_id": NATIVE_ID,
            "ts": SOURCE_TS,
            "observed_at": "2026-09-10T17:08:45.001Z",
            "payload": {
                "event": "agentStop",
                "hook_payload": {
                    "sessionId": NATIVE_ID,
                    "agentId": "child-agent-id",
                    "response": "42",
                },
            },
            "locator": {"observation_id": "child-stop", "event": "agentStop"},
        },
    ]
    batch: SourceBatch = {
        "source_key": paths["source_key"],
        "native_session_id": NATIVE_ID,
        "cwd": "/fixture/workspace",
        "records": records,
        "next_cursor": {"fixture": 1},
        "diagnostics": [],
    }
    commit_batch(web_config, paths, batch)
    return stored_session_id(paths, NATIVE_ID)


def test_generic_event_views_show_raw_child_and_hook_evidence(
    client, web_config, tmp_path: Path
) -> None:
    stored_id = _capture_synthetic_batch(web_config, tmp_path)

    session = client.get(f"/sessions/{stored_id}")
    tree = client.get(f"/sessions/{stored_id}/tree")
    detail = client.get(f"/sessions/{stored_id}/events/1")
    search = client.get("/search?q=alpha.txt&platform=copilot")

    assert (
        session.status_code == tree.status_code == detail.status_code == search.status_code == 200
    )
    assert b"copilot" in session.content
    assert b"copilot_transcript" in tree.content
    assert b"copilot_hook" in tree.content
    assert b"user_message" not in tree.content
    assert b"tool_call" not in tree.content
    assert b"child-agent-id" in detail.content
    assert NATIVE_ID.encode() in detail.content
    assert SOURCE_TS.encode() in detail.content
    assert b"child-message" in detail.content or b"child-stop" in detail.content
    assert b'"schema_version": 1' in detail.content
    assert b'"source_kind": "hook"' in detail.content
    assert stored_id.encode() in search.content
    assert b"alpha.txt" in search.content


def test_copilot_database_events_render_in_generic_tree(client, web_config, tmp_path: Path) -> None:
    paths = resolve_sources(tmp_path / "copilot-home")
    records: list[SourceRecord] = [
        {
            "source_id": f"{paths['source_key']}/{NATIVE_ID}/usage-row",
            "source_kind": "database",
            "native_session_id": NATIVE_ID,
            "ts": SOURCE_TS,
            "observed_at": "2026-09-10T17:08:45.000Z",
            "payload": {
                "table": "assistant_usage_events",
                "model": "gpt-4.1",
                "input_tokens": 100,
            },
            "locator": {"table": "assistant_usage_events", "rowid": 7},
        },
        {
            "source_id": f"{paths['source_key']}/{NATIVE_ID}/workspace-meta",
            "source_kind": "metadata",
            "native_session_id": NATIVE_ID,
            "ts": None,
            "observed_at": "2026-09-10T17:08:45.001Z",
            "payload": {
                "file": "workspace.yaml",
                "cwd": "/fixture/workspace",
            },
            "locator": {"file": "workspace.yaml"},
        },
    ]
    batch: SourceBatch = {
        "source_key": paths["source_key"],
        "native_session_id": NATIVE_ID,
        "cwd": "/fixture/workspace",
        "records": records,
        "next_cursor": {"fixture": 1},
        "diagnostics": [],
    }
    commit_batch(web_config, paths, batch)
    stored_id = stored_session_id(paths, NATIVE_ID)

    tree = client.get(f"/sessions/{stored_id}/tree")
    detail_db = client.get(f"/sessions/{stored_id}/events/0")
    detail_meta = client.get(f"/sessions/{stored_id}/events/1")

    assert tree.status_code == detail_db.status_code == detail_meta.status_code == 200
    assert b"copilot_database" in tree.content
    assert b"copilot_metadata" in tree.content
    assert b"assistant_usage_events" in detail_db.content
    assert b'"schema_version": 1' in detail_db.content
    assert b'"source_kind": "database"' in detail_db.content
    assert b"workspace.yaml" in detail_meta.content
    assert b'"source_kind": "metadata"' in detail_meta.content
    assert b"user_message" not in tree.content


def test_copilot_sessions_are_excluded_from_index_turn_query(
    client, web_config, tmp_path: Path
) -> None:
    stored_id = _capture_synthetic_batch(web_config, tmp_path)
    store = client.app.state.store
    meta = store.get_meta(stored_id)

    assert session_turns(meta, store) == []

    without_turn_filter = client.get("/?platform=copilot&since=2020-01-01")
    with_turn_query = client.get("/?platform=copilot&since=2020-01-01&turn_query=alpha.txt")

    assert without_turn_filter.status_code == with_turn_query.status_code == 200
    assert stored_id.encode() in without_turn_filter.content
    assert stored_id.encode() not in with_turn_query.content


def test_copilot_projected_turns_participate_in_index_turn_query(
    client, web_config, tmp_path: Path
) -> None:
    stored_id = seed_two_main_interaction_projection(web_config, tmp_path)
    store = client.app.state.store
    meta = store.get_meta(stored_id)
    turns = session_turns(meta, store)

    assert [turn["turn_id"] for turn in turns] == [TURN_ONE_ID, TURN_TWO_ID]

    first_turn_query = client.get("/?platform=copilot&since=2020-01-01&turn_query=alpha.txt")
    second_turn_query = client.get("/?platform=copilot&since=2020-01-01&turn_query=final%20sum")
    cross_turn_query = client.get(
        "/?platform=copilot&since=2020-01-01&turn_query=alpha.txt,final%20sum"
    )

    assert first_turn_query.status_code == second_turn_query.status_code == 200
    assert stored_id.encode() in first_turn_query.content
    assert stored_id.encode() in second_turn_query.content
    assert stored_id.encode() not in cross_turn_query.content


def test_copilot_session_usage_page_shows_projected_usage_rows(
    client, web_config, tmp_path: Path
) -> None:
    stored_id = seed_two_main_interaction_projection(web_config, tmp_path)

    usage = client.get(f"/sessions/{stored_id}/usage")

    assert usage.status_code == 200
    body = usage.text
    tbody = body.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    assert tbody.count("<tr>") == 2
    assert USAGE_MODEL_ONE in tbody
    assert USAGE_MODEL_TWO in tbody
    assert str(USAGE_TOKENS_ONE) in tbody
    assert str(USAGE_TOKENS_TWO) in tbody
    assert str(USAGE_TOKENS_TWO * 2) not in body


def test_copilot_session_usage_page_labels_native_billing_and_latency(
    client, web_config, tmp_path: Path
) -> None:
    stored_id = seed_observed_six_call_projection(web_config, tmp_path)

    usage = client.get(f"/sessions/{stored_id}/usage")

    assert usage.status_code == 200
    body = usage.text
    assert "copilot native billing (nano-AIU)" in body
    assert "duration_ms" in body
    assert "TTFT (ms)" in body
    assert "174125000" in body
    tbody = body.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    assert tbody.count("<tr>") == 6
    assert "operation.cost" not in body
    assert "$" not in tbody


def test_copilot_projected_session_search_tag_and_eval_routes(
    client, app, web_config, tmp_path: Path, monkeypatch
) -> None:
    stored_id = seed_two_main_interaction_projection(web_config, tmp_path)
    app.state.config = app.state.config.write_logfire_settings(
        LogfireSettings(api_key="dataset-key")
    )
    added: list[dict] = []

    class Client:
        def __init__(self, api_key):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def create_dataset(self, **kwargs):
            pass

        def add_cases(self, name, *, cases):
            added.extend(cases)

    package = ModuleType("logfire")
    experimental = ModuleType("logfire.experimental")
    api_client = ModuleType("logfire.experimental.api_client")
    api_client.LogfireAPIClient = Client
    monkeypatch.setitem(sys.modules, "logfire", package)
    monkeypatch.setitem(sys.modules, "logfire.experimental", experimental)
    monkeypatch.setitem(sys.modules, "logfire.experimental.api_client", api_client)

    session = client.get(f"/sessions/{stored_id}")
    tree = client.get(f"/sessions/{stored_id}/tree")
    detail = client.get(f"/sessions/{stored_id}/events/0")
    search = client.get("/search?q=alpha.txt&platform=copilot")
    tagged = client.post(f"/sessions/{stored_id}/events/0/tags", data={"tag": "review"})
    tagged_index = client.get("/?tag=review&since=2020-01-01")
    untagged_index = client.get("/?tag=missing-tag&since=2020-01-01")
    export = client.post(
        "/sessions/logfire-dataset",
        data={
            "dataset_name": "copilot-turns",
            "dataset_scope": "turn",
            "platform": "copilot",
            "since": "2020-01-01",
        },
    )

    assert session.status_code == tree.status_code == search.status_code == 200
    assert detail.status_code == 200
    assert b"copilot" in session.content
    assert b"/fixture/workspace" in session.content
    assert b"copilot_transcript" in tree.content
    assert b"user_message" not in tree.content
    assert b"tool_call" not in tree.content
    assert b"alpha.txt" in detail.content
    assert b'"schema_version": 1' in detail.content
    assert stored_id.encode() in search.content
    assert b"alpha.txt" in search.content
    assert tagged.status_code == 200
    assert b"review" in tagged.content
    assert tagged_index.status_code == untagged_index.status_code == 200
    assert stored_id.encode() in tagged_index.content
    assert stored_id.encode() not in untagged_index.content

    assert export.status_code == 200
    assert "Sent 2 turns" in export.text
    names = [case["name"] for case in added]
    assert names == [
        f"{stored_id}:{TURN_ONE_ID}",
        f"{stored_id}:{TURN_TWO_ID}",
    ]
    assert TURN_THREE_ID not in "".join(names)
    assert not any(case["name"].endswith(":child") for case in added)
    first, second = added
    assert "turn" in first["inputs"] and "turn" in second["inputs"]
    assert len(first["inputs"]["turn"]["events"]) == 3
    assert len(second["inputs"]["turn"]["events"]) == 2
    event_types = {event.get("t") for case in added for event in case["inputs"]["turn"]["events"]}
    assert event_types <= {
        "copilot_transcript",
        "copilot_database",
        "copilot_hook",
        "copilot_metadata",
    }
    assert "user_message" not in event_types
    assert "tool_call" not in event_types


def test_copilot_session_usage_page_has_no_token_rows_without_projection(
    client, web_config, tmp_path: Path
) -> None:
    paths = resolve_sources(tmp_path / "copilot-home")
    records: list[SourceRecord] = [
        {
            "source_id": f"{paths['source_key']}/{NATIVE_ID}/usage-row",
            "source_kind": "database",
            "native_session_id": NATIVE_ID,
            "ts": SOURCE_TS,
            "observed_at": "2026-09-10T17:08:45.000Z",
            "payload": {
                "table": "assistant_usage_events",
                "model": USAGE_MODEL,
                "input_tokens": USAGE_INPUT_TOKENS,
            },
            "locator": {"table": "assistant_usage_events", "rowid": 7},
        },
    ]
    batch: SourceBatch = {
        "source_key": paths["source_key"],
        "native_session_id": NATIVE_ID,
        "cwd": "/fixture/workspace",
        "records": records,
        "next_cursor": {"fixture": 1},
        "diagnostics": [],
    }
    commit_batch(web_config, paths, batch)
    stored_id = stored_session_id(paths, NATIVE_ID)

    usage = client.get(f"/sessions/{stored_id}/usage")

    assert usage.status_code == 200
    assert b"usage" in usage.content
    assert USAGE_MODEL.encode() not in usage.content
    assert b"input_tokens" not in usage.content
    assert str(USAGE_INPUT_TOKENS).encode() not in usage.content
