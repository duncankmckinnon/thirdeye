"""Generic web views render raw Copilot archive events without projections."""

from __future__ import annotations

from pathlib import Path

import pytest

from thirdeye.platforms.copilot.archive import commit_batch
from thirdeye.platforms.copilot.identity import resolve_sources, stored_session_id
from thirdeye.platforms.copilot.types import SourceBatch, SourceRecord

pytest.importorskip("starlette")

NATIVE_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"
SOURCE_TS = "2026-09-10T17:08:44.557Z"


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
                "schema_version": 1,
                "type": "user.message",
                "id": "child-message",
                "timestamp": SOURCE_TS,
                "agentId": "child-agent-id",
                "data": {"content": "Read alpha.txt and beta.txt as the explore child."},
            },
            "locator": {"file": "events.jsonl", "file_generation": "fixture-gen", "byte_offset": 512},
        },
        {
            "source_id": f"hook/{NATIVE_ID}/child-stop",
            "source_kind": "hook",
            "native_session_id": NATIVE_ID,
            "ts": SOURCE_TS,
            "observed_at": "2026-09-10T17:08:45.001Z",
            "payload": {
                "schema_version": 1,
                "event": "agentStop",
                "hook_payload": {"sessionId": NATIVE_ID, "agentId": "child-agent-id", "response": "42"},
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


def test_generic_event_views_show_raw_child_and_hook_evidence(client, web_config, tmp_path: Path) -> None:
    stored_id = _capture_synthetic_batch(web_config, tmp_path)

    session = client.get(f"/sessions/{stored_id}")
    tree = client.get(f"/sessions/{stored_id}/tree")
    detail = client.get(f"/sessions/{stored_id}/events/1")
    search = client.get("/search?q=alpha.txt&platform=copilot")

    assert session.status_code == tree.status_code == detail.status_code == search.status_code == 200
    assert b"copilot" in session.content
    assert b"copilot_transcript" in tree.content
    assert b"copilot_hook" in tree.content
    assert b"user_message" not in tree.content
    assert b"tool_call" not in tree.content
    assert b"child-agent-id" in detail.content
    assert NATIVE_ID.encode() in detail.content
    assert SOURCE_TS.encode() in detail.content
    assert b"child-message" in detail.content or b"child-stop" in detail.content
    assert stored_id.encode() in search.content
    assert b"alpha.txt" in search.content
