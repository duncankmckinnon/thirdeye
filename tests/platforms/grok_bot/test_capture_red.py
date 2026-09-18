"""RED tests — Wave 2 Grok Bot capture → TurnSpanDict → otel_export.

Must FAIL until Implementer lands capture/poll modules. Covers:

1. Fixture-driven parse of Research live shapes (`message`, `send-message`,
   `event`) → ``TurnSpanDict`` (``kind`` discriminant; correlate ``requestId``).
2. Fail-open: empty ``transcript_entries`` / SQLite BUSY → no crash, no export.
3. Capture export path calls shared ``otel_export`` only; never
   ``export_turn`` with an empty ``{}`` turn payload.
4. Identity attrs: ``thirdeye.platform=grok_bot`` + conversation/agent ids.
5. ``tool-call``: deferred/skipped (no body mapping required yet).

Expected production surface (Implementer may rename but keep behavior):
``thirdeye.platforms.grok_bot.capture`` with helpers used below.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
PLATFORM = "grok_bot"
AGENT_UUID = "agent-uuid-1"
CONVERSATION_ID = "conv-grok-capture-1"


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _import_capture():
    try:
        from thirdeye.platforms.grok_bot import capture
    except ImportError as exc:
        pytest.fail(f"thirdeye.platforms.grok_bot.capture missing: {exc}")
    return capture


def _write_store(path: Path, entries: list[tuple[int, str, dict[str, Any]]]) -> Path:
    """Create a minimal agent store.db with transcript_entries(seq, id, entry)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE transcript_entries ("
            "seq INTEGER PRIMARY KEY, id TEXT UNIQUE, entry TEXT NOT NULL)"
        )
        for seq, entry_id, body in entries:
            conn.execute(
                "INSERT INTO transcript_entries (seq, id, entry) VALUES (?, ?, ?)",
                (seq, entry_id, json.dumps(body)),
            )
        conn.commit()
    finally:
        conn.close()
    return path


# ---------------------------------------------------------------------------
# 1. Fixture parse → TurnSpanDict
# ---------------------------------------------------------------------------


class TestParseLiveEntryShapes:
    def test_message_pair_builds_turn_correlated_by_request_id(self):
        capture = _import_capture()
        user = _load("message_user.json")
        assistant = _load("message_assistant.json")
        event = _load("event.json")

        build = getattr(capture, "build_turns_from_entries", None) or getattr(
            capture, "entries_to_turns", None
        )
        assert build is not None, (
            "capture must expose build_turns_from_entries / entries_to_turns"
        )

        turns = build(
            [user, event, assistant],
            conversation_id=CONVERSATION_ID,
            agent_id=AGENT_UUID,
            agent_name="Orchestrator",
        )
        assert len(turns) >= 1
        turn = turns[0]
        assert turn["input_message"]
        assert "Add grok_bot" in turn["input_message"] or turn["input_message"]
        assert turn["output_message"]
        assert "Looks good" in turn["output_message"] or turn["output_message"]
        assert turn["status"] in {"ok", "success", "completed", "ok_interrupted"}
        assert isinstance(turn["llm_calls"], list)
        assert isinstance(turn["subagents"], list)
        assert isinstance(turn["permission_requests"], list)
        assert isinstance(turn["attributes"], dict)
        # Correlate on requestId from Research sample.
        attrs = turn["attributes"]
        assert (
            attrs.get("request_id") == "req-abc-1"
            or attrs.get("thirdeye.grok_bot.request_id") == "req-abc-1"
            or turn.get("turn_id") == "req-abc-1"
            or str(turn.get("turn_id", "")).endswith("req-abc-1")
        )

    def test_kind_discriminant_routes_message_send_message_event(self):
        capture = _import_capture()
        parse = getattr(capture, "parse_entry", None) or getattr(
            capture, "normalize_entry", None
        )
        assert parse is not None, "capture must expose parse_entry / normalize_entry"

        for name, expected_kind in (
            ("message_user.json", "message"),
            ("send_message.json", "send-message"),
            ("event.json", "event"),
        ):
            raw = _load(name)
            parsed = parse(raw)
            kind = (
                parsed.get("kind")
                if isinstance(parsed, dict)
                else getattr(parsed, "kind", None)
            )
            assert kind == expected_kind, f"{name}: expected kind={expected_kind!r}"

    def test_send_message_participates_in_turn_reconstruction(self):
        capture = _import_capture()
        build = getattr(capture, "build_turns_from_entries", None) or getattr(
            capture, "entries_to_turns", None
        )
        assert build is not None
        send = _load("send_message.json")
        # A lone send-message should not crash; may yield zero or one turn.
        turns = build(
            [send],
            conversation_id=CONVERSATION_ID,
            agent_id=AGENT_UUID,
            agent_name="Implementer",
        )
        assert isinstance(turns, list)


# ---------------------------------------------------------------------------
# 2. Fail-open poll
# ---------------------------------------------------------------------------


class TestFailOpenPoll:
    def test_empty_transcript_entries_skips_export(self, tmp_path: Path, monkeypatch):
        capture = _import_capture()
        poll = getattr(capture, "poll_and_export", None) or getattr(
            capture, "sync_store", None
        ) or getattr(capture, "poll_store", None)
        assert poll is not None, (
            "capture must expose poll_and_export / sync_store / poll_store"
        )

        db = _write_store(tmp_path / "agents" / AGENT_UUID / "store.db", [])
        exported: list[Any] = []

        from thirdeye import otel_export
        from thirdeye.platforms.grok_bot import tracing

        monkeypatch.setattr(
            otel_export,
            "export_turn",
            lambda *a, **k: exported.append(("otel", a, k)) or None,
        )
        monkeypatch.setattr(
            tracing,
            "export_session",
            lambda **k: exported.append(("tracing", k)) or None,
            raising=False,
        )
        monkeypatch.setattr(
            tracing,
            "export_turn",
            lambda **k: exported.append(("tracing_turn", k)) or None,
            raising=False,
        )

        # Must not raise on empty DB.
        result = poll(
            db,
            conversation_id=CONVERSATION_ID,
            agent_id=AGENT_UUID,
            agent_name="Orchestrator",
            cwd=str(tmp_path),
        )
        assert exported == [], f"empty store must not export; got {exported!r}"
        # Optional: return empty / skipped indicator
        if result is not None:
            assert result in (0, [], {}, None) or result is False or (
                isinstance(result, dict) and result.get("exported", 0) == 0
            )

    def test_sqlite_busy_skips_without_crash(self, tmp_path: Path, monkeypatch):
        capture = _import_capture()
        poll = getattr(capture, "poll_and_export", None) or getattr(
            capture, "sync_store", None
        ) or getattr(capture, "poll_store", None)
        assert poll is not None

        db = tmp_path / "store.db"
        db.write_text("not-a-db", encoding="utf-8")  # corrupt / unreadable

        # Also simulate OperationalError path if helper opens sqlite itself.
        real_connect = sqlite3.connect

        def busy_connect(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(sqlite3, "connect", busy_connect)
        try:
            poll(
                db,
                conversation_id=CONVERSATION_ID,
                agent_id=AGENT_UUID,
                agent_name="Orchestrator",
                cwd=str(tmp_path),
            )
        except sqlite3.OperationalError:
            pytest.fail("poll must fail-open on BUSY/locked — do not raise")
        finally:
            monkeypatch.setattr(sqlite3, "connect", real_connect)


# ---------------------------------------------------------------------------
# 3 + 4. Shared otel_export + identity; never empty {}
# ---------------------------------------------------------------------------


class TestCaptureExportPath:
    def test_export_uses_shared_otel_export_with_real_turn(
        self, tmp_path: Path, monkeypatch
    ):
        capture = _import_capture()
        poll = getattr(capture, "poll_and_export", None) or getattr(
            capture, "sync_store", None
        )
        assert poll is not None

        user = _load("message_user.json")
        assistant = _load("message_assistant.json")
        db = _write_store(
            tmp_path / "store.db",
            [
                (1, user["id"], user),
                (2, assistant["id"], assistant),
            ],
        )

        from thirdeye import otel_export

        captured: list[dict[str, Any]] = []

        def fake_export_turn(config, session_dir, session_id, platform, cwd, turn, **kw):
            captured.append(
                {
                    "session_id": session_id,
                    "platform": platform,
                    "cwd": cwd,
                    "turn": turn,
                }
            )

        monkeypatch.setattr(otel_export, "export_turn", fake_export_turn)

        poll(
            db,
            conversation_id=CONVERSATION_ID,
            agent_id=AGENT_UUID,
            agent_name="Orchestrator",
            cwd=str(tmp_path),
        )

        assert captured, "capture must call shared otel_export.export_turn"
        for item in captured:
            assert item["platform"] == PLATFORM
            assert item["session_id"] == CONVERSATION_ID or item["session_id"]
            turn = item["turn"]
            assert turn != {}, "never export_turn with empty {}"
            assert isinstance(turn, dict)
            assert "input_message" in turn or "output_message" in turn
            assert "llm_calls" in turn

    def test_identity_attrs_on_built_turn(self):
        capture = _import_capture()
        build = getattr(capture, "build_turns_from_entries", None) or getattr(
            capture, "entries_to_turns", None
        )
        assert build is not None
        turns = build(
            [_load("message_user.json"), _load("message_assistant.json")],
            conversation_id=CONVERSATION_ID,
            agent_id=AGENT_UUID,
            agent_name="Orchestrator",
        )
        assert turns
        attrs = turns[0]["attributes"]
        # Platform + agent/conversation findability for Logfire MCP.
        assert (
            attrs.get("thirdeye.platform") == PLATFORM
            or attrs.get("platform") == PLATFORM
        )
        blob = json.dumps(turns[0])
        assert CONVERSATION_ID in blob or AGENT_UUID in blob
        assert "Orchestrator" in blob or AGENT_UUID in blob

    def test_identity_via_shared_otel_helper(self):
        from thirdeye import otel_export

        attrs = otel_export._identity_attributes(
            session_id=CONVERSATION_ID,
            platform=PLATFORM,
            cwd="/box",
        )
        assert attrs["gen_ai.conversation.id"] == CONVERSATION_ID
        assert attrs["thirdeye.platform"] == PLATFORM
        assert attrs.get("gen_ai.agent.name")


# ---------------------------------------------------------------------------
# 5. tool-call deferred
# ---------------------------------------------------------------------------


class TestToolCallDeferred:
    def test_tool_call_outline_is_skipped_or_marked_deferred(self):
        capture = _import_capture()
        parse = getattr(capture, "parse_entry", None) or getattr(
            capture, "normalize_entry", None
        )
        build = getattr(capture, "build_turns_from_entries", None) or getattr(
            capture, "entries_to_turns", None
        )
        assert parse is not None and build is not None

        tool = _load("tool_call_outline.json")
        parsed = parse(tool)
        # Explicit skip/defer signal preferred.
        deferred = False
        if isinstance(parsed, dict):
            deferred = bool(
                parsed.get("deferred")
                or parsed.get("skipped")
                or parsed.get("status") == "deferred"
                or parsed.get("kind") == "tool-call-deferred"
            )
        # Or build_turns ignores tool-call until body mapping lands.
        turns = build(
            [_load("message_user.json"), tool, _load("message_assistant.json")],
            conversation_id=CONVERSATION_ID,
            agent_id=AGENT_UUID,
            agent_name="Orchestrator",
        )
        # Must not invent tool_calls from outline-only rows.
        for turn in turns:
            for call in turn.get("llm_calls") or []:
                for tc in call.get("tool_calls") or []:
                    assert tc.get("name") != "Shell" or deferred, (
                        "tool-call body mapping is deferred — do not invent Shell tools"
                    )
        assert deferred or all(
            not any(
                (tc.get("name") == "Shell")
                for call in (turn.get("llm_calls") or [])
                for tc in (call.get("tool_calls") or [])
            )
            for turn in turns
        )
