"""RED tests — Wave 4 passive Grok Bot capture (install once → auto export).

Must FAIL until Implementer lands passive watcher wiring. Duncan contract:
after ``thirdeye add --grok-bot`` / ``--cursor`` co-install, turns upload to
Logfire automatically — no manual ``poll_and_export`` happy path.

Covers:
1. Install starts/enables passive watcher (not marker-only).
2. New ``transcript_entries`` are exported via shared ``otel_export`` through
   the watcher/install contract (tests do not call ``poll_and_export`` as the
   user path).
3. ``remove --grok-bot`` stops watcher; ``remove --cursor`` co-stops it.
4. Empty/BUSY store → fail-open; never ``export_turn({})``.
5. Seq watermark → no duplicate exports on re-poll.
6. No grok entries in Cursor ``hooks.json``.

Expected surface (names may alias; behavior locked):
``GrokBotPlatform.install/uninstall``, ``is_watcher_running`` (or equivalent),
and ``thirdeye.platforms.grok_bot.watch`` (``tick`` / ``run_once``).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

PLATFORM = "grok_bot"
AGENT_UUID = "agent-uuid-passive-1"
CONVERSATION_ID = "conv-passive-1"
FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _write_store(path: Path, entries: list[tuple[int, str, dict[str, Any]]]) -> Path:
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


def _append_entry(path: Path, seq: int, entry_id: str, body: dict[str, Any]) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "INSERT INTO transcript_entries (seq, id, entry) VALUES (?, ?, ?)",
            (seq, entry_id, json.dumps(body)),
        )
        conn.commit()
    finally:
        conn.close()


def _platform(tmp_path: Path, **kwargs: Any):
    from thirdeye.platforms.grok_bot.install import GrokBotPlatform

    return GrokBotPlatform(state_dir=tmp_path / "grok_state", **kwargs)


def _watcher_running(platform: Any) -> bool:
    for name in ("is_watcher_running", "is_passive_running", "is_running"):
        fn = getattr(platform, name, None)
        if callable(fn):
            return bool(fn())
    # Module-level helpers
    try:
        from thirdeye.platforms.grok_bot import watch as watch_mod
    except ImportError:
        return False
    for name in ("is_running", "is_watcher_running", "status"):
        fn = getattr(watch_mod, name, None)
        if callable(fn):
            result = fn(platform) if name != "status" else fn()
            if isinstance(result, dict):
                return bool(result.get("running") or result.get("enabled"))
            return bool(result)
    return False


def _tick_watcher(platform: Any, *, agents_root: Path, **kwargs: Any) -> Any:
    """Advance the passive loop once (test harness — not the user happy path)."""
    try:
        from thirdeye.platforms.grok_bot import watch as watch_mod
    except ImportError as exc:
        pytest.fail(f"thirdeye.platforms.grok_bot.watch missing: {exc}")

    for name in ("tick", "run_once", "poll_once", "sync_once"):
        fn = getattr(watch_mod, name, None)
        if callable(fn):
            try:
                return fn(
                    platform,
                    agents_root=agents_root,
                    conversation_id=CONVERSATION_ID,
                    **kwargs,
                )
            except TypeError:
                try:
                    return fn(agents_root=agents_root, conversation_id=CONVERSATION_ID, **kwargs)
                except TypeError:
                    return fn()
    pytest.fail(
        "watch module must expose tick/run_once/poll_once for the passive loop"
    )


# ---------------------------------------------------------------------------
# 1 + 6: install enables passive watcher; no Cursor hooks.json
# ---------------------------------------------------------------------------


class TestInstallStartsPassiveWatcher:
    def test_install_enables_watcher_not_marker_only(self, tmp_path: Path):
        platform = _platform(tmp_path)
        assert not _watcher_running(platform)
        platform.install()
        assert platform.is_installed()
        assert _watcher_running(platform), (
            "install must start/enable passive watcher — marker-only is insufficient"
        )

    def test_add_grok_bot_cli_starts_watcher(self, tmp_path: Path, monkeypatch):
        from thirdeye.cli import main
        from thirdeye.commands import add as add_commands
        from thirdeye.platforms.grok_bot.install import GrokBotPlatform

        state = tmp_path / "grok_state"
        platform = GrokBotPlatform(state_dir=state)
        monkeypatch.setitem(add_commands.PLATFORMS, PLATFORM, lambda **_kw: platform)

        runner = CliRunner()
        result = runner.invoke(main, ["add", "--grok-bot"], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert platform.is_installed()
        assert _watcher_running(platform)

    def test_add_cursor_co_install_starts_grok_watcher(
        self, tmp_path: Path, monkeypatch
    ):
        from thirdeye.cli import main
        from thirdeye.commands import add as add_commands
        from thirdeye.platforms.cursor.install import CursorPlatform
        from thirdeye.platforms.grok_bot.install import GrokBotPlatform

        hooks = tmp_path / "hooks.json"
        monkeypatch.setattr(
            "thirdeye.platforms.cursor.install.shutil.which",
            lambda _name: None,
        )
        cursor = CursorPlatform(hooks_file=hooks)
        grok = GrokBotPlatform(state_dir=tmp_path / "grok_state")

        def resolve(flag: str, force: bool = False):
            if flag == "cursor":
                return cursor
            if flag == PLATFORM:
                return grok
            return add_commands.PLATFORMS[flag]()

        monkeypatch.setattr(add_commands, "_resolve_platform", resolve)

        runner = CliRunner()
        result = runner.invoke(main, ["add", "--cursor"], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert grok.is_installed()
        assert _watcher_running(grok)
        # 6: still not Cursor hooks.json
        data = json.loads(hooks.read_text(encoding="utf-8")) if hooks.exists() else {}
        blob = json.dumps(data).lower()
        assert "grok_bot" not in blob
        assert "grok-bot" not in blob or "thirdeye-cursor" in blob


# ---------------------------------------------------------------------------
# 2: passive path exports new entries without user calling poll_and_export
# ---------------------------------------------------------------------------


class TestPassiveExportWithoutManualPoll:
    def test_watcher_exports_new_entries_via_otel_export(
        self, tmp_path: Path, monkeypatch
    ):
        from thirdeye import otel_export

        platform = _platform(tmp_path)
        platform.install()
        assert _watcher_running(platform)

        agents_root = tmp_path / "agent-data" / "agents"
        db = _write_store(
            agents_root / AGENT_UUID / "store.db",
            [
                (1, "u1", _load("message_user.json")),
                (2, "a1", _load("message_assistant.json")),
            ],
        )

        captured: list[dict[str, Any]] = []

        def fake_export_turn(config, session_dir, session_id, platform_name, cwd, turn, **kw):
            captured.append(
                {
                    "session_id": session_id,
                    "platform": platform_name,
                    "turn": turn,
                }
            )

        monkeypatch.setattr(otel_export, "export_turn", fake_export_turn)

        # Critical: do NOT call capture.poll_and_export here — drive watcher tick.
        _tick_watcher(
            platform,
            agents_root=agents_root,
            agent_id=AGENT_UUID,
            agent_name="Orchestrator",
            cwd=str(tmp_path),
            store_path=db,
        )

        assert captured, "passive watcher must export via otel_export without manual poll"
        for item in captured:
            assert item["platform"] == PLATFORM
            assert item["turn"] != {}
            assert item["turn"].get("input_message") or item["turn"].get("output_message")


# ---------------------------------------------------------------------------
# 3: uninstall / remove --cursor stops watcher
# ---------------------------------------------------------------------------


class TestUninstallStopsWatcher:
    def test_uninstall_stops_watcher(self, tmp_path: Path):
        platform = _platform(tmp_path)
        platform.install()
        assert _watcher_running(platform)
        platform.uninstall()
        assert not _watcher_running(platform), "uninstall must stop the passive watcher"

    def test_remove_grok_bot_cli_stops_watcher(self, tmp_path: Path, monkeypatch):
        from thirdeye.cli import main
        from thirdeye.commands import add as add_commands
        from thirdeye.platforms.grok_bot.install import GrokBotPlatform

        platform = GrokBotPlatform(state_dir=tmp_path / "grok_state")
        platform.install()
        monkeypatch.setitem(add_commands.PLATFORMS, PLATFORM, lambda **_kw: platform)

        runner = CliRunner()
        result = runner.invoke(main, ["remove", "--grok-bot"], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert not _watcher_running(platform)

    def test_remove_cursor_co_stops_grok_watcher(self, tmp_path: Path, monkeypatch):
        """Q8 default: remove --cursor also stops the grok_bot watcher."""
        from thirdeye.cli import main
        from thirdeye.commands import add as add_commands
        from thirdeye.platforms.cursor.install import CursorPlatform
        from thirdeye.platforms.grok_bot.install import GrokBotPlatform

        hooks = tmp_path / "hooks.json"
        monkeypatch.setattr(
            "thirdeye.platforms.cursor.install.shutil.which",
            lambda _name: None,
        )
        cursor = CursorPlatform(hooks_file=hooks)
        grok = GrokBotPlatform(state_dir=tmp_path / "grok_state")
        cursor.install()
        grok.install()
        assert _watcher_running(grok)

        def resolve(flag: str, force: bool = False):
            if flag == "cursor":
                return cursor
            if flag == PLATFORM:
                return grok
            return add_commands.PLATFORMS[flag]()

        monkeypatch.setattr(add_commands, "_resolve_platform", resolve)

        runner = CliRunner()
        result = runner.invoke(main, ["remove", "--cursor"], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert not _watcher_running(grok), (
            "remove --cursor must co-stop the grok_bot passive watcher"
        )


# ---------------------------------------------------------------------------
# 4: fail-open
# ---------------------------------------------------------------------------


class TestPassiveFailOpen:
    def test_empty_store_tick_does_not_export_or_raise(
        self, tmp_path: Path, monkeypatch
    ):
        from thirdeye import otel_export

        platform = _platform(tmp_path)
        platform.install()
        agents_root = tmp_path / "agents"
        _write_store(agents_root / AGENT_UUID / "store.db", [])

        exported: list[Any] = []
        monkeypatch.setattr(
            otel_export,
            "export_turn",
            lambda *a, **k: exported.append((a, k)),
        )

        _tick_watcher(
            platform,
            agents_root=agents_root,
            agent_id=AGENT_UUID,
            agent_name="Orchestrator",
            cwd=str(tmp_path),
        )
        assert exported == []

    def test_busy_store_tick_fail_open(self, tmp_path: Path, monkeypatch):
        platform = _platform(tmp_path)
        platform.install()
        agents_root = tmp_path / "agents"
        (agents_root / AGENT_UUID).mkdir(parents=True)
        (agents_root / AGENT_UUID / "store.db").write_text("not-a-db", encoding="utf-8")

        real_connect = sqlite3.connect

        def busy_connect(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(sqlite3, "connect", busy_connect)
        try:
            _tick_watcher(
                platform,
                agents_root=agents_root,
                agent_id=AGENT_UUID,
                agent_name="Orchestrator",
                cwd=str(tmp_path),
            )
        except sqlite3.OperationalError:
            pytest.fail("passive tick must fail-open on BUSY — do not raise")
        finally:
            monkeypatch.setattr(sqlite3, "connect", real_connect)


# ---------------------------------------------------------------------------
# 5: seq watermark / no duplicate exports
# ---------------------------------------------------------------------------


class TestSeqWatermark:
    def test_repoll_does_not_reexport_same_seq(self, tmp_path: Path, monkeypatch):
        from thirdeye import otel_export

        platform = _platform(tmp_path)
        platform.install()
        agents_root = tmp_path / "agents"
        db = _write_store(
            agents_root / AGENT_UUID / "store.db",
            [
                (1, "u1", _load("message_user.json")),
                (2, "a1", _load("message_assistant.json")),
            ],
        )

        captured: list[Any] = []

        def fake_export_turn(*args, **kwargs):
            turn = args[5] if len(args) > 5 else kwargs.get("turn")
            captured.append(turn)

        monkeypatch.setattr(otel_export, "export_turn", fake_export_turn)

        _tick_watcher(
            platform,
            agents_root=agents_root,
            agent_id=AGENT_UUID,
            agent_name="Orchestrator",
            cwd=str(tmp_path),
            store_path=db,
        )
        first = len(captured)
        assert first >= 1

        _tick_watcher(
            platform,
            agents_root=agents_root,
            agent_id=AGENT_UUID,
            agent_name="Orchestrator",
            cwd=str(tmp_path),
            store_path=db,
        )
        assert len(captured) == first, (
            "seq watermark must prevent duplicate exports on re-poll"
        )

        # New seq still exports.
        _append_entry(
            db,
            3,
            "u2",
            {
                **_load("message_user.json"),
                "id": "entry_user_2",
                "requestId": "req-new-2",
                "content": "follow-up",
            },
        )
        _append_entry(
            db,
            4,
            "a2",
            {
                **_load("message_assistant.json"),
                "id": "entry_asst_2",
                "requestId": "req-new-2",
                "content": "ok",
            },
        )
        _tick_watcher(
            platform,
            agents_root=agents_root,
            agent_id=AGENT_UUID,
            agent_name="Orchestrator",
            cwd=str(tmp_path),
            store_path=db,
        )
        assert len(captured) > first
