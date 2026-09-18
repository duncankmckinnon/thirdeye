"""RED tests — Wave 4 RETARGET: store-mutation kick → export (Duncan Q1).

Duncan lock: passive like Claude/Cursor/Codex/Copilot — **event/action kick**,
not a boot/pidfile long-lived poll daemon as SoT.

Must FAIL until Implementer lands store-mutation kick wiring. Covers:
1. Install registers/enables **store-mutation kick** (not “daemon running after boot”).
2. Simulated ``store.db`` append / mtime / WAL change with kick enabled → export
   via shared ``otel_export`` without the test calling ``poll_and_export`` (or
   ``tick``) as the user happy path — the **kick** invokes capture.
3. Uninstall / ``remove --grok-bot`` (and ``remove --cursor`` co-stop) disables kick.
4. Fail-open empty/BUSY; seq watermark dedupe; no Cursor hooks.json grok entries.
5. Boot/pidfile-daemon-as-SoT assertions are obsolete (not required here).

Expected surface (aliases OK; behavior locked):
``register_store_kick`` / ``on_store_mutation`` / ``notify_store_change`` (or
equivalent) enabled by install; disabled by uninstall.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

PLATFORM = "grok_bot"
AGENT_UUID = "agent-uuid-kick-1"
CONVERSATION_ID = "conv-kick-1"
FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _write_store(path: Path, entries: list[tuple[int, str, dict[str, Any]]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
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
    # Ensure mtime/WAL identity change is observable.
    path.touch()


def _platform(tmp_path: Path):
    from thirdeye.platforms.grok_bot.install import GrokBotPlatform

    return GrokBotPlatform(state_dir=tmp_path / "grok_state")


def _kick_enabled(platform: Any) -> bool:
    """True when install registered store-mutation kick (not boot-daemon SoT)."""
    for name in (
        "is_store_kick_enabled",
        "is_kick_enabled",
        "is_action_indicator_enabled",
        "is_mutation_kick_enabled",
    ):
        fn = getattr(platform, name, None)
        if callable(fn):
            return bool(fn())
    try:
        from thirdeye.platforms.grok_bot import watch as watch_mod
    except ImportError:
        return False
    for name in (
        "is_store_kick_enabled",
        "is_kick_enabled",
        "kick_enabled",
        "is_action_indicator_enabled",
    ):
        fn = getattr(watch_mod, name, None)
        if callable(fn):
            try:
                return bool(fn(platform))
            except TypeError:
                return bool(fn())
    # Explicit kick registration handle.
    for name in ("store_kick", "mutation_kick", "action_indicator"):
        if getattr(platform, name, None) is not None:
            return True
        try:
            from thirdeye.platforms.grok_bot import watch as watch_mod

            if getattr(watch_mod, name, None) is not None:
                return True
        except ImportError:
            pass
    return False


def _invoke_store_kick(
    platform: Any,
    *,
    store_path: Path,
    agents_root: Path,
    **kwargs: Any,
) -> Any:
    """Fire the store-mutation / action-indicator path (not user poll_and_export)."""
    try:
        from thirdeye.platforms.grok_bot import watch as watch_mod
    except ImportError as exc:
        pytest.fail(f"thirdeye.platforms.grok_bot.watch missing: {exc}")

    for name in (
        "on_store_mutation",
        "notify_store_change",
        "handle_store_kick",
        "on_action_indicator",
        "kick_from_store_change",
    ):
        fn = getattr(watch_mod, name, None) or getattr(platform, name, None)
        if callable(fn):
            try:
                return fn(
                    platform,
                    store_path=store_path,
                    agents_root=agents_root,
                    conversation_id=CONVERSATION_ID,
                    agent_id=AGENT_UUID,
                    agent_name="Orchestrator",
                    cwd=str(agents_root.parent),
                    **kwargs,
                )
            except TypeError:
                try:
                    return fn(
                        store_path=store_path,
                        agents_root=agents_root,
                        conversation_id=CONVERSATION_ID,
                        agent_id=AGENT_UUID,
                        agent_name="Orchestrator",
                        cwd=str(agents_root.parent),
                        **kwargs,
                    )
                except TypeError:
                    return fn(store_path)

    pytest.fail(
        "watch/install must expose store-mutation kick "
        "(on_store_mutation / notify_store_change / handle_store_kick) — "
        "tick/run_once alone is not the Duncan Q1 happy path"
    )


# ---------------------------------------------------------------------------
# 1: install enables store-mutation kick (not boot daemon SoT)
# ---------------------------------------------------------------------------


class TestInstallRegistersStoreKick:
    def test_install_enables_store_mutation_kick_not_boot_daemon(self, tmp_path: Path):
        platform = _platform(tmp_path)
        assert not _kick_enabled(platform)
        platform.install()
        assert platform.is_installed()
        assert _kick_enabled(platform), (
            "install must register/enable store-mutation kick "
            "(watcher.enabled flag alone / boot pidfile daemon is not enough)"
        )

    def test_add_grok_bot_cli_enables_kick(self, tmp_path: Path, monkeypatch):
        from thirdeye.cli import main
        from thirdeye.commands import add as add_commands
        from thirdeye.platforms.grok_bot.install import GrokBotPlatform

        platform = GrokBotPlatform(state_dir=tmp_path / "grok_state")
        monkeypatch.setitem(add_commands.PLATFORMS, PLATFORM, lambda **_kw: platform)

        runner = CliRunner()
        result = runner.invoke(main, ["add", "--grok-bot"], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert _kick_enabled(platform)

    def test_add_cursor_co_install_enables_kick_without_hooks_json(
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
        assert _kick_enabled(grok)
        data = json.loads(hooks.read_text(encoding="utf-8")) if hooks.exists() else {}
        blob = json.dumps(data).lower()
        assert "grok_bot" not in blob


# ---------------------------------------------------------------------------
# 2: store mutation kick → otel_export (no user poll_and_export / tick)
# ---------------------------------------------------------------------------


class TestStoreMutationKickExports:
    def test_store_append_kick_exports_via_otel_without_manual_poll(
        self, tmp_path: Path, monkeypatch
    ):
        from thirdeye import otel_export

        platform = _platform(tmp_path)
        platform.install()
        assert _kick_enabled(platform)

        agents_root = tmp_path / "agent-data" / "agents"
        db = _write_store(agents_root / AGENT_UUID / "store.db", [])

        captured: list[dict[str, Any]] = []

        def fake_export_turn(config, session_dir, session_id, platform_name, cwd, turn, **kw):
            captured.append(
                {"session_id": session_id, "platform": platform_name, "turn": turn}
            )

        monkeypatch.setattr(otel_export, "export_turn", fake_export_turn)

        # Simulate bot activity: new transcript rows + observable store change.
        _append_entry(db, 1, "u1", _load("message_user.json"))
        _append_entry(db, 2, "a1", _load("message_assistant.json"))
        time.sleep(0.01)
        db.touch()

        # Must NOT call capture.poll_and_export or watch.tick as the user path.
        _invoke_store_kick(platform, store_path=db, agents_root=agents_root)

        assert captured, (
            "store-mutation kick must export via otel_export without manual poll_and_export"
        )
        for item in captured:
            assert item["platform"] == PLATFORM
            assert item["turn"] != {}
            assert item["turn"].get("input_message") or item["turn"].get("output_message")

    def test_kick_path_is_not_tick_alias_documentation_only(self, tmp_path: Path):
        """Retarget: a dedicated kick entrypoint must exist (tick alone is interim)."""
        try:
            from thirdeye.platforms.grok_bot import watch as watch_mod
        except ImportError as exc:
            pytest.fail(f"watch module missing: {exc}")

        kick_names = (
            "on_store_mutation",
            "notify_store_change",
            "handle_store_kick",
            "on_action_indicator",
            "kick_from_store_change",
        )
        assert any(callable(getattr(watch_mod, n, None)) for n in kick_names), (
            "need an explicit store-mutation/action kick API; "
            "tick/run_once alone does not satisfy Duncan Q1"
        )


# ---------------------------------------------------------------------------
# 3: uninstall / remove disables kick
# ---------------------------------------------------------------------------


class TestUninstallDisablesKick:
    def test_uninstall_disables_store_kick(self, tmp_path: Path):
        platform = _platform(tmp_path)
        platform.install()
        assert _kick_enabled(platform)
        platform.uninstall()
        assert not _kick_enabled(platform)
        assert not platform.is_installed(), "uninstall must clear enablement marker"

    def test_remove_grok_bot_cli_disables_kick(self, tmp_path: Path, monkeypatch):
        from thirdeye.cli import main
        from thirdeye.commands import add as add_commands
        from thirdeye.platforms.grok_bot.install import GrokBotPlatform

        platform = GrokBotPlatform(state_dir=tmp_path / "grok_state")
        platform.install()
        monkeypatch.setitem(add_commands.PLATFORMS, PLATFORM, lambda **_kw: platform)

        runner = CliRunner()
        result = runner.invoke(main, ["remove", "--grok-bot"], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert not _kick_enabled(platform)

    def test_remove_cursor_co_disables_grok_kick(self, tmp_path: Path, monkeypatch):
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
        assert _kick_enabled(grok)

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
        assert not _kick_enabled(grok), (
            "remove --cursor must co-disable grok_bot store-mutation kick"
        )


# ---------------------------------------------------------------------------
# 4: fail-open + watermark via kick path
# ---------------------------------------------------------------------------


class TestKickFailOpenAndWatermark:
    def test_kick_on_empty_store_fail_open(self, tmp_path: Path, monkeypatch):
        from thirdeye import otel_export

        platform = _platform(tmp_path)
        platform.install()
        agents_root = tmp_path / "agents"
        db = _write_store(agents_root / AGENT_UUID / "store.db", [])

        exported: list[Any] = []
        monkeypatch.setattr(
            otel_export, "export_turn", lambda *a, **k: exported.append((a, k))
        )

        _invoke_store_kick(platform, store_path=db, agents_root=agents_root)
        assert exported == []

    def test_kick_on_busy_store_fail_open(self, tmp_path: Path, monkeypatch):
        platform = _platform(tmp_path)
        platform.install()
        agents_root = tmp_path / "agents"
        db = agents_root / AGENT_UUID / "store.db"
        db.parent.mkdir(parents=True)
        db.write_text("not-a-db", encoding="utf-8")

        real_connect = sqlite3.connect

        def busy_connect(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(sqlite3, "connect", busy_connect)
        try:
            _invoke_store_kick(platform, store_path=db, agents_root=agents_root)
        except sqlite3.OperationalError:
            pytest.fail("store-mutation kick must fail-open on BUSY")
        finally:
            monkeypatch.setattr(sqlite3, "connect", real_connect)

    def test_kick_watermark_prevents_duplicate_export(
        self, tmp_path: Path, monkeypatch
    ):
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

        _invoke_store_kick(platform, store_path=db, agents_root=agents_root)
        first = len(captured)
        assert first >= 1

        # Same store identity / no new seq — kick again must not re-export.
        _invoke_store_kick(platform, store_path=db, agents_root=agents_root)
        assert len(captured) == first, "watermark must prevent duplicate exports"

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
        db.touch()
        _invoke_store_kick(platform, store_path=db, agents_root=agents_root)
        assert len(captured) > first


class TestKickMissingPathAndPlatformAttrs:
    def test_kick_on_missing_store_fail_open(self, tmp_path: Path, monkeypatch):
        from thirdeye import otel_export

        platform = _platform(tmp_path)
        platform.install()
        agents_root = tmp_path / "agents"
        agents_root.mkdir(parents=True)
        missing = agents_root / AGENT_UUID / "store.db"

        exported: list = []
        monkeypatch.setattr(
            otel_export, "export_turn", lambda *a, **k: exported.append((a, k))
        )
        _invoke_store_kick(platform, store_path=missing, agents_root=agents_root)
        assert exported == []

    def test_kick_export_sets_thirdeye_platform_grok_bot(
        self, tmp_path: Path, monkeypatch
    ):
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
        captured: list = []

        def fake_export_turn(config, session_dir, session_id, platform_name, cwd, turn, **kw):
            captured.append({"platform": platform_name, "turn": turn})

        monkeypatch.setattr(otel_export, "export_turn", fake_export_turn)
        _invoke_store_kick(platform, store_path=db, agents_root=agents_root)
        assert captured
        for item in captured:
            assert item["platform"] == PLATFORM
            attrs = (item["turn"] or {}).get("attributes") or {}
            assert (
                attrs.get("thirdeye.platform") == PLATFORM
                or attrs.get("platform") == PLATFORM
            ), "exported turn must carry thirdeye.platform=grok_bot"


class TestDocsPassiveHappyPath:
    def test_docs_happy_path_is_not_manual_poll_and_export(self):
        """Docs must not present poll_and_export as the primary happy path."""
        doc = Path(__file__).resolve().parents[3] / "docs" / "grok-bot.md"
        assert doc.is_file(), f"missing {doc}"
        text = doc.read_text(encoding="utf-8")
        # Happy-path section should describe kick/passive install, not lead with
        # "call poll_and_export yourself".
        lower = text.lower()
        assert "passive" in lower or "kick" in lower or "store" in lower
        # If poll_and_export appears, it must be demoted (advanced / optional).
        if "poll_and_export" in text:
            # Rough structure check: first occurrence should not be under a
            # primary "how to run" framing that contradicts install-once.
            idx = text.index("poll_and_export")
            window = text[max(0, idx - 400) : idx + 200].lower()
            assert any(
                marker in window
                for marker in (
                    "advanced",
                    "optional",
                    "manual",
                    "not required",
                    "do **not** need",
                    "do not need",
                    "library",
                )
            ), (
                "docs still present poll_and_export without demoting it from the happy path"
            )



# ---------------------------------------------------------------------------
# Wave 4 RC — real observer (Reviewer Blocking @ 6e45804)
# Mutation must trigger export WITHOUT test calling kick/tick/poll APIs.
# ---------------------------------------------------------------------------


def _wait_until(predicate, *, timeout: float = 3.0, interval: float = 0.05) -> bool:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class TestRealStoreObserver:
    """Install must arm a real FS observer over agents/*/store.db."""

    def test_install_reacts_to_store_mutation_without_test_invoking_kick(
        self, tmp_path: Path, monkeypatch
    ):
        """Writing store.db under the watched root must export without API calls."""
        from thirdeye import otel_export
        from thirdeye.platforms.grok_bot.install import GrokBotPlatform

        agents_root = tmp_path / "agent-data" / "agents"
        agents_root.mkdir(parents=True)
        state = tmp_path / "grok_state"

        # Prefer install(agents_root=...) or env; Implementer may choose either.
        monkeypatch.setenv("THIRDEYE_GROK_BOT_AGENTS_ROOT", str(agents_root))
        try:
            platform = GrokBotPlatform(state_dir=state, agents_root=agents_root)
        except TypeError:
            platform = GrokBotPlatform(state_dir=state)

        captured: list = []

        def fake_export_turn(config, session_dir, session_id, platform_name, cwd, turn, **kw):
            captured.append({"platform": platform_name, "turn": turn})

        monkeypatch.setattr(otel_export, "export_turn", fake_export_turn)

        platform.install()
        assert platform.is_installed()

        # Real mutation — do NOT call on_store_mutation / tick / run_once / poll_and_export.
        db = _write_store(
            agents_root / AGENT_UUID / "store.db",
            [
                (1, "u1", _load("message_user.json")),
                (2, "a1", _load("message_assistant.json")),
            ],
        )
        db.touch()

        assert _wait_until(lambda: len(captured) >= 1, timeout=4.0), (
            "install must arm a real observer (mtime/inotify/FSEvents or short-lived kick "
            "process) so store.db mutation exports without the test calling "
            "on_store_mutation / tick / run_once / poll_and_export"
        )
        assert captured[0]["platform"] == PLATFORM
        assert captured[0]["turn"] != {}

    def test_uninstall_stops_observer_so_later_mutation_does_not_export(
        self, tmp_path: Path, monkeypatch
    ):
        from thirdeye import otel_export
        from thirdeye.platforms.grok_bot.install import GrokBotPlatform

        agents_root = tmp_path / "agent-data" / "agents"
        agents_root.mkdir(parents=True)
        monkeypatch.setenv("THIRDEYE_GROK_BOT_AGENTS_ROOT", str(agents_root))
        try:
            platform = GrokBotPlatform(
                state_dir=tmp_path / "grok_state", agents_root=agents_root
            )
        except TypeError:
            platform = GrokBotPlatform(state_dir=tmp_path / "grok_state")

        captured: list = []
        monkeypatch.setattr(
            otel_export,
            "export_turn",
            lambda *a, **k: captured.append(a) or None,
        )

        platform.install()
        db = _write_store(
            agents_root / AGENT_UUID / "store.db",
            [
                (1, "u1", _load("message_user.json")),
                (2, "a1", _load("message_assistant.json")),
            ],
        )
        assert _wait_until(lambda: len(captured) >= 1, timeout=4.0), (
            "observer must fire at least once before uninstall (same contract as prior test)"
        )
        before = len(captured)

        platform.uninstall()
        assert not platform.is_installed()

        _append_entry(
            db,
            3,
            "u2",
            {
                **_load("message_user.json"),
                "id": "entry_user_2",
                "requestId": "req-after-uninstall",
                "content": "should not export",
            },
        )
        _append_entry(
            db,
            4,
            "a2",
            {
                **_load("message_assistant.json"),
                "id": "entry_asst_2",
                "requestId": "req-after-uninstall",
                "content": "nope",
            },
        )
        db.touch()

        # Give any stale observer time; must not grow.
        import time

        time.sleep(1.0)
        assert len(captured) == before, (
            "uninstall must stop the observer — later store mutations must not export"
        )

    def test_remove_cursor_stops_observer_reaction(
        self, tmp_path: Path, monkeypatch
    ):
        from thirdeye.cli import main
        from thirdeye.commands import add as add_commands
        from thirdeye import otel_export
        from thirdeye.platforms.cursor.install import CursorPlatform
        from thirdeye.platforms.grok_bot.install import GrokBotPlatform

        agents_root = tmp_path / "agent-data" / "agents"
        agents_root.mkdir(parents=True)
        monkeypatch.setenv("THIRDEYE_GROK_BOT_AGENTS_ROOT", str(agents_root))
        hooks = tmp_path / "hooks.json"
        monkeypatch.setattr(
            "thirdeye.platforms.cursor.install.shutil.which",
            lambda _name: None,
        )
        try:
            grok = GrokBotPlatform(
                state_dir=tmp_path / "grok_state", agents_root=agents_root
            )
        except TypeError:
            grok = GrokBotPlatform(state_dir=tmp_path / "grok_state")
        cursor = CursorPlatform(hooks_file=hooks)

        captured: list = []
        monkeypatch.setattr(
            otel_export,
            "export_turn",
            lambda *a, **k: captured.append(a) or None,
        )

        def resolve(flag: str, force: bool = False):
            if flag == "cursor":
                return cursor
            if flag == PLATFORM:
                return grok
            return add_commands.PLATFORMS[flag]()

        monkeypatch.setattr(add_commands, "_resolve_platform", resolve)

        cursor.install()
        grok.install()
        db = _write_store(
            agents_root / AGENT_UUID / "store.db",
            [
                (1, "u1", _load("message_user.json")),
                (2, "a1", _load("message_assistant.json")),
            ],
        )
        assert _wait_until(lambda: len(captured) >= 1, timeout=4.0)
        before = len(captured)

        runner = CliRunner()
        result = runner.invoke(main, ["remove", "--cursor"], catch_exceptions=False)
        assert result.exit_code == 0, result.output

        _append_entry(
            db,
            3,
            "u3",
            {
                **_load("message_user.json"),
                "id": "u3",
                "requestId": "req-post-remove",
                "content": "after remove",
            },
        )
        db.touch()
        import time

        time.sleep(1.0)
        assert len(captured) == before, (
            "remove --cursor must co-stop the grok observer"
        )
        # Still no hooks.json grok pollution.
        if hooks.exists():
            assert "grok_bot" not in hooks.read_text(encoding="utf-8").lower()



# ---------------------------------------------------------------------------
# Wave 4 RC² — detached worker survives install-process exit (Reviewer @ ac662fc)
# ---------------------------------------------------------------------------


_INSTALLER_SCRIPT = r"""
import os
import sys
from pathlib import Path

state = Path(os.environ["THIRDEYE_HOME"])
agents = Path(os.environ["THIRDEYE_GROK_BOT_AGENTS_ROOT"])
state.mkdir(parents=True, exist_ok=True)
agents.mkdir(parents=True, exist_ok=True)

from thirdeye.config import Config, LogfireSettings
from thirdeye.platforms.grok_bot.install import GrokBotPlatform

Config(root=state).write_logfire_settings(LogfireSettings(enabled=True, token="test-token"))
try:
    platform = GrokBotPlatform(state_dir=state / "platforms" / "grok_bot", agents_root=agents)
except TypeError:
    platform = GrokBotPlatform(state_dir=state / "platforms" / "grok_bot")
platform.install()
sys.exit(0)
"""


class TestDetachedObserverSurvivesInstallerExit:
    """In-process daemon threads die with the CLI; install must leave a detached worker."""

    def test_mutation_after_installer_exit_still_exports(self, tmp_path: Path):
        import subprocess
        import sys
        import time

        thirdeye_home = tmp_path / "thirdeye"
        agents_root = tmp_path / "agent-data" / "agents"
        agents_root.mkdir(parents=True)
        thirdeye_home.mkdir(parents=True)

        env = {
            **dict(**{k: v for k, v in __import__("os").environ.items()}),
            "THIRDEYE_HOME": str(thirdeye_home),
            "THIRDEYE_GROK_BOT_AGENTS_ROOT": str(agents_root),
        }
        # Ensure package import works in child.
        proc = subprocess.run(
            [sys.executable, "-c", _INSTALLER_SCRIPT],
            cwd=str(Path(__file__).resolve().parents[3]),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

        from thirdeye.paths import otel_jobs_dir

        jobs = otel_jobs_dir(thirdeye_home)
        before = set(jobs.glob("*.json")) if jobs.exists() else set()

        # Mutate after installer process is gone — no kick/tick calls from this test.
        db = _write_store(
            agents_root / AGENT_UUID / "store.db",
            [
                (1, "u1", _load("message_user.json")),
                (2, "a1", _load("message_assistant.json")),
            ],
        )
        db.touch()

        def new_jobs() -> bool:
            if not jobs.exists():
                return False
            return len(set(jobs.glob("*.json")) - before) >= 1

        assert _wait_until(new_jobs, timeout=6.0), (
            "after install process exits, a detached path-watch worker must still "
            "export on store.db mutation (in-process daemon thread is insufficient)"
        )

    def test_uninstall_via_subprocess_stops_detached_worker(self, tmp_path: Path):
        import subprocess
        import sys
        import time

        thirdeye_home = tmp_path / "thirdeye"
        agents_root = tmp_path / "agent-data" / "agents"
        agents_root.mkdir(parents=True)
        thirdeye_home.mkdir(parents=True)
        env = {
            **dict(**{k: v for k, v in __import__("os").environ.items()}),
            "THIRDEYE_HOME": str(thirdeye_home),
            "THIRDEYE_GROK_BOT_AGENTS_ROOT": str(agents_root),
        }
        proc = subprocess.run(
            [sys.executable, "-c", _INSTALLER_SCRIPT],
            cwd=str(Path(__file__).resolve().parents[3]),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr

        from thirdeye.paths import otel_jobs_dir

        jobs = otel_jobs_dir(thirdeye_home)
        # Seed one export so we know the worker was alive.
        db = _write_store(
            agents_root / AGENT_UUID / "store.db",
            [
                (1, "u1", _load("message_user.json")),
                (2, "a1", _load("message_assistant.json")),
            ],
        )
        assert _wait_until(
            lambda: jobs.exists() and any(jobs.glob("*.json")), timeout=6.0
        ), "detached worker never exported initial mutation"
        before = set(jobs.glob("*.json"))

        uninstall = r"""
import os, sys
from pathlib import Path
from thirdeye.platforms.grok_bot.install import GrokBotPlatform
state = Path(os.environ["THIRDEYE_HOME"]) / "platforms" / "grok_bot"
agents = Path(os.environ["THIRDEYE_GROK_BOT_AGENTS_ROOT"])
try:
    p = GrokBotPlatform(state_dir=state, agents_root=agents)
except TypeError:
    p = GrokBotPlatform(state_dir=state)
p.uninstall()
sys.exit(0)
"""
        proc2 = subprocess.run(
            [sys.executable, "-c", uninstall],
            cwd=str(Path(__file__).resolve().parents[3]),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc2.returncode == 0, proc2.stderr

        _append_entry(
            db,
            3,
            "u2",
            {
                **_load("message_user.json"),
                "id": "u2",
                "requestId": "post-uninstall",
                "content": "should not export",
            },
        )
        db.touch()
        time.sleep(1.5)
        after = set(jobs.glob("*.json")) if jobs.exists() else set()
        assert after == before, (
            "uninstall must stop the detached worker — later mutations must not export"
        )

    def test_default_agents_root_env_documented_or_honored(self, tmp_path: Path):
        """Optional: THIRDEYE_GROK_BOT_AGENTS_ROOT is the box agents root knob."""
        from thirdeye.platforms.grok_bot import install as install_mod

        assert hasattr(install_mod, "AGENTS_ROOT_ENV") or hasattr(
            install_mod, "DEFAULT_AGENTS_ROOT"
        ) or "THIRDEYE_GROK_BOT_AGENTS_ROOT" in Path(
            install_mod.__file__
        ).read_text(encoding="utf-8"), (
            "install module should name AGENTS_ROOT_ENV / default agents root "
            "for box-side observer arming"
        )

