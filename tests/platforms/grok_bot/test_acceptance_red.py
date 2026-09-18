"""RED acceptance tests for the grok_bot platform (Q4 co-install + export).

These tests must FAIL until the Implementer lands production code. They cover:

1. ``thirdeye add --cursor`` (and the setup wizard) always co-installs
   ``grok_bot``.
2. ``grok_bot`` is a distinct ``PLATFORMS`` entry — not Cursor ``hooks.json``
   entries.
3. Export emits GenAI-shaped identity attrs (conversation/session id, stable
   agent id, ``thirdeye.platform=grok_bot``), mirroring claude/codex/cursor.
4. Export reuses shared ``thirdeye.otel_export`` — no second export client.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner


GROK_BOT = "grok_bot"


# ---------------------------------------------------------------------------
# 1 + 2: registry + co-install (add --cursor / setup wizard)
# ---------------------------------------------------------------------------


class TestGrokBotPlatformRegistry:
    def test_grok_bot_is_registered_in_platforms(self):
        from thirdeye.commands.add import PLATFORMS

        assert GROK_BOT in PLATFORMS, (
            "grok_bot must be a separate PLATFORMS entry (not Cursor hooks.json)"
        )

    def test_grok_bot_platform_is_not_cursor(self):
        from thirdeye.commands.add import PLATFORMS
        from thirdeye.platforms.cursor.install import CursorPlatform

        assert GROK_BOT in PLATFORMS
        assert PLATFORMS[GROK_BOT] is not CursorPlatform
        # Class attribute on every Platform subclass (see CursorPlatform / ClaudePlatform).
        assert getattr(PLATFORMS[GROK_BOT], "name", None) == GROK_BOT


class TestCursorAddCoInstallsGrokBot:
    def test_add_cursor_cli_also_installs_grok_bot(self, tmp_path: Path, monkeypatch):
        """``thirdeye add --cursor`` must always co-install grok_bot (Q4)."""
        from thirdeye.cli import main
        from thirdeye.commands import add as add_commands
        from thirdeye.platforms.cursor.install import CursorPlatform

        hooks_file = tmp_path / "hooks.json"
        # Isolate Cursor hooks from the real ~/.cursor/hooks.json.
        monkeypatch.setattr(
            "thirdeye.platforms.cursor.install.shutil.which",
            lambda _name: None,
        )

        # Prefer constructing platforms with injectable paths when supported.
        cursor = CursorPlatform(hooks_file=hooks_file)
        monkeypatch.setattr(
            add_commands,
            "_resolve_platform",
            lambda flag, force=False: cursor if flag == "cursor" else add_commands.PLATFORMS[flag](),
        )

        # grok_bot platform must exist and accept an injectable state dir.
        assert GROK_BOT in add_commands.PLATFORMS
        grok_cls = add_commands.PLATFORMS[GROK_BOT]
        grok_state = tmp_path / "grok_bot"
        try:
            grok = grok_cls(state_dir=grok_state)  # type: ignore[call-arg]
        except TypeError:
            grok = grok_cls()

        # Re-bind resolve so cursor install is the one under test, and so we
        # can observe grok_bot installation side effects.
        installed: list[str] = []

        original_cursor_install = cursor.install

        def cursor_install_and_track() -> None:
            original_cursor_install()
            installed.append("cursor")

        monkeypatch.setattr(cursor, "install", cursor_install_and_track)

        # When add --cursor runs, production code should also install grok_bot.
        # Patch PLATFORMS[grok_bot] factory used by co-install if needed.
        real_grok_install = grok.install

        def grok_install_and_track() -> None:
            real_grok_install()
            installed.append(GROK_BOT)

        monkeypatch.setattr(grok, "install", grok_install_and_track)
        monkeypatch.setitem(add_commands.PLATFORMS, GROK_BOT, lambda **_kw: grok)

        runner = CliRunner()
        result = runner.invoke(main, ["add", "--cursor"], catch_exceptions=False)
        assert result.exit_code == 0, result.output

        assert "cursor" in installed
        assert GROK_BOT in installed, (
            "thirdeye add --cursor must always co-install grok_bot"
        )
        assert cursor.is_installed()
        assert grok.is_installed()

    def test_cursor_co_install_does_not_put_grok_bot_in_hooks_json(
        self, tmp_path: Path, monkeypatch
    ):
        """grok_bot is its own platform — Cursor hooks.json must not grow grok entries."""
        from thirdeye.commands import add as add_commands
        from thirdeye.platforms.cursor.constants import HOOK_BIN_NAME, TRACED_EVENTS
        from thirdeye.platforms.cursor.install import CursorPlatform

        hooks_file = tmp_path / "hooks.json"
        monkeypatch.setattr(
            "thirdeye.platforms.cursor.install.shutil.which",
            lambda _name: None,
        )
        cursor = CursorPlatform(hooks_file=hooks_file)

        assert GROK_BOT in add_commands.PLATFORMS
        grok_cls = add_commands.PLATFORMS[GROK_BOT]
        try:
            grok = grok_cls(state_dir=tmp_path / "grok_bot")  # type: ignore[call-arg]
        except TypeError:
            grok = grok_cls()

        # Simulate the intended production co-install sequence.
        cursor.install()
        grok.install()

        data = json.loads(hooks_file.read_text(encoding="utf-8"))
        hooks = data.get("hooks") or {}
        for event in TRACED_EVENTS:
            entries = hooks.get(event) or []
            for entry in entries:
                command = entry.get("command", "")
                assert GROK_BOT not in str(command).lower()
                assert "grok" not in Path(str(command)).name.lower() or HOOK_BIN_NAME in str(
                    command
                )

        # Still a separate installed platform.
        assert grok.is_installed()
        assert cursor.is_installed()


class TestSetupWizardCoInstallsGrokBot:
    def test_setup_platform_labels_include_grok_bot(self):
        from thirdeye.commands import setup as setup_commands

        labels = getattr(setup_commands, "_PLATFORM_LABELS", None)
        assert labels is not None
        assert GROK_BOT in labels

    def test_installing_cursor_via_setup_helper_co_installs_grok_bot(
        self, tmp_path: Path, monkeypatch
    ):
        """Setup wizard path that installs cursor must also install grok_bot."""
        from thirdeye.commands import add as add_commands
        from thirdeye.commands import setup as setup_commands
        from thirdeye.platforms.cursor.install import CursorPlatform

        hooks_file = tmp_path / "hooks.json"
        monkeypatch.setattr(
            "thirdeye.platforms.cursor.install.shutil.which",
            lambda _name: None,
        )
        cursor = CursorPlatform(hooks_file=hooks_file)
        assert GROK_BOT in add_commands.PLATFORMS
        grok_cls = add_commands.PLATFORMS[GROK_BOT]
        try:
            grok = grok_cls(state_dir=tmp_path / "grok_bot")  # type: ignore[call-arg]
        except TypeError:
            grok = grok_cls()

        installed: list[str] = []
        monkeypatch.setattr(
            cursor,
            "install",
            lambda: installed.append("cursor") or CursorPlatform.install(cursor),
        )
        monkeypatch.setattr(
            grok,
            "install",
            lambda: installed.append(GROK_BOT) or grok_cls.install(grok),
        )

        # Prefer the setup helper if it exists; otherwise call through add.
        install_fn = getattr(setup_commands, "_install_tracing", None) or getattr(
            setup_commands, "_install_platform", None
        )
        if install_fn is None:
            pytest.skip("setup tracing helper not found — add --cursor co-install still required")

        # Wire resolve so setup uses our temp-backed instances.
        monkeypatch.setattr(
            add_commands,
            "_resolve_platform",
            lambda name, force=False: {"cursor": cursor, GROK_BOT: grok}[name],
        )

        install_fn("cursor", cursor)

        assert "cursor" in installed
        assert GROK_BOT in installed, (
            "setup wizard cursor install must always co-install grok_bot"
        )


# ---------------------------------------------------------------------------
# 3: GenAI-shaped export identity attrs
# ---------------------------------------------------------------------------


class TestGrokBotExportIdentityAttrs:
    def test_identity_attributes_include_genai_session_agent_and_platform(self):
        from thirdeye import otel_export

        attrs = otel_export._identity_attributes(
            session_id="conv-grok-1",
            platform=GROK_BOT,
            cwd="/proj",
        )

        # Conversation / session id (GenAI semantic convention used by siblings).
        assert attrs.get("gen_ai.conversation.id") == "conv-grok-1"
        # Stable agent id — platform-scoped name (mirrors claude/codex/cursor).
        agent = attrs.get("gen_ai.agent.name")
        assert agent is not None
        assert GROK_BOT in str(agent) or str(agent).startswith("grok")
        # Platform attr must identify grok_bot.
        assert attrs.get("thirdeye.platform") == GROK_BOT

    def test_grok_bot_export_path_sets_identity_on_spans(self, tmp_path: Path, monkeypatch):
        """Platform export path must stamp GenAI identity attrs onto emitted spans."""
        # Import the grok_bot export/tracing surface once Implementer adds it.
        # Until then this import failure is the correct RED signal.
        try:
            import thirdeye.platforms.grok_bot.tracing as tracing
        except ImportError as exc:
            pytest.fail(f"grok_bot tracing module missing: {exc}")

        from thirdeye import otel_export

        captured: list[dict] = []

        def fake_export_spans(*args, **kwargs):
            captured.append({"args": args, "kwargs": kwargs})
            return True

        def fake_export_turn(*args, **kwargs):
            captured.append({"args": args, "kwargs": kwargs})
            return None

        monkeypatch.setattr(otel_export, "export_spans", fake_export_spans, raising=False)
        monkeypatch.setattr(otel_export, "export_turn", fake_export_turn, raising=False)

        # Prefer a small helper if the platform exposes one; else call export_turn
        # the same way sibling platforms do.
        export_fn = getattr(tracing, "export_session", None) or getattr(
            tracing, "export_turn", None
        )
        assert export_fn is not None, (
            "grok_bot tracing must expose an export entrypoint that uses otel_export"
        )

        # Minimal call — Implementer may adjust signature; the assertion below
        # is what matters for acceptance.
        try:
            export_fn(session_id="conv-grok-2", cwd=str(tmp_path))
        except TypeError:
            # Signature mismatch is OK for RED as long as module exists; try
            # keyword-heavy form used by shared export.
            export_fn(  # type: ignore[misc]
                session_id="conv-grok-2",
                platform=GROK_BOT,
                cwd=str(tmp_path),
            )

        assert captured, "grok_bot export must call shared otel_export (export_turn/export_spans)"
        # Platform argument on the shared client must be grok_bot.
        blob = repr(captured)
        assert GROK_BOT in blob


# ---------------------------------------------------------------------------
# 4: shared otel_export — no second export client
# ---------------------------------------------------------------------------


class TestGrokBotReusesSharedOtelExport:
    def test_grok_bot_modules_import_shared_otel_export(self):
        import importlib
        import inspect
        import sys

        try:
            package = importlib.import_module("thirdeye.platforms.grok_bot")
        except ImportError as exc:
            pytest.fail(f"thirdeye.platforms.grok_bot package missing: {exc}")

        # Collect source of grok_bot submodules that should talk to export.
        sources: list[str] = []
        for mod_name in list(sys.modules):
            if mod_name.startswith("thirdeye.platforms.grok_bot"):
                mod = sys.modules[mod_name]
                try:
                    sources.append(inspect.getsource(mod))
                except (OSError, TypeError):
                    continue

        # Also eagerly import likely modules so they appear above.
        for suffix in ("tracing", "export", "hooks", "install", "live_spans"):
            try:
                mod = importlib.import_module(f"thirdeye.platforms.grok_bot.{suffix}")
                sources.append(inspect.getsource(mod))
            except ImportError:
                continue

        assert sources, "expected at least one grok_bot module with source"
        joined = "\n".join(sources)
        assert "thirdeye.otel_export" in joined or "from thirdeye import otel_export" in joined, (
            "grok_bot must reuse shared thirdeye.otel_export"
        )
        # Guard against a second OTLP/Logfire client.
        for banned in (
            "OTLPSpanExporter(",
            "OTLPExporter(",
            "logfire.configure(",
            "TracerProvider(",
        ):
            assert banned not in joined, (
                f"grok_bot must not create a second export client ({banned})"
            )

    def test_otel_export_is_single_configure_site_for_grok_bot(self):
        """Configure/export client ownership stays in otel_export, not grok_bot."""
        import importlib
        import inspect

        otel_export = importlib.import_module("thirdeye.otel_export")
        assert hasattr(otel_export, "export_turn") or hasattr(otel_export, "export_spans")

        try:
            grok_tracing = importlib.import_module("thirdeye.platforms.grok_bot.tracing")
        except ImportError as exc:
            pytest.fail(f"grok_bot tracing missing: {exc}")

        src = inspect.getsource(grok_tracing)
        assert "logfire.configure" not in src
        assert "OTLPSpanExporter" not in src
