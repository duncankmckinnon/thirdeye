from __future__ import annotations

import json
from pathlib import Path

import pytest

from thirdeye.config import Config
from thirdeye.paths import session_dir
from thirdeye.platforms.codex import hooks as c_hooks
from thirdeye.platforms.codex.install import CodexPlatform
from thirdeye.store import Store
from thirdeye.usage.store import UsageStore

FIXTURE = Path(__file__).parent / "fixtures" / "usage" / "codex_rollout.jsonl"
FIXTURE_SID = "019fb579-cdda-7a03-86df-65c87b6c4ae2"


@pytest.fixture
def env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    # Sandbox the rollout sessions root so notify() never touches the
    # developer's real ~/.codex/sessions.
    sessions_root = tmp_path / "codex_sessions"
    sessions_root.mkdir()
    monkeypatch.setattr("thirdeye.platforms.codex.rollout.CODEX_SESSIONS_ROOT", sessions_root)
    return tmp_path


def _plant_rollout(env: Path, sid: str = FIXTURE_SID) -> Path:
    """Copy the real rollout fixture into the sandboxed sessions tree."""
    nested = env / "codex_sessions" / "2026" / "07" / "30"
    nested.mkdir(parents=True, exist_ok=True)
    dest = nested / f"rollout-2026-07-30T17-01-26-{sid}.jsonl"
    dest.write_bytes(FIXTURE.read_bytes())
    return dest


def _argv(monkeypatch, payload: dict) -> None:
    """Simulate Codex passing JSON as sys.argv[1]."""
    monkeypatch.setattr("sys.argv", ["thirdeye-codex-notify", json.dumps(payload)])


# -- install -------------------------------------------------------------------


class TestCodexInstall:
    def test_install_creates_config_file(self, tmp_path: Path):
        config_file = tmp_path / ".codex" / "config.toml"
        CodexPlatform(config_file=config_file).install()
        assert config_file.exists()

    def test_install_adds_notify_line(self, tmp_path: Path):
        config_file = tmp_path / ".codex" / "config.toml"
        CodexPlatform(config_file=config_file).install()
        text = config_file.read_text()
        assert "notify" in text
        assert "thirdeye-codex-notify" in text

    def test_install_notify_is_toml_array(self, tmp_path: Path):
        config_file = tmp_path / ".codex" / "config.toml"
        CodexPlatform(config_file=config_file).install()
        text = config_file.read_text()
        # Should be something like: notify = ['thirdeye-codex-notify']
        assert "notify = [" in text

    def test_install_idempotent(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        p = CodexPlatform(config_file=config_file)
        p.install()
        first = config_file.read_text()
        p.install()
        second = config_file.read_text()
        assert first == second

    def test_install_preserves_existing_content(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text('[model]\nprovider = "openai"\n')
        CodexPlatform(config_file=config_file).install()
        text = config_file.read_text()
        assert "thirdeye-codex-notify" in text
        assert 'provider = "openai"' in text


# -- full lifecycle: notify with agent-turn-complete ---------------------------


class TestCodexNotifyLifecycle:
    def _realistic_payload(self, thread_id: str = "codex-e2e-001") -> dict:
        return {
            "type": "agent-turn-complete",
            "thread-id": thread_id,
            "cwd": "/proj/codex",
            "input-messages": [{"role": "user", "content": "fix the bug in main.py"}],
            "last-assistant-message": {
                "role": "assistant",
                "content": "I've fixed the bug by correcting the indentation.",
            },
            "token_usage": {
                "input_tokens": 150,
                "output_tokens": 80,
            },
            "tool_calls": [
                {
                    "name": "edit",
                    "arguments": {"file": "main.py", "line": 42},
                }
            ],
        }

    def test_notify_stores_agent_turn_event(self, monkeypatch, env: Path):
        payload = self._realistic_payload()
        _argv(monkeypatch, payload)
        c_hooks.notify()

        store = Store(Config.load())
        events = list(store.reader("codex-e2e-001").iter_events())
        assert len(events) == 1
        assert events[0]["t"] == "agent_turn"

    def test_event_data_contains_expected_fields(self, monkeypatch, env: Path):
        payload = self._realistic_payload()
        _argv(monkeypatch, payload)
        c_hooks.notify()

        events = list(Store(Config.load()).reader("codex-e2e-001").iter_events())
        data = events[0]["data"]
        assert "input-messages" in data
        assert "last-assistant-message" in data
        assert "token_usage" in data
        assert "tool_calls" in data

    def test_event_data_excludes_routing_keys(self, monkeypatch, env: Path):
        payload = self._realistic_payload()
        _argv(monkeypatch, payload)
        c_hooks.notify()

        events = list(Store(Config.load()).reader("codex-e2e-001").iter_events())
        data = events[0]["data"]
        # cwd and thread-id are routing keys and should be stripped
        assert "cwd" not in data
        assert "thread-id" not in data

    def test_meta_platform_is_codex(self, monkeypatch, env: Path):
        payload = self._realistic_payload()
        _argv(monkeypatch, payload)
        c_hooks.notify()

        m = next(Store(Config.load()).list_sessions())
        assert m.platform == "codex"

    def test_meta_cwd_from_payload(self, monkeypatch, env: Path):
        payload = self._realistic_payload()
        _argv(monkeypatch, payload)
        c_hooks.notify()

        m = next(Store(Config.load()).list_sessions())
        assert m.cwd == "/proj/codex"

    def test_session_id_is_thread_id(self, monkeypatch, env: Path):
        payload = self._realistic_payload(thread_id="my-thread-42")
        _argv(monkeypatch, payload)
        c_hooks.notify()

        m = next(Store(Config.load()).list_sessions())
        assert m.session_id == "my-thread-42"


# -- notify with wrong type is silent no-op ------------------------------------


class TestCodexNotifyNoop:
    def test_non_agent_turn_type_is_noop(self, monkeypatch, env: Path):
        payload = {
            "type": "something-else",
            "thread-id": "noop-001",
            "cwd": "/p",
        }
        _argv(monkeypatch, payload)
        c_hooks.notify()
        assert list(Store(Config.load()).list_sessions()) == []

    def test_missing_type_is_noop(self, monkeypatch, env: Path):
        payload = {
            "thread-id": "noop-002",
            "cwd": "/p",
        }
        _argv(monkeypatch, payload)
        c_hooks.notify()
        assert list(Store(Config.load()).list_sessions()) == []

    def test_empty_argv_is_noop(self, monkeypatch, env: Path):
        monkeypatch.setattr("sys.argv", ["thirdeye-codex-notify"])
        c_hooks.notify()
        assert list(Store(Config.load()).list_sessions()) == []

    def test_invalid_json_argv_is_noop(self, monkeypatch, env: Path):
        monkeypatch.setattr("sys.argv", ["thirdeye-codex-notify", "not valid json"])
        c_hooks.notify()
        assert list(Store(Config.load()).list_sessions()) == []

    def test_missing_thread_id_is_noop(self, monkeypatch, env: Path):
        payload = {
            "type": "agent-turn-complete",
            "cwd": "/p",
        }
        _argv(monkeypatch, payload)
        c_hooks.notify()
        assert list(Store(Config.load()).list_sessions()) == []


# -- no stdout output ----------------------------------------------------------


class TestCodexNoStdout:
    def test_notify_produces_no_stdout(self, monkeypatch, env: Path, capsys):
        payload = {
            "type": "agent-turn-complete",
            "thread-id": "stdout-001",
            "cwd": "/p",
            "input-messages": [],
            "last-assistant-message": {"role": "assistant", "content": "ok"},
            "token_usage": {"input_tokens": 10, "output_tokens": 5},
            "tool_calls": [],
        }
        _argv(monkeypatch, payload)
        c_hooks.notify()
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_noop_notify_produces_no_stdout(self, monkeypatch, env: Path, capsys):
        monkeypatch.setattr("sys.argv", ["thirdeye-codex-notify"])
        c_hooks.notify()
        captured = capsys.readouterr()
        assert captured.out == ""


# -- multiple notifies ---------------------------------------------------------


class TestCodexMultipleNotifies:
    def test_multiple_notifies_same_thread(self, monkeypatch, env: Path):
        for i in range(3):
            payload = {
                "type": "agent-turn-complete",
                "thread-id": "multi-001",
                "cwd": "/p",
                "input-messages": [{"role": "user", "content": f"turn {i}"}],
                "last-assistant-message": {"role": "assistant", "content": f"resp {i}"},
                "token_usage": {"input_tokens": 10, "output_tokens": 5},
                "tool_calls": [],
            }
            _argv(monkeypatch, payload)
            c_hooks.notify()

        events = list(Store(Config.load()).reader("multi-001").iter_events())
        assert len(events) == 3
        assert all(e["t"] == "agent_turn" for e in events)

    def test_different_threads_are_independent(self, monkeypatch, env: Path):
        for tid in ["thread-a", "thread-b"]:
            payload = {
                "type": "agent-turn-complete",
                "thread-id": tid,
                "cwd": "/p",
                "input-messages": [],
                "last-assistant-message": {"role": "assistant", "content": "ok"},
                "token_usage": {},
                "tool_calls": [],
            }
            _argv(monkeypatch, payload)
            c_hooks.notify()

        store = Store(Config.load())
        metas = {m.session_id: m for m in store.list_sessions()}
        assert "thread-a" in metas
        assert "thread-b" in metas


# -- flexible key lookup -------------------------------------------------------


class TestCodexFlexibleKeys:
    def test_snake_case_thread_id(self, monkeypatch, env: Path):
        payload = {
            "type": "agent-turn-complete",
            "thread_id": "snake-001",
            "cwd": "/p",
        }
        _argv(monkeypatch, payload)
        c_hooks.notify()
        events = list(Store(Config.load()).reader("snake-001").iter_events())
        assert len(events) == 1

    def test_camel_case_thread_id(self, monkeypatch, env: Path):
        payload = {
            "type": "agent-turn-complete",
            "threadId": "camel-001",
            "cwd": "/p",
        }
        _argv(monkeypatch, payload)
        c_hooks.notify()
        events = list(Store(Config.load()).reader("camel-001").iter_events())
        assert len(events) == 1


# -- end-to-end: notify wires rollout events + usage ---------------------------


class TestCodexNotifyEndToEnd:
    """Regression guard for the original failure mode: Codex capture silently
    doing nothing. A single simulated notify against a resolvable rollout must
    produce high-fidelity events (tool_call / tool_result) AND token usage."""

    def _payload(self) -> dict:
        return {
            "type": "agent-turn-complete",
            "thread-id": FIXTURE_SID,
            "cwd": "/proj/codex",
            "input-messages": ["do the thing"],
            "last-assistant-message": "Done.",
        }

    def test_notify_produces_tool_events_and_usage(self, monkeypatch, env: Path):
        _plant_rollout(env)
        _argv(monkeypatch, self._payload())
        c_hooks.notify()

        events = list(Store(Config.load()).reader(FIXTURE_SID).iter_events())
        types = [e["t"] for e in events]
        assert "agent_turn" in types
        assert "tool_call" in types
        assert "tool_result" in types

        rows = list(UsageStore(session_dir(env, "codex", FIXTURE_SID)).iter_rows())
        assert rows, "expected non-zero usage rows from the rollout"
        assert all(r.platform == "codex" for r in rows)
        assert any(r.input_tokens and r.output_tokens for r in rows)
