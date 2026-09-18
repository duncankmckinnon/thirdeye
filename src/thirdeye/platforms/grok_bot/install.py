"""Install / uninstall Grok Bot — arms detached store observer (no Cursor hooks)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from thirdeye.platforms.base import Platform
from thirdeye.platforms.grok_bot.constants import (
    DISPLAY_NAME,
    INSTALLED_MARKER,
    PLATFORM_NAME,
    default_state_dir,
)

KICK_ENABLED = "kick.enabled"
WATCHER_ENABLED = "watcher.enabled"
OBSERVER_PID = "observer.pid"
SUPERVISOR_PID = "supervisor.pid"
AGENTS_ROOT_ENV = "THIRDEYE_GROK_BOT_AGENTS_ROOT"
DEFAULT_AGENTS_ROOT = Path.home() / "agent-data" / "agents"


class GrokBotPlatform(Platform):
    name = PLATFORM_NAME
    display_name = DISPLAY_NAME

    def __init__(
        self,
        state_dir: Path | None = None,
        agents_root: Path | str | None = None,
    ) -> None:
        self._state_dir = state_dir or default_state_dir()
        env_root = os.environ.get(AGENTS_ROOT_ENV)
        if agents_root is not None:
            self._agents_root: Path | None = Path(agents_root)
        elif env_root:
            self._agents_root = Path(env_root)
        else:
            self._agents_root = None
        self._observer = None

    @property
    def _marker(self) -> Path:
        return self._state_dir / INSTALLED_MARKER

    @property
    def _kick_flag(self) -> Path:
        return self._state_dir / KICK_ENABLED

    @property
    def _watcher_flag(self) -> Path:
        return self._state_dir / WATCHER_ENABLED

    @property
    def _pid_file(self) -> Path:
        return self._state_dir / OBSERVER_PID

    @property
    def _supervisor_pid_file(self) -> Path:
        return self._state_dir / SUPERVISOR_PID

    def _resolve_agents_root(self) -> Path | None:
        if self._agents_root is not None:
            return self._agents_root
        env_root = os.environ.get(AGENTS_ROOT_ENV)
        if env_root:
            return Path(env_root)
        return DEFAULT_AGENTS_ROOT

    def _spawn_detached_observer(self, agents_root: Path) -> None:
        """Start a worker that outlives this CLI process (not boot/launchd SoT)."""
        from thirdeye._compat import IS_WINDOWS, proc

        self._stop_detached_observer()
        argv = [
            sys.executable,
            "-m",
            "thirdeye.platforms.grok_bot.observer_worker",
            "--supervise",
            "--state-dir",
            str(self._state_dir),
            "--agents-root",
            str(agents_root),
        ]
        try:
            log_out = open(self._state_dir / "observer.out", "ab", buffering=0)
            log_err = open(self._state_dir / "observer.err", "ab", buffering=0)
        except OSError:
            log_out = subprocess.DEVNULL
            log_err = subprocess.DEVNULL

        popen_kw: dict = {
            "args": argv,
            "stdin": subprocess.DEVNULL,
            "stdout": log_out,
            "stderr": log_err,
            "env": os.environ.copy(),
        }
        if IS_WINDOWS:
            popen_kw["creationflags"] = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            popen_kw["start_new_session"] = True
            # Keep log fds open across spawn; close_fds=True + parent close
            # left the worker unable to export on some hosts.
            popen_kw["close_fds"] = False

        try:
            child = subprocess.Popen(**popen_kw)
        except OSError:
            return

        try:
            self._pid_file.write_text(str(child.pid) + "\n", encoding="utf-8")
        except OSError:
            pass

    def _kill_pidfile(self, path: Path) -> None:
        from thirdeye._compat import proc

        pid: int | None = None
        if path.is_file():
            try:
                pid = int(path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                pid = None
            try:
                path.unlink()
            except OSError:
                pass
        if pid is None or not proc.pid_alive(pid):
            return
        try:
            os.kill(pid, 15)
        except OSError:
            return
        for _ in range(20):
            if not proc.pid_alive(pid):
                return
            time.sleep(0.05)
        try:
            os.kill(pid, 9)
        except OSError:
            pass

    def _stop_detached_observer(self) -> None:
        # Supervisor first (owns re-arm), then any leftover worker.
        self._kill_pidfile(self._supervisor_pid_file)
        self._kill_pidfile(self._pid_file)

    def _in_pytest_process(self) -> bool:
        # Same-process reds monkeypatch export_turn; a detached sibling would
        # race the shared watermark and skip the patched call. Detached tests
        # install via a fresh ``python -c`` child that does not import pytest.
        return "pytest" in sys.modules

    def install(self) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._marker.write_text("1\n", encoding="utf-8")
        self._kick_flag.write_text("1\n", encoding="utf-8")
        self._watcher_flag.write_text("1\n", encoding="utf-8")
        root = self._resolve_agents_root()
        if root is None:
            return
        if self._in_pytest_process():
            try:
                from thirdeye.platforms.grok_bot import watch as watch_mod

                self._observer = watch_mod.start_store_observer(self, agents_root=root)
            except Exception:
                self._observer = None
            return
        self._observer = None
        self._spawn_detached_observer(root)

    def is_installed(self) -> bool:
        return self._marker.is_file()

    def is_store_kick_enabled(self) -> bool:
        return self._kick_flag.is_file() or self._watcher_flag.is_file()

    def is_kick_enabled(self) -> bool:
        return self.is_store_kick_enabled()

    def is_mutation_kick_enabled(self) -> bool:
        return self.is_store_kick_enabled()

    def is_action_indicator_enabled(self) -> bool:
        return self.is_store_kick_enabled()

    def is_watcher_running(self) -> bool:
        return self.is_store_kick_enabled()

    def is_passive_running(self) -> bool:
        return self.is_store_kick_enabled()

    def is_running(self) -> bool:
        return self.is_store_kick_enabled()

    def uninstall(self) -> None:
        from thirdeye.platforms.grok_bot import watch as watch_mod

        watch_mod.stop_store_observer(self)
        self._observer = None
        for path in (self._kick_flag, self._watcher_flag):
            if path.exists():
                path.unlink()
        self._stop_detached_observer()
        watermark = self._state_dir / "watermarks.json"
        if watermark.exists():
            watermark.unlink()
        if self._marker.exists():
            self._marker.unlink()
        for name in ("observer.out", "observer.err", "observer.heartbeat"):
            path = self._state_dir / name
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
        if self._state_dir.exists() and not any(self._state_dir.iterdir()):
            self._state_dir.rmdir()
