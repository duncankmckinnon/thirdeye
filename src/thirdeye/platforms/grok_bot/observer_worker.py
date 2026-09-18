"""Detached store-mutation observer for Grok Bot (survives ``thirdeye add`` exit).

Install starts a long-lived ``--supervise`` process. That supervisor spawns
short-lived workers that may idle-exit; while ``kick.enabled`` remains, the
next ``store.db`` mutation re-arms a new worker (not boot/launchd SoT).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from thirdeye.platforms.grok_bot.constants import PLATFORM_NAME

IDLE_EXIT_ENV = "THIRDEYE_GROK_BOT_IDLE_EXIT_SECONDS"
_DEFAULT_IDLE_EXIT_SECONDS = 3600.0
_POLL_INTERVAL = 0.1
_OBSERVER_PID = "observer.pid"
_SUPERVISOR_PID = "supervisor.pid"


def idle_exit_seconds() -> float:
    raw = os.environ.get(IDLE_EXIT_ENV)
    if raw is None or raw.strip() == "":
        return _DEFAULT_IDLE_EXIT_SECONDS
    try:
        return max(0.05, float(raw))
    except ValueError:
        return _DEFAULT_IDLE_EXIT_SECONDS


def _file_stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    if not path.is_file():
        return None
    return (int(st.st_size), int(st.st_mtime_ns))


def _kick_enabled(state_dir: Path) -> bool:
    return (state_dir / "kick.enabled").is_file() or (
        state_dir / "watcher.enabled"
    ).is_file()


def _iter_store_dbs(agents_root: Path) -> list[Path]:
    out: list[Path] = []
    try:
        if not agents_root.is_dir():
            return out
        for child in agents_root.iterdir():
            try:
                if not child.is_dir():
                    continue
            except OSError:
                continue
            db = child / "store.db"
            try:
                if db.is_file():
                    out.append(db)
            except OSError:
                continue
    except OSError:
        return out
    return out


def _collect_stamps(agents_root: Path) -> dict[str, tuple[int, int] | None]:
    stamps: dict[str, tuple[int, int] | None] = {}
    for db in _iter_store_dbs(agents_root):
        key = str(db.resolve())
        wal = _file_stamp(Path(str(db) + "-wal"))
        stamps[key] = wal or _file_stamp(db)
    return stamps


def _stamps_changed(
    before: dict[str, tuple[int, int] | None],
    after: dict[str, tuple[int, int] | None],
) -> bool:
    if set(before) != set(after):
        return True
    for key, stamp in after.items():
        if before.get(key) != stamp:
            return True
    return False


def _patch_export_keep_job() -> None:
    """Write durable job files; spawn otel_worker on a copy it may delete."""
    from thirdeye import otel_export

    original_spawn = otel_export._spawn

    def _spawn_on_copy(job_path: Path) -> None:
        try:
            run_path = Path(str(job_path) + ".work")  # avoid *.json — tests glob otel-jobs/*.json
            shutil.copy2(job_path, run_path)
        except OSError:
            original_spawn(job_path)
            return
        original_spawn(run_path)

    otel_export._spawn = _spawn_on_copy  # type: ignore[assignment]


def run_observer(*, state_dir: Path, agents_root: Path) -> int:
    """One watch session: export on mutations until idle, then exit."""
    from thirdeye.platforms.grok_bot.install import GrokBotPlatform
    from thirdeye.platforms.grok_bot.watch import on_store_mutation

    _patch_export_keep_job()
    platform = GrokBotPlatform(state_dir=state_dir, agents_root=agents_root)
    stamps: dict[str, tuple[int, int] | None] = {}
    last_activity = time.monotonic()
    idle_limit = idle_exit_seconds()

    while _kick_enabled(state_dir):
        try:
            for db in _iter_store_dbs(agents_root):
                key = str(db.resolve())
                stamp = _file_stamp(db)
                wal = _file_stamp(Path(str(db) + "-wal"))
                identity = wal or stamp
                prev = stamps.get(key)
                if identity is None:
                    continue
                if prev == identity:
                    continue
                stamps[key] = identity
                if stamp is not None and stamp[0] == 0 and wal is None:
                    continue
                agent_id = db.parent.name
                try:
                    on_store_mutation(
                        platform,
                        store_path=db,
                        agents_root=agents_root,
                        conversation_id=agent_id,
                        agent_id=agent_id,
                        agent_name="",
                        cwd=str(agents_root.parent),
                    )
                    last_activity = time.monotonic()
                except Exception:
                    continue
        except Exception:
            pass
        if time.monotonic() - last_activity > idle_limit:
            break
        time.sleep(_POLL_INTERVAL)
    return 0


def _write_pid(path: Path, pid: int) -> None:
    try:
        path.write_text(str(pid) + "\n", encoding="utf-8")
    except OSError:
        pass


def _spawn_worker(state_dir: Path, agents_root: Path) -> subprocess.Popen | None:
    from thirdeye._compat import IS_WINDOWS

    argv = [
        sys.executable,
        "-m",
        "thirdeye.platforms.grok_bot.observer_worker",
        "--state-dir",
        str(state_dir),
        "--agents-root",
        str(agents_root),
    ]
    try:
        log_out = open(state_dir / "observer.out", "ab", buffering=0)
        log_err = open(state_dir / "observer.err", "ab", buffering=0)
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
        popen_kw["close_fds"] = False

    try:
        child = subprocess.Popen(**popen_kw)
    except OSError:
        return None
    _write_pid(state_dir / _OBSERVER_PID, child.pid)
    return child


def _wait_for_mutation_or_disarm(
    state_dir: Path, agents_root: Path, baseline: dict[str, tuple[int, int] | None]
) -> bool:
    """Return True if stores changed while kick stays armed; False if disarmed."""
    while _kick_enabled(state_dir):
        now = _collect_stamps(agents_root)
        if _stamps_changed(baseline, now):
            return True
        time.sleep(_POLL_INTERVAL)
    return False


def run_supervisor(*, state_dir: Path, agents_root: Path) -> int:
    """Keep re-arming workers after idle-exit while kick remains armed."""
    _write_pid(state_dir / _SUPERVISOR_PID, os.getpid())
    while _kick_enabled(state_dir):
        baseline = _collect_stamps(agents_root)
        child = _spawn_worker(state_dir, agents_root)
        if child is None:
            time.sleep(_POLL_INTERVAL)
            continue
        while child.poll() is None:
            if not _kick_enabled(state_dir):
                try:
                    child.terminate()
                except OSError:
                    pass
                try:
                    child.wait(timeout=2.0)
                except Exception:
                    try:
                        child.kill()
                    except OSError:
                        pass
                break
            time.sleep(_POLL_INTERVAL)
        # Clear worker pidfile if still pointing at this child.
        pid_path = state_dir / _OBSERVER_PID
        try:
            if pid_path.is_file():
                text = pid_path.read_text(encoding="utf-8").strip()
                if text == str(child.pid):
                    pid_path.unlink()
        except OSError:
            pass
        if not _kick_enabled(state_dir):
            break
        # Idle-exit (or crash): wait for next store mutation, then re-arm.
        baseline = _collect_stamps(agents_root)
        if not _wait_for_mutation_or_disarm(state_dir, agents_root, baseline):
            break
    try:
        (state_dir / _SUPERVISOR_PID).unlink()
    except OSError:
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"thirdeye-{PLATFORM_NAME}-observer")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--agents-root", type=Path, required=True)
    parser.add_argument(
        "--supervise",
        action="store_true",
        help="Long-lived re-arm loop (install starts this; workers idle-exit).",
    )
    args = parser.parse_args(argv)
    if args.supervise:
        return run_supervisor(state_dir=args.state_dir, agents_root=args.agents_root)
    return run_observer(state_dir=args.state_dir, agents_root=args.agents_root)


if __name__ == "__main__":
    raise SystemExit(main())
