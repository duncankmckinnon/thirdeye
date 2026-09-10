"""Short-lived, coalesced follow-up capture for Copilot hook receipts.

The hook process must return promptly.  A hook therefore starts at most one
detached worker per source session; the worker makes a few bounded attempts to
pick up transcript or SQLite data which arrived just after the hook.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from thirdeye._compat import fsops, proc
from thirdeye._compat.locking import LockMode, LockTimeout, locked
from thirdeye.config import Config
from thirdeye.paths import session_dir
from thirdeye.platforms.copilot.identity import stored_session_id, validate_native_id
from thirdeye.platforms.copilot.state import lock_path
from thirdeye.platforms.copilot.types import SourcePaths
from thirdeye.usage.errlog import log_capture_error

_PLATFORM = "copilot"
_LEASE_FILENAME = "copilot.followup.json"
_LEASE_LOCK_FILENAME = "copilot.followup.lock"
_LEASE_SECONDS = 5.0
_LOCK_PROBE_TIMEOUT = 0.0
_INITIAL_BACKOFF_SECONDS = 0.05
_MAX_BACKOFF_SECONDS = 0.5


def _directory(config: Config, paths: SourcePaths, native_id: str) -> Path:
    return session_dir(config.root, _PLATFORM, stored_session_id(paths, native_id))


def _lease_path(directory: Path) -> Path:
    return directory / _LEASE_FILENAME


def _lease_lock_path(directory: Path) -> Path:
    return directory / _LEASE_LOCK_FILENAME


def _now() -> float:
    return time.time()


def _read_lease(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_lease(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, separators=(",", ":"), sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        fsops.replace(name, path)
        fsops.sync_directory(path.parent)
    except BaseException:
        fsops.unlink(Path(name), missing_ok=True)
        raise


def _claim_lease(config: Config, paths: SourcePaths, native_id: str) -> str | None:
    """Return a new generation, or ``None`` when a live worker owns it."""

    directory = _directory(config, paths, native_id)
    try:
        with locked(_lease_lock_path(directory), LockMode.EXCLUSIVE, timeout=_LOCK_PROBE_TIMEOUT):
            current = _read_lease(_lease_path(directory))
            if current is not None and isinstance(current.get("expires_at"), (int, float)):
                if float(current["expires_at"]) > _now():
                    return None
            generation = uuid4().hex
            _write_lease(
                _lease_path(directory),
                {"generation": generation, "expires_at": _now() + _LEASE_SECONDS},
            )
            return generation
    except (LockTimeout, OSError):
        return None


def _owns_lease(directory: Path, generation: str) -> bool:
    try:
        with locked(_lease_lock_path(directory), LockMode.EXCLUSIVE, timeout=_LOCK_PROBE_TIMEOUT):
            current = _read_lease(_lease_path(directory))
            return bool(current and current.get("generation") == generation)
    except (LockTimeout, OSError):
        return False


def _release_lease(directory: Path, generation: str) -> None:
    try:
        with locked(_lease_lock_path(directory), LockMode.EXCLUSIVE, timeout=_LOCK_PROBE_TIMEOUT):
            current = _read_lease(_lease_path(directory))
            if current is not None and current.get("generation") == generation:
                fsops.unlink(_lease_path(directory), missing_ok=True)
                fsops.sync_directory(directory)
    except (LockTimeout, OSError):
        return


def schedule_followup(config: Config, paths: SourcePaths, native_id: str) -> bool:
    """Coalesce hook follow-ups and spawn one detached, finite worker."""

    validate_native_id(native_id)
    generation = _claim_lease(config, paths, native_id)
    if generation is None:
        return False
    try:
        proc.spawn_detached(
            [
                sys.executable,
                "-m",
                "thirdeye.platforms.copilot.followup",
                "--source-home",
                paths["home"],
                "--session-id",
                native_id,
                "--config-root",
                str(config.root),
                "--generation",
                generation,
            ]
        )
    except Exception as exc:
        _release_lease(_directory(config, paths, native_id), generation)
        log_capture_error(
            thirdeye_home=config.root,
            phase="copilot_followup_spawn",
            error=exc,
            platform=_PLATFORM,
            session_id=native_id,
            silent_fallback=True,
        )
        return False
    return True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-home", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--config-root", required=True)
    parser.add_argument("--generation", required=True)
    return parser.parse_args(argv)


def _archive_lock_available(config: Config, paths: SourcePaths, native_id: str) -> bool:
    """Avoid starting a capture which is already known to block on its lock."""

    directory = _directory(config, paths, native_id)
    try:
        with locked(lock_path(directory), LockMode.EXCLUSIVE, timeout=_LOCK_PROBE_TIMEOUT):
            return True
    except (LockTimeout, OSError):
        return False


def _run(config: Config, paths: SourcePaths, native_id: str, generation: str) -> None:
    """Try follow-up capture for no longer than the lease window."""

    # Import only in runtime composition: source/archive modules remain
    # independent of this detached-worker mechanism.
    from thirdeye.platforms.copilot.capture import capture_session

    directory = _directory(config, paths, native_id)
    deadline = time.monotonic() + _LEASE_SECONDS
    delay = _INITIAL_BACKOFF_SECONDS
    try:
        while time.monotonic() < deadline and _owns_lease(directory, generation):
            if _archive_lock_available(config, paths, native_id):
                try:
                    result = capture_session(config, paths, native_id)
                except Exception as exc:
                    log_capture_error(
                        thirdeye_home=config.root,
                        phase="copilot_followup_capture",
                        error=exc,
                        platform=_PLATFORM,
                        session_id=native_id,
                        silent_fallback=True,
                    )
                else:
                    if result["errors"] == 0 and result["pending"] == 0:
                        return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, _MAX_BACKOFF_SECONDS)
    finally:
        _release_lease(directory, generation)


def main() -> None:
    """Detached entrypoint.  Its arguments contain paths and identity only."""

    try:
        args = _parse_args()
        from thirdeye.platforms.copilot.identity import resolve_sources

        paths = resolve_sources(Path(args.source_home))
        native_id = str(args.session_id)
        validate_native_id(native_id)
        config = Config(root=Path(args.config_root))
        _run(config, paths, native_id, str(args.generation))
    except Exception:
        # This worker is deliberately silent: diagnostics are kept locally and
        # a failed follow-up never changes Copilot's hook outcome.
        return


if __name__ == "__main__":
    main()
