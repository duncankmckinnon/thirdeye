from __future__ import annotations

import functools
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

from thirdeye.paths import usage_log_path


def log_capture_error(
    *,
    thirdeye_home: Path,
    phase: str,
    error: BaseException | None = None,
    message: str = "",
    platform: str = "",
    session_id: str = "",
    source_path: str = "",
    level: str = "warn",
    silent_fallback: bool = False,
) -> None:
    """Append one entry to <thirdeye_home>/logs/usage-errors.jsonl.

    Never raises. If the log itself can't be written, falls back to stderr
    unless ``silent_fallback`` is true.
    """
    try:
        log = usage_log_path(thirdeye_home)
        log.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "level": level,
            "platform": platform,
            "session_id": session_id,
            "phase": phase,
            "source_path": source_path,
            "error_class": type(error).__name__ if error is not None else "",
            "message": message or (str(error) if error else ""),
            "traceback": "".join(traceback.format_exception(error)) if error else "",
        }
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except Exception as fallback_err:
        if not silent_fallback:
            sys.stderr.write(
                f"[thirdeye usage] error log write failed: {fallback_err!r}; "
                f"original phase={phase!r} error={error!r}\n"
            )


def safe_capture(phase: str, platform: str):
    """Wrap a capture function so any exception is logged, never raised.

    The wrapped function MUST accept `thirdeye_home: Path` as a keyword
    argument; the decorator reads it from kwargs to know where to write the
    error log. If `thirdeye_home` is absent from kwargs, errors fall back to
    stderr.
    """

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                home = kwargs.get("thirdeye_home")
                sid = kwargs.get("session_id", "")
                if home is None:
                    sys.stderr.write(f"[thirdeye usage] {platform}/{phase}: {exc!r}\n")
                else:
                    log_capture_error(
                        thirdeye_home=home,
                        phase=phase,
                        error=exc,
                        platform=platform,
                        session_id=sid,
                    )
                return None

        return wrapper

    return deco
