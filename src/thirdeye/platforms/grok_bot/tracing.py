"""Grok Bot export entrypoints — reuse shared ``thirdeye.otel_export`` only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from thirdeye import otel_export
from thirdeye.platforms.grok_bot.constants import PLATFORM_NAME


def export_session(
    *,
    session_id: str,
    cwd: str = "",
    platform: str = PLATFORM_NAME,
    turn: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> None:
    """Export via shared otel_export.

    Requires a real turn dict (never ``{}``). When ``turn`` is omitted, builds a
    minimal completed TurnSpanDict so callers still hit shared export without
    the empty-payload footgun.
    """
    if turn == {}:
        return
    if turn is None:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
        turn = {
            "turn_id": session_id,
            "start_ts": ts,
            "end_ts": ts,
            "input_message": "",
            "output_message": "",
            "status": "completed",
            "llm_calls": [],
            "permission_requests": [],
            "subagents": [],
            "attributes": {"thirdeye.platform": platform},
        }
    session_dir = Path(cwd) if cwd else Path(".")
    try:
        from thirdeye.config import Config

        config = Config.load()
    except Exception:
        config = None  # type: ignore[assignment]
    otel_export.export_turn(
        config,  # type: ignore[arg-type]
        session_dir,
        session_id,
        platform,
        cwd or str(session_dir),
        turn,  # type: ignore[arg-type]
    )


def export_turn(
    *,
    session_id: str,
    cwd: str = "",
    platform: str = PLATFORM_NAME,
    turn: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Alias matching sibling platforms' naming."""
    export_session(
        session_id=session_id,
        cwd=cwd,
        platform=platform,
        turn=turn,
        **kwargs,
    )
