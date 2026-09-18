"""Grok Bot export entrypoints — reuse shared ``thirdeye.otel_export`` only.

Wave 1b: skeleton that stamps GenAI identity attrs via the shared client.
Wave 2: poll ``store.db`` transcript_entries and feed real turns here.
"""

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
    """Hand a session/turn to shared otel_export (fail-open; never blocks).

    ``turn`` is optional for the Wave 1b skeleton; Wave 2 passes a real
    ``TurnSpanDict``. Export still goes through ``otel_export.export_turn`` so
    identity attrs use ``thirdeye.platform=grok_bot``.
    """
    session_dir = Path(cwd) if cwd else Path(".")
    payload = turn if turn is not None else {}
    # Config is required by the real export_turn; when Logfire is disabled the
    # shared helper no-ops. Tests monkeypatch export_turn/export_spans.
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
        payload,  # type: ignore[arg-type]
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
