from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from thirdeye.turns import filter_turns

if TYPE_CHECKING:
    from thirdeye.meta import SessionMeta
    from thirdeye.store import Store


class DatasetExportError(RuntimeError):
    """A managed Logfire dataset could not be created."""


def _case(meta: SessionMeta, store: Store) -> dict[str, Any]:
    return {
        "name": meta.session_id,
        "inputs": {
            "session": asdict(meta),
            "events": list(store.reader(meta.session_id).iter_events()),
        },
    }


def _turn_case(turn: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": turn["id"],
        "inputs": {"turn": turn},
    }


def export_sessions(
    *,
    api_key: str,
    name: str,
    sessions: list[SessionMeta],
    store: Store,
    scope: str = "session",
    turn_id: str | None = None,
    turn_query: str | None = None,
) -> int:
    """Create a managed Logfire dataset with one case per session or turn."""
    try:
        from logfire.experimental.api_client import LogfireAPIClient
    except ImportError as exc:
        raise DatasetExportError(
            "Logfire dataset support is not installed; install with: pip install 'thrdi[logfire]'"
        ) from exc

    try:
        cases = (
            [_turn_case(turn) for turn in filter_turns(sessions, store, turn_id, turn_query)]
            if scope == "turn"
            else [_case(meta, store) for meta in sessions]
        )
        if not cases:
            raise DatasetExportError(f"No {scope}s match these filters; no dataset was created.")
        with LogfireAPIClient(api_key=api_key) as client:
            client.create_dataset(
                name=name,
                description=f"Captured thirdeye {scope}s exported from the sessions view.",
            )
            for start in range(0, len(cases), 100):
                client.add_cases(name, cases=cases[start : start + 100])
    except DatasetExportError:
        raise
    except Exception as exc:
        raise DatasetExportError(f"Logfire dataset export failed: {exc}") from exc
    return len(cases)
