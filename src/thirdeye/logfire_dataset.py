from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

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


def export_sessions(*, api_key: str, name: str, sessions: list[SessionMeta], store: Store) -> int:
    """Create a managed Logfire dataset with one case per captured session."""
    try:
        from logfire.experimental.api_client import LogfireAPIClient
    except ImportError as exc:
        raise DatasetExportError(
            "Logfire dataset support is not installed; install with: pip install 'thrdi[logfire]'"
        ) from exc

    try:
        with LogfireAPIClient(api_key=api_key) as client:
            client.create_dataset(
                name=name,
                description="Captured thirdeye sessions exported from the sessions view.",
            )
            for start in range(0, len(sessions), 100):
                cases = [_case(m, store) for m in sessions[start : start + 100]]
                client.add_cases(name, cases=cases)
    except DatasetExportError:
        raise
    except Exception as exc:
        raise DatasetExportError(f"Logfire dataset export failed: {exc}") from exc
    return len(sessions)
