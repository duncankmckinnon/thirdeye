from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 2


@dataclass
class SessionMeta:
    session_id: str
    platform: str
    cwd: str
    started_at: str
    ended_at: str | None
    status: str  # "open" | "closed" | "stale"
    event_count: int
    last_seq: int  # -1 if no events yet
    last_ts: str | None
    tag_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def write_meta(path: Path, meta: SessionMeta) -> None:
    payload = {"schema_version": SCHEMA_VERSION, **asdict(meta)}
    path.parent.mkdir(parents=True, exist_ok=True)
    # A fixed tmp name would let two concurrent writers for the same session
    # (e.g. overlapping subagent hook processes) collide: one's `os.replace`
    # can rename the other's tmp file away before it gets there, raising
    # FileNotFoundError and crashing that writer's hook process outright,
    # losing whatever event it was recording. `mkstemp` guarantees each
    # writer renames only a file it created itself.
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f"{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        os.unlink(tmp_name)
        raise


def read_meta(path: Path) -> SessionMeta | None:
    if not path.exists():
        return None
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    raw.pop("schema_version", None)
    raw.setdefault("extra", {})
    raw.setdefault("tag_count", 0)
    return SessionMeta(**raw)
