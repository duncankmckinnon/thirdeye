from __future__ import annotations

import contextlib
import fcntl
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from thirdeye.codec import encode_event
from thirdeye.index import IndexReader, IndexWriter, rebuild_index
from thirdeye.meta import SessionMeta, read_meta, write_meta
from thirdeye.paths import events_lock_path, events_path, index_path, meta_path


def _utc_iso_ms() -> str:
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def utc_iso_ms() -> str:
    return _utc_iso_ms()


def _has_content(path: Path) -> bool:
    """`path.exists()` followed by a separate `path.stat()` is a TOCTOU race:
    `rebuild_index` starts by unlinking the index file, so a checker can
    observe "it exists" and then have the file vanish before its own `stat()`
    call runs. A single `stat()` call, with a missing file treated as empty,
    has no such gap.
    """
    try:
        return path.stat().st_size > 0
    except FileNotFoundError:
        return False


class SessionWriter:
    def __init__(self, session_dir: Path, meta: SessionMeta) -> None:
        self.session_dir = session_dir
        self._events = events_path(session_dir)
        self._idx = index_path(session_dir)
        self._lock_path = events_lock_path(session_dir)
        self._meta_path = meta_path(session_dir)
        self._meta = meta
        self._repair_index_if_needed()
        self._index_w = IndexWriter(self._idx)
        self._next_seq = IndexReader(self._idx).count()

    def _repair_index_if_needed(self) -> None:
        """Rebuild the index if the log has data but the index doesn't (e.g.
        a crash wrote an event but never got to record its offset). Checked
        once unlocked to skip the common case cheaply, then re-checked under
        the lock before actually rebuilding: two processes can both observe
        "needs repair" before either acquires the lock, and a rebuild must
        never run concurrently with another writer's `append` -- it rescans
        the whole log from disk, so it would race an in-flight append
        landing mid-scan.
        """
        if not _has_content(self._events):
            return
        if _has_content(self._idx):
            return
        with self._locked():
            if _has_content(self._idx):
                return  # another process already repaired it while we waited
            rebuild_index(self._events, self._idx)

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    @classmethod
    def open(
        cls,
        session_dir: Path,
        *,
        session_id: str,
        platform: str,
        cwd: str,
        extra: dict[str, Any] | None = None,
    ) -> SessionWriter:
        session_dir.mkdir(parents=True, exist_ok=True)
        existing = read_meta(meta_path(session_dir))
        if existing is None:
            meta = SessionMeta(
                session_id=session_id,
                platform=platform,
                cwd=cwd,
                started_at=_utc_iso_ms(),
                ended_at=None,
                status="open",
                event_count=0,
                last_seq=-1,
                last_ts=None,
                extra=extra or {},
            )
        else:
            existing.status = "open"
            existing.ended_at = None
            meta = existing
        write_meta(meta_path(session_dir), meta)
        return cls(session_dir, meta)

    def append(self, t: str, data: Any = None) -> int:
        ts = _utc_iso_ms()
        with self._locked():
            # Every hook invocation is a fresh process constructing its own
            # SessionWriter, so `self._next_seq` can be stale by the time
            # this call actually runs -- re-derive the authoritative next
            # seq from the index itself while holding the lock, or two
            # concurrent writers can claim the same seq for different
            # events and their unsynchronized index appends can land out of
            # step with the log.
            seq = IndexReader(self._idx).count()
            event: dict[str, Any] = {"t": t, "ts": ts, "seq": seq}
            if data is not None:
                event["data"] = data
            frame = encode_event(event)

            with open(self._events, "ab") as fp:
                offset = fp.tell()
                fp.write(frame)
                fp.flush()
                os.fsync(fp.fileno())
            self._index_w.append(offset)

        self._next_seq = seq + 1
        self._meta.event_count = self._next_seq
        self._meta.last_seq = seq
        self._meta.last_ts = ts
        return seq

    def flush_and_detach(self) -> None:
        self._index_w.close()
        write_meta(self._meta_path, self._meta)

    def close(self, *, status: str = "closed") -> None:
        self._index_w.close()
        self._meta.status = status
        self._meta.ended_at = _utc_iso_ms()
        self._meta.event_count = self._next_seq
        self._meta.last_seq = self._next_seq - 1 if self._next_seq > 0 else -1
        write_meta(self._meta_path, self._meta)

    def __enter__(self) -> SessionWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
