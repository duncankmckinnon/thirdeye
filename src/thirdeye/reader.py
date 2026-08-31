from __future__ import annotations

import contextlib
import fcntl
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import zstandard as zstd

from thirdeye.codec import decode_event
from thirdeye.index import IndexReader
from thirdeye.paths import events_lock_path, events_path, index_path


class SessionReader:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self._events = events_path(session_dir)
        self._lock_path = events_lock_path(session_dir)
        self._idx = IndexReader(index_path(session_dir))
        self.truncated_tail: bool = False

    @contextlib.contextmanager
    def _locked_snapshot(self) -> Iterator[None]:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def get_event(self, seq: int) -> dict[str, Any]:
        if seq < 0:
            raise IndexError(f"seq {seq} out of range")
        with self._locked_snapshot():
            offset = self._idx.get(seq)
            next_offset = (
                self._idx.get(seq + 1)
                if seq + 1 < self._idx.count()
                else self._events.stat().st_size
            )
            with open(self._events, "rb") as f:
                f.seek(offset)
                frame = f.read(next_offset - offset)
        return decode_event(frame)

    def iter_events(
        self,
        *,
        types: Iterable[str] | None = None,
        seq_range: tuple[int, int] | None = None,
    ) -> Iterator[dict[str, Any]]:
        type_set = set(types) if types is not None else None
        with self._locked_snapshot():
            offsets = self._idx.all_offsets()
            log_size = self._events.stat().st_size if self._events.exists() else 0
            offsets.append(log_size)  # sentinel

            start, end = seq_range if seq_range is not None else (0, len(offsets) - 1)
            if start >= end or not self._events.exists():
                return
            frames: list[bytes] = []
            with open(self._events, "rb") as f:
                for seq in range(start, min(end, len(offsets) - 1)):
                    f.seek(offsets[seq])
                    frames.append(f.read(offsets[seq + 1] - offsets[seq]))

        for frame in frames:
            try:
                event = decode_event(frame)
            except zstd.ZstdError:
                self.truncated_tail = True
                return
            if type_set is not None and event.get("t") not in type_set:
                continue
            yield event
