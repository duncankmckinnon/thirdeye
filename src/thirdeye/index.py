from __future__ import annotations

import contextlib
import os
import struct
from pathlib import Path

import zstandard as zstd

from thirdeye._compat import fsops

_ENTRY_FMT = "<Q"
_ENTRY_SIZE = 8


class IndexWriter:
    """Append-only writer for the offset index.

    Holds no file handle between calls: ``__init__`` only ensures the file
    exists, ``append`` opens/flushes/fsyncs/closes per call, and ``close`` is a
    no-op kept for interface compatibility. This keeps every writer honest about
    the exclusive store lock -- an instance created before ``rebuild_index``
    replaces the file will transparently append to the replacement.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        # Create the file if missing without truncating an existing one, and
        # without keeping the handle open.
        with open(path, "ab"):
            pass

    def append(self, offset: int) -> None:
        with open(self.path, "ab") as fp:
            fp.write(struct.pack(_ENTRY_FMT, offset))
            fp.flush()
            os.fsync(fp.fileno())

    def close(self) -> None:
        """No-op: open-per-append leaves nothing buffered to lose."""

    def __enter__(self) -> IndexWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class IndexReader:
    def __init__(self, path: Path) -> None:
        self.path = path

    def count(self) -> int:
        if not self.path.exists():
            return 0
        return self.path.stat().st_size // _ENTRY_SIZE

    def get(self, seq: int) -> int:
        if seq < 0 or seq >= self.count():
            raise IndexError(f"seq {seq} out of range (count={self.count()})")
        with open(self.path, "rb") as fp:
            fp.seek(seq * _ENTRY_SIZE)
            return struct.unpack(_ENTRY_FMT, fp.read(_ENTRY_SIZE))[0]

    def all_offsets(self) -> list[int]:
        if not self.path.exists():
            return []
        with open(self.path, "rb") as fp:
            data = fp.read()
        return [
            struct.unpack(_ENTRY_FMT, data[i : i + _ENTRY_SIZE])[0]
            for i in range(0, len(data), _ENTRY_SIZE)
        ]


def rebuild_index(events_log: Path, idx_path: Path) -> int:
    """Walk events.alog frame-by-frame; rewrite idx_path. Returns event count.

    Builds the fresh index into a sibling temp file and atomically replaces the
    destination, so a crash mid-rebuild leaves the previous index untouched
    rather than a truncated one.
    """
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    offsets: list[int] = []
    if events_log.exists() and events_log.stat().st_size > 0:
        data = events_log.read_bytes()
        pos = 0
        while pos < len(data):
            offsets.append(pos)
            dobj = zstd.ZstdDecompressor().decompressobj()
            try:
                dobj.decompress(data[pos:])
            except zstd.ZstdError:
                offsets.pop()
                break
            remaining = len(dobj.unused_data)
            pos = len(data) - remaining

    tmp_path = idx_path.with_name(f"{idx_path.name}.rebuild-{os.getpid()}.tmp")
    with IndexWriter(tmp_path) as w:
        for off in offsets:
            w.append(off)
    try:
        fsops.replace(tmp_path, idx_path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise
    return len(offsets)
