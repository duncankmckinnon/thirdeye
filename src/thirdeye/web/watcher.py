"""asyncio file watcher for live-tailing session events.

Uses mtime-based polling rather than inotify/kqueue. Trade-off: polling
costs a couple of stat() calls per second per open connection, but keeps
zero new dependencies and avoids platform-specific code paths. Watchers
are per-SSE-connection, so the cost is bounded by the number of open
browser tabs — negligible for a local single-user tool.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from thirdeye.reader import SessionReader


async def tail_events(
    reader: SessionReader,
    *,
    start_seq: int = -1,
    poll_interval: float = 0.5,
) -> AsyncIterator[dict[str, Any]]:
    """Yield new events as they're appended to the session.

    Polls the underlying event file; on each tick, re-reads events and
    yields any whose seq > last_yielded_seq. Caller is responsible for
    stopping iteration (e.g. when the session closes or the client
    disconnects).
    """
    last_seq = start_seq
    while True:
        try:
            events = list(reader.iter_events())
        except FileNotFoundError:
            await asyncio.sleep(poll_interval)
            continue
        for ev in events:
            if ev.get("seq", -1) > last_seq:
                yield ev
                last_seq = ev["seq"]
        await asyncio.sleep(poll_interval)
