from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("starlette")


def test_tail_yields_all_existing_events(web_store):
    sid = "01J9WATCH01"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "a"})
        w.append("assistant_message", {"text": "b"})
    web_store.close_session(sid, platform="claude")

    from thirdeye.web.watcher import tail_events

    reader = web_store.reader(sid)

    async def collect_first_n(n: int) -> list[int]:
        seqs: list[int] = []
        async for ev in tail_events(reader, start_seq=-1, poll_interval=0.01):
            seqs.append(ev["seq"])
            if len(seqs) >= n:
                break
        return seqs

    seqs = asyncio.run(collect_first_n(2))
    assert sorted(seqs) == seqs  # ordered ascending
    assert len(seqs) == 2


def test_tail_survives_missing_file_window(tmp_path):
    from thirdeye.reader import SessionReader
    from thirdeye.web.watcher import tail_events

    missing_dir = tmp_path / "missing"
    reader = SessionReader(missing_dir)

    async def step_once() -> None:
        gen = tail_events(reader, start_seq=-1, poll_interval=0.01)
        try:
            await asyncio.wait_for(gen.__anext__(), timeout=0.05)
        except asyncio.TimeoutError:
            pass  # expected — no events, no errors
        await gen.aclose()

    asyncio.run(step_once())


def test_tail_skips_events_at_or_before_start_seq(web_store):
    """Resuming with last_seq=N must skip seqs <= N (no double-delivery)."""
    sid = "01J9WATCH02"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "a"})  # seq 0
        w.append("assistant_message", {"text": "b"})  # seq 1
        w.append("user_message", {"prompt": "c"})  # seq 2
    web_store.close_session(sid, platform="claude")

    from thirdeye.web.watcher import tail_events

    reader = web_store.reader(sid)

    async def collect_after(start: int, n: int) -> list[int]:
        seqs: list[int] = []
        async for ev in tail_events(reader, start_seq=start, poll_interval=0.01):
            seqs.append(ev["seq"])
            if len(seqs) >= n:
                break
        return seqs

    # start_seq=0 means "I have seq 0; give me everything > 0"
    seqs = asyncio.run(collect_after(0, 2))
    assert seqs == [1, 2]
