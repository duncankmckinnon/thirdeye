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
