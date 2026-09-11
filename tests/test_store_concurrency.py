"""Concurrency stress tests for the process-shared session store."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from thirdeye.config import Config
from thirdeye.index import IndexReader
from thirdeye.paths import index_path, session_dir
from thirdeye.reader import SessionReader
from thirdeye.store import Store

_WRITER_COUNT = 8
_EVENTS_PER_WRITER = 25
_SESSION_ID = "CONCURRENT_SESSION"
_PLATFORM = "claude"
_CWD = "/concurrency-test"
# Generous margin: this spawns _WRITER_COUNT real Python interpreters that
# contend for the same on-disk lock, and a busy or oversubscribed CI runner
# (observed on GitHub's Windows runners) can push interpreter start-up and
# lock-wait time well past what a quiet dev machine sees.
_WRITER_TIMEOUT = 90
_WRITER_SCRIPT = f"""
from pathlib import Path
import sys
import time

from thirdeye.config import Config
from thirdeye.store import Store

root = Path(sys.argv[1])
start_signal = Path(sys.argv[2])
writer_id = int(sys.argv[3])

while not start_signal.exists():
    time.sleep(0.001)

store = Store(Config(root=root))
with store.open_session(
    {_SESSION_ID!r}, platform={_PLATFORM!r}, cwd={_CWD!r}
) as writer:
    for event_id in range({_EVENTS_PER_WRITER}):
        writer.append("concurrent_event", {{"writer": writer_id, "event": event_id}})
"""


def _spawn_writers(tmp_path: Path) -> tuple[list[subprocess.Popen[str]], float]:
    start_signal = tmp_path / "start-writers"
    deadline = time.monotonic() + _WRITER_TIMEOUT
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _WRITER_SCRIPT,
                str(tmp_path),
                str(start_signal),
                str(writer_id),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        for writer_id in range(_WRITER_COUNT)
    ]
    start_signal.touch()
    return processes, deadline


def _assert_writers_succeeded(processes: list[subprocess.Popen[str]], deadline: float) -> None:
    failures: list[str] = []
    for process in processes:
        try:
            stdout, stderr = process.communicate(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                process.kill()
            stdout, stderr = process.communicate()
            failures.append(f"writer timed out\nstdout:\n{stdout}\nstderr:\n{stderr}")
            continue
        if process.returncode:
            failures.append(
                f"writer exited {process.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )
    assert not failures, "\n\n".join(failures)


def test_concurrent_writers_preserve_seq_continuity(tmp_path: Path) -> None:
    processes, deadline = _spawn_writers(tmp_path)
    _assert_writers_succeeded(processes, deadline)

    store = Store(Config(root=tmp_path))
    events = list(store.reader(_SESSION_ID).iter_events())
    expected_count = _WRITER_COUNT * _EVENTS_PER_WRITER
    sd = session_dir(tmp_path, _PLATFORM, _SESSION_ID)

    assert len(events) == expected_count
    assert sorted(event["seq"] for event in events) == list(range(expected_count))
    assert IndexReader(index_path(sd)).count() == len(events)


def test_reader_never_sees_partial_frame(tmp_path: Path) -> None:
    store = Store(Config(root=tmp_path))
    with store.open_session(_SESSION_ID, platform=_PLATFORM, cwd=_CWD):
        pass

    processes, deadline = _spawn_writers(tmp_path)
    sd = session_dir(tmp_path, _PLATFORM, _SESSION_ID)
    reader_iterations = 0
    try:
        while any(process.poll() is None for process in processes):
            reader = SessionReader(sd)
            events = list(reader.iter_events())
            assert reader.truncated_tail is False
            assert [event["seq"] for event in events] == list(range(len(events)))
            assert all(event["t"] == "concurrent_event" for event in events)
            reader_iterations += 1
            time.sleep(0.001)
    finally:
        _assert_writers_succeeded(processes, deadline)

    reader = SessionReader(sd)
    events = list(reader.iter_events())

    assert reader_iterations > 0
    assert reader.truncated_tail is False
    assert len(events) == _WRITER_COUNT * _EVENTS_PER_WRITER
    assert [event["seq"] for event in events] == list(range(len(events)))
    assert all(event["t"] == "concurrent_event" for event in events)
