"""The CLI's stdout must be UTF-8 whatever the platform's locale encoding is.

`thirdeye` emits arbitrary session text -- prompts, tool output, and search's
"…" truncation marker -- and its `--json` modes are meant to be piped. CPython
picks the *locale* encoding for a redirected stdout, which on a stock Windows
runner is cp1252, so without an explicit override that output silently changes
byte-for-byte with the machine's codepage: "…" ships as the lone byte 0x85 and
any UTF-8 consumer downstream fails to decode it.

These tests stand in for a Windows locale by forcing the child's encoding via
PYTHONIOENCODING, which reaches the same code path the codepage does.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from thirdeye.config import Config
from thirdeye.store import Store

# A prompt long enough that `search`'s 80-char window has to elide both ends,
# so the snippet is wrapped in the non-ASCII "…" marker, and carrying non-ASCII
# text of its own that no single-byte codepage could round-trip.
_FILLER = "context padding that pushes the match away from both ends. "
_PROMPT = _FILLER * 3 + "naïve café 日本語 xylophone_unique_word " + _FILLER * 3


@pytest.fixture
def populated_home(tmp_path: Path) -> Path:
    home = tmp_path / "thirdeye"
    with Store(Config(root=home)).open_session(
        "01J9G7XK4P", platform="claude", cwd="/proj"
    ) as writer:
        writer.append("user_message", _PROMPT)
    return home


def _run(*args: str, home: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["THIRDEYE_HOME"] = str(home)
    env["PYTHONIOENCODING"] = "cp1252"
    return subprocess.run(
        [sys.executable, "-m", "thirdeye", *args],
        env=env,
        check=False,
        capture_output=True,
    )


def test_search_output_is_utf8_under_a_single_byte_locale(populated_home: Path) -> None:
    result = _run("search", "xylophone_unique_word", home=populated_home)

    assert result.returncode == 0, result.stderr
    text = result.stdout.decode("utf-8")
    assert "01J9G7XK4P" in text
    assert "…" in text, "search snippet should still carry its truncation marker"


def test_json_output_is_utf8_under_a_single_byte_locale(populated_home: Path) -> None:
    """`--json` is the piped, machine-read path, so it must decode cleanly."""
    result = _run("events", "01J9G7XK4P", "--json", home=populated_home)

    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in result.stdout.decode("utf-8").splitlines() if line]
    assert any("naïve café 日本語" in json.dumps(e, ensure_ascii=False) for e in events)
