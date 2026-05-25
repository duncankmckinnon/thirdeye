from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time

import pytest

pytest.importorskip("starlette")


def _wait_for_port(port: int, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"server on {port} never came up")


def test_thirdeye_ui_smoke(tmp_path):
    # The CLI is a subprocess, so we can't pass a Config in-process.
    # Use THIRDEYE_HOME (honored by Config.load()) to point the subprocess
    # at an isolated dir. The subprocess opens NO browser and binds port 0
    # so it's safe to run in CI / restricted environments.
    env = os.environ.copy()
    env["THIRDEYE_HOME"] = str(tmp_path)
    # Make sure no real ~/.thirdeye config bleeds in.
    env.pop("THIRDEYE_CONFIG", None)
    (tmp_path / "traces").mkdir()
    proc = subprocess.Popen(
        [sys.executable, "-m", "thirdeye", "ui", "--port", "0", "--no-browser"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,  # line-buffered so readline() works deterministically
    )
    try:
        port = None
        deadline = time.time() + 5.0
        while time.time() < deadline and port is None:
            line = proc.stdout.readline()
            if not line:
                break
            m = re.search(r"http://127\.0\.0\.1:(\d+)", line)
            if m:
                port = int(m.group(1))
        assert port is not None, "did not see bound port in stdout"
        _wait_for_port(port)
        import urllib.request

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2.0) as resp:
            assert resp.status == 200
    finally:
        proc.terminate()
        proc.wait(timeout=5.0)
