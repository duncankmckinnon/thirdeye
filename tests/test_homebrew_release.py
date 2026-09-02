from __future__ import annotations

import hashlib
import http.server
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).parents[1]
DOWNLOAD_SCRIPT = ROOT / ".github/actions/update-homebrew-tap/download-pypi-sdist.sh"


class _ArchiveHandler(http.server.BaseHTTPRequestHandler):
    payload = b""
    failures_remaining = 0
    requests = 0

    def do_GET(self) -> None:
        type(self).requests += 1
        if type(self).failures_remaining:
            type(self).failures_remaining -= 1
            self.send_error(503)
            return

        self.send_response(200)
        self.send_header("Content-Length", str(len(type(self).payload)))
        self.end_headers()
        self.wfile.write(type(self).payload)

    def log_message(self, format: str, *args: object) -> None:
        pass


@contextmanager
def archive_server(
    payload: bytes, *, failures: int = 0
) -> Iterator[tuple[str, type[_ArchiveHandler]]]:
    class Handler(_ArchiveHandler):
        pass

    Handler.payload = payload
    Handler.failures_remaining = failures
    Handler.requests = 0
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/thrdi.tar.gz", Handler
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def run_download(url: str, destination: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(DOWNLOAD_SCRIPT), url, str(destination)],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PYPI_DOWNLOAD_RETRIES": "2",
            "PYPI_DOWNLOAD_RETRY_DELAY": "0",
        },
        check=False,
    )


def test_download_retries_http_failure_and_hashes_archive(tmp_path: Path) -> None:
    payload = b"a real source distribution"
    destination = tmp_path / "thrdi.tar.gz"

    with archive_server(payload, failures=1) as (url, handler):
        result = run_download(url, destination)

    assert result.returncode == 0, result.stderr
    assert handler.requests == 2
    assert destination.read_bytes() == payload
    assert result.stdout.strip() == hashlib.sha256(payload).hexdigest()


def test_download_rejects_empty_archive(tmp_path: Path) -> None:
    destination = tmp_path / "thrdi.tar.gz"

    with archive_server(b"") as (url, _handler):
        result = run_download(url, destination)

    assert result.returncode != 0
    assert "empty" in result.stderr.lower()
