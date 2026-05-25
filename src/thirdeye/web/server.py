from __future__ import annotations

import socket
import threading
import time
import webbrowser

import click
import uvicorn

from thirdeye.web.app import create_app


def _pick_port(host: str, requested: int) -> int:
    if requested != 0:
        return requested
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _open_browser_later(url: str, delay: float = 0.4) -> None:
    def _open() -> None:
        time.sleep(delay)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()


def run_server(*, host: str, port: int, open_browser: bool) -> None:
    port = _pick_port(host, port)
    url = f"http://{host}:{port}"
    click.echo(f"thirdeye ui: {url}")
    if open_browser:
        _open_browser_later(url)
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)
