from __future__ import annotations

import click


def _server_options(fn):
    fn = click.option(
        "--no-browser",
        is_flag=True,
        default=False,
        help="Don't open the system browser automatically.",
    )(fn)
    fn = click.option(
        "--port",
        default=0,
        show_default=True,
        type=int,
        help="Port to bind. 0 means pick a free port.",
    )(fn)
    fn = click.option(
        "--host",
        default="127.0.0.1",
        show_default=True,
        help="Bind address. Default loopback only.",
    )(fn)
    return fn


def _launch(host: str, port: int, no_browser: bool) -> None:
    try:
        from thirdeye.web.server import run_server
    except ImportError as e:
        msg = str(e)
        if "starlette" in msg or "uvicorn" in msg or "jinja2" in msg:
            raise click.ClickException(
                "The UI requires the 'ui' extra. Install with:\n    pip install 'thrdi[ui]'"
            ) from e
        raise

    run_server(host=host, port=port, open_browser=not no_browser)


@click.command(name="ui", help="Launch the thirdeye local UI in your browser.")
@_server_options
def ui(host: str, port: int, no_browser: bool) -> None:
    _launch(host, port, no_browser)


@click.command(
    name="serve",
    help="Alias for `thirdeye ui`. Launch the thirdeye local UI in your browser.",
)
@_server_options
def serve(host: str, port: int, no_browser: bool) -> None:
    _launch(host, port, no_browser)
