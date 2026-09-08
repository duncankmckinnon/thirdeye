"""Process stdio that stays UTF-8 whatever the platform's locale encoding is.

Every *file* thirdeye writes is opened with an explicit ``encoding="utf-8"``,
but process stdout/stderr are handed to us already constructed, and CPython
builds them from the locale encoding whenever they are redirected. On POSIX
that is effectively always UTF-8; on Windows it is the ANSI codepage -- cp1252
on a stock en-US machine, cp932 on a Japanese one.

That makes the CLI's output vary byte-for-byte with the machine it runs on.
thirdeye prints arbitrary recorded session text, so the failure is not
hypothetical: a snippet marker like "…" is emitted as the lone byte 0x85 under
cp1252 and breaks any UTF-8 consumer downstream, while text a single-byte
codepage cannot represent at all (CJK, most emoji) raises UnicodeEncodeError
and takes the whole command down. ``--json`` output, which exists to be piped,
is the worst-affected path.

Reconfiguring the streams to UTF-8 is deliberately unconditional rather than
Windows-only: the invariant we want is "thirdeye speaks UTF-8", not "thirdeye
patches Windows", and on a POSIX box already in UTF-8 the call is a no-op.
"""

from __future__ import annotations

import codecs
import sys
from typing import IO, Any


def _is_utf8(encoding: str | None) -> bool:
    if not encoding:
        return False
    try:
        return codecs.lookup(encoding).name == "utf-8"
    except LookupError:
        return False


def _reconfigure(stream: IO[Any] | None) -> None:
    # Streams reach us in more shapes than the annotation suggests: None under
    # pythonw, a plain in-memory object under pytest's capture and click's
    # CliRunner, or an already-detached stream. Anything without a working
    # reconfigure() is left exactly as it is -- a stream we cannot set to UTF-8
    # is not a reason to fail the command the user actually asked for.
    if stream is None or _is_utf8(getattr(stream, "encoding", None)):
        return
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8")
    except (OSError, ValueError, AttributeError):
        return


def force_utf8_stdio() -> None:
    """Put ``sys.stdout``/``sys.stderr`` into UTF-8 unless they already are.

    Call once at process start, before anything writes output.
    """
    _reconfigure(sys.stdout)
    _reconfigure(sys.stderr)
