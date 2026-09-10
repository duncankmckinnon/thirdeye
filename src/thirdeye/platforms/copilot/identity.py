"""Pure source-home and native-session identity helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .constants import COPILOT_HOME_ENV
from .types import SourcePaths

# Stored session IDs keep the first 16 hex characters of the SHA-256 source key
# for path readability.  The full 64-character digest remains the identity.
SOURCE_KEY_PREFIX_LEN = 16
SOURCE_KEY_DIGEST_LEN = 64


def _canonical_path(path: Path) -> Path:
    """Return a stable, absolute path without requiring the path to exist."""

    return path.expanduser().resolve(strict=False)


def _normalized_path(path: Path) -> str:
    """Normalize the canonical path for source-home identity on this platform."""

    return os.path.normcase(os.fspath(_canonical_path(path)))


def _source_key(home: Path) -> str:
    return hashlib.sha256(_normalized_path(home).encode("utf-8")).hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_paths(paths: SourcePaths) -> Path:
    """Reject forged identities or recording paths outside their source home."""

    home = _canonical_path(Path(paths["home"]))
    if paths["source_key"] != _source_key(home):
        raise ValueError("source_key does not match the canonical source home")

    for field in ("session_root", "database"):
        candidate = _canonical_path(Path(paths[field]))
        if not _is_within(candidate, home):
            raise ValueError(f"{field} escapes the selected Copilot home")
    return home


def resolve_sources(source_home: Path | None = None) -> SourcePaths:
    """Resolve one Copilot home at invocation time.

    Explicit input wins over ``COPILOT_HOME``; the default is
    ``Path.home() / '.copilot'``.  The returned paths are canonical strings so
    aliases resolving to the same home share a source identity.
    """

    requested = source_home
    if requested is None:
        configured = os.environ.get(COPILOT_HOME_ENV)
        requested = Path(configured) if configured else Path.home() / ".copilot"

    home = _canonical_path(requested)
    paths: SourcePaths = {
        "home": os.fspath(home),
        "source_key": _source_key(home),
        "session_root": os.fspath(home / "session-state"),
        "database": os.fspath(home / "session-store.db"),
    }
    _validate_paths(paths)
    return paths


def validate_native_id(native_id: str) -> None:
    """Ensure a native session ID can never select a path outside its home."""

    if not isinstance(native_id, str) or not native_id or native_id.strip() != native_id:
        raise ValueError("native session ID must be a non-empty, trimmed string")
    if native_id in {".", ".."}:
        raise ValueError("native session ID must not be a traversal segment")
    if any(character in native_id for character in ("/", "\\", "\x00", ":")):
        raise ValueError("native session ID contains a path separator or invalid path character")
    if any(ord(character) < 32 for character in native_id):
        raise ValueError("native session ID contains a control character")


def stored_session_id(paths: SourcePaths, native_id: str) -> str:
    """Return the stable thirdeye ID for a native ID within one source home.

    The stored ID uses only :data:`SOURCE_KEY_PREFIX_LEN` hex characters of
    the source key.  This function checks that *one* ``SourcePaths`` value is
    internally consistent: the full digest matches the canonical home, and
    recording paths stay inside that home.  That check does not detect two
    legitimate homes whose SHA-256 digests share a prefix.  Both would still
    produce the same stored ID.  On reuse the archive must compare the full
    ``source_key`` retained in session metadata with the candidate home and
    refuse to merge a prefix collision.
    """

    _validate_paths(paths)
    validate_native_id(native_id)
    return f"copilot-{paths['source_key'][:SOURCE_KEY_PREFIX_LEN]}-{native_id}"


def source_keys_share_stored_prefix(left: str, right: str) -> bool:
    """Return True when two distinct full source keys would collide in stored IDs.

    Archive reuse must apply this comparison (or an equivalent full-key check
    against retained metadata).  :func:`stored_session_id` cannot: it never
    sees the other home.
    """

    for key in (left, right):
        if not isinstance(key, str) or len(key) != SOURCE_KEY_DIGEST_LEN:
            raise ValueError("source_key must be a 64-character SHA-256 hex digest")
    return left != right and left[:SOURCE_KEY_PREFIX_LEN] == right[:SOURCE_KEY_PREFIX_LEN]
