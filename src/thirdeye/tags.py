from __future__ import annotations

import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from thirdeye.paths import tags_path

_TAG_RE = re.compile(r"^[a-z0-9_\-.#]{1,64}$")

_HASHTAG_RE = re.compile(r"(?<!\w)#([a-zA-Z][\w-]{0,63})")


def _utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_tag(tag: str) -> str:
    """Lowercase, validate, return canonical form. Raise ValueError on invalid input."""
    orig = tag
    if not isinstance(tag, str):
        raise ValueError(f"invalid tag {orig!r}: must match [a-z0-9_-]{{1,64}}")
    stripped = tag.strip()
    canonical = stripped.lower()
    if canonical != stripped or not _TAG_RE.match(canonical):
        raise ValueError(f"invalid tag '{orig}': must match [a-z0-9_-]{{1,64}}")
    return canonical


def extract_hashtags(text: str) -> set[str]:
    """Return the set of unique tags found in `text`, lowercased & validated."""
    if not isinstance(text, str) or not text:
        return set()
    found: set[str] = set()
    for m in _HASHTAG_RE.finditer(text):
        candidate = m.group(1).lower()
        try:
            found.add(validate_tag(candidate))
        except ValueError:
            continue
    return found


class TagStore:
    """Append-only sidecar tag store for one session.

    File: <session_dir>/tags.jsonl   one JSON object per line:
        {"seq": <int>, "tag": <str>, "op": "add"|"remove",
         "source": "manual"|"auto", "at": "<utc-iso>"}
    """

    def __init__(self, session_dir: Path) -> None:
        self._session_dir = session_dir
        self._path = tags_path(session_dir)

    @property
    def path(self) -> Path:
        return self._path

    def _append(self, seq: int, tag: str, op: str, source: str) -> None:
        canonical = validate_tag(tag)
        entry = {
            "seq": seq,
            "tag": canonical,
            "op": op,
            "source": source,
            "at": _utc_iso(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, default=str, ensure_ascii=False) + "\n"
        with open(self._path, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def add(self, seq: int, tag: str, *, source: str = "manual") -> None:
        self._append(seq, tag, "add", source)

    def remove(self, seq: int, tag: str, *, source: str = "manual") -> None:
        self._append(seq, tag, "remove", source)

    def _replay(self) -> dict[int, set[str]]:
        result: dict[int, set[str]] = {}
        if not self._path.exists():
            return result
        with open(self._path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except (ValueError, json.JSONDecodeError):
                    print(
                        f"warning: corrupt tag entry in {self._path}: {line!r}",
                        file=sys.stderr,
                    )
                    continue
                if (
                    not isinstance(entry, dict)
                    or "seq" not in entry
                    or "tag" not in entry
                    or "op" not in entry
                ):
                    print(
                        f"warning: corrupt tag entry in {self._path}: {line!r}",
                        file=sys.stderr,
                    )
                    continue
                seq = entry["seq"]
                tag = entry["tag"]
                op = entry["op"]
                bucket = result.setdefault(seq, set())
                if op == "add":
                    bucket.add(tag)
                elif op == "remove":
                    bucket.discard(tag)
        return {seq: tags for seq, tags in result.items() if tags}

    def tags_for(self, seq: int) -> set[str]:
        """Current effective tag set for one event seq."""
        return self._replay().get(seq, set())

    def all_tags(self) -> dict[int, set[str]]:
        """{seq: {tag, ...}} for every event with at least one current tag."""
        return self._replay()

    def unique_tags(self) -> set[str]:
        """Distinct tags currently effective across the session."""
        result: set[str] = set()
        for tags in self._replay().values():
            result |= tags
        return result

    def tagged_seq_count(self) -> int:
        """Number of distinct seqs with at least one current tag."""
        return len(self._replay())
