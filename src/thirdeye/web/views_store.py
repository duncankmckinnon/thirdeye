from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

NAME_RE = re.compile(r"^[a-zA-Z0-9 _-]{1,64}$")
PATH_RE = re.compile(r"^\?[A-Za-z0-9._~%&=+\-]*$")
ALLOWED_PAGES: frozenset[str] = frozenset({"sessions"})


def now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SavedView:
    name: str
    path: str
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SavedView:
        return cls(
            name=str(d["name"]),
            path=str(d["path"]),
            created_at=str(d["created_at"]),
        )


class ViewStore:
    """One JSON file per page under <root>/views/<page>.json."""

    def __init__(self, root: Path, page: str) -> None:
        if page not in ALLOWED_PAGES:
            raise ValueError(f"unknown page: {page!r}")
        self.page = page
        self.file = root / "views" / f"{page}.json"

    def list(self) -> list[SavedView]:
        if not self.file.exists():
            return []
        try:
            data = json.loads(self.file.read_text(encoding="utf-8"))
            return [SavedView.from_dict(v) for v in data.get("views", [])]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return []

    def save(self, view: SavedView) -> None:
        if not NAME_RE.match(view.name):
            raise ValueError(f"invalid view name: {view.name!r}")
        if not PATH_RE.match(view.path):
            raise ValueError(f"invalid view path: {view.path!r}")
        existing = [v for v in self.list() if v.name != view.name]
        existing.append(view)
        self._write(existing)

    def delete(self, name: str) -> None:
        existing = [v for v in self.list() if v.name != name]
        self._write(existing)

    def _write(self, views: list[SavedView]) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(self.file.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {"views": [v.to_dict() for v in views]},
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        os.replace(tmp, self.file)
