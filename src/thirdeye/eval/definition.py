from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from thirdeye.paths import eval_def_path, eval_defs_dir

SHIPPED_NAMES = ("default", "token-efficiency", "tool-quality")


@dataclass(frozen=True)
class EvalDefinition:
    name: str
    description: str
    directive: str
    default_agent: str = "claude"
    output_schema: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalDefinition:
        return cls(
            name=str(d["name"]),
            description=str(d.get("description", "")),
            directive=str(d["directive"]),
            default_agent=str(d.get("default_agent", "claude")),
            output_schema=str(d.get("output_schema", "v1")),
        )

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)


def _shipped_path(name: str) -> Path | None:
    """Return path to a shipped definition file, if any."""
    if name not in SHIPPED_NAMES:
        return None
    root = resources.files("thirdeye").joinpath("eval/defs")
    p = Path(str(root)) / f"{name}.yaml"
    return p if p.is_file() else None


def load_definition(thirdeye_home: Path, name: str) -> EvalDefinition:
    """Load a definition by name. Lazily materializes shipped defaults.

    Raises FileNotFoundError if the name is neither in the user's home nor a
    shipped default.
    """
    user_path = eval_def_path(thirdeye_home, name)
    if not user_path.is_file():
        shipped = _shipped_path(name)
        if shipped is None:
            raise FileNotFoundError(
                f"no eval definition named '{name}' (checked {user_path} and shipped defaults)"
            )
        user_path.parent.mkdir(parents=True, exist_ok=True)
        user_path.write_text(shipped.read_text(encoding="utf-8"), encoding="utf-8")
    data = yaml.safe_load(user_path.read_text(encoding="utf-8")) or {}
    return EvalDefinition.from_dict(data)


def save_definition(thirdeye_home: Path, defn: EvalDefinition, *, force: bool = False) -> Path:
    """Write a definition to the user's home. Atomic via tmp + rename."""
    path = eval_def_path(thirdeye_home, defn.name)
    if path.exists() and not force:
        raise FileExistsError(f"definition '{defn.name}' already exists at {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(defn.to_yaml(), encoding="utf-8")
    os.replace(tmp, path)
    return path


def delete_definition(thirdeye_home: Path, name: str) -> bool:
    """Remove a user's definition. Returns True if removed, False if absent.

    Does not affect shipped defaults — they're restored on next load.
    """
    path = eval_def_path(thirdeye_home, name)
    if path.is_file():
        path.unlink()
        return True
    return False


def list_definitions(thirdeye_home: Path) -> list[str]:
    """Return sorted list of definition names available (user + shipped union)."""
    names: set[str] = set(SHIPPED_NAMES)
    d = eval_defs_dir(thirdeye_home)
    if d.is_dir():
        for f in d.glob("*.yaml"):
            names.add(f.stem)
    return sorted(names)


def validate_directive_yaml(text: str) -> tuple[bool, str]:
    """Validate the YAML body of an eval definition without writing.

    Returns (True, "") on success. On failure, returns (False, <message>).
    """
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as e:
        return False, f"YAML parse error: {e}"
    if not isinstance(loaded, dict):
        return False, "definition must be a YAML mapping"
    required = ("name", "directive")
    missing = [k for k in required if k not in loaded]
    if missing:
        return False, f"missing required keys: {', '.join(missing)}"
    if not isinstance(loaded.get("directive"), str) or not loaded["directive"].strip():
        return False, "'directive' must be a non-empty string"
    return True, ""
