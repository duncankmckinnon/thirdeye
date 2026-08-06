from __future__ import annotations

import json
import shutil
from pathlib import Path

from thirdeye.platforms.codex.rollout import end_offset, iter_frames, resolve_rollout

FIXTURE = Path(__file__).parent / "fixtures" / "usage" / "codex_rollout.jsonl"
EXPECTED = Path(__file__).parent / "fixtures" / "usage" / "codex_rollout.expected.json"

# The session id carried in the fixture's session_meta frame and its filename.
FIXTURE_SID = "019fb579-cdda-7a03-86df-65c87b6c4ae2"
FIXTURE_NAME = f"rollout-2026-07-30T17-01-26-{FIXTURE_SID}.jsonl"


def _make_tree(root: Path, name: str = FIXTURE_NAME, *, src: Path = FIXTURE) -> Path:
    """Copy `src` into a Codex-style YYYY/MM/DD tree under `root`."""
    nested = root / "2026" / "07" / "30"
    nested.mkdir(parents=True, exist_ok=True)
    dest = nested / name
    shutil.copy(src, dest)
    return dest


# -- resolve_rollout -----------------------------------------------------------


def test_wellformed_tree_resolves(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    dest = _make_tree(root)
    assert resolve_rollout(FIXTURE_SID, root) == dest.resolve()


def test_session_id_with_star_rejected(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    _make_tree(root)
    # Prove a glob interpolating the id *would* have matched the real file — the
    # exact injection the guard exists to stop.
    would_match = list(root.rglob(f"rollout-*-{'*'}.jsonl"))
    assert would_match, "precondition: a glob with '*' must match the fixture"
    assert resolve_rollout("*", root) is None


def test_session_id_with_dotdot_rejected(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    _make_tree(root)
    assert resolve_rollout("..", root) is None


def test_session_id_with_slash_rejected(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    _make_tree(root)
    assert resolve_rollout("a/b", root) is None


def test_session_id_with_bracket_rejected(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    _make_tree(root)
    assert resolve_rollout("abc[123]", root) is None


def test_empty_session_id_rejected(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    _make_tree(root)
    assert resolve_rollout("", root) is None


def test_session_meta_mismatch_rejected(tmp_path: Path) -> None:
    # Filename names "deadbeef" but the file's session_meta names FIXTURE_SID.
    root = tmp_path / "sessions"
    _make_tree(root, name="rollout-2026-07-30T17-01-26-deadbeef.jsonl")
    assert resolve_rollout("deadbeef", root) is None


def test_no_session_meta_accepted_on_filename(tmp_path: Path) -> None:
    sid = "nometasession"
    lines = FIXTURE.read_text().splitlines(keepends=True)
    kept = [ln for ln in lines if json.loads(ln).get("type") != "session_meta"]
    root = tmp_path / "sessions"
    nested = root / "2026" / "07" / "30"
    nested.mkdir(parents=True)
    dest = nested / f"rollout-2026-07-30T17-01-26-{sid}.jsonl"
    dest.write_text("".join(kept))
    assert resolve_rollout(sid, root) == dest.resolve()


def test_symlink_escape_rejected(tmp_path: Path) -> None:
    sid = "escapesession"
    name = f"rollout-2026-07-30T17-01-26-{sid}.jsonl"
    # Real file lives outside the sessions root.
    outside = tmp_path / "outside"
    outside.mkdir()
    real = outside / name
    shutil.copy(FIXTURE, real)

    root = tmp_path / "sessions"
    nested = root / "2026" / "07" / "30"
    nested.mkdir(parents=True)
    (nested / name).symlink_to(real)

    assert resolve_rollout(sid, root) is None


def test_missing_sessions_root_returns_none(tmp_path: Path) -> None:
    assert resolve_rollout(FIXTURE_SID, tmp_path / "does_not_exist") is None


# -- iter_frames / end_offset --------------------------------------------------


def test_iter_frames_yields_all(tmp_path: Path) -> None:
    dest = _make_tree(tmp_path / "sessions")
    total_lines = json.loads(EXPECTED.read_text())["total_lines"]
    frames = list(iter_frames(dest, 0))
    assert len(frames) == total_lines
    assert frames[0][0] == 0
    offsets = [off for off, _ in frames]
    assert offsets == sorted(offsets)
    assert len(set(offsets)) == len(offsets)


def test_iter_frames_from_mid_offset(tmp_path: Path) -> None:
    dest = _make_tree(tmp_path / "sessions")
    frames = list(iter_frames(dest, 0))
    start_off = frames[5][0]
    later = list(iter_frames(dest, start_off))
    assert later == frames[5:]
    assert later[0][0] == start_off


def test_malformed_line_skipped(tmp_path: Path) -> None:
    p = tmp_path / "r.jsonl"
    p.write_text('{"type":"a"}\nnot json at all\n{"type":"b"}\n')
    frames = list(iter_frames(p, 0))
    assert [f["type"] for _, f in frames] == ["a", "b"]


def test_truncated_final_line_not_yielded(tmp_path: Path) -> None:
    p = tmp_path / "r.jsonl"
    first = b'{"type":"a"}\n'
    p.write_bytes(first + b'{"type":"b')  # truncated, no newline
    frames = list(iter_frames(p, 0))
    assert [f["type"] for _, f in frames] == ["a"]
    assert end_offset(p, 0) == len(first)


def test_end_offset_full_file_equals_size(tmp_path: Path) -> None:
    dest = _make_tree(tmp_path / "sessions")
    assert end_offset(dest, 0) == dest.stat().st_size
