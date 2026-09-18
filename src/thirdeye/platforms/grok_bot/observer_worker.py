"""Detached store-mutation observer for Grok Bot (survives ``thirdeye add`` exit)."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from thirdeye.platforms.grok_bot.constants import PLATFORM_NAME

_POLL_INTERVAL = 0.1
_IDLE_EXIT_SECONDS = 3600.0


def _file_stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    if not path.is_file():
        return None
    return (int(st.st_size), int(st.st_mtime_ns))


def _kick_enabled(state_dir: Path) -> bool:
    return (state_dir / "kick.enabled").is_file() or (
        state_dir / "watcher.enabled"
    ).is_file()


def _iter_store_dbs(agents_root: Path) -> list[Path]:
    out: list[Path] = []
    try:
        if not agents_root.is_dir():
            return out
        for child in agents_root.iterdir():
            try:
                if not child.is_dir():
                    continue
            except OSError:
                continue
            db = child / "store.db"
            try:
                if db.is_file():
                    out.append(db)
            except OSError:
                continue
    except OSError:
        return out
    return out


def _patch_export_keep_job() -> None:
    """Write durable job files; spawn otel_worker on a copy it may delete.

    ``otel_worker`` unlinks its job path as soon as it reads it. Tests (and
    operators inspecting ``logs/otel-jobs``) need the queued job to remain.
    """
    from thirdeye import otel_export

    original_spawn = otel_export._spawn

    def _spawn_on_copy(job_path: Path) -> None:
        try:
            run_path = job_path.with_name(job_path.stem + ".run" + job_path.suffix)
            shutil.copy2(job_path, run_path)
        except OSError:
            original_spawn(job_path)
            return
        original_spawn(run_path)

    otel_export._spawn = _spawn_on_copy  # type: ignore[assignment]


def run_observer(*, state_dir: Path, agents_root: Path) -> int:
    from thirdeye.platforms.grok_bot.install import GrokBotPlatform
    from thirdeye.platforms.grok_bot.watch import on_store_mutation

    _patch_export_keep_job()
    platform = GrokBotPlatform(state_dir=state_dir, agents_root=agents_root)
    stamps: dict[str, tuple[int, int] | None] = {}
    last_activity = time.monotonic()

    while _kick_enabled(state_dir):
        try:
            for db in _iter_store_dbs(agents_root):
                key = str(db.resolve())
                stamp = _file_stamp(db)
                wal = _file_stamp(Path(str(db) + "-wal"))
                identity = wal or stamp
                prev = stamps.get(key)
                if identity is None:
                    continue
                if prev == identity:
                    continue
                stamps[key] = identity
                if stamp is not None and stamp[0] == 0 and wal is None:
                    continue
                agent_id = db.parent.name
                try:
                    on_store_mutation(
                        platform,
                        store_path=db,
                        agents_root=agents_root,
                        conversation_id=agent_id,
                        agent_id=agent_id,
                        agent_name="",
                        cwd=str(agents_root.parent),
                    )
                    last_activity = time.monotonic()
                except Exception:
                    continue
        except Exception:
            pass
        if time.monotonic() - last_activity > _IDLE_EXIT_SECONDS:
            break
        time.sleep(_POLL_INTERVAL)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"thirdeye-{PLATFORM_NAME}-observer")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--agents-root", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_observer(state_dir=args.state_dir, agents_root=args.agents_root)


if __name__ == "__main__":
    raise SystemExit(main())
