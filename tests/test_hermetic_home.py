"""The autouse guard in conftest.py keeps every test off the developer's own
home directory: thirdeye's config and store, the agents' config and session
data, and any THIRDEYE_* setting exported in the shell that runs pytest.

`test_load_without_capture_env` once failed only on machines whose real
~/.thirdeye/config.yaml or shell set capture-env patterns.
"""

from __future__ import annotations

import os
from pathlib import Path

from thirdeye.config import Config


def _inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def test_config_loads_from_a_temporary_home(tmp_path: Path):
    assert _inside(Config.load().root, tmp_path)


def test_home_directory_is_temporary(tmp_path: Path):
    assert _inside(Path.home(), tmp_path)


def test_no_thirdeye_setting_leaks_in_from_the_shell():
    leaked = {
        key
        for key in os.environ
        if key.startswith("THIRDEYE_")
        and key
        not in {"THIRDEYE_HOME", "THIRDEYE_CURSOR_TRANSCRIPT_ROOTS", "THIRDEYE_CURSOR_CHAT_ROOTS"}
    }
    assert leaked == set()
    assert "COPILOT_HOME" not in os.environ


def test_codex_sessions_root_is_temporary(tmp_path: Path):
    from thirdeye.platforms.codex import rollout, usage

    assert _inside(rollout.CODEX_SESSIONS_ROOT, tmp_path)
    assert _inside(usage.CODEX_SESSIONS_ROOT, tmp_path)


def test_gemini_orphan_config_is_temporary(tmp_path: Path):
    from thirdeye.commands import add

    assert all(_inside(path, tmp_path) for path in add.ORPHAN_CONFIG_PATHS)
