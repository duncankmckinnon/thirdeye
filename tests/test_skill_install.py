from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from thirdeye.cli import main
from thirdeye.commands.skill import _install_state, _list_bundled_skills, add, skills_group


@pytest.fixture
def fake_skill(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "bundle" / "use-thirdeye"
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: use-thirdeye\n---\n")
    return skill_dir


def _run(fake_skill: Path, args: list[str]) -> object:
    runner = CliRunner()
    with patch("thirdeye.commands.skill._bundled_skill_root", return_value=fake_skill):
        return runner.invoke(add, args, catch_exceptions=False)


def _install_state_for(fake_skill: Path, dest: Path) -> str:
    with patch("thirdeye.commands.skill._bundled_skill_root", return_value=fake_skill):
        return _install_state("use-thirdeye", dest)


def test_install_creates_symlink_at_default(
    fake_skill: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = _run(fake_skill, [])
    assert result.exit_code == 0
    dest = tmp_path / ".agents" / "skills" / "use-thirdeye"
    assert _install_state_for(fake_skill, dest) == "installed"


def test_install_idempotent(
    fake_skill: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _run(fake_skill, [])
    result = _run(fake_skill, [])
    assert result.exit_code == 0
    assert "already installed" in result.output


def test_install_rejects_existing_without_force(
    fake_skill: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dest = tmp_path / ".agents" / "skills" / "use-thirdeye"
    dest.parent.mkdir(parents=True)
    dest.write_text("not a symlink")
    result = _run(fake_skill, [])
    assert result.exit_code != 0
    assert "already exists" in result.output


def test_install_force_replaces(
    fake_skill: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dest = tmp_path / ".agents" / "skills" / "use-thirdeye"
    dest.parent.mkdir(parents=True)
    dest.write_text("not a symlink")
    result = _run(fake_skill, ["--force"])
    assert result.exit_code == 0
    assert _install_state_for(fake_skill, dest) == "installed"


def test_install_custom_target_folder(fake_skill: Path, tmp_path: Path) -> None:
    custom = tmp_path / "custom-skills"
    result = _run(fake_skill, ["-p", str(custom)])
    assert result.exit_code == 0
    installed = custom / "use-thirdeye"
    assert _install_state_for(fake_skill, installed) == "installed"


def test_install_expands_user(
    fake_skill: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # expanduser() reads HOME on POSIX but USERPROFILE on Windows (ntpath never
    # consults HOME), so set both to redirect "~" on either host.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    result = _run(fake_skill, ["--path", "~/.claude/skills"])
    assert result.exit_code == 0
    assert (
        _install_state_for(fake_skill, tmp_path / ".claude" / "skills" / "use-thirdeye")
        == "installed"
    )


def test_install_rejects_custom_path_with_agent_flag(fake_skill: Path, tmp_path: Path) -> None:
    result = _run(fake_skill, ["-p", str(tmp_path / "skills"), "--claude"])
    assert result.exit_code != 0
    assert "cannot be combined" in result.output


def test_skill_list_command() -> None:
    result = CliRunner().invoke(skills_group, ["list"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "use-thirdeye" in result.output


def test_install_all_bundled_skills_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dest_root = tmp_path / "skills"
    result = CliRunner().invoke(skills_group, ["add", "-p", str(dest_root)], catch_exceptions=False)
    assert result.exit_code == 0
    bundled = _list_bundled_skills()
    assert bundled, "expected at least one bundled skill"
    for name in bundled:
        entry = dest_root / name
        assert entry.is_symlink() or entry.is_dir()


def test_install_only_single_skill(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        skills_group,
        ["add", "-p", str(tmp_path / "skills"), "--only", "use-thirdeye"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert (tmp_path / "skills" / "use-thirdeye").exists()


def test_install_unknown_only_errors(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        skills_group,
        ["add", "-p", str(tmp_path / "skills"), "--only", "nonexistent-skill"],
    )
    assert result.exit_code != 0
    assert "unknown skill" in result.output


def test_install_claude_and_codex_together(
    fake_skill: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = _run(fake_skill, ["--claude", "--codex"])
    assert result.exit_code == 0
    assert (
        _install_state_for(fake_skill, tmp_path / ".claude" / "skills" / "use-thirdeye")
        == "installed"
    )
    assert (
        _install_state_for(fake_skill, tmp_path / ".codex" / "skills" / "use-thirdeye")
        == "installed"
    )


def test_plural_top_level_command_replaces_singular() -> None:
    plural = CliRunner().invoke(main, ["skills", "list"], catch_exceptions=False)
    singular = CliRunner().invoke(main, ["skill", "list"])
    assert plural.exit_code == 0
    assert "use-thirdeye" in plural.output
    assert singular.exit_code != 0


def test_long_path_option_accepts_equals_syntax(fake_skill: Path, tmp_path: Path) -> None:
    target = tmp_path / "skills"
    result = _run(fake_skill, [f"--path={target}"])
    assert result.exit_code == 0
    assert _install_state_for(fake_skill, target / "use-thirdeye") == "installed"


def test_copy_fallback_when_symlink_unsupported(
    fake_skill: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def unsupported_symlink(
        self: Path, target: str | Path, target_is_directory: bool = False
    ) -> None:
        raise OSError("directory symlinks unsupported")

    monkeypatch.setattr(Path, "symlink_to", unsupported_symlink)
    result = _run(fake_skill, [])

    assert result.exit_code == 0
    dest = tmp_path / ".agents" / "skills" / "use-thirdeye"
    assert dest.is_dir()
    assert (dest / "SKILL.md").is_file()
    assert (dest / ".thirdeye-skill-src").read_text(encoding="utf-8") == str(fake_skill.resolve())


def test_install_state_installed_for_copy(fake_skill: Path, tmp_path: Path) -> None:
    dest = tmp_path / "skills" / "use-thirdeye"
    shutil.copytree(fake_skill, dest)
    (dest / ".thirdeye-skill-src").write_text(str(fake_skill.resolve()), encoding="utf-8")

    assert _install_state_for(fake_skill, dest) == "installed"


def test_install_state_conflict_for_foreign_marker(fake_skill: Path, tmp_path: Path) -> None:
    dest = tmp_path / "skills" / "use-thirdeye"
    shutil.copytree(fake_skill, dest)
    (dest / ".thirdeye-skill-src").write_text(str(tmp_path / "different-source"), encoding="utf-8")

    assert _install_state_for(fake_skill, dest) == "conflict"


def test_force_replaces_stale_copy(
    fake_skill: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dest = tmp_path / ".agents" / "skills" / "use-thirdeye"
    dest.parent.mkdir(parents=True)
    shutil.copytree(fake_skill, dest)
    (dest / ".thirdeye-skill-src").write_text("/stale/source", encoding="utf-8")
    (dest / "stale-file").write_text("stale", encoding="utf-8")

    def unsupported_symlink(
        self: Path, target: str | Path, target_is_directory: bool = False
    ) -> None:
        raise OSError("directory symlinks unsupported")

    monkeypatch.setattr(Path, "symlink_to", unsupported_symlink)
    result = _run(fake_skill, ["--force"])

    assert result.exit_code == 0
    assert dest.is_dir()
    assert not (dest / "stale-file").exists()
    assert (dest / "SKILL.md").is_file()
    assert (dest / ".thirdeye-skill-src").read_text(encoding="utf-8") == str(fake_skill.resolve())
