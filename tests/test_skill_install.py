from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from thirdeye.cli import main
from thirdeye.commands.skill import _list_bundled_skills, add, skills_group


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


def test_install_creates_symlink_at_default(
    fake_skill: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = _run(fake_skill, [])
    assert result.exit_code == 0
    dest = tmp_path / ".agents" / "skills" / "use-thirdeye"
    assert dest.is_symlink()
    assert dest.resolve() == fake_skill.resolve()


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
    assert dest.is_symlink()


def test_install_custom_target_folder(fake_skill: Path, tmp_path: Path) -> None:
    custom = tmp_path / "custom-skills"
    result = _run(fake_skill, ["-p", str(custom)])
    assert result.exit_code == 0
    installed = custom / "use-thirdeye"
    assert installed.is_symlink()
    assert installed.resolve() == fake_skill.resolve()


def test_install_expands_user(
    fake_skill: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    result = _run(fake_skill, ["--path", "~/.claude/skills"])
    assert result.exit_code == 0
    assert (tmp_path / ".claude" / "skills" / "use-thirdeye").is_symlink()


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
    assert (tmp_path / ".claude" / "skills" / "use-thirdeye").is_symlink()
    assert (tmp_path / ".codex" / "skills" / "use-thirdeye").is_symlink()


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
    assert (target / "use-thirdeye").is_symlink()
