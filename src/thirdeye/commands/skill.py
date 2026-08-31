from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path

import click


@click.group(name="skills", help="Manage the bundled thirdeye agent skills.")
def skills_group() -> None:
    pass


DEFAULT_TARGET = Path(".agents/skills")
CLAUDE_TARGET = Path(".claude/skills")
CODEX_TARGET = Path(".codex/skills")


def _bundled_skills_root() -> Path:
    """Return the absolute path to the bundled skills package directory."""
    return Path(str(resources.files("thirdeye").joinpath("skills")))


def _list_bundled_skills() -> list[str]:
    """Return the names of bundled skills (subdirs of skills/ containing SKILL.md)."""
    root = _bundled_skills_root()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "SKILL.md").is_file())


def _bundled_skill_root(name: str = "use-thirdeye") -> Path:
    """Return the absolute path to a named bundled skill directory."""
    path = _bundled_skills_root() / name
    if not path.is_dir():
        raise click.ClickException(f"bundled skill not found at {path} — reinstall thirdeye")
    return path.resolve()


def _install_one(name: str, dest: Path, *, force: bool) -> str:
    """Symlink the named bundled skill at `dest`. Returns a status message."""
    source = _bundled_skill_root(name).resolve()
    dest = dest.expanduser().absolute()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_symlink() and dest.resolve() == source:
        return f"{name} skill already installed at {dest}"

    if dest.exists() or dest.is_symlink():
        if not force:
            raise click.ClickException(f"'{dest}' already exists — pass --force to replace")
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)

    dest.symlink_to(source, target_is_directory=True)
    return f"Installed {name} skill at {dest}"


def _install_state(name: str, dest: Path) -> str:
    """Return ``installed``, ``missing``, or ``conflict`` for a skill target."""
    source = _bundled_skill_root(name).resolve()
    dest = dest.expanduser().absolute()
    try:
        if dest.is_symlink() and dest.resolve() == source:
            return "installed"
    except OSError:
        pass
    if dest.exists() or dest.is_symlink():
        return "conflict"
    return "missing"


@skills_group.command(name="add")
@click.option(
    "-p",
    "--path",
    "custom_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Install into a custom skills folder.",
)
@click.option("--claude", is_flag=True, help="Install into .claude/skills.")
@click.option("--codex", is_flag=True, help="Install into .codex/skills.")
@click.option(
    "--only",
    "only",
    multiple=True,
    help="Install only the named skill (repeatable). Defaults to all bundled skills.",
)
@click.option("--force", is_flag=True, help="Replace an existing entry at the destination.")
def add(
    custom_path: Path | None,
    claude: bool,
    codex: bool,
    only: tuple[str, ...],
    force: bool,
) -> None:
    """Symlink bundled thirdeye skills into one or more agent skill folders.

    With no target option, installs into `.agents/skills`. `--claude` and
    `--codex` may be combined. Use `-p/--path` for one custom skills folder.
    """
    if custom_path is not None and (claude or codex):
        raise click.UsageError("-p/--path cannot be combined with --claude or --codex")
    if custom_path is not None:
        targets = [custom_path]
    elif claude or codex:
        targets = []
        if claude:
            targets.append(CLAUDE_TARGET)
        if codex:
            targets.append(CODEX_TARGET)
    else:
        targets = [DEFAULT_TARGET]

    bundled = _list_bundled_skills()
    if only:
        unknown = [n for n in only if n not in bundled]
        if unknown:
            raise click.ClickException(
                f"unknown skill(s): {', '.join(unknown)}; "
                f"available: {', '.join(bundled) if bundled else '(none)'}"
            )
        # Preserve user-given order but de-dupe.
        names = list(dict.fromkeys(only))
    else:
        names = bundled

    if not names:
        raise click.ClickException("no bundled skills found")

    for target in targets:
        for skill_name in names:
            msg = _install_one(skill_name, target / skill_name, force=force)
            click.echo(msg)


@skills_group.command(name="list", help="List the names of bundled thirdeye skills.")
def list_cmd() -> None:
    for name in _list_bundled_skills():
        click.echo(name)
