from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import click


class LogfireAuthError(RuntimeError):
    """Logfire login or write-token minting failed."""


def mint_write_token() -> str:
    """Run Logfire login if needed, then mint a project write token."""
    client = _authenticated_client()
    try:
        projects = client.get_user_projects()
    except LogfireAuthError:
        raise
    except Exception as exc:
        raise LogfireAuthError(f"could not list Logfire projects: {exc}") from exc
    return _write_token_for_projects(projects, create_token=client.create_write_token)


def _authenticated_client() -> Any:
    click.echo("Signing in to Logfire...")
    result = subprocess.run([sys.executable, "-m", "logfire", "auth"])
    if result.returncode != 0:
        raise LogfireAuthError("Logfire authentication failed")
    return _logfire_client_from_saved_user_token()


def _logfire_client_from_saved_user_token() -> Any:
    try:
        from logfire._internal.client import LogfireClient
    except ImportError as exc:
        raise LogfireAuthError(
            "the `logfire` package is not installed. Install with: pip install 'thrdi[logfire]'"
        ) from exc
    try:
        return LogfireClient.from_url(None)
    except Exception as exc:
        raise LogfireAuthError(f"could not use saved Logfire login: {exc}") from exc


def _write_token_for_projects(
    projects: Sequence[Mapping[str, Any]],
    *,
    create_token: Callable[[str, str], Mapping[str, Any]],
) -> str:
    if not projects:
        raise LogfireAuthError("no writable Logfire projects on this account")
    if len(projects) == 1:
        organization = str(projects[0]["organization_name"])
        project_name = str(projects[0]["project_name"])
        click.echo(f"Using Logfire project {organization}/{project_name}")
    else:
        for index, project in enumerate(projects, start=1):
            click.echo(f"  {index}. {project['organization_name']}/{project['project_name']}")
        index = click.prompt(
            "Select a Logfire project",
            type=click.IntRange(1, len(projects)),
        )
        organization = str(projects[index - 1]["organization_name"])
        project_name = str(projects[index - 1]["project_name"])
    try:
        creds = create_token(organization, project_name)
    except LogfireAuthError:
        raise
    except Exception as exc:
        raise LogfireAuthError(f"could not create a Logfire write token: {exc}") from exc
    token = creds.get("token") if isinstance(creds, Mapping) else None
    if not token:
        raise LogfireAuthError("Logfire did not return a write token")
    return str(token)
