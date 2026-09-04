from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import click


class LogfireAuthError(RuntimeError):
    """Logfire login or write-token minting failed."""


def mint_write_token(*, force_auth: bool = False) -> str:
    """Run Logfire login if needed, then mint a project write token."""
    client = _authenticated_client(force_auth=force_auth)
    try:
        projects = client.get_user_projects()
    except LogfireAuthError:
        raise
    except Exception as exc:
        raise LogfireAuthError(f"could not list Logfire projects: {exc}") from exc
    if projects:
        return _write_token_for_projects(projects, create_token=client.create_write_token)
    return _write_token_for_new_or_named_project(client)


def _authenticated_client(*, force_auth: bool = False) -> Any:
    if force_auth:
        subprocess.run([sys.executable, "-m", "logfire", "auth", "logout"])
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


def _organization_names(client: Any) -> list[str]:
    names: list[str] = []
    try:
        for item in client.get_user_organizations() or []:
            name = item.get("organization_name") if isinstance(item, Mapping) else None
            if name:
                names.append(str(name))
    except Exception:
        names = []
    if names:
        return names
    try:
        me = client.get_user_information() or {}
    except Exception as exc:
        raise LogfireAuthError(f"could not read Logfire account: {exc}") from exc
    if not isinstance(me, Mapping):
        return []
    for key in ("default_organization", "personal_organization"):
        org = me.get(key) or {}
        if isinstance(org, Mapping):
            name = org.get("organization_name")
            if name and str(name) not in names:
                names.append(str(name))
    return names


def _write_token_for_new_or_named_project(client: Any) -> str:
    organizations = _organization_names(client)
    if not organizations:
        raise LogfireAuthError(
            "this Logfire account has no organization. "
            "Open the Logfire UI to finish setup, then retry, "
            "or paste a gateway key in thirdeye ui settings"
        )
    if not click.confirm(
        "No writable Logfire projects were listed. Create or attach a project?",
        default=True,
    ):
        raise LogfireAuthError(
            "no writable Logfire projects on this account; "
            "paste a gateway key in thirdeye ui settings if you already have one"
        )
    if len(organizations) == 1:
        organization = organizations[0]
        click.echo(f"Using Logfire organization {organization}")
    else:
        for index, name in enumerate(organizations, start=1):
            click.echo(f"  {index}. {name}")
        index = click.prompt(
            "Select a Logfire organization",
            type=click.IntRange(1, len(organizations)),
        )
        organization = organizations[index - 1]
    project_name = str(click.prompt("Logfire project name", default="thirdeye")).strip()
    if not project_name:
        raise LogfireAuthError("a Logfire project name is required")
    try:
        creds = client.create_write_token(organization, project_name)
    except Exception:
        try:
            creds = client.create_new_project(organization, project_name)
        except Exception as exc:
            raise LogfireAuthError(
                f"could not create Logfire project {organization}/{project_name}: {exc}"
            ) from exc
    token = creds.get("token") if isinstance(creds, Mapping) else None
    if not token:
        raise LogfireAuthError("Logfire did not return a write token")
    return str(token)
