from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from thirdeye.logfire_auth import LogfireAuthError, mint_write_token


class _FakeClient:
    def __init__(self, projects: list[dict[str, str]], token: str = "minted-token") -> None:
        self.projects = projects
        self.token = token
        self.created: list[tuple[str, str]] = []

    def get_user_projects(self) -> list[dict[str, str]]:
        return self.projects

    def create_write_token(self, organization: str, project_name: str) -> dict[str, str]:
        self.created.append((organization, project_name))
        return {"token": self.token, "project_name": project_name}


def test_mint_uses_only_project_without_prompt(monkeypatch: pytest.MonkeyPatch):
    client = _FakeClient([{"organization_name": "acme", "project_name": "traces"}])
    monkeypatch.setattr(
        "thirdeye.logfire_auth._authenticated_client", lambda force_auth=False: client
    )

    assert mint_write_token() == "minted-token"
    assert client.created == [("acme", "traces")]


def test_mint_prompts_when_multiple_projects(monkeypatch: pytest.MonkeyPatch):
    client = _FakeClient(
        [
            {"organization_name": "acme", "project_name": "one"},
            {"organization_name": "acme", "project_name": "two"},
        ]
    )
    monkeypatch.setattr(
        "thirdeye.logfire_auth._authenticated_client", lambda force_auth=False: client
    )
    monkeypatch.setattr("thirdeye.logfire_auth.click.prompt", lambda *args, **kwargs: 2)

    assert mint_write_token() == "minted-token"
    assert client.created == [("acme", "two")]


def test_mint_errors_when_account_has_no_projects(monkeypatch: pytest.MonkeyPatch):
    client = _FakeClient([])
    monkeypatch.setattr(
        "thirdeye.logfire_auth._authenticated_client", lambda force_auth=False: client
    )

    with pytest.raises(LogfireAuthError, match="no writable"):
        mint_write_token()
    assert client.created == []


def test_mint_errors_when_write_token_missing(monkeypatch: pytest.MonkeyPatch):
    client = _FakeClient([{"organization_name": "acme", "project_name": "traces"}])
    client.create_write_token = lambda organization, project_name: {"project_name": project_name}
    monkeypatch.setattr(
        "thirdeye.logfire_auth._authenticated_client", lambda force_auth=False: client
    )

    with pytest.raises(LogfireAuthError, match="did not return a write token"):
        mint_write_token()


def test_mint_wraps_project_list_errors(monkeypatch: pytest.MonkeyPatch):
    class Boom:
        def get_user_projects(self):
            raise RuntimeError("network down")

    monkeypatch.setattr(
        "thirdeye.logfire_auth._authenticated_client", lambda force_auth=False: Boom()
    )

    with pytest.raises(LogfireAuthError, match="could not list Logfire projects"):
        mint_write_token()


def test_mint_wraps_create_token_errors(monkeypatch: pytest.MonkeyPatch):
    client = _FakeClient([{"organization_name": "acme", "project_name": "traces"}])

    def boom(organization: str, project_name: str) -> dict[str, str]:
        raise RuntimeError("denied")

    client.create_write_token = boom
    monkeypatch.setattr(
        "thirdeye.logfire_auth._authenticated_client", lambda force_auth=False: client
    )

    with pytest.raises(LogfireAuthError, match="could not create a Logfire write token"):
        mint_write_token()


def test_authenticate_runs_logfire_auth_then_builds_client(monkeypatch: pytest.MonkeyPatch):
    run = MagicMock(return_value=SimpleNamespace(returncode=0))
    client = object()
    monkeypatch.setattr("thirdeye.logfire_auth.subprocess.run", run)
    monkeypatch.setattr(
        "thirdeye.logfire_auth._logfire_client_from_saved_user_token", lambda: client
    )

    from thirdeye.logfire_auth import _authenticated_client

    assert _authenticated_client() is client
    command = run.call_args.args[0]
    assert command == [sys.executable, "-m", "logfire", "auth"]
    assert run.call_count == 1


def test_authenticate_logs_out_before_auth_when_forced(monkeypatch: pytest.MonkeyPatch):
    run = MagicMock(return_value=SimpleNamespace(returncode=0))
    client = object()
    monkeypatch.setattr("thirdeye.logfire_auth.subprocess.run", run)
    monkeypatch.setattr(
        "thirdeye.logfire_auth._logfire_client_from_saved_user_token", lambda: client
    )

    from thirdeye.logfire_auth import _authenticated_client

    assert _authenticated_client(force_auth=True) is client
    commands = [call.args[0] for call in run.call_args_list]
    assert commands == [
        [sys.executable, "-m", "logfire", "auth", "logout"],
        [sys.executable, "-m", "logfire", "auth"],
    ]


def test_authenticate_ignores_logout_failure_when_forcing(monkeypatch: pytest.MonkeyPatch):
    run = MagicMock(side_effect=[SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)])
    client = object()
    monkeypatch.setattr("thirdeye.logfire_auth.subprocess.run", run)
    monkeypatch.setattr(
        "thirdeye.logfire_auth._logfire_client_from_saved_user_token", lambda: client
    )

    from thirdeye.logfire_auth import _authenticated_client

    assert _authenticated_client(force_auth=True) is client
    assert run.call_count == 2


def test_authenticate_raises_when_logfire_auth_fails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "thirdeye.logfire_auth.subprocess.run",
        MagicMock(return_value=SimpleNamespace(returncode=1)),
    )

    from thirdeye.logfire_auth import _authenticated_client

    with pytest.raises(LogfireAuthError, match="authentication failed"):
        _authenticated_client()


def test_saved_user_token_client_uses_logfire_sdk(monkeypatch: pytest.MonkeyPatch):
    logfire = pytest.importorskip("logfire")
    fake = object()
    monkeypatch.setattr(
        logfire._internal.client.LogfireClient,
        "from_url",
        classmethod(lambda cls, base_url: fake),
    )

    from thirdeye.logfire_auth import _logfire_client_from_saved_user_token

    assert _logfire_client_from_saved_user_token() is fake


def test_saved_user_token_client_wraps_sdk_errors(monkeypatch: pytest.MonkeyPatch):
    logfire = pytest.importorskip("logfire")

    def boom(cls, base_url):
        raise RuntimeError("expired")

    monkeypatch.setattr(
        logfire._internal.client.LogfireClient,
        "from_url",
        classmethod(boom),
    )

    from thirdeye.logfire_auth import _logfire_client_from_saved_user_token

    with pytest.raises(LogfireAuthError, match="could not use saved Logfire login"):
        _logfire_client_from_saved_user_token()
