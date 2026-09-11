from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from thirdeye.commands.views import views_group


@pytest.fixture
def env(tmp_path: Path, monkeypatch) -> dict:
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    return {"THIRDEYE_HOME": str(tmp_path)}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestViewsSave:
    def test_save_writes_file(self, tmp_path: Path, runner: CliRunner, env: dict) -> None:
        r = runner.invoke(
            views_group,
            ["save", "--page", "sessions", "--name", "foo", "--path", "?since=7d"],
            env=env,
        )
        assert r.exit_code == 0, r.output
        f = tmp_path / "views" / "sessions.json"
        assert f.exists()
        data = json.loads(f.read_text())
        names = [v["name"] for v in data["views"]]
        assert "foo" in names


class TestViewsList:
    def test_list_prints_saved(self, tmp_path: Path, runner: CliRunner, env: dict) -> None:
        runner.invoke(
            views_group,
            ["save", "--page", "sessions", "--name", "foo", "--path", "?since=7d"],
            env=env,
        )
        r = runner.invoke(views_group, ["list", "--page", "sessions"], env=env)
        assert r.exit_code == 0, r.output
        lines = [ln for ln in r.output.splitlines() if ln.strip()]
        assert len(lines) == 1
        parts = lines[0].split("\t")
        assert parts[0] == "foo"
        assert parts[1] == "?since=7d"
        # created_at present
        assert parts[2]


class TestViewsDelete:
    def test_delete_removes(self, tmp_path: Path, runner: CliRunner, env: dict) -> None:
        runner.invoke(
            views_group,
            ["save", "--page", "sessions", "--name", "foo", "--path", "?since=7d"],
            env=env,
        )
        r = runner.invoke(
            views_group,
            ["delete", "--page", "sessions", "--name", "foo"],
            env=env,
        )
        assert r.exit_code == 0, r.output
        r = runner.invoke(views_group, ["list", "--page", "sessions"], env=env)
        assert r.exit_code == 0
        assert "foo" not in r.output


class TestViewsSaveInvalidName:
    def test_invalid_name_fails(self, tmp_path: Path, runner: CliRunner, env: dict) -> None:
        r = runner.invoke(
            views_group,
            ["save", "--page", "sessions", "--name", "bad/name", "--path", "?since=7d"],
            env=env,
        )
        assert r.exit_code != 0
        assert "invalid" in r.output.lower()


class TestViewsSaveUnknownPage:
    def test_unknown_page_rejected(self, tmp_path: Path, runner: CliRunner, env: dict) -> None:
        r = runner.invoke(
            views_group,
            ["save", "--page", "unknown", "--name", "foo", "--path", "?since=7d"],
            env=env,
        )
        assert r.exit_code != 0
