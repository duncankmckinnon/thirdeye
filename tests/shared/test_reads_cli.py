from __future__ import annotations

import json
from pathlib import Path

import yaml
from click.testing import CliRunner

from thirdeye.cli import main
from thirdeye.config import Config
from thirdeye.paths import meta_path, session_dir
from thirdeye.store import Store
from thirdeye.tags import TagStore


def _env(tmp_path: Path) -> dict:
    return {"THIRDEYE_HOME": str(tmp_path)}


def _seed(runner: CliRunner, env: dict, platform: str, sid: str, events: list[dict]) -> None:
    payload = "\n".join(json.dumps(e) for e in events) + "\n"
    r = runner.invoke(
        main,
        ["ingest", "--platform", platform, "--session-id", sid, "--cwd", "/proj"],
        input=payload,
        env=env,
    )
    assert r.exit_code == 0, r.output


class TestListHarnessAlias:
    def test_harness_filters_like_platform(self, tmp_path: Path):
        runner = CliRunner()
        env = _env(tmp_path)
        _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "x"}])
        _seed(runner, env, "cursor", "02ABCDEF12", [{"t": "y"}])
        r = runner.invoke(main, ["list", "--harness", "claude"], env=env)
        assert r.exit_code == 0, r.output
        assert "01J9G7XK4P" in r.output
        assert "02ABCDEF12" not in r.output

    def test_harness_and_platform_agree(self, tmp_path: Path):
        runner = CliRunner()
        env = _env(tmp_path)
        _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "x"}])
        r = runner.invoke(main, ["list", "--harness", "claude", "--platform", "claude"], env=env)
        assert r.exit_code == 0, r.output
        assert "01J9G7XK4P" in r.output

    def test_harness_and_platform_disagree(self, tmp_path: Path):
        runner = CliRunner()
        env = _env(tmp_path)
        _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "x"}])
        r = runner.invoke(main, ["list", "--harness", "claude", "--platform", "cursor"], env=env)
        assert r.exit_code != 0
        assert "disagree" in r.output.lower()

    def test_both_flags_in_help(self, tmp_path: Path):
        runner = CliRunner()
        r = runner.invoke(main, ["list", "--help"], env=_env(tmp_path))
        assert r.exit_code == 0
        assert "--harness" in r.output
        assert "--platform" in r.output


class TestListTagFilter:
    def _tag_event(self, tmp_path: Path, platform: str, sid: str, seq: int, tag: str) -> None:
        store = Store(Config(root=tmp_path))
        sd = session_dir(store.config.root, platform, sid)
        TagStore(sd).add(seq, tag)

    def test_tag_match(self, tmp_path: Path):
        runner = CliRunner()
        env = _env(tmp_path)
        _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": "hi"}])
        self._tag_event(tmp_path, "claude", "01J9G7XK4P", 0, "foo")
        r = runner.invoke(main, ["list", "--tag", "foo"], env=env)
        assert r.exit_code == 0, r.output
        assert "01J9G7XK4P" in r.output

    def test_tag_no_match(self, tmp_path: Path):
        runner = CliRunner()
        env = _env(tmp_path)
        _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": "hi"}])
        self._tag_event(tmp_path, "claude", "01J9G7XK4P", 0, "foo")
        r = runner.invoke(main, ["list", "--tag", "bar"], env=env)
        assert r.exit_code == 0, r.output
        assert "01J9G7XK4P" not in r.output

    def test_multiple_tags_anded(self, tmp_path: Path):
        runner = CliRunner()
        env = _env(tmp_path)
        _seed(
            runner,
            env,
            "claude",
            "01J9G7XK4P",
            [{"t": "msg", "data": "a"}, {"t": "msg", "data": "b"}],
        )
        # tags on different events; session-level AND should still match (each tag
        # present in at least one event of the session).
        self._tag_event(tmp_path, "claude", "01J9G7XK4P", 0, "foo")
        self._tag_event(tmp_path, "claude", "01J9G7XK4P", 1, "bar")
        r = runner.invoke(main, ["list", "--tag", "foo", "--tag", "bar"], env=env)
        assert r.exit_code == 0, r.output
        assert "01J9G7XK4P" in r.output

        r2 = runner.invoke(main, ["list", "--tag", "foo", "--tag", "baz"], env=env)
        assert r2.exit_code == 0, r2.output
        assert "01J9G7XK4P" not in r2.output

    def test_invalid_tag_errors(self, tmp_path: Path):
        runner = CliRunner()
        env = _env(tmp_path)
        _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "x"}])
        r = runner.invoke(main, ["list", "--tag", "NOT VALID!"], env=env)
        assert r.exit_code != 0


class TestListDateFilter:
    def _set_meta_timestamps(
        self,
        tmp_path: Path,
        platform: str,
        sid: str,
        *,
        started_at: str,
        last_ts: str,
    ) -> None:
        store = Store(Config(root=tmp_path))
        sd = session_dir(store.config.root, platform, sid)
        mpath = meta_path(sd)
        raw = yaml.safe_load(mpath.read_text())
        raw["started_at"] = started_at
        raw["last_ts"] = last_ts
        mpath.write_text(yaml.safe_dump(raw, sort_keys=False))

    def test_since_excludes_old(self, tmp_path: Path):
        runner = CliRunner()
        env = _env(tmp_path)
        _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": "old"}])
        self._set_meta_timestamps(
            tmp_path,
            "claude",
            "01J9G7XK4P",
            started_at="1990-01-01T00:00:00.000Z",
            last_ts="1990-01-01T00:00:00.000Z",
        )
        r = runner.invoke(main, ["list", "--since", "7d"], env=env)
        assert r.exit_code == 0, r.output
        assert "01J9G7XK4P" not in r.output

    def test_since_includes_recent(self, tmp_path: Path):
        runner = CliRunner()
        env = _env(tmp_path)
        _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": "new"}])
        # ingest just wrote current timestamps; no need to mutate
        r = runner.invoke(main, ["list", "--since", "7d"], env=env)
        assert r.exit_code == 0, r.output
        assert "01J9G7XK4P" in r.output

    def test_until_excludes_future(self, tmp_path: Path):
        runner = CliRunner()
        env = _env(tmp_path)
        _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": "new"}])
        r = runner.invoke(main, ["list", "--until", "1990-01-01"], env=env)
        assert r.exit_code == 0, r.output
        assert "01J9G7XK4P" not in r.output


class TestListInvalidSince:
    def test_invalid_since(self, tmp_path: Path):
        runner = CliRunner()
        env = _env(tmp_path)
        r = runner.invoke(main, ["list", "--since", "blarg"], env=env)
        assert r.exit_code != 0
        assert "--since" in r.output
        assert "blarg" in r.output

    def test_invalid_until(self, tmp_path: Path):
        runner = CliRunner()
        env = _env(tmp_path)
        r = runner.invoke(main, ["list", "--until", "blarg"], env=env)
        assert r.exit_code != 0
        assert "--until" in r.output


class TestListJsonFlag:
    def test_json_flag_accepted(self, tmp_path: Path):
        runner = CliRunner()
        env = _env(tmp_path)
        _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "x"}])
        r = runner.invoke(main, ["list", "--json"], env=env)
        assert r.exit_code == 0, r.output
        # default output is jsonl; parse first line.
        obj = json.loads(r.output.strip().splitlines()[0])
        assert obj["session_id"] == "01J9G7XK4P"


class TestSearchSameFlags:
    def _tag_event(self, tmp_path: Path, platform: str, sid: str, seq: int, tag: str) -> None:
        store = Store(Config(root=tmp_path))
        sd = session_dir(store.config.root, platform, sid)
        TagStore(sd).add(seq, tag)

    def test_search_accepts_all_flags(self, tmp_path: Path):
        runner = CliRunner()
        env = _env(tmp_path)
        _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": "needle"}])
        r = runner.invoke(
            main,
            [
                "search",
                "needle",
                "--harness",
                "claude",
                "--since",
                "7d",
                "--until",
                "9999-01-01",
            ],
            env=env,
        )
        assert r.exit_code == 0, r.output
        assert "01J9G7XK4P" in r.output

    def test_search_harness_filter(self, tmp_path: Path):
        runner = CliRunner()
        env = _env(tmp_path)
        _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": "needle"}])
        _seed(runner, env, "cursor", "02ABCDEF12", [{"t": "msg", "data": "needle"}])
        r = runner.invoke(main, ["search", "needle", "--harness", "claude"], env=env)
        assert r.exit_code == 0, r.output
        assert "01J9G7XK4P" in r.output
        assert "02ABCDEF12" not in r.output

    def test_search_harness_platform_disagree(self, tmp_path: Path):
        runner = CliRunner()
        env = _env(tmp_path)
        _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": "needle"}])
        r = runner.invoke(
            main,
            ["search", "needle", "--harness", "claude", "--platform", "cursor"],
            env=env,
        )
        assert r.exit_code != 0
        assert "disagree" in r.output.lower()

    def test_search_invalid_since(self, tmp_path: Path):
        runner = CliRunner()
        env = _env(tmp_path)
        r = runner.invoke(main, ["search", "x", "--since", "blarg"], env=env)
        assert r.exit_code != 0
        assert "--since" in r.output

    def test_search_tag_filter(self, tmp_path: Path):
        runner = CliRunner()
        env = _env(tmp_path)
        _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": "needle"}])
        self._tag_event(tmp_path, "claude", "01J9G7XK4P", 0, "foo")
        r = runner.invoke(main, ["search", "needle", "--tag", "foo"], env=env)
        assert r.exit_code == 0, r.output
        assert "01J9G7XK4P" in r.output

        r2 = runner.invoke(main, ["search", "needle", "--tag", "bar"], env=env)
        assert r2.exit_code == 0, r2.output
        assert "01J9G7XK4P" not in r2.output
