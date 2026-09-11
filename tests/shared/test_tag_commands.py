from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from thirdeye.cli import main
from thirdeye.commands.tags import tag, tags
from thirdeye.config import Config
from thirdeye.meta import read_meta
from thirdeye.paths import meta_path, session_dir, tags_path
from thirdeye.store import Store


@pytest.fixture
def env(tmp_path: Path, monkeypatch) -> dict:
    monkeypatch.setenv("THIRDEYE_HOME", str(tmp_path))
    return {"THIRDEYE_HOME": str(tmp_path)}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _seed(tmp_path: Path, sid: str, platform: str = "claude", n: int = 2) -> None:
    store = Store(Config(root=tmp_path))
    with store.open_session(sid, platform=platform, cwd="/proj") as w:
        for i in range(n):
            w.append("user_message", f"event-{i}")


class TestTagAdd:
    def test_add_writes_ops_and_updates_meta(
        self, tmp_path: Path, runner: CliRunner, env: dict
    ) -> None:
        _seed(tmp_path, "abcSESSION1")
        r = runner.invoke(tag, ["abc", "0", "--add", "alpha,beta"], env=env)
        assert r.exit_code == 0, r.output

        tp = tags_path(session_dir(tmp_path, "claude", "abcSESSION1"))
        lines = [ln for ln in tp.read_text().splitlines() if ln]
        assert len(lines) == 2
        ops = [json.loads(ln) for ln in lines]
        assert {o["tag"] for o in ops} == {"alpha", "beta"}
        assert all(o["op"] == "add" and o["seq"] == 0 for o in ops)

        meta = read_meta(meta_path(session_dir(tmp_path, "claude", "abcSESSION1")))
        assert meta is not None
        assert meta.tag_count == 1


class TestTagRemove:
    def test_remove_reduces_tag_count(self, tmp_path: Path, runner: CliRunner, env: dict) -> None:
        _seed(tmp_path, "abcSESSION1")
        r = runner.invoke(tag, ["abc", "0", "--add", "alpha,beta"], env=env)
        assert r.exit_code == 0, r.output
        r = runner.invoke(tag, ["abc", "0", "--remove", "alpha,beta"], env=env)
        assert r.exit_code == 0, r.output

        meta = read_meta(meta_path(session_dir(tmp_path, "claude", "abcSESSION1")))
        assert meta is not None
        assert meta.tag_count == 0


class TestTagAddInvalidName:
    def test_invalid_tag_name_fails(self, tmp_path: Path, runner: CliRunner, env: dict) -> None:
        _seed(tmp_path, "abcSESSION1")
        r = runner.invoke(tag, ["abc", "0", "--add", "Bad Tag"], env=env)
        assert r.exit_code != 0
        assert "invalid tag" in r.output
        assert "Bad Tag" in r.output


class TestTagSeqOutOfRange:
    def test_seq_too_high_fails(self, tmp_path: Path, runner: CliRunner, env: dict) -> None:
        _seed(tmp_path, "abcSESSION1", n=2)
        r = runner.invoke(tag, ["abc", "999", "--add", "x"], env=env)
        assert r.exit_code != 0
        assert "seq 999 not found" in r.output

    def test_seq_negative_fails(self, tmp_path: Path, runner: CliRunner, env: dict) -> None:
        _seed(tmp_path, "abcSESSION1", n=2)
        r = runner.invoke(tag, ["abc", "-1", "--add", "x"], env=env)
        # negative number may be parsed as a flag — accept either failure mode
        assert r.exit_code != 0


class TestTagPrefixResolve:
    def test_unique_prefix_resolves(self, tmp_path: Path, runner: CliRunner, env: dict) -> None:
        _seed(tmp_path, "01JABCDEF")
        r = runner.invoke(tag, ["01", "0", "--add", "alpha"], env=env)
        assert r.exit_code == 0, r.output

    def test_ambiguous_prefix_fails(self, tmp_path: Path, runner: CliRunner, env: dict) -> None:
        _seed(tmp_path, "01JABCDEF")
        _seed(tmp_path, "01JGHIJKL", platform="cursor")
        r = runner.invoke(tag, ["01", "0", "--add", "alpha"], env=env)
        assert r.exit_code != 0
        assert "ambiguous" in r.output.lower()

    def test_no_match_fails(self, tmp_path: Path, runner: CliRunner, env: dict) -> None:
        _seed(tmp_path, "01JABCDEF")
        r = runner.invoke(tag, ["ZZZ", "0", "--add", "alpha"], env=env)
        assert r.exit_code != 0


class TestTagListMode:
    def test_list_prints_sorted_json_lines(
        self, tmp_path: Path, runner: CliRunner, env: dict
    ) -> None:
        _seed(tmp_path, "abcSESSION1", n=4)
        runner.invoke(tag, ["abc", "2", "--add", "gamma"], env=env)
        runner.invoke(tag, ["abc", "0", "--add", "alpha,beta"], env=env)
        runner.invoke(tag, ["abc", "1", "--add", "zeta"], env=env)

        r = runner.invoke(tag, ["abc", "--list"], env=env)
        assert r.exit_code == 0, r.output
        lines = [ln for ln in r.output.splitlines() if ln.strip()]
        objs = [json.loads(ln) for ln in lines]
        assert [o["seq"] for o in objs] == [0, 1, 2]
        assert objs[0]["tags"] == ["alpha", "beta"]
        assert objs[1]["tags"] == ["zeta"]
        assert objs[2]["tags"] == ["gamma"]


class TestTagListModeRejectsSeq:
    def test_list_with_seq_fails(self, tmp_path: Path, runner: CliRunner, env: dict) -> None:
        _seed(tmp_path, "abcSESSION1")
        r = runner.invoke(tag, ["abc", "0", "--list"], env=env)
        assert r.exit_code != 0


class TestTagsCommandTerse:
    def test_terse_global_inventory(self, tmp_path: Path, runner: CliRunner, env: dict) -> None:
        _seed(tmp_path, "sess0001A", n=3)
        _seed(tmp_path, "sess0002B", platform="cursor", n=2)

        # In sess0001A: tag "alpha" on seq 0 and 1; "beta" on seq 0
        runner.invoke(tag, ["sess0001A", "0", "--add", "alpha,beta"], env=env)
        runner.invoke(tag, ["sess0001A", "1", "--add", "alpha"], env=env)
        # In sess0002B: tag "alpha" on seq 0 only
        runner.invoke(tag, ["sess0002B", "0", "--add", "alpha"], env=env)

        r = runner.invoke(tags, [], env=env)
        assert r.exit_code == 0, r.output
        lines = [ln for ln in r.output.splitlines() if ln.strip()]
        # alpha has 3 total events (2 in sess1, 1 in sess2) across 2 sessions
        # beta has 1 event across 1 session
        parsed = [ln.split("\t") for ln in lines]
        as_dict = {row[0]: (int(row[1]), int(row[2])) for row in parsed}
        assert as_dict["alpha"] == (3, 2)
        assert as_dict["beta"] == (1, 1)
        # alpha (higher count) should come before beta
        assert parsed[0][0] == "alpha"
        assert parsed[1][0] == "beta"


class TestTagsCommandJson:
    def test_json_lines(self, tmp_path: Path, runner: CliRunner, env: dict) -> None:
        _seed(tmp_path, "sess0001A", n=2)
        runner.invoke(tag, ["sess0001A", "0", "--add", "alpha,beta"], env=env)

        r = runner.invoke(tags, ["--json"], env=env)
        assert r.exit_code == 0, r.output
        lines = [ln for ln in r.output.splitlines() if ln.strip()]
        objs = [json.loads(ln) for ln in lines]
        tags_seen = {o["tag"]: o for o in objs}
        assert tags_seen["alpha"]["events"] == 1
        assert tags_seen["alpha"]["sessions"] == 1
        assert tags_seen["beta"]["events"] == 1
        assert tags_seen["beta"]["sessions"] == 1


class TestTagAfterClose:
    def test_tag_after_close_works(self, tmp_path: Path, runner: CliRunner, env: dict) -> None:
        _seed(tmp_path, "abcSESSION1")
        store = Store(Config(root=tmp_path))
        store.close_session("abcSESSION1", platform="claude")

        meta = read_meta(meta_path(session_dir(tmp_path, "claude", "abcSESSION1")))
        assert meta is not None
        assert meta.status == "closed"

        r = runner.invoke(tag, ["abc", "0", "--add", "alpha"], env=env)
        assert r.exit_code == 0, r.output

        tp = tags_path(session_dir(tmp_path, "claude", "abcSESSION1"))
        assert tp.exists()
        meta_after = read_meta(meta_path(session_dir(tmp_path, "claude", "abcSESSION1")))
        assert meta_after is not None
        assert meta_after.tag_count == 1


class TestRegistration:
    def test_tag_and_tags_registered(self, tmp_path: Path, runner: CliRunner, env: dict) -> None:
        r = runner.invoke(main, ["--help"], env=env)
        assert r.exit_code == 0
        assert "tag" in r.output
        assert "tags" in r.output
