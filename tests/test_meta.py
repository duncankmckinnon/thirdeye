"""Tests for thirdeye.meta: SessionMeta dataclass and atomic YAML read/write."""

from __future__ import annotations

import threading
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from thirdeye.meta import SCHEMA_VERSION, SessionMeta, read_meta, write_meta


def _sample(**over) -> SessionMeta:
    base = dict(
        session_id="01J9G7XK4P",
        platform="claude",
        cwd="/tmp/proj",
        started_at="2026-04-30T17:00:00.000Z",
        ended_at=None,
        status="open",
        event_count=0,
        last_seq=-1,
        last_ts=None,
        extra={"model": "opus"},
    )
    base.update(over)
    return SessionMeta(**base)


# -- Roundtrip -----------------------------------------------------------------


class TestWriteThenRead:
    def test_roundtrip_basic(self, tmp_path: Path):
        p = tmp_path / "meta.yaml"
        m = _sample()
        write_meta(p, m)
        assert read_meta(p) == m

    def test_roundtrip_with_ended_at(self, tmp_path: Path):
        p = tmp_path / "meta.yaml"
        m = _sample(ended_at="2026-04-30T18:00:00.000Z", status="closed")
        write_meta(p, m)
        assert read_meta(p) == m

    def test_roundtrip_empty_extra(self, tmp_path: Path):
        p = tmp_path / "meta.yaml"
        m = _sample(extra={})
        write_meta(p, m)
        assert read_meta(p) == m

    def test_roundtrip_complex_extra(self, tmp_path: Path):
        p = tmp_path / "meta.yaml"
        extra = {"model": "opus", "tags": ["debug", "test"], "nested": {"a": 1}}
        m = _sample(extra=extra)
        write_meta(p, m)
        assert read_meta(p) == m

    def test_roundtrip_high_event_count(self, tmp_path: Path):
        p = tmp_path / "meta.yaml"
        m = _sample(event_count=999999, last_seq=999998)
        write_meta(p, m)
        assert read_meta(p) == m

    def test_roundtrip_preserves_all_fields(self, tmp_path: Path):
        p = tmp_path / "meta.yaml"
        m = _sample(
            session_id="01JBCDEFGH1234567890ABCDEF",
            platform="cursor",
            cwd="/home/user/project",
            started_at="2026-01-01T00:00:00.000Z",
            ended_at="2026-01-01T01:00:00.000Z",
            status="closed",
            event_count=42,
            last_seq=41,
            last_ts="2026-01-01T00:59:59.999Z",
            extra={"key": "value"},
        )
        write_meta(p, m)
        got = read_meta(p)
        assert got.session_id == m.session_id
        assert got.platform == m.platform
        assert got.cwd == m.cwd
        assert got.started_at == m.started_at
        assert got.ended_at == m.ended_at
        assert got.status == m.status
        assert got.event_count == m.event_count
        assert got.last_seq == m.last_seq
        assert got.last_ts == m.last_ts
        assert got.extra == m.extra


# -- Atomic write --------------------------------------------------------------


class TestAtomicWrite:
    def test_no_tmp_file_left_behind(self, tmp_path: Path):
        p = tmp_path / "meta.yaml"
        write_meta(p, _sample())
        assert not (tmp_path / "meta.yaml.tmp").exists()
        assert p.exists()

    def test_creates_parent_directories(self, tmp_path: Path):
        p = tmp_path / "deep" / "nested" / "dir" / "meta.yaml"
        write_meta(p, _sample())
        assert p.exists()
        assert read_meta(p) == _sample()

    def test_overwrites_existing_file(self, tmp_path: Path):
        p = tmp_path / "meta.yaml"
        write_meta(p, _sample(status="open"))
        write_meta(p, _sample(status="closed", ended_at="2026-04-30T18:00:00.000Z"))
        got = read_meta(p)
        assert got.status == "closed"
        assert got.ended_at == "2026-04-30T18:00:00.000Z"


# -- Concurrent writers ----------------------------------------------------------


class TestConcurrentWrites:
    def test_concurrent_writers_do_not_crash(self, tmp_path: Path):
        """Multiple hook processes for the same session (e.g. concurrent
        subagents) call write_meta for the same path at once. The tmp file
        must be unique per write attempt, or one writer's `os.replace` can
        consume the tmp file another writer is still using, raising
        FileNotFoundError and crashing that writer's hook process entirely --
        silently dropping whatever event it was recording.
        """
        p = tmp_path / "meta.yaml"
        write_meta(p, _sample())

        errors: list[str] = []
        errors_lock = threading.Lock()

        def worker(tag: str) -> None:
            for i in range(200):
                try:
                    write_meta(p, _sample(event_count=i, last_seq=i - 1, extra={"w": tag}))
                except Exception as e:
                    with errors_lock:
                        errors.append(f"{tag}: {type(e).__name__}: {e}")
                    return

        threads = [threading.Thread(target=worker, args=(tag,)) for tag in "ABCD"]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert read_meta(p) is not None


# -- read_meta edge cases ------------------------------------------------------


class TestReadMeta:
    def test_missing_file_returns_none(self, tmp_path: Path):
        assert read_meta(tmp_path / "absent.yaml") is None

    def test_strips_schema_version(self, tmp_path: Path):
        """schema_version is written to YAML but not part of the dataclass."""
        p = tmp_path / "meta.yaml"
        write_meta(p, _sample())
        with open(p) as f:
            raw = yaml.safe_load(f)
        assert "schema_version" in raw
        got = read_meta(p)
        assert not hasattr(got, "schema_version") or "schema_version" not in asdict(got)

    def test_defaults_extra_when_missing_in_yaml(self, tmp_path: Path):
        """If extra key is absent from the YAML, read_meta should default it to {}."""
        p = tmp_path / "meta.yaml"
        data = {
            "schema_version": SCHEMA_VERSION,
            "session_id": "01J9G7XK4P",
            "platform": "claude",
            "cwd": "/tmp/proj",
            "started_at": "2026-04-30T17:00:00.000Z",
            "ended_at": None,
            "status": "open",
            "event_count": 0,
            "last_seq": -1,
            "last_ts": None,
        }
        with open(p, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        got = read_meta(p)
        assert got is not None
        assert got.extra == {}


# -- YAML file content ---------------------------------------------------------


class TestYamlContent:
    def test_schema_version_in_file(self, tmp_path: Path):
        p = tmp_path / "meta.yaml"
        write_meta(p, _sample())
        with open(p) as f:
            raw = yaml.safe_load(f)
        assert raw["schema_version"] == SCHEMA_VERSION

    def test_schema_version_is_first_key(self, tmp_path: Path):
        p = tmp_path / "meta.yaml"
        write_meta(p, _sample())
        text = p.read_text()
        first_line = text.strip().splitlines()[0]
        assert first_line.startswith("schema_version:")

    def test_yaml_is_valid(self, tmp_path: Path):
        p = tmp_path / "meta.yaml"
        write_meta(p, _sample())
        with open(p) as f:
            raw = yaml.safe_load(f)
        assert isinstance(raw, dict)
        assert raw["session_id"] == "01J9G7XK4P"
        assert raw["platform"] == "claude"

    def test_yaml_keys_not_sorted(self, tmp_path: Path):
        """sort_keys=False should preserve insertion order."""
        p = tmp_path / "meta.yaml"
        write_meta(p, _sample())
        text = p.read_text()
        lines = [l.split(":")[0] for l in text.strip().splitlines() if not l.startswith(" ")]
        sv_idx = lines.index("schema_version")
        sid_idx = lines.index("session_id")
        assert sv_idx < sid_idx


# -- SessionMeta dataclass -----------------------------------------------------


class TestSessionMetaDataclass:
    def test_fields_accessible(self):
        m = _sample()
        assert m.session_id == "01J9G7XK4P"
        assert m.platform == "claude"
        assert m.cwd == "/tmp/proj"
        assert m.started_at == "2026-04-30T17:00:00.000Z"
        assert m.ended_at is None
        assert m.status == "open"
        assert m.event_count == 0
        assert m.last_seq == -1
        assert m.last_ts is None
        assert m.extra == {"model": "opus"}

    def test_extra_defaults_to_empty_dict(self):
        m = SessionMeta(
            session_id="X",
            platform="p",
            cwd="/",
            started_at="ts",
            ended_at=None,
            status="open",
            event_count=0,
            last_seq=-1,
            last_ts=None,
        )
        assert m.extra == {}

    def test_equality(self):
        a = _sample()
        b = _sample()
        assert a == b

    def test_inequality_on_field_change(self):
        a = _sample()
        b = _sample(status="closed")
        assert a != b

    def test_asdict(self):
        m = _sample()
        d = asdict(m)
        assert d["session_id"] == "01J9G7XK4P"
        assert d["extra"] == {"model": "opus"}
        assert isinstance(d, dict)


# -- SCHEMA_VERSION constant ----------------------------------------------------


class TestSchemaVersion:
    def test_schema_version_is_int(self):
        assert isinstance(SCHEMA_VERSION, int)

    def test_schema_version_is_two(self):
        assert SCHEMA_VERSION == 2


# -- tag_count field -----------------------------------------------------------


class TestSessionMetaTagCount:
    def test_tag_count_defaults_to_zero(self):
        m = SessionMeta(
            session_id="X",
            platform="p",
            cwd="/",
            started_at="ts",
            ended_at=None,
            status="open",
            event_count=0,
            last_seq=-1,
            last_ts=None,
        )
        assert m.tag_count == 0

    def test_tag_count_explicit_value(self):
        m = _sample(tag_count=5)
        assert m.tag_count == 5


class TestWriteMetaEmitsTagCount:
    def test_tag_count_written_to_yaml(self, tmp_path: Path):
        p = tmp_path / "meta.yaml"
        m = _sample(tag_count=3)
        write_meta(p, m)
        with open(p) as f:
            raw = yaml.safe_load(f)
        assert raw["tag_count"] == 3
        assert raw["schema_version"] == 2


class TestReadV1Compat:
    def test_v1_yaml_reads_with_tag_count_zero(self, tmp_path: Path):
        p = tmp_path / "meta.yaml"
        data = {
            "schema_version": 1,
            "session_id": "01J9G7XK4P",
            "platform": "claude",
            "cwd": "/tmp/proj",
            "started_at": "2026-04-30T17:00:00.000Z",
            "ended_at": None,
            "status": "open",
            "event_count": 0,
            "last_seq": -1,
            "last_ts": None,
            "extra": {},
        }
        with open(p, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        got = read_meta(p)
        assert got is not None
        assert got.tag_count == 0


class TestReadV2:
    def test_v2_yaml_reads_tag_count(self, tmp_path: Path):
        p = tmp_path / "meta.yaml"
        data = {
            "schema_version": 2,
            "session_id": "01J9G7XK4P",
            "platform": "claude",
            "cwd": "/tmp/proj",
            "started_at": "2026-04-30T17:00:00.000Z",
            "ended_at": None,
            "status": "open",
            "event_count": 0,
            "last_seq": -1,
            "last_ts": None,
            "tag_count": 7,
            "extra": {},
        }
        with open(p, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        got = read_meta(p)
        assert got is not None
        assert got.tag_count == 7


# -- Platform values -----------------------------------------------------------


class TestPlatformValues:
    @pytest.mark.parametrize("platform", ["claude", "cursor", "codex", "gemini", "copilot"])
    def test_roundtrip_various_platforms(self, tmp_path: Path, platform: str):
        p = tmp_path / "meta.yaml"
        m = _sample(platform=platform)
        write_meta(p, m)
        assert read_meta(p) == m


# -- Status values -------------------------------------------------------------


class TestStatusValues:
    @pytest.mark.parametrize("status", ["open", "closed", "stale"])
    def test_roundtrip_various_statuses(self, tmp_path: Path, status: str):
        p = tmp_path / "meta.yaml"
        m = _sample(status=status)
        write_meta(p, m)
        assert read_meta(p) == m
