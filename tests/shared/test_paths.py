"""Tests for thirdeye.paths — pure path-layout helpers."""

from pathlib import Path

from thirdeye.paths import (
    eval_agents_config_path,
    eval_def_path,
    eval_defs_dir,
    eval_job_log_path,
    eval_job_path,
    evals_jobs_dir,
    evals_jsonl_path,
    events_path,
    index_path,
    meta_path,
    platform_dir,
    session_dir,
    sessions_root,
    usage_db_path,
    usage_jsonl_path,
    usage_log_path,
    usage_state_path,
)


class TestSessionsRoot:
    def test_appends_traces(self):
        assert sessions_root(Path("/tmp/thirdeye")) == Path("/tmp/thirdeye/traces")

    def test_preserves_home_path(self):
        home = Path("/home/user/.thirdeye")
        assert sessions_root(home) == home / "traces"


class TestPlatformDir:
    def test_claude(self):
        assert platform_dir(Path("/tmp/thirdeye"), "claude") == Path("/tmp/thirdeye/traces/claude")

    def test_cursor(self):
        assert platform_dir(Path("/tmp/thirdeye"), "cursor") == Path("/tmp/thirdeye/traces/cursor")

    def test_nested_under_sessions_root(self):
        home = Path("/tmp/thirdeye")
        assert platform_dir(home, "gemini") == sessions_root(home) / "gemini"


class TestSessionDir:
    def test_basic(self):
        assert session_dir(Path("/tmp/thirdeye"), "claude", "01J9G7") == Path(
            "/tmp/thirdeye/traces/claude/01J9G7"
        )

    def test_full_ulid(self):
        ulid = "01J9G7XK4PABCDEFGHJKMNPQRS"
        result = session_dir(Path("/tmp/thirdeye"), "codex", ulid)
        assert result == Path(f"/tmp/thirdeye/traces/codex/{ulid}")

    def test_nested_under_platform_dir(self):
        home = Path("/tmp/thirdeye")
        sid = "01J9G7XK4P"
        assert session_dir(home, "claude", sid) == (platform_dir(home, "claude") / sid)


class TestEventFiles:
    def setup_method(self):
        self.sd = Path("/tmp/thirdeye/traces/claude/01J9G7")

    def test_events_path(self):
        assert events_path(self.sd) == self.sd / "events.alog"

    def test_index_path(self):
        assert index_path(self.sd) == self.sd / "events.idx"

    def test_meta_path(self):
        assert meta_path(self.sd) == self.sd / "meta.yaml"

    def test_file_extensions_are_distinct(self):
        paths = {events_path(self.sd), index_path(self.sd), meta_path(self.sd)}
        assert len(paths) == 3

    def test_all_under_session_dir(self):
        for fn in (events_path, index_path, meta_path):
            assert fn(self.sd).parent == self.sd


class TestUsagePaths:
    def test_usage_jsonl_path(self):
        sd = Path("/x/.thirdeye/traces/claude/abc")
        assert usage_jsonl_path(sd) == sd / "usage.jsonl"

    def test_usage_state_path(self):
        sd = Path("/x/.thirdeye/traces/claude/abc")
        assert usage_state_path(sd) == sd / "usage.state.json"

    def test_usage_db_path(self):
        home = Path("/home/user/.thirdeye")
        assert usage_db_path(home) == home / "usage.db"

    def test_usage_log_path_is_under_logs_dir(self):
        home = Path("/home/user/.thirdeye")
        assert usage_log_path(home) == home / "logs" / "usage-errors.jsonl"

    def test_helpers_compose_with_session_dir(self):
        home = Path("/x/.thirdeye")
        sd = session_dir(home, "claude", "abc123")
        assert usage_jsonl_path(sd) == home / "traces" / "claude" / "abc123" / "usage.jsonl"
        assert usage_state_path(sd) == home / "traces" / "claude" / "abc123" / "usage.state.json"


class TestEvalPaths:
    def test_evals_jsonl_path(self):
        sd = Path("/x/.thirdeye/traces/claude/abc")
        assert evals_jsonl_path(sd) == sd / "evals.jsonl"

    def test_evals_jobs_dir(self):
        sd = Path("/x/.thirdeye/traces/claude/abc")
        assert evals_jobs_dir(sd) == sd / "evals.jobs"

    def test_eval_job_path(self):
        sd = Path("/x/.thirdeye/traces/claude/abc")
        assert eval_job_path(sd, "01J7XYZ") == sd / "evals.jobs" / "01J7XYZ.json"

    def test_eval_job_log_path(self):
        sd = Path("/x/.thirdeye/traces/claude/abc")
        assert eval_job_log_path(sd, "01J7XYZ") == sd / "evals.jobs" / "01J7XYZ.log"

    def test_eval_defs_dir(self):
        home = Path("/home/user/.thirdeye")
        assert eval_defs_dir(home) == home / "evals" / "defs"

    def test_eval_def_path(self):
        home = Path("/home/user/.thirdeye")
        assert eval_def_path(home, "default") == home / "evals" / "defs" / "default.yaml"

    def test_eval_agents_config_path(self):
        home = Path("/home/user/.thirdeye")
        assert eval_agents_config_path(home) == home / "eval-agents.yaml"

    def test_helpers_compose_with_session_dir(self):
        home = Path("/x/.thirdeye")
        sd = session_dir(home, "claude", "abc123")
        assert evals_jsonl_path(sd) == home / "traces" / "claude" / "abc123" / "evals.jsonl"
