from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from thirdeye.cli import main
from thirdeye.reader import SessionReader


def _runner_with_home(tmp_path: Path) -> tuple[CliRunner, dict]:
    return CliRunner(), {"THIRDEYE_HOME": str(tmp_path)}


# -- help / version -----------------------------------------------------------


def test_help(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    r = runner.invoke(main, ["--help"], env=env)
    assert r.exit_code == 0
    assert "thirdeye" in r.output.lower()


def test_ingest_help(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    r = runner.invoke(main, ["ingest", "--help"], env=env)
    assert r.exit_code == 0
    assert "--platform" in r.output
    assert "--session-id" in r.output
    assert "--cwd" in r.output


# -- ingest basic flow --------------------------------------------------------


def test_ingest_writes_session(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    payload = (
        json.dumps({"t": "user_message", "data": "hi"})
        + "\n"
        + json.dumps({"t": "assistant_message", "data": "hello"})
        + "\n"
    )
    r = runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "01J9G7XK4P", "--cwd", "/p"],
        input=payload,
        env=env,
    )
    assert r.exit_code == 0, r.output

    sd = tmp_path / "traces" / "claude" / "01J9G7XK4P"
    assert (sd / "events.alog").exists()
    assert (sd / "events.idx").stat().st_size == 16  # 2 events * 8 bytes each
    assert (sd / "meta.yaml").exists()


def test_ingest_event_count_in_stderr(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    payload = json.dumps({"t": "msg", "data": "hi"}) + "\n"
    r = runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "SID1"],
        input=payload,
        env=env,
    )
    assert r.exit_code == 0
    # stderr should contain the session id and event count
    assert "SID1" in (r.output + getattr(r, "stderr", ""))
    assert "1 events" in (r.output + getattr(r, "stderr", ""))


def test_ingest_prints_session_id_to_stdout(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    payload = json.dumps({"t": "msg"}) + "\n"
    r = runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "MYSID"],
        input=payload,
        env=env,
    )
    assert r.exit_code == 0
    # The last line of stdout should be the session ID alone
    assert "MYSID" in r.output


# -- session-id auto-generation -----------------------------------------------


def test_ingest_generates_session_id_when_omitted(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    payload = json.dumps({"t": "msg", "data": "x"}) + "\n"
    r = runner.invoke(
        main,
        ["ingest", "--platform", "claude"],
        input=payload,
        env=env,
    )
    assert r.exit_code == 0
    # Output should contain a ULID (26 chars, alphanumeric)
    lines = r.output.strip().splitlines()
    generated_id = lines[-1].strip()
    assert len(generated_id) == 26
    # Session dir should exist under the generated ID
    sd = tmp_path / "traces" / "claude" / generated_id
    assert sd.is_dir()


# -- cwd default --------------------------------------------------------------


def test_ingest_cwd_defaults_to_dot(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    payload = json.dumps({"t": "msg"}) + "\n"
    r = runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "CWD1"],
        input=payload,
        env=env,
    )
    assert r.exit_code == 0
    # meta.yaml should record cwd as "."
    import yaml

    meta = yaml.safe_load((tmp_path / "traces" / "claude" / "CWD1" / "meta.yaml").read_text())
    assert meta["cwd"] == "."


def test_ingest_cwd_explicit(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    payload = json.dumps({"t": "msg"}) + "\n"
    r = runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "CWD2", "--cwd", "/my/project"],
        input=payload,
        env=env,
    )
    assert r.exit_code == 0
    import yaml

    meta = yaml.safe_load((tmp_path / "traces" / "claude" / "CWD2" / "meta.yaml").read_text())
    assert meta["cwd"] == "/my/project"


# -- blank lines / empty input ------------------------------------------------


def test_ingest_skips_blank_lines(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    payload = (
        "\n"
        + json.dumps({"t": "msg", "data": "one"})
        + "\n"
        + "\n"
        + "   \n"
        + json.dumps({"t": "msg", "data": "two"})
        + "\n"
        + "\n"
    )
    r = runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "BLANK1"],
        input=payload,
        env=env,
    )
    assert r.exit_code == 0
    # Only 2 real events, blank lines skipped
    sd = tmp_path / "traces" / "claude" / "BLANK1"
    assert (sd / "events.idx").stat().st_size == 16  # 2 * 8


def test_ingest_empty_stdin(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    r = runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "EMPTY1"],
        input="",
        env=env,
    )
    assert r.exit_code == 0
    # 0 events written
    assert "0 events" in (r.output + getattr(r, "stderr", ""))


def test_ingest_handles_crlf_input(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    payload = (
        json.dumps({"t": "user_message", "data": "café"})
        + "\r\n"
        + json.dumps({"t": "assistant_message", "data": "日本語"})
        + "\r\n"
    )
    r = runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "CRLF1"],
        input=payload,
        env=env,
    )

    assert r.exit_code == 0, r.output
    events = list(SessionReader(tmp_path / "traces" / "claude" / "CRLF1").iter_events())
    assert [event["data"] for event in events] == ["café", "日本語"]
    assert all("\r" not in event["data"] for event in events)


# -- event field defaults -----------------------------------------------------


def test_ingest_defaults_t_to_event(tmp_path: Path):
    """When `t` is absent from the JSON line, it defaults to "event"."""
    runner, env = _runner_with_home(tmp_path)
    payload = json.dumps({"data": "something"}) + "\n"
    r = runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "TDEF1"],
        input=payload,
        env=env,
    )
    assert r.exit_code == 0
    # Verify a single event was written
    sd = tmp_path / "traces" / "claude" / "TDEF1"
    assert (sd / "events.idx").stat().st_size == 8  # 1 event


def test_ingest_data_none_when_absent(tmp_path: Path):
    """When `data` is absent from the JSON line, data=None is passed to append."""
    runner, env = _runner_with_home(tmp_path)
    payload = json.dumps({"t": "ping"}) + "\n"
    r = runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "DNIL1"],
        input=payload,
        env=env,
    )
    assert r.exit_code == 0
    sd = tmp_path / "traces" / "claude" / "DNIL1"
    assert (sd / "events.idx").stat().st_size == 8


# -- platform required --------------------------------------------------------


def test_ingest_requires_platform(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    payload = json.dumps({"t": "msg"}) + "\n"
    r = runner.invoke(
        main,
        ["ingest"],
        input=payload,
        env=env,
    )
    assert r.exit_code != 0
    assert "platform" in r.output.lower() or "required" in r.output.lower()


# -- multiple events ----------------------------------------------------------


def test_ingest_many_events(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    n = 50
    lines = [json.dumps({"t": f"evt_{i}", "data": {"n": i}}) for i in range(n)]
    payload = "\n".join(lines) + "\n"
    r = runner.invoke(
        main,
        ["ingest", "--platform", "cursor", "--session-id", "MANY1"],
        input=payload,
        env=env,
    )
    assert r.exit_code == 0
    sd = tmp_path / "traces" / "cursor" / "MANY1"
    assert (sd / "events.idx").stat().st_size == n * 8


# -- session closed after ingest ----------------------------------------------


def test_ingest_closes_session(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    payload = json.dumps({"t": "msg", "data": "hi"}) + "\n"
    r = runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "CLOSE1"],
        input=payload,
        env=env,
    )
    assert r.exit_code == 0
    import yaml

    meta = yaml.safe_load((tmp_path / "traces" / "claude" / "CLOSE1" / "meta.yaml").read_text())
    assert meta["status"] == "closed"


# -- events are readable after ingest -----------------------------------------


def test_ingest_events_readable(tmp_path: Path):
    """After ingest, the stored events should be readable via SessionReader."""
    runner, env = _runner_with_home(tmp_path)
    payload = (
        json.dumps({"t": "user_message", "data": "hello"})
        + "\n"
        + json.dumps({"t": "assistant_message", "data": "world"})
        + "\n"
    )
    r = runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "READ1", "--cwd", "/proj"],
        input=payload,
        env=env,
    )
    assert r.exit_code == 0

    from thirdeye.config import Config
    from thirdeye.store import Store

    store = Store(Config(root=tmp_path))
    reader = store.reader("READ1")
    events = list(reader.iter_events())
    assert len(events) == 2
    assert events[0]["t"] == "user_message"
    assert events[0]["data"] == "hello"
    assert events[1]["t"] == "assistant_message"
    assert events[1]["data"] == "world"


# -- different platforms -------------------------------------------------------


def test_ingest_different_platforms(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    for platform in ["claude", "cursor", "codex"]:
        payload = json.dumps({"t": "msg", "data": platform}) + "\n"
        r = runner.invoke(
            main,
            ["ingest", "--platform", platform, "--session-id", f"SID_{platform}"],
            input=payload,
            env=env,
        )
        assert r.exit_code == 0, f"failed for {platform}: {r.output}"
    # All three session dirs exist
    for platform in ["claude", "cursor", "codex"]:
        assert (tmp_path / "traces" / platform / f"SID_{platform}").is_dir()


# -- invalid JSON input -------------------------------------------------------


def test_ingest_invalid_json_fails(tmp_path: Path):
    """Invalid JSON on stdin should cause a non-zero exit."""
    runner, env = _runner_with_home(tmp_path)
    r = runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "BADJSON"],
        input="not valid json\n",
        env=env,
    )
    assert r.exit_code != 0


def test_ingest_partial_valid_json_stops_at_bad_line(tmp_path: Path):
    """If good lines precede a bad line, the command should fail on the bad line."""
    runner, env = _runner_with_home(tmp_path)
    payload = json.dumps({"t": "ok", "data": "fine"}) + "\n" + "BAD LINE\n"
    r = runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "PARTIAL1"],
        input=payload,
        env=env,
    )
    assert r.exit_code != 0


# -- event data roundtrip with defaults ----------------------------------------


def test_ingest_default_t_value_readable(tmp_path: Path):
    """When t is omitted, the stored event should have t='event'."""
    runner, env = _runner_with_home(tmp_path)
    payload = json.dumps({"data": "no_type"}) + "\n"
    r = runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "TROUND"],
        input=payload,
        env=env,
    )
    assert r.exit_code == 0

    from thirdeye.config import Config
    from thirdeye.store import Store

    store = Store(Config(root=tmp_path))
    reader = store.reader("TROUND")
    events = list(reader.iter_events())
    assert len(events) == 1
    assert events[0]["t"] == "event"
    assert events[0]["data"] == "no_type"


def test_ingest_data_none_roundtrip(tmp_path: Path):
    """When data is omitted, stored event should not have a data key (or have None)."""
    runner, env = _runner_with_home(tmp_path)
    payload = json.dumps({"t": "ping"}) + "\n"
    r = runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "DROUND"],
        input=payload,
        env=env,
    )
    assert r.exit_code == 0

    from thirdeye.config import Config
    from thirdeye.store import Store

    store = Store(Config(root=tmp_path))
    reader = store.reader("DROUND")
    events = list(reader.iter_events())
    assert len(events) == 1
    assert events[0]["t"] == "ping"
    # data=None means the key is either absent or None
    assert events[0].get("data") is None


# -- only-blank-lines input ---------------------------------------------------


def test_ingest_only_blank_lines(tmp_path: Path):
    """Stdin with only blank/whitespace lines should write 0 events."""
    runner, env = _runner_with_home(tmp_path)
    r = runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "BLANKS"],
        input="\n\n  \n\t\n",
        env=env,
    )
    assert r.exit_code == 0
    assert "0 events" in (r.output + getattr(r, "stderr", ""))


# -- complex data types -------------------------------------------------------


def test_ingest_complex_data(tmp_path: Path):
    """Events with nested dicts, lists, and various types should roundtrip."""
    runner, env = _runner_with_home(tmp_path)
    complex_data = {
        "nested": {"a": [1, 2, 3]},
        "flag": True,
        "count": 42,
        "label": None,
    }
    payload = json.dumps({"t": "complex", "data": complex_data}) + "\n"
    r = runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "CPLX1"],
        input=payload,
        env=env,
    )
    assert r.exit_code == 0

    from thirdeye.config import Config
    from thirdeye.store import Store

    store = Store(Config(root=tmp_path))
    reader = store.reader("CPLX1")
    events = list(reader.iter_events())
    assert len(events) == 1
    assert events[0]["data"] == complex_data


# -- meta.yaml platform field --------------------------------------------------


def test_ingest_meta_records_platform(tmp_path: Path):
    """meta.yaml should record the platform passed via --platform."""
    runner, env = _runner_with_home(tmp_path)
    payload = json.dumps({"t": "msg"}) + "\n"
    r = runner.invoke(
        main,
        ["ingest", "--platform", "gemini", "--session-id", "PLAT1"],
        input=payload,
        env=env,
    )
    assert r.exit_code == 0
    import yaml

    meta = yaml.safe_load((tmp_path / "traces" / "gemini" / "PLAT1" / "meta.yaml").read_text())
    assert meta["platform"] == "gemini"
    assert meta["session_id"] == "PLAT1"


# =============================================================================
# Read commands
# =============================================================================


def _seed(runner: CliRunner, env: dict, platform: str, sid: str, events: list[dict]) -> None:
    payload = "\n".join(json.dumps(e) for e in events) + "\n"
    runner.invoke(
        main,
        ["ingest", "--platform", platform, "--session-id", sid, "--cwd", "/proj"],
        input=payload,
        env=env,
    )


# -- list command --------------------------------------------------------------


def test_list_default_jsonl(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "user_message", "data": "hi"}])
    r = runner.invoke(main, ["list"], env=env)
    assert r.exit_code == 0
    parsed = json.loads(r.output.strip().splitlines()[0])
    assert parsed["session_id"] == "01J9G7XK4P"
    assert parsed["platform"] == "claude"


def test_list_filter_platform(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "x"}])
    _seed(runner, env, "cursor", "02ABCDEF12", [{"t": "y"}])
    r = runner.invoke(main, ["list", "--platform", "cursor"], env=env)
    assert "02ABCDEF12" in r.output
    assert "01J9G7XK4P" not in r.output


def test_list_filter_cwd(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "x"}])
    # Seed a second session with different cwd
    payload = json.dumps({"t": "y"}) + "\n"
    runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "02ABCDEF12", "--cwd", "/other"],
        input=payload,
        env=env,
    )
    r = runner.invoke(main, ["list", "--cwd", "/other"], env=env)
    assert "02ABCDEF12" in r.output
    assert "01J9G7XK4P" not in r.output


def test_list_filter_status(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "x"}])
    r = runner.invoke(main, ["list", "--status", "closed"], env=env)
    assert "01J9G7XK4P" in r.output
    r2 = runner.invoke(main, ["list", "--status", "open"], env=env)
    assert "01J9G7XK4P" not in r2.output


def test_list_tree_mode(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "x"}, {"t": "y"}])
    r = runner.invoke(main, ["list", "--tree"], env=env)
    assert r.exit_code == 0
    assert "01J9G7XK4P" in r.output
    assert "[claude]" in r.output
    assert "events=2" in r.output


def test_list_empty_store(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    r = runner.invoke(main, ["list"], env=env)
    assert r.exit_code == 0
    assert r.output.strip() == ""


def test_list_multiple_sessions(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "x"}])
    _seed(runner, env, "cursor", "02ABCDEF12", [{"t": "y"}])
    _seed(runner, env, "codex", "03GHIJKL34", [{"t": "z"}])
    r = runner.invoke(main, ["list"], env=env)
    assert r.exit_code == 0
    lines = r.output.strip().splitlines()
    assert len(lines) == 3


def test_list_help(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    r = runner.invoke(main, ["list", "--help"], env=env)
    assert r.exit_code == 0
    assert "--platform" in r.output
    assert "--cwd" in r.output
    assert "--status" in r.output
    assert "--tree" in r.output


# -- events command ------------------------------------------------------------


def test_events_terse(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(
        runner,
        env,
        "claude",
        "01J9G7XK4P",
        [{"t": "user_message", "data": "hello"}, {"t": "assistant_message", "data": "hi"}],
    )
    r = runner.invoke(main, ["events", "01J9"], env=env)
    assert r.exit_code == 0
    lines = r.output.strip().splitlines()
    assert lines[0] == "0 user_message hello"
    assert lines[1] == "1 assistant_message hi"


def test_events_json(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "user_message", "data": "hi"}])
    r = runner.invoke(main, ["events", "01J9", "--json"], env=env)
    assert r.exit_code == 0
    obj = json.loads(r.output.strip())
    assert obj["t"] == "user_message"


def test_events_tree(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "user_message", "data": "hi"}])
    r = runner.invoke(main, ["events", "01J9", "--tree"], env=env)
    assert r.exit_code == 0
    assert "#0 user_message" in r.output


def test_events_type_filter(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(
        runner,
        env,
        "claude",
        "01J9G7XK4P",
        [
            {"t": "user_message", "data": "a"},
            {"t": "assistant_message", "data": "b"},
            {"t": "user_message", "data": "c"},
        ],
    )
    r = runner.invoke(main, ["events", "01J9", "--type", "user_message"], env=env)
    assert r.exit_code == 0
    lines = r.output.strip().splitlines()
    assert len(lines) == 2
    assert "assistant_message" not in r.output


def test_events_multiple_type_filters(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(
        runner,
        env,
        "claude",
        "01J9G7XK4P",
        [
            {"t": "user_message", "data": "a"},
            {"t": "assistant_message", "data": "b"},
            {"t": "tool_call", "data": "c"},
        ],
    )
    r = runner.invoke(
        main, ["events", "01J9", "--type", "user_message", "--type", "tool_call"], env=env
    )
    assert r.exit_code == 0
    lines = r.output.strip().splitlines()
    assert len(lines) == 2
    assert "assistant_message" not in r.output


def test_events_width_truncation(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    long_data = "x" * 200
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": long_data}])
    r = runner.invoke(main, ["events", "01J9", "--width", "50"], env=env)
    assert r.exit_code == 0
    line = r.output.strip().splitlines()[0]
    assert len(line) <= 50


def test_events_width_zero_unlimited(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    long_data = "x" * 200
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": long_data}])
    r = runner.invoke(main, ["events", "01J9", "--width", "0"], env=env)
    assert r.exit_code == 0
    line = r.output.strip().splitlines()[0]
    assert long_data in line


def test_events_prefix_resolution(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": "hi"}])
    # Full ID should also work
    r = runner.invoke(main, ["events", "01J9G7XK4P"], env=env)
    assert r.exit_code == 0
    assert "msg" in r.output


def test_events_no_match_prefix(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": "hi"}])
    r = runner.invoke(main, ["events", "ZZZZZZ"], env=env)
    assert r.exit_code != 0


def test_events_help(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    r = runner.invoke(main, ["events", "--help"], env=env)
    assert r.exit_code == 0
    assert "--type" in r.output
    assert "--json" in r.output
    assert "--tree" in r.output
    assert "--width" in r.output


# -- show command --------------------------------------------------------------


def test_show_uses_events(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(
        runner,
        env,
        "claude",
        "01J9G7XK4P",
        [{"t": "user_message", "data": "a"}, {"t": "assistant_message", "data": "b"}],
    )
    r = runner.invoke(main, ["show", "01J9"], env=env)
    assert "user_message" in r.output and "assistant_message" in r.output


def test_show_json_mode(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": "hi"}])
    r = runner.invoke(main, ["show", "01J9", "--json"], env=env)
    assert r.exit_code == 0
    obj = json.loads(r.output.strip())
    assert obj["t"] == "msg"


def test_show_tree_mode(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": "hi"}])
    r = runner.invoke(main, ["show", "01J9", "--tree"], env=env)
    assert r.exit_code == 0
    assert "#0 msg" in r.output


# -- tail command --------------------------------------------------------------


def test_tail(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    events_payload = [{"t": "user_message", "data": str(i)} for i in range(5)]
    _seed(runner, env, "claude", "01J9G7XK4P", events_payload)
    r = runner.invoke(main, ["tail", "01J9", "-n", "2"], env=env)
    assert r.exit_code == 0
    lines = r.output.strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("3")
    assert lines[1].startswith("4")


def test_tail_default_n(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    events_payload = [{"t": "msg", "data": str(i)} for i in range(15)]
    _seed(runner, env, "claude", "01J9G7XK4P", events_payload)
    r = runner.invoke(main, ["tail", "01J9"], env=env)
    assert r.exit_code == 0
    lines = r.output.strip().splitlines()
    assert len(lines) == 10  # default is 10


def test_tail_n_larger_than_total(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    events_payload = [{"t": "msg", "data": str(i)} for i in range(3)]
    _seed(runner, env, "claude", "01J9G7XK4P", events_payload)
    r = runner.invoke(main, ["tail", "01J9", "-n", "100"], env=env)
    assert r.exit_code == 0
    lines = r.output.strip().splitlines()
    assert len(lines) == 3  # only 3 events exist


def test_tail_json_mode(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    events_payload = [{"t": "msg", "data": str(i)} for i in range(5)]
    _seed(runner, env, "claude", "01J9G7XK4P", events_payload)
    r = runner.invoke(main, ["tail", "01J9", "-n", "1", "--json"], env=env)
    assert r.exit_code == 0
    obj = json.loads(r.output.strip())
    assert obj["data"] == "4"


def test_tail_tree_mode(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    events_payload = [{"t": "msg", "data": str(i)} for i in range(5)]
    _seed(runner, env, "claude", "01J9G7XK4P", events_payload)
    r = runner.invoke(main, ["tail", "01J9", "-n", "1", "--tree"], env=env)
    assert r.exit_code == 0
    assert "#4 msg" in r.output


# -- event command -------------------------------------------------------------


def test_event_returns_full_data(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    big = "x" * 500
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "blob", "data": big}])
    r = runner.invoke(main, ["event", "01J9", "0"], env=env)
    assert r.exit_code == 0
    assert big in r.output


def test_event_json_format(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": "hello"}])
    r = runner.invoke(main, ["event", "01J9", "0"], env=env)
    assert r.exit_code == 0
    obj = json.loads(r.output)
    assert obj["t"] == "msg"
    assert obj["data"] == "hello"
    assert "seq" in obj
    assert "ts" in obj


def test_event_field_extraction(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(
        runner,
        env,
        "claude",
        "01J9G7XK4P",
        [{"t": "msg", "data": {"content": "hello", "role": "user"}}],
    )
    r = runner.invoke(main, ["event", "01J9", "0", "--field", "content"], env=env)
    assert r.exit_code == 0
    assert r.output.strip() == "hello"


def test_event_field_extraction_nested(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    nested = {"inner": {"deep": True}}
    _seed(
        runner,
        env,
        "claude",
        "01J9G7XK4P",
        [{"t": "msg", "data": {"payload": nested, "label": "test"}}],
    )
    r = runner.invoke(main, ["event", "01J9", "0", "--field", "payload"], env=env)
    assert r.exit_code == 0
    obj = json.loads(r.output.strip())
    assert obj == nested


def test_event_field_not_found(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": {"content": "hello"}}])
    r = runner.invoke(main, ["event", "01J9", "0", "--field", "nonexistent"], env=env)
    assert r.exit_code != 0


def test_event_field_on_string_data(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": "plain string"}])
    r = runner.invoke(main, ["event", "01J9", "0", "--field", "anything"], env=env)
    assert r.exit_code != 0


def test_event_field_scalar_types(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(
        runner,
        env,
        "claude",
        "01J9G7XK4P",
        [{"t": "msg", "data": {"count": 42, "flag": True, "rate": 3.14}}],
    )
    # int
    r = runner.invoke(main, ["event", "01J9", "0", "--field", "count"], env=env)
    assert r.exit_code == 0
    assert r.output.strip() == "42"
    # bool
    r = runner.invoke(main, ["event", "01J9", "0", "--field", "flag"], env=env)
    assert r.exit_code == 0
    assert r.output.strip() == "True"
    # float
    r = runner.invoke(main, ["event", "01J9", "0", "--field", "rate"], env=env)
    assert r.exit_code == 0
    assert r.output.strip() == "3.14"


def test_event_second_seq(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(
        runner,
        env,
        "claude",
        "01J9G7XK4P",
        [{"t": "first", "data": "a"}, {"t": "second", "data": "b"}],
    )
    r = runner.invoke(main, ["event", "01J9", "1"], env=env)
    assert r.exit_code == 0
    obj = json.loads(r.output)
    assert obj["t"] == "second"
    assert obj["data"] == "b"


# -- search command ------------------------------------------------------------


def test_search(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "user_message", "data": "find me here"}])
    r = runner.invoke(main, ["search", "find me"], env=env)
    assert r.exit_code == 0
    obj = json.loads(r.output.strip().splitlines()[0])
    assert obj["session_id"] == "01J9G7XK4P"


def test_search_no_match(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": "hello"}])
    r = runner.invoke(main, ["search", "zzzznotfound"], env=env)
    assert r.exit_code == 0
    assert r.output.strip() == ""


def test_search_case_insensitive(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": "Hello World"}])
    r = runner.invoke(main, ["search", "hello world"], env=env)
    assert r.exit_code == 0
    assert "01J9G7XK4P" in r.output


def test_search_platform_filter(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": "secret"}])
    _seed(runner, env, "cursor", "02ABCDEF12", [{"t": "msg", "data": "secret"}])
    r = runner.invoke(main, ["search", "secret", "--platform", "cursor"], env=env)
    assert r.exit_code == 0
    assert "02ABCDEF12" in r.output
    assert "01J9G7XK4P" not in r.output


def test_search_cwd_filter(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": "target"}])
    payload = json.dumps({"t": "msg", "data": "target"}) + "\n"
    runner.invoke(
        main,
        ["ingest", "--platform", "claude", "--session-id", "02ABCDEF12", "--cwd", "/other"],
        input=payload,
        env=env,
    )
    r = runner.invoke(main, ["search", "target", "--cwd", "/proj"], env=env)
    assert "01J9G7XK4P" in r.output
    assert "02ABCDEF12" not in r.output


def test_search_across_sessions(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "msg", "data": "needle"}])
    _seed(runner, env, "cursor", "02ABCDEF12", [{"t": "msg", "data": "needle"}])
    r = runner.invoke(main, ["search", "needle"], env=env)
    assert r.exit_code == 0
    lines = r.output.strip().splitlines()
    assert len(lines) == 2


def test_search_hit_fields(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "user_message", "data": "find this"}])
    r = runner.invoke(main, ["search", "find this"], env=env)
    obj = json.loads(r.output.strip().splitlines()[0])
    assert "session_id" in obj
    assert "platform" in obj
    assert "seq" in obj
    assert "t" in obj
    assert "snippet" in obj


# -- stats command -------------------------------------------------------------


def test_stats_global(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "x", "data": 1}])
    _seed(runner, env, "cursor", "02ABCDEF12", [{"t": "y", "data": 2}, {"t": "z", "data": 3}])
    r = runner.invoke(main, ["stats"], env=env)
    obj = json.loads(r.output.strip())
    assert obj["session_count"] == 2
    assert obj["event_count"] == 3


def test_stats_session(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "x"}, {"t": "y"}, {"t": "z"}])
    r = runner.invoke(main, ["stats", "01J9"], env=env)
    assert r.exit_code == 0
    obj = json.loads(r.output.strip())
    assert obj["session_id"] == "01J9G7XK4P"
    assert obj["platform"] == "claude"
    assert obj["event_count"] == 3


def test_stats_empty_store(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    r = runner.invoke(main, ["stats"], env=env)
    assert r.exit_code == 0
    obj = json.loads(r.output.strip())
    assert obj["session_count"] == 0
    assert obj["event_count"] == 0


def test_stats_includes_bytes(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "x", "data": "content"}])
    r = runner.invoke(main, ["stats", "01J9"], env=env)
    obj = json.loads(r.output.strip())
    assert "bytes_compressed" in obj
    assert obj["bytes_compressed"] > 0


def test_stats_nonexistent_session(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    _seed(runner, env, "claude", "01J9G7XK4P", [{"t": "x"}])
    r = runner.invoke(main, ["stats", "ZZZZZZ"], env=env)
    assert r.exit_code != 0


# -- command registration checks -----------------------------------------------


def test_all_read_commands_registered(tmp_path: Path):
    runner, env = _runner_with_home(tmp_path)
    r = runner.invoke(main, ["--help"], env=env)
    assert r.exit_code == 0
    for cmd in ["list", "show", "events", "tail", "event", "search", "stats"]:
        assert cmd in r.output, f"command {cmd!r} not in --help output"
