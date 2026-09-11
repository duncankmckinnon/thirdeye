from __future__ import annotations

import json
from pathlib import Path

from thirdeye.eval.definition import EvalDefinition
from thirdeye.eval.prompt import build_prompt
from thirdeye.paths import session_dir, usage_jsonl_path


def _defn(directive: str = "evaluate this") -> EvalDefinition:
    return EvalDefinition(name="t", description="", directive=directive)


def _seed_usage(sd: Path, rows: list[dict]) -> None:
    sd.mkdir(parents=True, exist_ok=True)
    with usage_jsonl_path(sd).open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_prompt_includes_directive(tmp_path: Path):
    p = build_prompt(
        thirdeye_home=tmp_path,
        platform="claude",
        session_id="abc",
        definition=_defn("DIR_FIRST_BLOCK"),
    )
    assert "DIR_FIRST_BLOCK" in p
    assert p.index("DIR_FIRST_BLOCK") < p.index("=== Session being evaluated ===")


def test_prompt_includes_output_contract(tmp_path: Path):
    p = build_prompt(
        thirdeye_home=tmp_path,
        platform="claude",
        session_id="abc",
        definition=_defn(),
    )
    assert "=== Required output ===" in p
    assert '"verdict": "pass" | "warn" | "fail"' in p


def test_prompt_includes_session_id_and_platform(tmp_path: Path):
    p = build_prompt(
        thirdeye_home=tmp_path,
        platform="claude",
        session_id="abc123",
        definition=_defn(),
    )
    assert "session_id: abc123" in p
    assert "platform:   claude" in p


def test_prompt_usage_summary_when_sidecar_exists(tmp_path: Path):
    sd = session_dir(tmp_path, "claude", "abc")
    _seed_usage(
        sd,
        [
            {
                "session_id": "abc",
                "seq": 0,
                "ts": "t",
                "platform": "claude",
                "model": "claude-opus-4-7",
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
            },
            {
                "session_id": "abc",
                "seq": 1,
                "ts": "t",
                "platform": "claude",
                "model": "claude-opus-4-7",
                "input_tokens": 200,
                "output_tokens": 20,
                "total_tokens": 220,
            },
        ],
    )
    p = build_prompt(
        thirdeye_home=tmp_path,
        platform="claude",
        session_id="abc",
        definition=_defn(),
    )
    assert "=== Usage summary ===" in p
    assert "turns: 2" in p
    assert "claude-opus-4-7 (2 turns)" in p
    assert "300" in p
    assert "330" in p


def test_prompt_omits_usage_summary_when_no_sidecar(tmp_path: Path):
    p = build_prompt(
        thirdeye_home=tmp_path,
        platform="claude",
        session_id="abc",
        definition=_defn(),
    )
    assert "=== Usage summary ===" not in p


def test_prompt_event_timeline_when_lines_passed(tmp_path: Path):
    lines = ["0 session_start", "1 user_message hello", "2 stop"]
    p = build_prompt(
        thirdeye_home=tmp_path,
        platform="claude",
        session_id="abc",
        definition=_defn(),
        event_lines=lines,
    )
    assert "=== Event timeline (condensed) ===" in p
    for line in lines:
        assert line in p


def test_prompt_event_timeline_omitted_when_no_lines(tmp_path: Path):
    p = build_prompt(
        thirdeye_home=tmp_path,
        platform="claude",
        session_id="abc",
        definition=_defn(),
    )
    assert "=== Event timeline ===" in p
    assert "use `thirdeye events abc`" in p


def test_prompt_truncates_long_timeline(tmp_path: Path):
    lines = [f"seq={i}" for i in range(300)]
    p = build_prompt(
        thirdeye_home=tmp_path,
        platform="claude",
        session_id="abc",
        definition=_defn(),
        event_lines=lines,
        max_timeline_lines=20,
    )
    assert "lines elided" in p
    assert p.count("seq=") < 300


def test_prompt_tool_inventory_present(tmp_path: Path):
    p = build_prompt(
        thirdeye_home=tmp_path,
        platform="claude",
        session_id="abc",
        definition=_defn(),
    )
    assert "=== Tool inventory ===" in p
    assert "thirdeye" in p
    assert "sqlite3" in p
    assert str(tmp_path / "usage.db") in p


def test_prompt_block_order(tmp_path: Path):
    p = build_prompt(
        thirdeye_home=tmp_path,
        platform="claude",
        session_id="abc",
        definition=_defn("DIRECTIVE_TEXT"),
    )
    order = [
        "DIRECTIVE_TEXT",
        "=== Session being evaluated ===",
        "=== Tool inventory ===",
        "=== Required output ===",
    ]
    positions = [p.index(s) for s in order]
    assert positions == sorted(positions)


def test_prompt_handles_malformed_usage_lines(tmp_path: Path):
    sd = session_dir(tmp_path, "claude", "abc")
    sd.mkdir(parents=True)
    usage_jsonl_path(sd).write_text(
        json.dumps(
            {
                "session_id": "abc",
                "seq": 0,
                "ts": "t",
                "platform": "claude",
                "model": "m",
                "input_tokens": 5,
                "output_tokens": 1,
                "total_tokens": 6,
            }
        )
        + "\n"
        "this is not json\n"
    )
    p = build_prompt(
        thirdeye_home=tmp_path,
        platform="claude",
        session_id="abc",
        definition=_defn(),
    )
    assert "turns: 1" in p
