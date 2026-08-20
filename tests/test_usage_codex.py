from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from thirdeye.paths import (
    session_dir,
    usage_log_path,
    usage_state_path,
)
from thirdeye.platforms.codex.usage import (
    capture_usage_codex,
    parse_new_usage_rows_codex,
    persist_usage_rows_codex,
)
from thirdeye.usage.read import iter_calls
from thirdeye.usage.store import UsageStore

FIXTURES = Path(__file__).parent / "fixtures" / "usage"
FIXTURE = FIXTURES / "codex_rollout.jsonl"
FIXTURE_V0626 = FIXTURES / "codex_rollout_v0626.jsonl"

# The session id named by each fixture's session_meta frame. resolve_rollout
# verifies the id against session_meta, so these must match the real data.
SID = "019fb579-cdda-7a03-86df-65c87b6c4ae2"
SID_V0626 = "019f0542-0112-7583-bdbe-e55f44ef80b5"


@pytest.fixture
def expected() -> dict:
    return json.loads((FIXTURES / "codex_rollout.expected.json").read_text())


def _plant(root: Path, fixture: Path, sid: str) -> Path:
    """Copy `fixture` under a dated dir with a resolvable rollout filename."""
    nested = root / "2026" / "07" / "30"
    nested.mkdir(parents=True, exist_ok=True)
    dest = nested / f"rollout-2026-07-30T17-01-26-{sid}.jsonl"
    shutil.copy(fixture, dest)
    return dest


@pytest.fixture
def codex_root(tmp_path: Path) -> Path:
    root = tmp_path / "codex_sessions"
    _plant(root, FIXTURE, SID)
    return root


def _token_count_line(
    cum_total: int,
    inp: int,
    out: int,
    *,
    cached: int | None = None,
    reasoning: int | None = None,
    cache_write: int | None = None,
    ts: str = "2026-07-31T00:00:00.000Z",
) -> str:
    last: dict = {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}
    if cached is not None:
        last["cached_input_tokens"] = cached
    if reasoning is not None:
        last["reasoning_output_tokens"] = reasoning
    if cache_write is not None:
        last["cache_write_input_tokens"] = cache_write
    frame = {
        "timestamp": ts,
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": {"total_tokens": cum_total},
                "last_token_usage": last,
            },
        },
    }
    return json.dumps(frame)


def _meta_line(sid: str) -> str:
    return json.dumps({"type": "session_meta", "payload": {"id": sid, "model_provider": "openai"}})


def _turn_context_line(model: str) -> str:
    return json.dumps({"type": "turn_context", "payload": {"model": model}})


def test_yields_one_row_per_token_count_frame(
    tmp_path: Path, codex_root: Path, expected: dict
) -> None:
    rows = capture_usage_codex(
        thirdeye_home=tmp_path, session_id=SID, triggering_seq=5, sessions_root=codex_root
    )
    assert rows == expected["token_count_frames"]

    sd = session_dir(tmp_path, "codex", SID)
    # The raw sidecar mirrors one line per frame; read.py collapses repeats.
    raw = list(UsageStore(sd).iter_rows())
    assert len(raw) == expected["token_count_frames"]
    calls = list(iter_calls(sd))
    assert len({c.call_id for c in calls}) == expected["expected_calls"]
    # The fixture contains a repeat report, so a call collapses.
    assert expected["token_count_frames"] > expected["expected_calls"]


def test_reconciliation_invariant(tmp_path: Path, codex_root: Path, expected: dict) -> None:
    capture_usage_codex(
        thirdeye_home=tmp_path, session_id=SID, triggering_seq=1, sessions_root=codex_root
    )
    sd = session_dir(tmp_path, "codex", SID)

    collapsed_sum = sum(c.input_tokens + c.output_tokens for c in iter_calls(sd))
    naive_sum = sum(r.input_tokens + r.output_tokens for r in UsageStore(sd).iter_rows())

    # After dedup the per-call deltas sum to the final cumulative total exactly.
    assert collapsed_sum == expected["final_cumulative_total_tokens"]
    # The naive per-frame sum overcounts because of the repeat report.
    assert naive_sum == expected["naive_per_frame_sum"]
    # A regression to naive summing must fail loudly, not silently.
    assert naive_sum != collapsed_sum


def test_sample_call_matches(tmp_path: Path, codex_root: Path, expected: dict) -> None:
    capture_usage_codex(
        thirdeye_home=tmp_path, session_id=SID, triggering_seq=1, sessions_root=codex_root
    )
    sd = session_dir(tmp_path, "codex", SID)
    sample = expected["sample_call"]
    target = f"cum:{sample['cumulative_total_tokens']}"
    row = next(c for c in iter_calls(sd) if c.call_id == target)

    assert row.input_tokens == sample["input_tokens"]
    assert row.cache_read_input_tokens == sample["cached_input_tokens"]
    assert row.reasoning_output_tokens == sample["reasoning_output_tokens"]
    assert row.output_tokens == sample["output_tokens"]
    assert row.input_tokens + row.output_tokens == sample["total_tokens"]


def test_input_is_cache_inclusive_no_addition(
    tmp_path: Path, codex_root: Path, expected: dict
) -> None:
    capture_usage_codex(
        thirdeye_home=tmp_path, session_id=SID, triggering_seq=1, sessions_root=codex_root
    )
    sd = session_dir(tmp_path, "codex", SID)

    # Index the raw last_token_usage.input_tokens by cumulative watermark.
    raw_input: dict[str, int] = {}
    for line in FIXTURE.read_text().splitlines():
        if not line.strip():
            continue
        frame = json.loads(line)
        payload = frame.get("payload") or {}
        if frame.get("type") == "event_msg" and payload.get("type") == "token_count":
            info = payload["info"]
            raw_input[f"cum:{info['total_token_usage']['total_tokens']}"] = info[
                "last_token_usage"
            ]["input_tokens"]

    for row in iter_calls(sd):
        # Codex's inclusion invariant: input already contains cached tokens.
        assert row.input_tokens >= (row.cache_read_input_tokens or 0)
        # No cache addition happened: input equals the raw value verbatim.
        assert row.input_tokens == raw_input[row.call_id]


def test_v0626_cache_creation_is_none(tmp_path: Path) -> None:
    root = tmp_path / "codex_sessions"
    _plant(root, FIXTURE_V0626, SID_V0626)
    rows = capture_usage_codex(
        thirdeye_home=tmp_path, session_id=SID_V0626, triggering_seq=1, sessions_root=root
    )
    assert rows > 0
    sd = session_dir(tmp_path, "codex", SID_V0626)
    written = list(UsageStore(sd).iter_rows())
    assert written
    # cache_write_input_tokens is absent pre-2026-07-30 → None, never 0.
    assert all(r.cache_creation_input_tokens is None for r in written)


def test_response_model_resolves_from_turn_context(
    tmp_path: Path, codex_root: Path, expected: dict
) -> None:
    capture_usage_codex(
        thirdeye_home=tmp_path, session_id=SID, triggering_seq=1, sessions_root=codex_root
    )
    sd = session_dir(tmp_path, "codex", SID)
    models = {r.response_model for r in UsageStore(sd).iter_rows()}
    assert models == {expected["resolved_model"]}
    assert "unknown" not in models


def test_offset_advances_incrementally(tmp_path: Path) -> None:
    root = tmp_path / "root" / "2026" / "07" / "30"
    root.mkdir(parents=True)
    rp = root / f"rollout-2026-07-30T00-00-00-{SID}.jsonl"
    rp.write_text(
        _meta_line(SID)
        + "\n"
        + _turn_context_line("gpt-5.5")
        + "\n"
        + _token_count_line(100, 90, 10)
        + "\n"
    )
    sessions_root = tmp_path / "root"

    first = capture_usage_codex(
        thirdeye_home=tmp_path, session_id=SID, triggering_seq=1, sessions_root=sessions_root
    )
    assert first == 1
    sd = session_dir(tmp_path, "codex", SID)
    off1 = json.loads(usage_state_path(sd).read_text())["rollout_offset"]
    assert off1 > 0

    # Append a new frame — only the new row is captured.
    with rp.open("a") as f:
        f.write(_token_count_line(250, 140, 10) + "\n")
    second = capture_usage_codex(
        thirdeye_home=tmp_path, session_id=SID, triggering_seq=2, sessions_root=sessions_root
    )
    assert second == 1

    # Unchanged file — nothing new.
    third = capture_usage_codex(
        thirdeye_home=tmp_path, session_id=SID, triggering_seq=3, sessions_root=sessions_root
    )
    assert third == 0


def test_idempotent_from_reset_offset(tmp_path: Path, codex_root: Path, expected: dict) -> None:
    sd = session_dir(tmp_path, "codex", SID)
    capture_usage_codex(
        thirdeye_home=tmp_path, session_id=SID, triggering_seq=1, sessions_root=codex_root
    )
    # Reset the offset so the second pass re-reads the whole file.
    store = UsageStore(sd)
    store.write_state(rollout_offset=0)
    capture_usage_codex(
        thirdeye_home=tmp_path, session_id=SID, triggering_seq=2, sessions_root=codex_root
    )

    raw = list(store.iter_rows())
    assert len(raw) == 2 * expected["token_count_frames"]
    # Duplicates collapse to the same distinct call count through iter_calls.
    assert len({c.call_id for c in iter_calls(sd)}) == expected["expected_calls"]


def test_no_rollout_returns_zero_and_logs_open_source(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    rows = capture_usage_codex(
        thirdeye_home=tmp_path, session_id="nope", triggering_seq=1, sessions_root=empty
    )
    assert rows == 0
    log = usage_log_path(tmp_path)
    assert log.exists()
    entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    assert any(e["phase"] == "open_source" for e in entries)


def test_rollout_path_bypasses_resolution(tmp_path: Path) -> None:
    # A rollout placed OUTSIDE any sessions_root — resolution would never find it.
    rp = tmp_path / "loose" / "some-rollout.jsonl"
    rp.parent.mkdir(parents=True)
    rp.write_text(_turn_context_line("gpt-5.5") + "\n" + _token_count_line(100, 90, 10) + "\n")
    rows = capture_usage_codex(
        thirdeye_home=tmp_path,
        session_id="sid",
        triggering_seq=1,
        sessions_root=tmp_path / "does-not-exist",
        rollout_path=str(rp),
    )
    assert rows == 1
    sd = session_dir(tmp_path, "codex", "sid")
    assert json.loads(usage_state_path(sd).read_text())["rollout_path"] == str(rp)


def test_model_param_used_without_turn_context(tmp_path: Path) -> None:
    rp = tmp_path / "loose" / "r.jsonl"
    rp.parent.mkdir(parents=True)
    # No turn_context frame at all.
    rp.write_text(_token_count_line(100, 90, 10) + "\n")
    capture_usage_codex(
        thirdeye_home=tmp_path,
        session_id="sid",
        triggering_seq=1,
        rollout_path=str(rp),
        model="gpt-forced",
    )
    sd = session_dir(tmp_path, "codex", "sid")
    rows = list(UsageStore(sd).iter_rows())
    assert rows
    assert all(r.response_model == "gpt-forced" for r in rows)


def test_truncated_final_line_ignored_and_offset_intact(tmp_path: Path) -> None:
    rp = tmp_path / "loose" / "r.jsonl"
    rp.parent.mkdir(parents=True)
    good = _token_count_line(100, 90, 10) + "\n"
    # Trailing line has no newline — a rollout mid-write.
    truncated = _token_count_line(250, 140, 10)
    rp.write_text(good + truncated)

    first = capture_usage_codex(
        thirdeye_home=tmp_path, session_id="sid", triggering_seq=1, rollout_path=str(rp)
    )
    assert first == 1
    sd = session_dir(tmp_path, "codex", "sid")
    offset = json.loads(usage_state_path(sd).read_text())["rollout_offset"]
    assert offset == len(good.encode())

    # Complete the truncated line — it is now captured, offset was not corrupted.
    with rp.open("a") as f:
        f.write("\n")
    second = capture_usage_codex(
        thirdeye_home=tmp_path, session_id="sid", triggering_seq=2, rollout_path=str(rp)
    )
    assert second == 1


def test_safe_capture_swallows_unexpected_error(
    tmp_path: Path, codex_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import thirdeye.platforms.codex.usage as mod

    monkeypatch.setattr(
        mod,
        "_extract_usage_row",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("oops")),
    )
    result = capture_usage_codex(
        thirdeye_home=tmp_path, session_id=SID, triggering_seq=1, sessions_root=codex_root
    )
    assert result is None
    assert "RuntimeError" in usage_log_path(tmp_path).read_text()


# --- parse/persist split: lets the caller aggregate usage onto the
# triggering event's own span before it's built (see otel_export.py's module
# docstring and notify() in codex/hooks.py) -------------------------------


def test_parse_is_pure_and_does_not_persist(tmp_path: Path, codex_root: Path) -> None:
    first = parse_new_usage_rows_codex(
        thirdeye_home=tmp_path, session_id=SID, sessions_root=codex_root
    )
    second = parse_new_usage_rows_codex(
        thirdeye_home=tmp_path, session_id=SID, sessions_root=codex_root
    )
    assert first is not None and second is not None
    assert first == second
    sd = session_dir(tmp_path, "codex", SID)
    assert not usage_state_path(sd).exists()


def test_parse_no_rollout_returns_none_path_and_logs(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    parsed = parse_new_usage_rows_codex(
        thirdeye_home=tmp_path, session_id="nope", sessions_root=empty
    )
    assert parsed == ([], None, None, None)
    log = usage_log_path(tmp_path)
    entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    assert any(e["phase"] == "open_source" for e in entries)


def test_persist_stamps_seq_onto_stored_rows(tmp_path: Path, codex_root: Path) -> None:
    parsed = parse_new_usage_rows_codex(
        thirdeye_home=tmp_path, session_id=SID, sessions_root=codex_root
    )
    assert parsed is not None
    rows, resolved_path, new_offset, last_model = parsed
    assert resolved_path is not None and new_offset is not None
    assert all(row.seq == 0 for row in rows)  # placeholder, pre-persist

    n = persist_usage_rows_codex(
        thirdeye_home=tmp_path,
        session_id=SID,
        rows=rows,
        resolved_path=resolved_path,
        new_offset=new_offset,
        last_model=last_model,
        triggering_seq=42,
    )
    assert n == len(rows)
    sd = session_dir(tmp_path, "codex", SID)
    stored = list(UsageStore(sd).iter_rows())
    assert all(row.seq == 42 for row in stored)


def test_parse_then_persist_matches_capture_usage_codex(
    tmp_path: Path, codex_root: Path
) -> None:
    """The split path and the combined compat wrapper must agree exactly on
    what ends up in the sidecar, for the same fixture and triggering_seq.
    """
    combined_home = tmp_path / "combined"
    split_home = tmp_path / "split"

    capture_usage_codex(
        thirdeye_home=combined_home, session_id=SID, triggering_seq=7, sessions_root=codex_root
    )
    parsed = parse_new_usage_rows_codex(
        thirdeye_home=split_home, session_id=SID, sessions_root=codex_root
    )
    assert parsed is not None
    rows, resolved_path, new_offset, last_model = parsed
    persist_usage_rows_codex(
        thirdeye_home=split_home,
        session_id=SID,
        rows=rows,
        resolved_path=resolved_path,
        new_offset=new_offset,
        last_model=last_model,
        triggering_seq=7,
    )

    combined_rows = sorted(
        r.call_id for r in UsageStore(session_dir(combined_home, "codex", SID)).iter_rows()
    )
    split_rows = sorted(
        r.call_id for r in UsageStore(session_dir(split_home, "codex", SID)).iter_rows()
    )
    assert combined_rows == split_rows
