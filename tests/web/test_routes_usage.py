from __future__ import annotations

from pathlib import Path

import pytest

from thirdeye.meta import SessionMeta, write_meta
from thirdeye.paths import meta_path, session_dir
from thirdeye.usage.store import UsageStore
from thirdeye.usage.types import UsageRow

pytest.importorskip("starlette")


def _make_session(root: Path, *, sid: str, platform: str, started_at: str) -> Path:
    sd = session_dir(root, platform, sid)
    sd.mkdir(parents=True, exist_ok=True)
    meta = SessionMeta(
        session_id=sid,
        platform=platform,
        cwd="/tmp/proj",
        started_at=started_at,
        ended_at=None,
        status="open",
        event_count=0,
        last_seq=-1,
        last_ts=started_at,
    )
    write_meta(meta_path(sd), meta)
    return sd


def _row(
    *,
    sid: str,
    platform: str = "claude",
    ts: str = "2026-05-05T10:00:00.000Z",
    call_id: str,
    input_tokens: int = 100,
    output_tokens: int = 10,
    response_model: str = "claude-opus-4-7",
    cache_read_input_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
    reasoning_output_tokens: int | None = None,
) -> UsageRow:
    return UsageRow(
        session_id=sid,
        seq=0,
        call_id=call_id,
        ts=ts,
        platform=platform,
        provider_name="anthropic",
        response_model=response_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
    )


def test_global_usage_200(client):
    r = client.get("/usage")
    assert r.status_code == 200


def test_session_usage_200(client, web_store):
    sid = "01J9USG01"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "x"})
    web_store.close_session(sid, platform="claude")

    r = client.get(f"/sessions/{sid}/usage")
    assert r.status_code == 200


def test_session_usage_404(client):
    r = client.get("/sessions/bogus/usage")
    assert r.status_code == 404


def test_global_platform_filter_only_claude_and_codex(client):
    """The platform filter offers only claude and codex — gemini is gone."""
    r = client.get("/usage")
    assert r.status_code == 200
    body = r.text
    assert 'value="claude"' in body
    assert 'value="codex"' in body
    assert 'value="gemini"' not in body
    assert ">gemini<" not in body


def test_session_usage_renders_per_call_rows_with_model(client, web_config):
    """Per-call rows show the response_model in a model column."""
    sd = _make_session(
        web_config.root, sid="modelsess1", platform="claude", started_at="2026-05-05T10:00:00.000Z"
    )
    UsageStore(sd).append(
        [_row(sid="modelsess1", call_id="c1", response_model="claude-sonnet-4-6")]
    )

    r = client.get("/sessions/modelsess1/usage")
    assert r.status_code == 200
    body = r.text
    assert ">model<" in body
    assert "claude-sonnet-4-6" in body


def test_session_usage_absent_cache_renders_dash_zero_renders_zero(client, web_config):
    """An unreported cache value renders '-'; a reported zero renders '0'."""
    sd = _make_session(
        web_config.root, sid="cachesess1", platform="claude", started_at="2026-05-05T10:00:00.000Z"
    )
    UsageStore(sd).append(
        [
            _row(
                sid="cachesess1",
                call_id="c1",
                cache_read_input_tokens=None,
                cache_creation_input_tokens=0,
            )
        ]
    )

    r = client.get("/sessions/cachesess1/usage")
    assert r.status_code == 200
    body = r.text
    # Absent cache_read renders as a dash cell; zero cache_creation renders as 0.
    assert '<td class="num">-</td>' in body
    assert '<td class="num">0</td>' in body


def test_session_usage_deduplicates_repeated_call_id(client, web_config):
    """Rows sharing a call_id collapse to one via iter_calls."""
    sd = _make_session(
        web_config.root, sid="dupsess1", platform="claude", started_at="2026-05-05T10:00:00.000Z"
    )
    UsageStore(sd).append(
        [
            _row(sid="dupsess1", call_id="same", input_tokens=100, output_tokens=10),
            _row(sid="dupsess1", call_id="same", input_tokens=100, output_tokens=10),
            _row(sid="dupsess1", call_id="same", input_tokens=100, output_tokens=10),
        ]
    )

    r = client.get("/sessions/dupsess1/usage")
    assert r.status_code == 200
    # Three identical frames, one logical call -> one <tr> in the tbody.
    body = r.text
    tbody = body.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    assert tbody.count("<tr>") == 1
