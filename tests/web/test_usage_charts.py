from __future__ import annotations

import json
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


def _row(*, sid: str, platform: str, ts: str, input_tokens: int = 100,
         output_tokens: int = 10) -> UsageRow:
    return UsageRow(
        session_id=sid,
        seq=0,
        ts=ts,
        platform=platform,
        model="claude-opus-4-7",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _extract_chart_json(body: bytes) -> dict:
    text = body.decode("utf-8")
    marker = '<script type="application/json" id="chart-data">'
    start = text.index(marker) + len(marker)
    end = text.index("</script>", start)
    return json.loads(text[start:end])


def test_global_usage_empty_store(client) -> None:
    r = client.get("/usage")
    assert r.status_code == 200
    body = r.content
    assert b'class="aggregate-value">0</div>' in body
    data = _extract_chart_json(body)
    assert data == {"days": [], "input": [], "output": [], "sessions": []}


def test_global_usage_platform_and_date_filter(client, web_config) -> None:
    sd_claude = _make_session(
        web_config.root, sid="claude1", platform="claude",
        started_at="2026-05-05T10:00:00.000Z",
    )
    UsageStore(sd_claude).append(
        [_row(sid="claude1", platform="claude", ts="2026-05-05T10:00:00.000Z",
              input_tokens=100, output_tokens=10)]
    )
    sd_codex = _make_session(
        web_config.root, sid="codex1", platform="codex",
        started_at="2026-05-06T10:00:00.000Z",
    )
    UsageStore(sd_codex).append(
        [_row(sid="codex1", platform="codex", ts="2026-05-06T10:00:00.000Z",
              input_tokens=999, output_tokens=99)]
    )

    r = client.get("/usage?platform=claude&since=2026-05-01&until=2026-05-10")
    assert r.status_code == 200
    data = _extract_chart_json(r.content)

    assert "2026-05-05" in data["days"]
    idx = data["days"].index("2026-05-05")
    assert data["input"][idx] == 100
    assert data["output"][idx] == 10
    assert data["sessions"][idx] == 1

    assert sum(data["input"]) == 100
    assert sum(data["output"]) == 10
    assert sum(data["sessions"]) == 1


def test_global_usage_invalid_since_treated_as_default(client) -> None:
    r = client.get("/usage?since=nonsense")
    assert r.status_code == 200
