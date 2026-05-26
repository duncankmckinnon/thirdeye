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


def _row(
    *, sid: str, platform: str, ts: str, input_tokens: int = 100, output_tokens: int = 10
) -> UsageRow:
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
    assert set(data.keys()) == {"days", "input", "output", "sessions"}
    n = len(data["days"])
    assert n >= 1
    assert len(data["input"]) == n
    assert len(data["output"]) == n
    assert len(data["sessions"]) == n
    assert all(v == 0 for v in data["input"])
    assert all(v == 0 for v in data["output"])
    assert all(v == 0 for v in data["sessions"])
    assert all(isinstance(d, str) and len(d) == 10 for d in data["days"])


def test_global_usage_platform_and_date_filter(client, web_config) -> None:
    sd_claude = _make_session(
        web_config.root,
        sid="claude1",
        platform="claude",
        started_at="2026-05-05T10:00:00.000Z",
    )
    UsageStore(sd_claude).append(
        [
            _row(
                sid="claude1",
                platform="claude",
                ts="2026-05-05T10:00:00.000Z",
                input_tokens=100,
                output_tokens=10,
            )
        ]
    )
    sd_codex = _make_session(
        web_config.root,
        sid="codex1",
        platform="codex",
        started_at="2026-05-06T10:00:00.000Z",
    )
    UsageStore(sd_codex).append(
        [
            _row(
                sid="codex1",
                platform="codex",
                ts="2026-05-06T10:00:00.000Z",
                input_tokens=999,
                output_tokens=99,
            )
        ]
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


def test_global_usage_invalid_until_treated_as_default(client) -> None:
    r = client.get("/usage?until=garbage")
    assert r.status_code == 200
    data = _extract_chart_json(r.content)
    assert len(data["days"]) >= 1


def test_global_usage_totals_include_all_platforms(client, web_config) -> None:
    for sid, platform, ts, ip, op in [
        ("c1", "claude", "2026-05-05T10:00:00.000Z", 100, 10),
        ("c2", "claude", "2026-05-05T11:00:00.000Z", 50, 5),
        ("x1", "codex", "2026-05-06T10:00:00.000Z", 30, 3),
    ]:
        sd = _make_session(web_config.root, sid=sid, platform=platform, started_at=ts)
        UsageStore(sd).append(
            [_row(sid=sid, platform=platform, ts=ts, input_tokens=ip, output_tokens=op)]
        )

    r = client.get("/usage?since=2026-05-01&until=2026-05-10")
    assert r.status_code == 200
    body = r.content.decode("utf-8")
    assert "180" in body
    assert "18" in body
    assert "3" in body

    data = _extract_chart_json(r.content)
    assert sum(data["input"]) == 180
    assert sum(data["output"]) == 18
    assert sum(data["sessions"]) == 3


def test_global_usage_platform_codex_filter(client, web_config) -> None:
    sd_claude = _make_session(
        web_config.root,
        sid="cl1",
        platform="claude",
        started_at="2026-05-05T10:00:00.000Z",
    )
    UsageStore(sd_claude).append(
        [
            _row(
                sid="cl1",
                platform="claude",
                ts="2026-05-05T10:00:00.000Z",
                input_tokens=100,
                output_tokens=10,
            )
        ]
    )
    sd_codex = _make_session(
        web_config.root,
        sid="cx1",
        platform="codex",
        started_at="2026-05-05T10:00:00.000Z",
    )
    UsageStore(sd_codex).append(
        [
            _row(
                sid="cx1",
                platform="codex",
                ts="2026-05-05T10:00:00.000Z",
                input_tokens=42,
                output_tokens=4,
            )
        ]
    )

    r = client.get("/usage?platform=codex&since=2026-05-01&until=2026-05-10")
    assert r.status_code == 200
    data = _extract_chart_json(r.content)
    assert sum(data["input"]) == 42
    assert sum(data["output"]) == 4
    assert sum(data["sessions"]) == 1


def test_global_usage_filters_echoed_in_form(client) -> None:
    r = client.get("/usage?platform=gemini&since=7d&until=today")
    assert r.status_code == 200
    body = r.content.decode("utf-8")
    assert 'name="since"' in body
    assert 'value="7d"' in body
    assert 'name="until"' in body
    assert 'value="today"' in body
    assert (
        '<option value="gemini"\n            selected>gemini</option>' in body
        or 'value="gemini"' in body
        and "selected" in body
    )


def test_global_usage_includes_chart_script_and_canvases(client) -> None:
    r = client.get("/usage")
    assert r.status_code == 200
    body = r.content.decode("utf-8")
    assert "/static/chart.min.js" in body
    assert 'id="tokens-chart"' in body
    assert 'id="sessions-chart"' in body
    assert 'id="chart-data"' in body


def test_global_usage_rolling_since(client, web_config) -> None:
    from datetime import UTC, datetime, timedelta

    one_day = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    three_day = (datetime.now(UTC) - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    sd = _make_session(
        web_config.root,
        sid="r1",
        platform="claude",
        started_at=one_day,
    )
    UsageStore(sd).append(
        [
            _row(sid="r1", platform="claude", ts=one_day, input_tokens=7, output_tokens=1),
        ]
    )
    sd2 = _make_session(
        web_config.root,
        sid="r3",
        platform="claude",
        started_at=three_day,
    )
    UsageStore(sd2).append(
        [
            _row(sid="r3", platform="claude", ts=three_day, input_tokens=3, output_tokens=2),
        ]
    )
    r = client.get("/usage?since=7d&until=1h")
    assert r.status_code == 200
    data = _extract_chart_json(r.content)
    assert len(data["days"]) >= 1
    assert sum(data["input"]) == 10
    assert sum(data["output"]) == 3
    assert sum(data["sessions"]) == 2


def test_global_usage_chart_data_arrays_aligned(client, web_config) -> None:
    sd = _make_session(
        web_config.root,
        sid="s1",
        platform="claude",
        started_at="2026-05-05T10:00:00.000Z",
    )
    UsageStore(sd).append(
        [
            _row(
                sid="s1",
                platform="claude",
                ts="2026-05-05T10:00:00.000Z",
                input_tokens=10,
                output_tokens=1,
            ),
            _row(
                sid="s1",
                platform="claude",
                ts="2026-05-07T10:00:00.000Z",
                input_tokens=20,
                output_tokens=2,
            ),
        ]
    )

    r = client.get("/usage?since=2026-05-01&until=2026-05-10")
    assert r.status_code == 200
    data = _extract_chart_json(r.content)
    n = len(data["days"])
    assert n == len(data["input"]) == len(data["output"]) == len(data["sessions"])
    assert "2026-05-05" in data["days"]
    assert "2026-05-07" in data["days"]
    idx5 = data["days"].index("2026-05-05")
    idx7 = data["days"].index("2026-05-07")
    assert data["input"][idx5] == 10
    assert data["input"][idx7] == 20
    assert data["output"][idx5] == 1
    assert data["output"][idx7] == 2


def test_global_usage_session_view_unchanged(client, web_store) -> None:
    sid = "01J9USG01"
    with web_store.open_session(sid, platform="claude", cwd="/p") as w:
        w.append("user_message", {"prompt": "x"})
    web_store.close_session(sid, platform="claude")
    r = client.get(f"/sessions/{sid}/usage")
    assert r.status_code == 200
