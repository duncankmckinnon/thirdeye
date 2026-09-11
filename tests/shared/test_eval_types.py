from __future__ import annotations

import re

from thirdeye.eval._ulid import ulid_now
from thirdeye.eval.result import EvalResult, Finding, parse_envelope

# --- Finding ---


def test_finding_round_trip():
    f = Finding(seq=42, severity="warn", note="x", category="tokens")
    assert Finding.from_dict(f.to_dict()) == f


def test_finding_seq_none_round_trip():
    f = Finding(seq=None, severity="info", note="session-level")
    assert Finding.from_dict(f.to_dict()) == f


def test_finding_coerces_unknown_severity_to_info():
    f = Finding.from_dict({"seq": 1, "severity": "critical", "note": "n"})
    assert f.severity == "info"


# --- EvalResult ---


def _result(**overrides) -> EvalResult:
    base = dict(
        id="01J7XYZ",
        session_id="abc",
        definition="default",
        agent="claude",
        agent_model="claude-sonnet-4-6",
        agent_session_id="eval-sid",
        started_at="2026-05-16T01:42:00Z",
        ended_at="2026-05-16T01:42:18Z",
        duration_ms=18432,
        verdict="warn",
        summary="ok",
    )
    base.update(overrides)
    return EvalResult(**base)


def test_result_round_trip():
    r = _result(scores={"overall": 7.0}, findings=[Finding(seq=1, severity="warn", note="x")])
    decoded = EvalResult.from_dict(r.to_dict())
    assert decoded == r


def test_result_coerces_unknown_verdict_to_unknown():
    r = EvalResult.from_dict(
        {
            "id": "x",
            "session_id": "s",
            "definition": "d",
            "agent": "a",
            "started_at": "t",
            "ended_at": "t",
            "verdict": "weird",
        }
    )
    assert r.verdict == "unknown"


def test_result_empty_collections_default():
    r = _result()
    assert r.scores == {} and r.findings == [] and r.cost == {}


# --- parse_envelope ---


def test_parse_envelope_extracts_json_and_narrative():
    text = '```json\n{"verdict": "pass", "summary": "ok"}\n```\n\nFollowed by narrative.'
    env, narrative = parse_envelope(text)
    assert env == {"verdict": "pass", "summary": "ok"}
    assert "Followed by narrative" in narrative


def test_parse_envelope_returns_none_when_no_fence():
    env, narrative = parse_envelope("just markdown, no json")
    assert env is None
    assert narrative == "just markdown, no json"


def test_parse_envelope_returns_none_on_malformed_json():
    env, narrative = parse_envelope("```json\n{not valid}\n```")
    assert env is None
    assert "not valid" in narrative


def test_parse_envelope_keeps_narrative_before_fence():
    text = 'Preamble.\n\n```json\n{"verdict": "pass"}\n```\n'
    env, narrative = parse_envelope(text)
    assert env == {"verdict": "pass"}
    assert "Preamble" in narrative


def test_parse_envelope_empty_input():
    env, narrative = parse_envelope("")
    assert env is None
    assert narrative == ""


# --- ulid_now ---


def test_ulid_format():
    u = ulid_now()
    assert len(u) == 26
    assert re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", u)


def test_ulid_monotonic_within_a_loop():
    ids = [ulid_now() for _ in range(20)]
    # Time-prefixed ULIDs from the same process should be lexicographically
    # non-decreasing (they may tie at the millisecond level).
    assert ids == sorted(ids) or all(ids[i][:10] <= ids[i + 1][:10] for i in range(19))


def test_ulid_uniqueness_under_load():
    ids = {ulid_now() for _ in range(1000)}
    assert len(ids) == 1000
