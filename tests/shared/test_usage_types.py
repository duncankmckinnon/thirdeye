from __future__ import annotations

import json

import pytest

from thirdeye.usage.types import ATTRIBUTE_KEYS, UsageRow

# The nine gen_ai.* spec names, written as literals. Do NOT derive this from
# ATTRIBUTE_KEYS — a hardcoded set is what makes the conformance test meaningful.
SPEC_GEN_AI_KEYS = {
    "gen_ai.provider.name",
    "gen_ai.operation.name",
    "gen_ai.conversation.id",
    "gen_ai.response.model",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.cache_read.input_tokens",
    "gen_ai.usage.cache_creation.input_tokens",
    "gen_ai.usage.reasoning.output_tokens",
}

ENVELOPE_KEYS = {"session_id", "seq", "call_id", "ts", "platform"}


def make_row(**overrides) -> UsageRow:
    defaults = dict(
        session_id="abc123",
        seq=0,
        call_id="msg_001",
        ts="2026-05-15T00:00:00.000Z",
        platform="claude",
        provider_name="anthropic",
        response_model="claude-opus-4-8",
        input_tokens=100,
        output_tokens=10,
    )
    defaults.update(overrides)
    return UsageRow(**defaults)


def full_row() -> UsageRow:
    return make_row(
        cache_read_input_tokens=50,
        cache_creation_input_tokens=25,
        reasoning_output_tokens=5,
    )


def test_round_trip_all_optionals_set() -> None:
    row = full_row()
    assert UsageRow.from_dict(row.to_dict()) == row


def test_round_trip_all_optionals_none() -> None:
    row = make_row()
    assert row.cache_read_input_tokens is None
    assert UsageRow.from_dict(row.to_dict()) == row


def test_round_trip_via_json() -> None:
    row = full_row()
    decoded = UsageRow.from_dict(json.loads(json.dumps(row.to_dict())))
    assert decoded == row


def test_to_dict_omits_none_optionals() -> None:
    d = make_row().to_dict()
    assert "gen_ai.usage.cache_read.input_tokens" not in d
    assert "gen_ai.usage.cache_creation.input_tokens" not in d
    assert "gen_ai.usage.reasoning.output_tokens" not in d


def test_zero_optional_serializes_as_zero() -> None:
    """absent-vs-zero: 0 means 'reported as none', which must be kept."""
    d = make_row(cache_read_input_tokens=0).to_dict()
    assert d["gen_ai.usage.cache_read.input_tokens"] == 0
    assert UsageRow.from_dict(d).cache_read_input_tokens == 0


def test_attributes_only_gen_ai_keys() -> None:
    attrs = full_row().attributes()
    assert all(k.startswith("gen_ai.") for k in attrs)


def test_attributes_includes_conversation_id() -> None:
    row = make_row(session_id="sess-xyz")
    assert row.attributes()["gen_ai.conversation.id"] == "sess-xyz"


def test_attributes_has_no_envelope_keys() -> None:
    attrs = full_row().attributes()
    assert not (ENVELOPE_KEYS & set(attrs))


def test_total_tokens_is_derived() -> None:
    row = make_row(input_tokens=100, output_tokens=10)
    assert row.total_tokens == 110
    d = row.to_dict()
    assert "total_tokens" not in d
    assert "gen_ai.usage.total_tokens" not in d


def test_expected_claude_serialization() -> None:
    row = UsageRow(
        session_id="acb30f50",
        seq=36,
        call_id="msg_011Cd",
        ts="2026-08-05T16:53:52.790Z",
        platform="claude",
        provider_name="anthropic",
        response_model="claude-opus-4-8",
        input_tokens=324429,
        output_tokens=1230,
        cache_read_input_tokens=304110,
        cache_creation_input_tokens=20317,
    )
    assert row.to_dict() == {
        "session_id": "acb30f50",
        "seq": 36,
        "call_id": "msg_011Cd",
        "ts": "2026-08-05T16:53:52.790Z",
        "platform": "claude",
        "gen_ai.provider.name": "anthropic",
        "gen_ai.operation.name": "chat",
        "gen_ai.conversation.id": "acb30f50",
        "gen_ai.response.model": "claude-opus-4-8",
        "gen_ai.usage.input_tokens": 324429,
        "gen_ai.usage.output_tokens": 1230,
        "gen_ai.usage.cache_read.input_tokens": 304110,
        "gen_ai.usage.cache_creation.input_tokens": 20317,
    }


def test_conformance_only_spec_gen_ai_keys() -> None:
    """Every gen_ai.* key emitted must be one of the nine literal spec names.

    A typo or invented attribute must fail here.
    """
    d = full_row().to_dict()
    gen_ai_keys = {k for k in d if k.startswith("gen_ai.")}
    assert gen_ai_keys <= SPEC_GEN_AI_KEYS
    # The full row exercises all nine.
    assert gen_ai_keys == SPEC_GEN_AI_KEYS


def test_attribute_keys_map_matches_spec() -> None:
    """ATTRIBUTE_KEYS values are a subset of the spec names (all but conversation.id)."""
    assert set(ATTRIBUTE_KEYS.values()) == SPEC_GEN_AI_KEYS - {"gen_ai.conversation.id"}


def test_from_dict_missing_session_id_raises() -> None:
    d = full_row().to_dict()
    del d["session_id"]
    with pytest.raises(KeyError):
        UsageRow.from_dict(d)


def test_is_frozen() -> None:
    row = make_row()
    with pytest.raises(Exception):  # FrozenInstanceError, version-dependent
        row.seq = 99  # type: ignore[misc]
