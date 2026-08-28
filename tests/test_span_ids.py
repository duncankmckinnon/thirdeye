import hashlib
import json
import os
import subprocess
import sys

import pytest

import thirdeye.span_ids as span_ids
from thirdeye.span_ids import (
    chat_span_id,
    root_span_id_for_session,
    tool_span_id,
    trace_id_for_session,
    turn_span_id,
)

PERSON = b"thirdeye-span"


def expected_id(value: str, digest_size: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=digest_size, person=PERSON).digest()
    return int.from_bytes(digest, "big") or 1


@pytest.mark.parametrize(
    ("derive", "expected_input", "digest_size"),
    [
        (lambda: trace_id_for_session("claude", "session-123"), "claude/session-123", 16),
        (lambda: root_span_id_for_session("claude", "session-123"), "claude/session-123/root", 8),
        (lambda: turn_span_id("claude", "session-123", 42), "claude/session-123/turn/42", 8),
        (
            lambda: chat_span_id("claude", "session-123", "message-456"),
            "claude/session-123/call/message-456",
            8,
        ),
        (
            lambda: tool_span_id("claude", "session-123", "tool-789"),
            "claude/session-123/tool/tool-789",
            8,
        ),
    ],
)
def test_ids_follow_platform_scoped_derivation_contract(derive, expected_input, digest_size):
    assert derive() == expected_id(expected_input, digest_size)


def test_same_session_id_differs_across_platforms():
    claude_ids = (
        trace_id_for_session("claude", "shared-session"),
        root_span_id_for_session("claude", "shared-session"),
        turn_span_id("claude", "shared-session", 7),
        chat_span_id("claude", "shared-session", "message"),
        tool_span_id("claude", "shared-session", "tool"),
    )
    cursor_ids = (
        trace_id_for_session("cursor", "shared-session"),
        root_span_id_for_session("cursor", "shared-session"),
        turn_span_id("cursor", "shared-session", 7),
        chat_span_id("cursor", "shared-session", "message"),
        tool_span_id("cursor", "shared-session", "tool"),
    )

    assert all(
        claude_id != cursor_id for claude_id, cursor_id in zip(claude_ids, cursor_ids, strict=True)
    )


def test_non_ascii_inputs_are_encoded_as_utf8():
    assert chat_span_id("cursør-🖱", "sessiøn-👁", "méssage-雪") == expected_id(
        "cursør-🖱/sessiøn-👁/call/méssage-雪", 8
    )


def test_same_platform_inputs_are_deterministic():
    calls = [
        lambda: trace_id_for_session("claude", "session"),
        lambda: root_span_id_for_session("claude", "session"),
        lambda: turn_span_id("claude", "session", 7),
        lambda: chat_span_id("claude", "session", "message"),
        lambda: tool_span_id("claude", "session", "tool"),
    ]

    for derive in calls:
        assert derive() == derive() == derive()


def test_ids_are_stable_across_python_hash_seeds():
    code = """
import json
from thirdeye.span_ids import (
    chat_span_id,
    root_span_id_for_session,
    tool_span_id,
    trace_id_for_session,
    turn_span_id,
)

print(json.dumps([
    trace_id_for_session("claude", "session"),
    root_span_id_for_session("claude", "session"),
    turn_span_id("claude", "session", 7),
    chat_span_id("claude", "session", "message"),
    tool_span_id("claude", "session", "tool"),
]))
"""

    outputs = []
    for seed in ("1", "987654321"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        outputs.append(json.loads(result.stdout))

    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            lambda: trace_id_for_session("claude", "session-a"),
            lambda: trace_id_for_session("claude", "session-b"),
        ),
        (
            lambda: root_span_id_for_session("claude", "session-a"),
            lambda: root_span_id_for_session("claude", "session-b"),
        ),
        (
            lambda: turn_span_id("claude", "session-a", 1),
            lambda: turn_span_id("claude", "session-b", 1),
        ),
        (
            lambda: chat_span_id("claude", "session-a", "1"),
            lambda: chat_span_id("claude", "session-b", "1"),
        ),
        (
            lambda: tool_span_id("claude", "session-a", "1"),
            lambda: tool_span_id("claude", "session-b", "1"),
        ),
        (
            lambda: turn_span_id("claude", "session", 1),
            lambda: turn_span_id("claude", "session", 2),
        ),
        (
            lambda: chat_span_id("claude", "session", "message-a"),
            lambda: chat_span_id("claude", "session", "message-b"),
        ),
        (
            lambda: tool_span_id("claude", "session", "tool-a"),
            lambda: tool_span_id("claude", "session", "tool-b"),
        ),
        (
            lambda: turn_span_id("claude", "session", 1),
            lambda: chat_span_id("claude", "session", "1"),
        ),
        (
            lambda: turn_span_id("claude", "session", 1),
            lambda: tool_span_id("claude", "session", "1"),
        ),
        (
            lambda: chat_span_id("claude", "session", "1"),
            lambda: tool_span_id("claude", "session", "1"),
        ),
    ],
)
def test_ids_are_distinct_across_inputs_and_domains(first, second):
    assert first() != second()


def test_ids_are_nonzero_and_fit_otel_widths():
    trace_ids = [trace_id_for_session("claude", value) for value in ("", "session-a", "session-b")]
    span_ids = [
        root_span_id_for_session("claude", ""),
        root_span_id_for_session("claude", "session"),
        turn_span_id("claude", "session", 0),
        turn_span_id("claude", "session", 2**63),
        chat_span_id("claude", "session", ""),
        chat_span_id("claude", "session", "message"),
        tool_span_id("claude", "session", ""),
        tool_span_id("claude", "session", "tool"),
    ]

    assert all(0 < value < 2**128 for value in trace_ids)
    assert all(0 < value < 2**64 for value in span_ids)


def test_all_zero_digest_is_replaced_with_one(monkeypatch):
    class ZeroDigest:
        def __init__(self, digest_size):
            self.digest_size = digest_size

        def digest(self):
            return b"\x00" * self.digest_size

    def fake_blake2b(data, *, digest_size, person):
        assert isinstance(data, bytes)
        assert person == PERSON
        return ZeroDigest(digest_size)

    monkeypatch.setattr(span_ids.hashlib, "blake2b", fake_blake2b)

    assert trace_id_for_session("claude", "session") == 1
    assert root_span_id_for_session("claude", "session") == 1
    assert turn_span_id("claude", "session", 1) == 1
    assert chat_span_id("claude", "session", "message") == 1
    assert tool_span_id("claude", "session", "tool") == 1


@pytest.mark.parametrize(
    ("derive", "legacy_args"),
    [
        (trace_id_for_session, ("session",)),
        (root_span_id_for_session, ("session",)),
        (turn_span_id, ("session", 1)),
        (chat_span_id, ("session", "message")),
        (tool_span_id, ("session", "tool")),
    ],
)
def test_platform_argument_is_required_without_legacy_overloads(derive, legacy_args):
    with pytest.raises(TypeError):
        derive(*legacy_args)
