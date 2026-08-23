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
        (lambda: trace_id_for_session("session-123"), "session-123", 16),
        (lambda: root_span_id_for_session("session-123"), "session-123/root", 8),
        (lambda: turn_span_id("session-123", 42), "session-123/turn/42", 8),
        (lambda: chat_span_id("session-123", "message-456"), "session-123/call/message-456", 8),
        (lambda: tool_span_id("session-123", "tool-789"), "session-123/tool/tool-789", 8),
    ],
)
def test_ids_follow_the_blake2b_derivation_contract(derive, expected_input, digest_size):
    assert derive() == expected_id(expected_input, digest_size)


def test_non_ascii_inputs_are_encoded_as_utf8():
    assert chat_span_id("sessiøn-👁", "méssage-雪") == expected_id("sessiøn-👁/call/méssage-雪", 8)


def test_same_inputs_are_deterministic():
    calls = [
        lambda: trace_id_for_session("session"),
        lambda: root_span_id_for_session("session"),
        lambda: turn_span_id("session", 7),
        lambda: chat_span_id("session", "message"),
        lambda: tool_span_id("session", "tool"),
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
    trace_id_for_session("session"),
    root_span_id_for_session("session"),
    turn_span_id("session", 7),
    chat_span_id("session", "message"),
    tool_span_id("session", "tool"),
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
    "first, second",
    [
        (trace_id_for_session("session-a"), trace_id_for_session("session-b")),
        (root_span_id_for_session("session-a"), root_span_id_for_session("session-b")),
        (turn_span_id("session-a", 1), turn_span_id("session-b", 1)),
        (chat_span_id("session-a", "1"), chat_span_id("session-b", "1")),
        (tool_span_id("session-a", "1"), tool_span_id("session-b", "1")),
        (turn_span_id("session", 1), turn_span_id("session", 2)),
        (chat_span_id("session", "message-a"), chat_span_id("session", "message-b")),
        (tool_span_id("session", "tool-a"), tool_span_id("session", "tool-b")),
        (turn_span_id("session", 1), chat_span_id("session", "1")),
        (turn_span_id("session", 1), tool_span_id("session", "1")),
        (chat_span_id("session", "1"), tool_span_id("session", "1")),
    ],
)
def test_ids_are_distinct_across_inputs_and_domains(first, second):
    assert first != second


def test_ids_are_nonzero_and_fit_otel_widths():
    trace_ids = [trace_id_for_session(value) for value in ("", "session-a", "session-b")]
    span_ids = [
        root_span_id_for_session(""),
        root_span_id_for_session("session"),
        turn_span_id("session", 0),
        turn_span_id("session", 2**63),
        chat_span_id("session", ""),
        chat_span_id("session", "message"),
        tool_span_id("session", ""),
        tool_span_id("session", "tool"),
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

    assert trace_id_for_session("session") == 1
    assert root_span_id_for_session("session") == 1
    assert turn_span_id("session", 1) == 1
    assert chat_span_id("session", "message") == 1
    assert tool_span_id("session", "tool") == 1
