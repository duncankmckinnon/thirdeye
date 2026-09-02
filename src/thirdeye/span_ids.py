"""Deterministic OTel trace/span ids derived from stable session data.

Span ids used to be minted by the OTel SDK and read back after the fact, which
forced every span to exist before anything could point at it. That works for a
whole-turn export built in one process, but not for live emission: a tool span
emitted mid-turn has to name the ``chat`` span that requested it as its parent,
and that chat span is not exported until the turn ends.

So ids are derived instead of minted. Every id here is a pure function of data
any process already holds — the platform, session id, turn sequence number,
transcript ``message.id``, and ``tool_use_id`` — so a live tool span can name a
parent that does not exist yet, and a lost or corrupt open-turn marker is
non-fatal: the turn span id is recomputable from the platform, session id, and
turn seq alone.

Derivation is ``blake2b`` over a domain-separated string, personalised with a
fixed constant so these digests can never collide with any other hash use in
the codebase. **Never use Python's built-in ``hash()`` here** — it is
randomised per process under ``PYTHONHASHSEED``, so two processes would derive
different ids for the same input and the whole scheme would silently fall
apart.
"""

from __future__ import annotations

import hashlib

# blake2b personalisation (max 16 bytes). Domain-separates every digest in this
# module from any other blake2b use, in this codebase or elsewhere.
_PERSON = b"thirdeye-span"

_TRACE_ID_BYTES = 16  # OTel trace ids are 128-bit
_SPAN_ID_BYTES = 8  # OTel span ids are 64-bit


def _derive(key: str, digest_size: int) -> int:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=digest_size, person=_PERSON).digest()
    value = int.from_bytes(digest, "big")
    # OTel treats an all-zero trace/span id as "invalid" and drops the span, so
    # substitute 1 in the (astronomically unlikely) case a digest lands on zero.
    return value or 1


def trace_id_for_session(platform: str, session_id: str) -> int:
    """128-bit trace id anchoring every span in one thirdeye session."""
    return _derive(f"{platform}/{session_id}", _TRACE_ID_BYTES)


def root_span_id_for_session(platform: str, session_id: str) -> int:
    """64-bit span id of the session's root ``session`` span."""
    return _derive(f"{platform}/{session_id}/root", _SPAN_ID_BYTES)


def turn_span_id(platform: str, session_id: str, turn_seq: int) -> int:
    """64-bit span id of the ``invoke_agent`` span for turn ``turn_seq``."""
    return _derive(f"{platform}/{session_id}/turn/{turn_seq}", _SPAN_ID_BYTES)


def chat_span_id(platform: str, session_id: str, message_id: str) -> int:
    """64-bit span id of the ``chat`` span for LLM message ``message_id``."""
    return _derive(f"{platform}/{session_id}/call/{message_id}", _SPAN_ID_BYTES)


def tool_span_id(platform: str, session_id: str, tool_use_id: str) -> int:
    """64-bit span id of the ``tool`` span for tool call ``tool_use_id``."""
    return _derive(f"{platform}/{session_id}/tool/{tool_use_id}", _SPAN_ID_BYTES)


def interaction_span_id(platform: str, session_id: str, interaction_id: str) -> int:
    """64-bit span id of the ``interaction`` span for interaction ``interaction_id``."""
    return _derive(f"{platform}/{session_id}/interaction/{interaction_id}", _SPAN_ID_BYTES)
