from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Mapping from dataclass field name to its dotted OpenTelemetry GenAI spec key.
# All gen_ai.* attributes are Development status (moved to a dedicated repo in
# semconv v1.42.0). These are the exact spec names — no abbreviations, no
# invented attributes.
ATTRIBUTE_KEYS: dict[str, str] = {
    "provider_name": "gen_ai.provider.name",
    "operation_name": "gen_ai.operation.name",
    "response_model": "gen_ai.response.model",
    "input_tokens": "gen_ai.usage.input_tokens",
    "output_tokens": "gen_ai.usage.output_tokens",
    "cache_read_input_tokens": "gen_ai.usage.cache_read.input_tokens",
    "cache_creation_input_tokens": "gen_ai.usage.cache_creation.input_tokens",
    "reasoning_output_tokens": "gen_ai.usage.reasoning.output_tokens",
}

# The thirdeye envelope fields — NOT gen_ai attributes. Serialized as bare names.
_ENVELOPE_KEYS: tuple[str, ...] = ("session_id", "seq", "call_id", "ts", "platform")

# gen_ai.conversation.id is always emitted (mirrors session_id) but is not a
# stored dataclass field, so it lives outside ATTRIBUTE_KEYS.
_CONVERSATION_ID_KEY = "gen_ai.conversation.id"

# Optional gen_ai.* fields, absent-vs-zero: an unreported attribute is an absent
# key (None), never 0.
_OPTIONAL_FIELDS: tuple[str, ...] = (
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_output_tokens",
)


@dataclass(frozen=True)
class UsageRow:
    """One model call's usage, shaped as OpenTelemetry GenAI semantic conventions.

    Serializes to dotted ``gen_ai.*`` spec keys plus a small thirdeye envelope.
    There is no ``total_tokens`` attribute in the spec — it is a derived property,
    never stored or serialized.
    """

    # --- thirdeye envelope (NOT gen_ai attributes) ---
    session_id: str
    seq: int
    call_id: str
    ts: str
    platform: str
    # --- gen_ai.* attributes ---
    provider_name: str
    response_model: str
    input_tokens: int
    output_tokens: int
    operation_name: str = "chat"
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None

    @property
    def total_tokens(self) -> int:
        """Derived, never stored or serialized."""
        return self.input_tokens + self.output_tokens

    def attributes(self) -> dict[str, Any]:
        """Only the gen_ai.* attributes, omitting any whose value is None.

        Always includes gen_ai.conversation.id (mirrors session_id). This method
        is the whole future OTLP exporter; keep it pure.
        """
        out: dict[str, Any] = {_CONVERSATION_ID_KEY: self.session_id}
        for field_name, key in ATTRIBUTE_KEYS.items():
            value = getattr(self, field_name)
            if value is None:
                continue
            out[key] = value
        return out

    def to_dict(self) -> dict[str, Any]:
        """Envelope keys (bare names) merged with attributes()."""
        out: dict[str, Any] = {name: getattr(self, name) for name in _ENVELOPE_KEYS}
        out.update(self.attributes())
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UsageRow:
        """Inverse of to_dict. Missing optional attribute keys become None.

        Raises KeyError / ValueError on a malformed row.
        """
        optional: dict[str, int | None] = {}
        for field_name in _OPTIONAL_FIELDS:
            key = ATTRIBUTE_KEYS[field_name]
            optional[field_name] = int(d[key]) if key in d else None
        return cls(
            session_id=str(d["session_id"]),
            seq=int(d["seq"]),
            call_id=str(d["call_id"]),
            ts=str(d["ts"]),
            platform=str(d["platform"]),
            provider_name=str(d[ATTRIBUTE_KEYS["provider_name"]]),
            response_model=str(d[ATTRIBUTE_KEYS["response_model"]]),
            input_tokens=int(d[ATTRIBUTE_KEYS["input_tokens"]]),
            output_tokens=int(d[ATTRIBUTE_KEYS["output_tokens"]]),
            operation_name=str(d[ATTRIBUTE_KEYS["operation_name"]]),
            **optional,
        )
