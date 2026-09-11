"""Versioned, lossless source contracts shared by Copilot V1 and V2.

These TypedDicts deliberately describe raw source evidence.  They do not imply
turn reconstruction, usage accounting, or any semantic correlation.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from thirdeye.tracing.model import TurnSpanDict
from thirdeye.usage.types import UsageRow

from .constants import SOURCE_SCHEMA_VERSION

# The value placed beside every archived SourceRecord in a thirdeye event data
# envelope.  It is deliberately separate from a Copilot CLI version.
SCHEMA_VERSION = SOURCE_SCHEMA_VERSION


class SourcePaths(TypedDict):
    """Canonical source-home identity and its Copilot recording locations."""

    home: str
    source_key: str
    session_root: str
    database: str


class SourceRecord(TypedDict):
    """One immutable observation from a Copilot source domain."""

    source_id: str
    source_kind: str  # transcript | database | hook | metadata
    native_session_id: str
    ts: str | None  # Source time when valid; never invented.
    observed_at: str
    payload: dict[str, Any]
    locator: dict[str, Any]


class SourceBatch(TypedDict):
    """The composed capture boundary consumed by the durable archive."""

    source_key: str
    native_session_id: str
    cwd: str | None
    records: list[SourceRecord]
    next_cursor: dict[str, Any]
    diagnostics: list[dict[str, Any]]


class SyncResult(TypedDict):
    """Counts returned by a capture operation."""

    sessions: int
    records_written: int
    duplicate_records: int
    pending: int
    errors: int


class SourceSlice(TypedDict):
    """A bounded read from one source; composed into a :class:`SourceBatch`."""

    records: list[SourceRecord]
    next_cursor: dict[str, Any]
    diagnostics: list[dict[str, Any]]
    cwd: str | None
    exhausted: bool


# V2 projection contracts ---------------------------------------------------
#
# These are deliberately additive to the V1 source envelopes above.  Raw
# source records are immutable input; projections are replaceable derived
# state.  A source ID, database generation, and content revision are therefore
# always retained alongside a derived result rather than being replaced by a
# timestamp or import-order identity.

AttributionStatus = Literal["matched", "pending", "ambiguous", "conflicting"]
DiagnosticSeverity = Literal["info", "warning", "error"]


class SourceReference(TypedDict):
    """A role-labelled pointer back to immutable archived evidence.

    ``role`` is a stable consumer-facing label such as ``"user_prompt"``,
    ``"assistant_message"``, ``"tool_execution"``, ``"usage_row"``, or
    ``"finish"``.  It describes evidence only; it never upgrades a heuristic
    correlation to proof.
    """

    source_id: str
    source_kind: str
    role: str


class FinishEvidence(TypedDict):
    """One completion observation supporting a semantic call candidate."""

    source_id: str
    kind: str
    value: str | None


class DatabaseRevision(TypedDict):
    """Database identity for one accounting snapshot.

    ``logical_call_id`` is derived from table, database generation, and row
    primary key.  ``content_revision`` selects the newest snapshot for that
    logical call; a correction replaces its derived UsageRow instead of adding
    another charge.
    """

    table: str
    primary_key: str
    generation: str
    content_revision: str


class NormalizedEvent(TypedDict):
    """One semantic event with stable identity and source provenance."""

    id: str
    kind: str
    ts: str | None
    source_ids: list[str]
    source_references: list[SourceReference]
    attributes: dict[str, Any]


class CallCandidate(TypedDict):
    """A possible assistant-call span reconstructed from transcript evidence.

    IDs are native/source-derived (for example a transcript assistant-message
    event ID), never a bare ``turnId``, chronological parent ID, tool name, or
    list position.  Nullable fields remain null when the archive does not
    establish them.
    """

    call_id: str
    stored_turn_id: str | None
    interaction_id: str | None
    agent_id: str | None
    parent_tool_call_id: str | None
    model: str | None
    source_ids: list[str]
    source_references: list[SourceReference]
    start_ts: str | None
    end_ts: str | None
    tool_call_ids: list[str]
    finish_evidence: list[FinishEvidence]


class AccountingCandidate(TypedDict):
    """A database call before it is attributed to an assistant message.

    ``turn_index`` is the native user-turn index and is not a transcript
    ``turnId``.  ``provider`` is null when the database has no provider;
    normalization must preserve that uncertainty (a UsageRow may use the
    explicit ``"unknown"`` provider sentinel required by its existing shape).
    Missing usage produces no UsageRow rather than a zero-valued one.
    """

    usage_source_id: str
    logical_call_id: str
    turn_index: int | None
    agent_id: str | None
    parent_tool_call_id: str | None
    model: str | None
    provider: str | None
    source_ids: list[str]
    source_references: list[SourceReference]
    revision: DatabaseRevision
    timestamp: str | None
    finish_reason: str | None
    supplemental_metrics: dict[str, Any]


class PendingItem(TypedDict):
    """An explicit capability gap or unresolved relationship."""

    id: str
    kind: str
    reason: str
    source_ids: list[str]
    evidence: list[str]


class ProjectionDiagnostic(TypedDict):
    """Content-free derived-state diagnostic safe for status output."""

    code: str
    severity: DiagnosticSeverity
    message: str
    source_ids: list[str]
    details: dict[str, Any]


class Attribution(TypedDict):
    """The durable result of joining one accounting record to semantics."""

    usage_source_id: str
    stored_turn_id: str | None
    agent_id: str | None
    call_id: str | None
    status: AttributionStatus
    evidence: list[str]


class SemanticProjection(TypedDict):
    """Pure transcript/hook reconstruction; it does not account for tokens."""

    events: list[dict[str, Any]]
    turns: list[TurnSpanDict]
    call_candidates: list[CallCandidate]
    pending: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]


class AccountingProjection(TypedDict):
    """Pure database normalization; it does not claim a message join."""

    usage_rows: list[UsageRow]
    candidates: list[AccountingCandidate]
    diagnostics: list[dict[str, Any]]


class Projection(TypedDict):
    """Combined local-only V2 projection, serializable via UsageRow.to_dict."""

    normalized_events: list[dict[str, Any]]
    turns: list[TurnSpanDict]
    usage_rows: list[UsageRow]
    attributions: list[Attribution]
    pending: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]


class ProjectedTurnRecord(TypedDict):
    """The existing ``session_turns`` view shape for completed main turns only.

    Child-agent evidence remains nested in the main turn's trace/events and is
    never emitted here as an independent human interaction.
    """

    id: str
    turn_id: str
    session_id: str
    platform: str
    cwd: str
    start_seq: int | None
    end_seq: int | None
    start_ts: str | None
    end_ts: str | None
    events: list[dict[str, Any]]
