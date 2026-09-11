"""Versioned source envelopes (V1) and derived projection contracts (V2).

V1 TypedDicts describe immutable archived evidence.  They remain the input to
every V2 algorithm.  V2 TypedDicts describe replaceable derived state:
normalized events, turn reconstruction, independent database accounting, and
attribution.  A source ID, database generation, and content revision are
always retained alongside a derived result rather than being replaced by a
timestamp or import-order identity.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

from thirdeye.tracing.model import TurnSpanDict
from thirdeye.usage.types import UsageRow

from .constants import SOURCE_SCHEMA_VERSION

# The value placed beside every archived SourceRecord in a thirdeye event data
# envelope.  It is deliberately separate from a Copilot CLI version.
SCHEMA_VERSION = SOURCE_SCHEMA_VERSION

# Derived projection state is versioned independently of the V1 envelope.
# Rebuilds may drop and recreate this document without rewriting raw events.
PROJECTION_SCHEMA_VERSION = 2


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
# Stable derived identities use native/source identities, never import time or
# mutable list indices.  Encoding rules (see also the reconciliation README):
#
# Transcript source IDs are V1 ``{full_source_key}/{native_id}/{event_id}``.
# Database source IDs are V1
# ``copilot-db:{full_source_key}:{native_id}:{table}:{quoted_canonical_pk}:{revision}``
# and do not embed generation.  Hook source IDs are V1
# ``hook/{native_id}/{observation_id}``.
#
# ``<source-key>`` in every derived ID is the full 64-character SHA-256 source
# home digest, not the 16-character prefix used only in stored session IDs.
#
# Semantic event: ``copilot:event:<source-id>`` for the primary event of that
# record.  When one source record yields extra events, append a native suffix:
# ``copilot:event:<source-id>:tool:<toolCallId>`` for each
# ``data.toolRequests[].toolCallId`` on ``assistant.message``.  A tool
# start/complete pair already has two source IDs, so each uses the unsuffixed
# form.  If no native suffix exists, append ``:<SourceReferenceRole>``.
#
# Semantic call: ``copilot:call:<assistant-message-source-id>``.
# Main turn: ``copilot:turn:<full-source-key>:<native-session>:<interaction-id>``.
#
# Logical database call / accounting span:
# ``copilot:usage:<full-source-key>:<table>:<quoted-generation>:<quoted-canonical-pk>``.
# ``quoted-generation`` and ``quoted-canonical-pk`` are
# ``urllib.parse.quote(..., safe="")`` of the locator generation and of
# ``json.dumps(locator.primary_key, sort_keys=True, separators=(",", ":"))``
# respectively, matching V1 percent-quoting of composite keys.  Generation is
# ``sha256:<hex>`` and therefore contains a colon; quoting turns that colon
# into ``%3A``.  Content revision is excluded.  Because V1 source IDs omit
# generation, an identical row seen in a new database file deduplicates to the
# first archived record; the logical ID uses that first record's locator
# generation.  A different generation with different content is a new logical
# call (row-ID reuse), never a relabel of the previous call.
#
# ``Attribution.usage_source_id`` is the newest revision's source ID.
# Durable attribution is keyed by ``logical_call_id``.

AttributionStatus = Literal["matched", "pending", "ambiguous", "conflicting"]
AttributionJoinKind = Literal["direct", "inferred"]
DiagnosticSeverity = Literal["info", "warning", "error"]
EventClassification = Literal["main", "title_generation", "checkpoint", "shutdown_validation"]
Initiator = Literal["user", "agent", "sub-agent"]

NormalizedEventKind = Literal[
    "user_prompt",
    "assistant_message",
    "tool_request",
    "tool_execution_start",
    "tool_execution_complete",
    "tool_execution_failure",
    "permission_request",
    "permission_decision",
    "notification",
    "prompt_transformation",
    "compaction",
    "abort",
    "error",
    "session_start",
    "session_end",
    "session_shutdown",
    "subagent_started",
    "subagent_completed",
    "auxiliary_model_call",
    "unknown",
]

SourceReferenceRole = Literal[
    "user_prompt",
    "assistant_message",
    "tool_request",
    "tool_execution",
    "tool_result",
    "permission_request",
    "permission_decision",
    "usage_row",
    "finish",
    "hook",
    "shutdown",
    "checkpoint",
    "compaction",
    "auxiliary_model",
    "nested_child",
]

FinishEvidenceKind = Literal[
    "assistant_turn_end",
    "agent_stop_hook",
    "finish_reason_db",
    "abort",
    "error",
]

PendingItemKind = Literal[
    "open_interaction",
    "missing_usage",
    "unmatched_usage",
    "missing_source_capability",
    "missing_identity",
    "incomplete_tool_pair",
    "delayed_row",
]

DiagnosticCode = Literal[
    "usage_revision_conflict",
    "usage_row_id_reuse",
    "shutdown_total_mismatch",
    "checkpoint_not_additive",
    "auxiliary_excluded_from_main",
    "capability_gap",
    "inferred_join",
    "unknown_provider",
    "missing_usage_fields",
    "open_interaction",
]

# Attribution.evidence strings are ``key:value`` with these keys.  ``call_id``
# may repeat when status is ambiguous.  Inferred joins always include
# ``join_kind:inferred``; a future native assistant-message/provider-call id
# uses ``join_kind:direct``.
AttributionEvidenceKey = Literal[
    "join_kind",
    "interaction_id",
    "agent_id",
    "model",
    "turn_index",
    "parent_tool_call_id",
    "finish",
    "order",
    "call_id",
    "logical_call_id",
    "usage_source_id",
    "revision",
    "initiator",
    "tool_call_id",
]


class SourceReference(TypedDict):
    """A role-labelled pointer back to immutable archived evidence.

    ``role`` describes evidence only; it never upgrades a heuristic correlation
    to proof.
    """

    source_id: str
    source_kind: str
    role: SourceReferenceRole


class FinishEvidence(TypedDict):
    """One completion observation supporting a semantic call candidate.

    Observed Copilot transcripts finish a model cycle with
    ``assistant.turn_end``, which carries only ``turnId``.  There is no
    ``agent.stop`` transcript event and no finish reason on that record.
    """

    source_id: str
    kind: FinishEvidenceKind
    value: str | None


class DatabaseRevision(TypedDict):
    """Database identity for one accounting snapshot.

    ``primary_key`` is the percent-quoted canonical JSON of the V1 locator's
    ``primary_key`` value (an int for observed ``assistant_usage_events`` rows,
    an object for composite keys).  ``generation`` is the unquoted locator
    generation (``sha256:<hex>``).  ``logical_call_id`` quotes both.
    ``content_revision`` selects the newest snapshot for that logical call; a
    correction replaces its derived UsageRow instead of adding another charge.
    """

    table: str
    primary_key: str
    generation: str
    content_revision: str


class NormalizedEvent(TypedDict):
    """One semantic event with stable identity and source provenance.

    Auxiliary ``model.*`` title-generation records use
    ``kind="auxiliary_model_call"`` and ``classification="title_generation"``.
    They must not appear as main ``call_candidates`` and must not inflate
    conversation token totals.
    """

    id: str
    kind: NormalizedEventKind
    classification: EventClassification
    ts: str | None
    source_ids: list[str]
    source_references: list[SourceReference]
    attributes: dict[str, Any]


class CallCandidate(TypedDict):
    """A possible assistant-call span reconstructed from transcript evidence.

    IDs are native/source-derived (for example a transcript assistant-message
    event ID), never a bare ``turnId``, chronological parent ID, tool name, or
    list position.  Nullable fields remain null when the archive does not
    establish them.  ``tool_call_ids`` come from
    ``data.toolRequests[].toolCallId``, not from a ``toolCallIds`` array.
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


class SupplementalMetrics(TypedDict, total=False):
    """Copilot-only row fields that are not UsageRow columns.

    Keys are omitted when the source value is missing or JSON-null.  Do not
    store nulls.  Nano-AI-unit billing stays here and is never converted into
    estimated model-price USD.
    """

    total_nano_aiu: int
    request_multiplier: float
    duration_ms: int
    time_to_first_token_ms: float
    output_ttft_ms: float
    inter_token_latency_ms: float
    initiator: Initiator
    api_endpoint: str
    reasoning_effort: str
    content_filter_triggered: int
    token_details_json: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int


class AccountingCandidate(TypedDict):
    """A database call before it is attributed to an assistant message.

    ``turn_index`` is the native user-turn index and is not a transcript
    ``turnId``.  ``provider`` is null when the database has no provider;
    normalization must preserve that uncertainty (a UsageRow may use the
    explicit ``"unknown"`` provider sentinel required by its existing shape).
    Missing usage produces no UsageRow rather than a zero-valued one.

    A UsageRow is emitted only when timestamp, model, input_tokens, and
    output_tokens are all present.  A null timestamp or model keeps the
    candidate and adds ``missing_usage_fields``; no row.  Partial usage
    (input present, output missing) likewise yields no row: present counts
    go into ``supplemental_metrics`` only.  In-memory ``UsageRow.seq`` is ``0``
    until persistence stamps the Store seq of the newest archived revision.
    The SQLite primary key is never ``seq``, including when it happens to be
    an integer.
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
    supplemental_metrics: SupplementalMetrics


class PendingItem(TypedDict):
    """An explicit capability gap or unresolved relationship."""

    id: str
    kind: PendingItemKind
    reason: str
    source_ids: list[str]
    evidence: list[str]


class ProjectionDiagnostic(TypedDict):
    """Content-free derived-state diagnostic safe for status output."""

    code: DiagnosticCode
    severity: DiagnosticSeverity
    message: str
    source_ids: list[str]
    details: dict[str, Any]


class Attribution(TypedDict):
    """The durable result of joining one accounting record to semantics.

    ``usage_source_id`` is the newest revision's source ID.  ``logical_call_id``
    is the durable key across content revisions.  ``join_kind`` is ``inferred``
    or ``direct`` only when ``status`` is ``matched``; otherwise null.

    ``status="conflicting"`` is a join conflict: competing semantic assignments
    or a join that would rebind a logical call after tokens were emitted.
    Incompatible database revisions are ``ProjectionDiagnostic``
    ``usage_revision_conflict`` and quarantine the logical call; they do not
    use ``AttributionStatus`` by themselves.
    """

    usage_source_id: str
    logical_call_id: str
    stored_turn_id: str | None
    agent_id: str | None
    call_id: str | None
    status: AttributionStatus
    join_kind: AttributionJoinKind | None
    evidence: list[str]


class SemanticProjection(TypedDict):
    """Pure transcript/hook reconstruction; it does not account for tokens."""

    events: list[NormalizedEvent]
    turns: list[TurnSpanDict]
    call_candidates: list[CallCandidate]
    pending: list[PendingItem]
    diagnostics: list[ProjectionDiagnostic]


class AccountingProjection(TypedDict):
    """Pure database normalization; it does not claim a message join."""

    usage_rows: list[UsageRow]
    candidates: list[AccountingCandidate]
    diagnostics: list[ProjectionDiagnostic]


class Projection(TypedDict):
    """Combined local-only V2 projection, serializable via UsageRow.to_dict."""

    normalized_events: list[NormalizedEvent]
    turns: list[TurnSpanDict]
    usage_rows: list[UsageRow]
    attributions: list[Attribution]
    pending: list[PendingItem]
    diagnostics: list[ProjectionDiagnostic]


class ProjectedTurnRecord(TypedDict):
    """The existing ``session_turns`` view shape for completed main turns only.

    ``events`` are Store events (``seq``, ``t``, ``ts``, ``data``), not
    normalized semantic stubs.  ``t`` is ``copilot_transcript``,
    ``copilot_database``, ``copilot_hook``, or ``copilot_metadata``.  ``data``
    is the V1 envelope ``{schema_version, source_record}``.  ``start_seq`` /
    ``end_seq`` are those events' seq values.  ``filter_turns`` and
    ``logfire_dataset._turn_case`` consume this shape.

    Child-agent evidence remains nested in the main turn's events / trace and
    is never emitted here as an independent human interaction.
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


class OpenInteractionState(TypedDict):
    """Unfinished interaction retained across incremental archive partitions."""

    interaction_id: str
    agent_id: str | None
    stored_turn_id: str
    source_ids: list[str]
    last_event_source_id: str | None
    start_ts: str | None
    pending_tool_call_ids: list[str]


class LogicalCallState(TypedDict):
    """Durable accounting identity for one logical database call.

    ``generation`` is the first archived record's locator generation.
    ``metrics_digest`` is ``sha256:`` of canonical JSON over the row's
    input/output/cache/reasoning/nano-AIU fields so a later revision can be
    compared without treating row-ID reuse as the same call.
    """

    logical_call_id: str
    generation: str
    content_revision: str
    metrics_digest: str
    usage_source_id: str


class SemanticProjectionState(TypedDict):
    """Incremental semantic replay state stored under projection state."""

    open_interactions: dict[str, OpenInteractionState]


class AccountingProjectionState(TypedDict):
    """Incremental accounting replay state stored under projection state."""

    logical_calls: dict[str, LogicalCallState]


class ProjectionState(TypedDict):
    """Persisted derived state.  Separate from the V1 archive envelope.

    ``projection_schema_version`` is :data:`PROJECTION_SCHEMA_VERSION`.  It is
    not the V1 ``schema_version`` field on source envelopes.
    """

    projection_schema_version: int
    archive_source_ids: list[str]
    semantic_state: SemanticProjectionState
    accounting_state: AccountingProjectionState
    projection_revision: str
    commit_result: NotRequired[dict[str, int]]
