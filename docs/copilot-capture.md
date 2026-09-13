# Copilot CLI capture and reconciliation

thirdeye's Copilot integration is local, archive-first CLI capture. It supports
GitHub Copilot CLI recordings; it does not claim support for native VS Code,
Copilot cloud history, or Copilot as an evaluator/Ask backend.

## V1 archive to V2 projections

V1 capture stores immutable source envelopes in the normal local session Store.
Their event types are `copilot_transcript`, `copilot_database`, `copilot_hook`,
and `copilot_metadata`; each has a version-1 envelope and preserves its source
ID, native session ID, source timestamp when available, observation time,
payload, and locator. Transcript records retain native event IDs and content;
database records retain table/row/revision identities; hook records remain
observations rather than asserted tool executions.

V2 is a separate, versioned derived state. It reads the retained archive rather
than the current Copilot home, so it can reprocess an older captured session
without launching an agent or retaining the original transcript/SQLite files.
It does not duplicate the raw records in default semantic views.

```bash
# Rebuild every retained Copilot archive locally.
thirdeye copilot reconcile

# Rebuild one archive's derived indexes only.
thirdeye copilot reconcile --session-id <stored-session-id> --rebuild
```

`--rebuild` replaces derived projections only. It does not delete V1 raw source
events, change stored-session/source identities, or reset export eligibility or
delivery history.

## Raw and derived views

Generic event commands continue to expose the retained raw evidence. Copilot
main-turn views, searches, evaluation inputs, and usage views consume V2's
normalized projections, so raw transcript/database/hook evidence does not show
up a second time as a semantic event. Completed main interactions are the
projected user-turn records; child-agent events and tools remain nested evidence
within their owning main interaction.

The `status` command separates source capability/ingestion diagnostics from
projection and export state. In particular, pending, ambiguous, and conflicting
joins are local accounting states, and a configured or queued export is not a
successful remote delivery.

## Source evidence and turn reconstruction

For the observed Copilot CLI 1.0.83 corpus, a user interaction is identified by
`interactionId` with agent identity, not by a bare transcript `turnId` (which
resets). A child is attached through `agentId`, `parentToolCallId`, and the
parent task's `subagent.started.data.toolCallId`; interleaved transcript events
therefore remain in the correct subtree. Tool requests/executions are paired by
their invocation IDs, which distinguishes simultaneous identical calls.

The retained external hook stream is supplementary. Its pre/post tool payloads
do not provide an invocation ID, prompt/stop hooks can use a child agent ID in
the session field, and its coverage need not equal transcript hook coverage.
It must not be used as proof of an exact tool pairing or as a session-ID-only
turn state machine. A prompt may also arrive before `SessionStart`.

Unknown transcript schemas/events remain source evidence and searchable raw
records. A missing identity, missing completion, permission decision, abort,
compaction, or otherwise incomplete record produces an explicit pending or
diagnostic result rather than an invented completed turn.

## Usage accounting and attribution

`assistant_usage_events` database rows are the primary per-call accounting
source. Input counts include cache tokens when supplied; missing usage stays
missing rather than becoming zero. Checkpoints and shutdown totals validate the
database ledger but never add a second charge. Auxiliary `model.*` title
generation is classified separately and cannot inflate conversation totals.
Unknown model providers remain `unknown`; Copilot nano-AI-unit billing is kept
separate from any estimated USD model price.

Each logical database call has a stable accounting identity across revisions.
An authoritative row correction replaces its local derived `UsageRow`; a
generation/row-ID reuse or incompatible correction is quarantined as a
conflict, not counted as another call. Delayed rows remain eligible for a later
reconciliation.

The observed database schema has user `turn_index`, agent ID, parent tool-call
ID, model, ordering, finish/tool evidence, and token/billing/latency fields,
but no shared assistant-message/provider-call ID. As a result, attribution is
deliberately conservative:

- `matched` is direct only with direct evidence, or inferred only when the
  interaction, agent, model, order, and finish/tool evidence form one uniquely
  consistent candidate. Its evidence is retained with the attribution.
- `pending` means more source evidence may resolve ownership.
- `ambiguous` lists competing candidates and chooses none.
- `conflicting` quarantines incompatible revision or join evidence.

An unmatched terminal usage row still appears in local accounting. It can be
represented as an explicit user-turn/agent accounting span, or as session-level
accounting when no user-turn ownership is known; thirdeye never fabricates a
user turn merely to place tokens.

## Export eligibility, retries, and corrections

Local reconciliation never exports history by default. `sync --export` or
`reconcile --export` explicitly opts the selected completed retained history
into export eligibility. When live export is configured, first activation also
records a durable boundary: already-terminal history stays local-only, while
an interaction open at activation becomes eligible if it completes later.
The boundary survives restart and derived-state rebuild.

Export assembly writes durable, deterministic local jobs. Matched accounting is
placed on its chat span; unmatched accounting uses one explicit accounting span.
The placement ledger prevents a logical token identity from being emitted in
both locations. If fallback accounting has already been delivered, a later
local match cannot re-export those tokens on the chat span. Pending rows stay
retryable; a correction after confirmed delivery is reported as an accounting
conflict rather than silently adding a new charge.

Remote OpenTelemetry delivery is not transactionally exactly once. A process
can crash after a remote flush and before durable acknowledgement, so a retry
may resend a deterministic span. Durable jobs, deterministic IDs, and delivery
claims minimize that window and allow recovery, but cannot prove remote receipt
in every crash case. `status` reports queued work and last errors separately
from confirmed delivery.

## Known source limitations

The checked-in 1.0.83 corpus establishes two main interactions, one explore
child, five tool executions, twenty external hook observations, and six
database usage calls. The six database calls reconcile to the observed shutdown
totals, but the lack of a native shared assistant-message/provider-call ID
means comprehensive local capture does not make every message join exact.

Interactive, folder-trusted capture produced hooks in the probe. Earlier
non-interactive probes did not; the cause was not proven, so this is not a
general claim that non-interactive hooks are unsupported. Watch/import remains
useful without hooks. Synthetic schema-derived fixtures cover permission,
compaction, abort, unknown-version, delayed-row, revision, retry, identical
concurrent-tool, nested-child, and uncertain-attribution behavior. They are
deterministic regression inputs, not live validation. Optional future live
tests remain separate and require no CI credentials.
