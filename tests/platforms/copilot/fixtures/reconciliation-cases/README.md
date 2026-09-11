# Copilot V2 reconciliation contracts

These JSON documents are serialized DTO examples for projection consumers.
`input_records` arrays are V1 `SourceRecord` envelopes and may be passed
directly to `build_semantics` / `build_accounting`. They are not generated
archives: production tests that need a Store still create archives through
the V1 capture APIs from the observed sibling corpus or from these records.

Placeholders for the live home digest and database file-stat generation are
documented in `identities.json`. Observed token totals remain authoritative
in `../assistant-usage-events.json` and `../usage.json`.

| File | Role |
|---|---|
| `identities.json` | Shared source-key / generation / ID templates |
| `semantic-projection.json` | Transcript `SourceRecord` input and `SemanticProjection` output, including a `TurnSpanDict` |
| `accounting-projection.json` | Database `SourceRecord` input and `AccountingProjection` output |
| `storage.json` | `ProjectionState` plus `read_projected_turns` Store-event shape |
| `transport.json` | Generic `AccountingCallSpanDict`, turn-owned unmatched job, session job, ledger |
| `observed-six-calls.json` | Six observed rows, inferred call mapping, shutdown totals |
| `attributions.json` | Matched (inferred), pending, ambiguous, conflicting examples |
| `diagnostics.json` | `PendingItem` and `ProjectionDiagnostic` examples |
| `cases.json` | Concrete records and expected outputs for every required edge case |

## V1 source-record shapes

Transcript IDs are `{full_source_key}/{native_id}/{event_id}`. Payloads are
the raw JSONL object plus `schema_version`. Event type is `type` (not
`event`); `id`, `timestamp`, `parentId`, and `agentId` are top-level.
Tool IDs are `data.toolRequests[].toolCallId`. Model cycles end with
`assistant.turn_end` (`data.turnId` only). Locators use `file`,
`file_generation`, `byte_offset`, `byte_length`, and `native_event_id`.

Database IDs are
`copilot-db:{full_source_key}:{native_id}:{table}:{quoted_canonical_pk}:{revision}`
with no generation. Locators include `database`. `ts` comes from the row's
`created_at`. V1 source IDs omit generation, so an identical row in a new
database file deduplicates; the logical call ID uses the first archived
record's locator generation.

Hook IDs are `hook/{native_id}/{observation_id}`.

## Derived identities

`<source-key>` is always the full 64-character SHA-256 digest, never the
16-character stored-session prefix.

- semantic event: `copilot:event:<source-id>`
- extra events from one record: `copilot:event:<source-id>:tool:<toolCallId>`
  (or `:<role>` when no native suffix exists)
- semantic call: `copilot:call:<assistant-message-source-id>`
- main turn: `copilot:turn:<full-source-key>:<native-session>:<interaction-id>`
- logical call / accounting span:
  `copilot:usage:<full-source-key>:<table>:<quoted-generation>:<quoted-canonical-pk>`

`quoted-generation` and `quoted-canonical-pk` are `urllib.parse.quote(..., safe="")`
of the locator generation (`sha256:<hex>`, colon becomes `%3A`) and of
canonical JSON of `locator.primary_key`. Content revision is excluded from
the logical ID. `Attribution.usage_source_id` is the newest revision's source
ID; durable attribution is keyed by `logical_call_id`.

## UsageRow gaps

A `UsageRow` is emitted only when timestamp, model, input_tokens, and
output_tokens are all present. Otherwise keep the `AccountingCandidate` and
add `missing_usage_fields`. Present partial counts go in `supplemental_metrics`.
Omit JSON-null metric keys (absent, never null). In-memory `seq` is `0` until
persistence stamps the Store seq of the newest archived revision. The SQLite
primary key is never `seq`. Unknown provider is the UsageRow sentinel
`unknown`; candidate `provider` stays null when the database has none.
Nano-AI-unit billing stays in supplemental metrics.

## Attribution

Statuses are `matched`, `pending`, `ambiguous`, and `conflicting`.
`join_kind` is `inferred` or `direct` only on a match. Evidence strings are
`key:value` using `AttributionEvidenceKey`. Ambiguous listings include every
candidate `call_id:...` and choose none.

`Attribution.status=conflicting` is a join conflict. Incompatible database
revisions are `ProjectionDiagnostic.code=usage_revision_conflict` and
quarantine the logical call.

## Turns, storage, export

`read_projected_turns` returns Store events (`seq` / `t` / `ts` / `data`)
for completed main interactions only. Child evidence stays inside them.

`AccountingCallSpanDict.call_id` set means export on that chat span; null
means a user-turn or agent accounting span. `LlmCallSpanDict.usage` stays
empty whenever `accounting_calls` is present. Span IDs:

- chat-span: existing chat span for `call_id`
- turn-accounting-span: `accounting:{session_id}:{turn_id}:{accounting_id}`
- session-accounting-span: `accounting:{session_id}:{accounting_id}`

Queued or configured export is not successful delivery. Auxiliary
`model.*` title-generation records use `kind=auxiliary_model_call` and
`classification=title_generation` and cannot inflate the six-call total.
