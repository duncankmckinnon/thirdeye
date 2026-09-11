# Copilot V2 reconciliation contracts

These JSON documents are static DTO examples for projection consumers. They
are not generated archives and are not executable fixtures. Production tests
must create archives through the V1 capture APIs, using the observed sibling
corpus or focused `SourceRecord` inputs.

`semantic-projection.json` is a pure transcript/hook input and its expected
semantic output. `accounting-projection.json` is a pure SQLite input and its
expected accounting output. `storage.json` gives the persisted projection
state and the public main-turn read shape. `transport.json` gives the generic,
platform-independent accounting-span and durable export-job shapes.

`observed-six-calls.json` maps all six observed `assistant_usage_events` rows
from `../assistant-usage-events.json` to source-derived logical identities;
the original data remains the authoritative token/billing corpus. `cases.json`
labels the required synthetic edge cases. A case records a required outcome,
not a timestamp heuristic: a close timestamp, tool name, parentId, bare
transcript `turnId`, or row order is never exact-join proof.

Derived identities are strings whose components are native/source identities:

- semantic event: `copilot:event:<source-id>`;
- main turn: `copilot:turn:<source-key>:<native-session>:<interaction-id>`;
- semantic call: `copilot:call:<assistant-message-source-id>`;
- logical database call/accounting span: `copilot:usage:<source-key>:<table>:<generation>:<primary-key>`.

The content revision is deliberately excluded from a logical database call
identity. A corrected row replaces its normalized `UsageRow`; a different
database generation makes a different identity, preventing row-ID reuse from
being relabelled. `UsageRow` values cross a disk/process boundary only through
their existing `to_dict` / `from_dict` serializers. Unknown providers are
stored as the explicit `unknown` UsageRow provider sentinel; absent usage is
absent, never a zero row. Nano-AI-unit billing stays in supplemental metrics,
not a guessed USD price.

Attribution statuses are exactly `matched`, `pending`, `ambiguous`, and
`conflicting`. An inferred `matched` join lists all supporting evidence. A
fallback accounting export has a deterministic accounting span ID and is
ledgered, so a later local match cannot export the same tokens again on a chat
span.
