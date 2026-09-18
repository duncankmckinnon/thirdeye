# Grok Bot tracing

Thirdeye can export Grok Bot agent turns to Pydantic Logfire with the same
GenAI span shape used for Claude Code, Codex, and Cursor: session →
`invoke_agent` → chat / tools.

Grok Bot runs on hosted computers. Cursor IDE `hooks.json` does **not** see
those runs, so thirdeye installs a separate `grok_bot` platform beside Cursor
and reuses the shared `otel_export` path (no second Logfire auth or OTLP
client).

## Install (always with Cursor)

Installing Cursor tracing always co-installs Grok Bot (same step, not a
separate opt-in):

```bash
thirdeye add --cursor
```

The guided setup wizard does the same when it installs Cursor tracing.
You can also install Grok Bot alone:

```bash
thirdeye add --grok-bot
```

Install records an enablement marker under
`$THIRDEYE_HOME/platforms/grok_bot/` (default `~/.thirdeye/platforms/grok_bot/`).
It does **not** add entries to `~/.cursor/hooks.json`.

Enable Logfire export if you have not already (`thirdeye logfire enable`).
Grok Bot spans use the same write token and project as other platforms.

## Capture path

On each Grok Bot computer, thirdeye polls the local agent store:

```text
/home/box/agent-data/agents/<agentUuid>/store.db
  → transcript_entries(seq, id, entry)
```

Mapped entry kinds (Wave 2):

| `kind` | Role |
|--------|------|
| `message` | User / assistant turn text (`role`, `content`, `requestId`) |
| `send-message` | Recognized; does not invent a turn by itself |
| `event` | Correlated by `requestId`; does not invent I/O text |
| `tool-call` | **Deferred** — outline rows are skipped until full tool body mapping lands |

Entries that share a `requestId` become one `TurnSpanDict`. Completed turns are
handed to `thirdeye.otel_export.export_turn` (detached, fail-open). Empty or
missing turns are never exported as `{}`.

### Fail-open behavior

Temporal harness boxes often have an empty local `store.db` while live
transcript state lives on the server tail. That is expected:

- Empty `transcript_entries` → skip the cycle; do not export.
- SQLite `BUSY` / locked / corrupt DB → skip the cycle; do not raise into the
  agent loop.

When local rows appear, they are the preferred evidence for GenAI turn trees.
Until then, treat empty local DB as normal — not a misconfiguration.

### Required attributes (Logfire MCP findability)

Every exported turn carries (at least):

| Attribute | Source |
|-----------|--------|
| `thirdeye.platform` | Always `grok_bot` |
| `gen_ai.conversation.id` | Session / conversation id passed to capture |
| `gen_ai.agent.name` | Agent profile name (Logfire Agents page) |
| `thirdeye.grok_bot.agent_id` | Agent directory UUID |
| `thirdeye.grok_bot.request_id` / `request_id` | Entry `requestId` used for turn correlation |

Prompt and tool detail follow existing thirdeye scrubbing / capture-env
settings. Do not put gateway tokens, encryption keys, or other secrets on
span attributes.

## Caveats

- **tool-call bodies** are not mapped yet. Outline-only `tool-call` rows are
  skipped so thirdeye does not invent Shell (or other) tool spans from
  incomplete data.
- Continuous deploy should add a last-seen `seq` watermark so re-polls do not
  duplicate spans in Logfire (tracked follow-up).
- Do not conflate Grok Bot with Grok Build CLI / `GROK_EXTERNAL_OTEL`, or with
  Cursor Enterprise Action Recording OTel (logs/metrics, not GenAI turn trees).

## Logfire MCP: query examples for alignment

Orchestrator (and other agents) can use Logfire MCP `query_run` against the
`records` table to find Grok Bot sessions and reconstruct trajectories for
instruction / brief improvements. There is no first-class "get transcript"
tool — filter on the attributes above. `query_run` windows are typically on
the order of ~14 days.

### 1. Recent Grok Bot agent turns

```sql
SELECT
  start_timestamp,
  span_name,
  attributes->>'gen_ai.conversation.id' AS conversation_id,
  attributes->>'gen_ai.agent.name' AS agent_name,
  attributes->>'thirdeye.grok_bot.agent_id' AS agent_id,
  attributes->>'thirdeye.grok_bot.request_id' AS request_id
FROM records
WHERE attributes->>'thirdeye.platform' = 'grok_bot'
  AND span_name = 'invoke_agent'
ORDER BY start_timestamp DESC
LIMIT 50
```

### 2. One conversation’s turn tree (trajectory)

Replace the conversation id:

```sql
SELECT
  start_timestamp,
  span_name,
  parent_span_id,
  span_id,
  attributes->>'gen_ai.agent.name' AS agent_name,
  attributes->>'thirdeye.grok_bot.request_id' AS request_id,
  attributes->>'gen_ai.input.messages' AS input_messages,
  attributes->>'gen_ai.output.messages' AS output_messages
FROM records
WHERE attributes->>'thirdeye.platform' = 'grok_bot'
  AND attributes->>'gen_ai.conversation.id' = '<conversation-id>'
ORDER BY start_timestamp ASC
LIMIT 500
```

Use this shape to walk session → `invoke_agent` → chat/tool children for a
single bot conversation before rewriting that agent’s brief.

### 3. Filter by agent (multi-bot)

```sql
SELECT
  start_timestamp,
  attributes->>'gen_ai.conversation.id' AS conversation_id,
  attributes->>'gen_ai.agent.name' AS agent_name,
  attributes->>'thirdeye.grok_bot.agent_id' AS agent_id,
  span_name
FROM records
WHERE attributes->>'thirdeye.platform' = 'grok_bot'
  AND (
    attributes->>'gen_ai.agent.name' = 'Orchestrator'
    OR attributes->>'thirdeye.grok_bot.agent_id' = '<agent-uuid>'
  )
ORDER BY start_timestamp DESC
LIMIT 100
```

Keep bots distinct: always key on agent id (and conversation id), not a single
shared agent name for the whole fleet.

## Related

- Shared export: `thirdeye.otel_export`
- Platform package: `thirdeye.platforms.grok_bot` (`install`, `capture`, `tracing`)
- Logfire setup skill: `thirdeye-logfire`
