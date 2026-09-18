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

Install writes an enablement marker under
`$THIRDEYE_HOME/platforms/grok_bot/` (default `~/.thirdeye/platforms/grok_bot/`)
and **arms a store-mutation kick** (marker-gated, idle-exiting) so local
`store.db` activity exports automatically. It does **not** add entries to
`~/.cursor/hooks.json`, and it is **not** a boot/pidfile long-lived daemon.

Enable Logfire export if you have not already (`thirdeye logfire enable`).
Grok Bot spans use the same write token and project as other platforms.

## Passive tracing (store-mutation kick)

After install (or Cursor co-install on the box), thirdeye **arms a box-level
store-mutation kick**. When an agent `store.db` under the agent-data root
changes (new `transcript_entries`), a short sync builds `TurnSpanDict`s and
hands them to `thirdeye.otel_export.export_turn` (detached, fail-open) with a
**seq watermark** so re-kicks do not duplicate spans. The kick is
**idle-exiting** — not a boot-managed daemon. You do **not** need to call a
library function for the happy path.

Uninstall (`thirdeye remove --grok-bot`, and `remove --cursor` co-stop) disarms
the kick and clears its state.

### Advanced: manual one-shot poll (optional / not required)

For debugging only, you can drive a single library cycle without the kick:

```python
from pathlib import Path
from thirdeye.platforms.grok_bot.capture import poll_and_export

poll_and_export(
    Path("/home/box/agent-data/agents/<agentUuid>/store.db"),  # example path
    conversation_id="<conversation-id>",
    agent_id="<agentUuid>",
    agent_name="Orchestrator",
    cwd="/home/box",
)
```

Prefer the install-armed store-mutation kick for passive tracing.

## Capture path

When `poll_and_export` runs, it reads the local agent store. A typical layout
on a Grok Bot computer looks like:

```text
/home/box/agent-data/agents/<agentUuid>/store.db   # example; use the box's real agent-data root
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
passed only to `thirdeye.otel_export.export_turn` (detached, fail-open) — never
an empty `{}` payload.

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
`records` table (DataFusion SQL) to find Grok Bot sessions and reconstruct
trajectories for instruction / brief improvements. There is no first-class
"get transcript" tool — filter on the attributes above. `query_run` windows
are typically ≤14 days. After a hit, `project_logfire_link(trace_id=…)` opens
the trace for human review.

### 1. Recent Grok Bot sessions by agent

```sql
SELECT start_timestamp, trace_id,
       attributes->>'gen_ai.conversation.id' AS conversation_id,
       attributes->>'gen_ai.agent.name' AS agent_name,
       attributes->>'thirdeye.platform' AS platform
FROM records
WHERE attributes->>'thirdeye.platform' = 'grok_bot'
ORDER BY start_timestamp DESC
LIMIT 50;
```

### 2. One conversation trajectory (alignment loop)

```sql
SELECT start_timestamp, span_name, duration,
       attributes->>'gen_ai.operation.name' AS op,
       attributes->>'gen_ai.conversation.id' AS conversation_id
FROM records
WHERE attributes->>'thirdeye.platform' = 'grok_bot'
  AND attributes->>'gen_ai.conversation.id' = '<conversation-id>'
ORDER BY start_timestamp;
```

Walk session → `invoke_agent` → chat/tool children before rewriting that
agent’s brief. Tool-call bodies are still unmapped in capture.

### 3. Multi-bot filter by agent id/name

```sql
SELECT attributes->>'gen_ai.agent.name' AS agent_name,
       attributes->>'gen_ai.conversation.id' AS conversation_id,
       count(*) AS spans
FROM records
WHERE attributes->>'thirdeye.platform' = 'grok_bot'
  AND (attributes->>'gen_ai.agent.name' = '<agent-name>'
       OR attributes->>'thirdeye.grok_bot.agent_id' = '<agent-uuid>')
GROUP BY 1, 2
ORDER BY spans DESC;
```

Keep bots distinct: key on agent id and conversation id, not one shared name
for the whole fleet.

## Related

- Shared export: `thirdeye.otel_export`
- Platform package: `thirdeye.platforms.grok_bot` (`install`, `capture`, `tracing`)
- Logfire setup skill: `thirdeye-logfire`
