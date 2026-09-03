---
name: thirdeye-logfire
description: Set up, verify, disable, or troubleshoot thirdeye's live Pydantic Logfire export and Logfire MCP access for Claude Code, Codex, or Cursor. Use for thirdeye Logfire tokens, missing spans, export status, MCP connection, and integration diagnostics; do not use for instrumenting unrelated applications directly with the Logfire SDK.
---

# Set up thirdeye Logfire export

Use thirdeye's built-in integration. Do not add application-level Logfire
instrumentation when the request is only about exporting thirdeye traces.

## Inspect before changing

Run:

```bash
thirdeye logfire status
```

Treat export as active only when `package installed`, `enabled`, and `active`
are all true. Respect `THIRDEYE_HOME`; configuration is persisted in
`<thirdeye_home>/config.yaml` and applies to future captured turns.

## Enable

If the Logfire package is missing, install thirdeye's optional dependency in
the environment that owns the `thirdeye` executable. Use the installation
method appropriate to that environment, such as:

```bash
pip install 'thrdi[logfire]'
```

Enabling requires a Logfire project write token (gateway key):

```bash
thirdeye logfire enable
```

The command prompts for the write token with hidden input so it is not placed
in shell history or process arguments.

Never invent, search broadly for, print, or commit a token. Ask the user to
provide it when it is unavailable, and explain that thirdeye persists it in
its config. The user can enter it through the command's hidden prompt or the
Settings page in `thirdeye ui`.

Ensure tracing hooks are installed for each requested harness (`thirdeye add
--codex`, `--claude`, or another supported platform). Logfire export does not
create sessions by itself.

## Verify end to end

After enabling, confirm status again, then complete a new agent turn. Export
is asynchronous, so allow the detached worker a brief opportunity to flush.
Verify both sides:

- The session and completed turn exist locally with `thirdeye list` and
  `thirdeye events <sid>`.
- The corresponding trace appears in the configured Logfire project.

For local Cursor IDE and CLI sessions, expect the child of a `Task` call to be
its own Task-parented `invoke_agent` span — including background, parallel, and
nested children. CLI tool calls attach when they share the derived child
generation. IDE tool calls live in the child conversation and are joined when
that session's transcript ends with `turn_ended` (Cursor often skips
`subagentStop` for backgrounded Tasks). Missing or unreadable transcripts still
produce the Task-parented child turn from lifecycle hooks. When a child
transcript is present it backfills the first user and last assistant text; it
does not supply span timing, tool IDs, token usage, or extra model-call
boundaries unless those events were captured in the child session. When
assistant text is absent, the child output falls back to a summary.

Do not claim success from `status` alone: it validates local configuration,
not delivery. If direct Logfire access is unavailable, state that the remote
appearance still needs user verification.

## Cursor trace content

New Cursor turns export the captured interaction stream into Logfire. Each
thirdeye session is one trace with this hierarchy:

`session → invoke_agent → chat → tool`

Interaction spans (reasoning, user and assistant messages, subagent lifecycle
events, and similar) nest under the turn's `invoke_agent` span. Each chat span
carries cumulative conversation context in `gen_ai.input.messages` — the full
session history up to that call's response, not just the current turn.

thirdeye exports complete raw captured payloads on interaction spans as
structured `thirdeye.interaction.payload` attributes (user prompts, assistant
text, reasoning, and lifecycle details). Tool arguments and results export on
tool spans (`thirdeye.tool.call.payload` / `thirdeye.tool.result.payload`) and
in chat `gen_ai.input.messages`. thirdeye does not scrub or truncate this
content before export; Pydantic Logfire applies its own scrubbing at ingestion
(for example passwords and API keys). thirdeye only exempts legitimate uses of
the word "session" in captured agent content from Logfire's default scrub
patterns.

Live interaction and tool spans can arrive before their turn or chat parents.
At `turn_stop`, thirdeye uploads the turn and chat spans; Logfire joins
temporarily orphaned children by trace and span ID. thirdeye never keeps OTel
spans open across hook processes.

Identical reasoning events from the same hook callback are deduplicated: when
kind, timestamp, generation ID, correlation ID, and normalized content all
match, thirdeye emits one span and records duplicate source sequences in
`thirdeye.interaction.duplicate_seqs`. Repeated reasoning with the same text at
a different timestamp or with distinct IDs remains separate spans.

Only newly captured turns receive this interaction export. Existing Logfire
traces are not retroactively updated or backfilled.

## Troubleshoot missing spans

Check `<thirdeye_home>/logs/usage-errors.jsonl` for recent entries whose
`phase` contains `logfire`, `otel`, or `export`. Keep tokens and unrelated
captured content out of diagnostic output. Distinguish these failure classes:

- No local session: repair or reinstall the platform tracing hooks.
- No completed turn: finish a new turn; export is triggered from captured
  turn and live-span events.
- Inactive status: install the optional package, enable export, or supply a
  token as indicated by `thirdeye logfire status`.
- Worker/export error: report the phase, exception class, and concise message,
  then address that concrete cause.
- Local export succeeds but nothing appears remotely: verify project/token
  selection and network access, then generate one fresh turn rather than
  repeatedly changing configuration.

Disable export without deleting the saved token using:

```bash
thirdeye logfire disable
```

Report what was changed, the resulting status, what was verified locally,
and whether remote delivery was actually observed.

## Logfire MCP access

The export integration writes telemetry; the Logfire MCP server lets Claude
Code or Codex query it. These are separate connections with separate
authentication. When the user requests MCP setup, read
[references/mcp-server.md](references/mcp-server.md) and follow only the
section for their client and Logfire region.
