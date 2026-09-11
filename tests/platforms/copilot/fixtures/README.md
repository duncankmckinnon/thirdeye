# Copilot CLI capture probe

Captured 2026-09-10 on macOS using Copilot CLI 1.0.83, with automatic model selection (resolved to gpt-5.6-luna). These are observed fixtures for adapter design, not an implemented adapter or a regression test suite.

## Successful scenario

An isolated temporary Git repository contained `alpha.txt` (`alpha = 17`) and `beta.txt` (`beta = 25`). An interactive CLI session was launched from that directory. Folder trust was accepted for that session only. Available tools were restricted to `view,task`; both were allowed. Built-in MCP servers and remote session export were disabled.

1. Ask for two separate reads in parallel and their sum.
2. Ask an explore subagent to read the same two files and report their sum.
3. Exit normally with `/exit`.

Both top-level answers and the subagent answer were `42`. The transcript verifies overlapping top-level reads, five tool start/completion pairs (four reads plus the task invocation), two top-level user prompts, and one completed subagent.

Repository hooks used version 1, camelCase event names, `type: command`, `bash`, and `timeoutSec: 5`. Each invoked a Python recorder that read stdin JSON and wrote a separate timestamp/PID-named file. It emitted no stdout and caught recorder errors. Captured events: sessionStart, userPromptSubmitted, preToolUse, postToolUse, agentStop, subagentStart, subagentStop, sessionEnd. Failure/error hooks were configured but not exercised.

## Files

- `hooks.jsonl`: 20 actual external hook invocations, ordered by recorder filename. `registered_event` is the configured camelCase event; `captured_ns` is recorder wall time; `payload` is Copilot input.
- `events.jsonl`: 76 retained transcript records, in original file order.
- `usage.json`: the final transcript `session.shutdown.data` object, including overall and per-agent usage. This is a duplicate extraction for convenience; do not count it again when aggregating the transcript.

Sanitization replaces the local home/workspace paths and removes system messages, usage checkpoint internals, `model.*` events, transformed prompts, reasoning text/blocks, opaque/encrypted provider fields, API call IDs, tool telemetry, and server-tools metadata. Event IDs, timestamps, agent IDs, interaction IDs, tool-call IDs, and measured usage are retained. This is deliberately a filtered transcript, not a byte-for-byte raw capture; parentId references may point to omitted records. Raw captures remain in `/private/tmp/thirdeye-copilot-probe` and the original CLI session state.

## Integration findings

- External pre/post tool hooks have no invocation ID in this run. The transcript provides `toolCallId` on requests and executions. Correlate parallel tools using the transcript, not just tool names.
- `turnId` denotes a model/tool cycle and resets across user requests. Group user interactions using `interactionId` plus agent identity.
- Subagent events are interleaved in the same transcript and carry top-level `agentId`; `subagent.started` links to the parent task via `data.toolCallId`. Child records also expose `parentToolCallId` where applicable.
- The child generated its own user-prompt and agent-stop hooks using the parent's session ID and transcript path. There are three prompt hooks and three stop hooks for two top-level user requests. External hook payloads alone cannot reliably distinguish these turns.
- SessionStart arrived after the first user-prompt hook. Initialization must tolerate that ordering.
- There are 20 external recorder files but only 18 hook.start/hook.end pairs in the final transcript. Do not assume those two streams have one-to-one coverage.
- Assistant text and model names are present in assistant.message. Final session.shutdown includes input/output/cache/reasoning usage and per-agent metrics. Availability of per-call usage at Stop time has not been established by this probe.

## Limitations and unsuccessful probes

Earlier non-interactive `-p` runs completed but produced no external hook captures. This remained true with both PascalCase/exec and camelCase/bash configurations, after committing the temporary hook file, and when launching directly from the fixture directory. Interactive mode with session folder trust produced the successful fixture. This narrows the issue to runtime mode/trust/loading behavior but does not isolate its exact cause; it does not prove non-interactive hooks are universally unsupported.

PascalCase compatibility, user-level installation, interrupted/error turns, repeated identical concurrent tool arguments, and other Copilot versions/runtimes remain unverified. No permanent user-level hooks or thirdeye adapter code were installed.

## Persisted database follow-up

Read-only inspection of `~/.copilot/session-store.db` found six `assistant_usage_events` rows for this session: four main-agent calls and two explore calls. `assistant-usage-events.json` retains those rows. Their input/output/cache-read/cache-write/reasoning totals and nano-AI-unit total exactly match session.shutdown; per-agent input totals also match. Database `turn_index` groups these into the two actual user interactions (0 and 1), unlike transcript turnId, which indexes model cycles and restarts.

The table includes agent_id, parent_tool_call_id, model, tokens, billing details, call duration, time to first token/output, inter-token latency, initiator, endpoint, reasoning effort, finish reason, and creation time. The installed event schema marks assistant.usage as ephemeral: it is absent from events.jsonl, but these database rows persist it independently. Both interactive and the earlier non-interactive test sessions have usage rows. The table lacks the transcript assistant message ID/provider call ID, so exact per-message joins require additional care; grouping by session/turn/agent is explicit.

Persisted usage checkpoint events contain aggregate billing and cache-frontier diagnostics, including latest input/cache counts, tool schema hashes, system-segment token estimates, cache TTL, and completion time. They are snapshots, not a complete per-call ledger. In this session each top-level checkpoint follows agentStop in file order, so a Stop-time read can precede the checkpoint.

The raw transcript also contains two auxiliary model.model_call_success records for gpt-4o-mini session-title generation, with request/response content, usage and latency. These were omitted from the sanitized transcript. Do not treat them as the main-agent model-call history or add their usage to the six-row total without explicitly accounting for auxiliary calls.

The database also has sessions, turns, checkpoints, session_files, session_refs and full-text search tables. Our session has two complete turns but no session_files rows despite four file reads, so the database's discovery/index tables do not replace raw tool execution events. Files beside the transcript include workspace.yaml (identity/repo/title), checkpoints/index.md (empty here), and rewind-file-snapshots/tracking.json (tracking metadata only here).
