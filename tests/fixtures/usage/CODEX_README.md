# Codex rollout usage fixtures

Two **scrubbed, trimmed subsets of real Codex CLI rollout files**, used to test
the Codex usage extractor and event capture. They replace an earlier
hand-authored fiction (`{"type":"response_item","payload":{"input_tokens":1000,…}}`)
that encoded a token shape Codex has never written and let the broken extractor
pass a green test suite while returning zero rows from every real rollout.

Every kept frame is copied verbatim from a real rollout; only prose, arguments,
outputs, and filesystem paths are scrubbed. No frame is hand-authored.

The machine-readable companion `codex_rollout.expected.json` is what the
`codex-usage` and `codex-events` tasks load and assert against. Every number in
it, and every number below, is measured from the shipped `.jsonl` files.

## `codex_rollout.jsonl`

The primary fixture. Exercises the extractor's per-call reconciliation, including
the **repeat `token_count` report** that a naive per-frame sum gets wrong.

### Provenance

- **Source:** `~/.codex/sessions/2026/07/30/rollout-2026-07-30T17-01-26-019fb579-cdda-7a03-86df-65c87b6c4ae2.jsonl`
- **CLI version:** `codex-cli 0.146.0-alpha.9.2` (from `session_meta.payload.cli_version`).
- **Model:** `gpt-5.6-sol` (from `turn_context.payload.model`).
- Original file: 448 lines, 1,286,618 bytes.

### Token facts (verified against the shipped fixture)

- **Codex's `input_tokens` already includes cached tokens** — e.g. the final
  call reports `input_tokens: 198800`, `cached_input_tokens: 192256`,
  `output_tokens: 622`, and `input + output == total_tokens == 199422`. This
  matches the OTel rule, so Codex needs no cache arithmetic (unlike Claude).
- `last_token_usage` is a per-call delta; `total_token_usage` is a cumulative
  running total and must never be summed.
- `reasoning_output_tokens` is a subset of `output_tokens`.
- The 2026-07-30 schema carries `cache_write_input_tokens` in both usage blocks.
- **The fixture contains one repeat report.** The cumulative
  `total_token_usage.total_tokens` series repeats the value **4,784,765** on two
  consecutive `token_count` frames while the running total stays flat. Because
  of it, summing every frame's `last_token_usage.total_tokens` (the naive
  extractor) yields **6,832,295**, overcounting the true final cumulative
  **6,694,163** by **138,132**. Reproducing this discrepancy is the whole point
  of this fixture — dedup must happen in `usage/read.py`.

### Trim strategy (by frame *type*, never by sampling the token series)

- **Kept: all 81 `event_msg/token_count` frames, verbatim and in order.** The
  reconciliation invariant (`sum(last_token_usage) == final cumulative` only
  after dedup) depends on the complete series, including the repeat pair.
- Kept: the first `session_meta` frame and one `turn_context` frame (the largest
  frames in the file — aggressively scrubbed).
- Kept: 3 matched `function_call` / `function_call_output` pairs and 3 matched
  `custom_tool_call` / `custom_tool_call_output` pairs, correlated by `call_id`.
- Kept: one `event_msg/user_message` and one `event_msg/agent_message`.
- Dropped: every `response_item/message`, `response_item/reasoning`,
  `world_state`, `thread_settings_applied`, `task_started`, `task_complete`,
  `web_search_end`, `patch_apply_end`, `mcp_tool_call_end`, and `tool_search`
  frame — the bulk of the bytes; no task asserts on them.

Original relative order is preserved. There was no `turn_aborted` frame in this
rollout, so none is present.

### Resulting census (measured on the shipped fixture)

| Property | Value |
|---|---|
| total lines | 97 |
| `event_msg/token_count` | 81 |
| `response_item/function_call` / `_output` | 3 / 3 |
| `response_item/custom_tool_call` / `_output` | 3 / 3 |
| `event_msg/user_message` | 1 |
| `event_msg/agent_message` | 1 |
| `session_meta` | 1 |
| `turn_context` | 1 |
| distinct cumulative totals | 80 |
| expected de-duplicated calls | 80 |
| final cumulative `total_tokens` | 6,694,163 |
| naive per-frame sum | 6,832,295 |
| expected tool-call events (call family) | 6 |
| expected tool-result events (output family) | 6 |
| file size | 75,030 bytes |

## `codex_rollout_v0626.jsonl`

A much smaller fixture proving the **older token schema still parses**.

### Provenance

- **Source:** `~/.codex/sessions/2026/06/26/rollout-2026-06-26T11-47-19-019f0542-0112-7583-bdbe-e55f44ef80b5.jsonl`
- **CLI version:** `codex-cli 0.141.0` (from `session_meta.payload.cli_version`).
- **Model:** `gpt-5.5`.

Its `last_token_usage` / `total_token_usage` carry only
`input_tokens, cached_input_tokens, output_tokens, reasoning_output_tokens,
total_tokens` — **no `cache_write_input_tokens`**. `non_cached_input_tokens`
does not exist in any real rollout and is absent here too.

Kept: all 4 `token_count` frames, the `session_meta` frame, one `turn_context`,
one matched `function_call` / `function_call_output` pair, one `user_message`,
and one `agent_message`. 10 lines, 4,537 bytes.

The two fixtures are intentionally **not** normalized to the same schema — the
schema difference (presence vs absence of `cache_write_input_tokens`) is the
point.

## Scrubbing rules applied

This repository is public. Rollouts contain real prompts, source code, shell
commands, and filesystem paths.

**Preserved exactly** (assertions depend on them): `timestamp`, `type`,
`payload.type`, the whole `payload.info` subtree on `token_count` frames
(including `rate_limits`), `payload.model` on `turn_context`, `payload.id` and
`payload.session_id` on `session_meta` (kept consistent with the filename UUID
so rollout-resolution tests hold), and `call_id` on every tool frame.

**Replaced:**

- all message / prompt / reasoning text → `"[scrubbed]"`
- `session_meta.payload.base_instructions` → `"[scrubbed]"`
- tool-call `arguments` / `input` → `"{}"`; tool-call/output `output` → `"[scrubbed]"`
- every `cwd` and `workspace_roots` entry → `/scrubbed`
- `session_meta.payload.git.repository_url` → `/scrubbed`
- on `turn_context`, dropped `developer_instructions`, `permission_profile`,
  `sandbox_policy`, `file_system_sandbox_policy`, and `collaboration_mode`
  (large, carry environment detail); kept `model`, `turn_id`, `approval_policy`,
  `effort`, `summary`
- any residual absolute `/Users/…` path anywhere → `/scrubbed`, and any residual
  personal-name token → `scrubbed`

No `duncanmckinnon`, `/Users/`, `Desktop`, or `Documents` string remains in
either fixture.

## Verification (real output)

Run in-process (this sandbox has no `jq`; Python reproduces the same checks):

```
tests/fixtures/usage/codex_rollout.jsonl: 97 lines all valid JSON, 75030 bytes
tests/fixtures/usage/codex_rollout_v0626.jsonl: 10 lines all valid JSON, 4537 bytes
REPEAT cumulative values: [4784765]
naive last_token_usage sum: 6832295
final cumulative total: 6694163
differ: True overcount: 138132
cache_write in fixture1: 162      # 81 frames x 2 usage blocks
cache_write in fixture2: 0
PII in codex_rollout.jsonl : []
PII in codex_rollout_v0626.jsonl : []
```

- Each `.jsonl` is ≤ 150 KB (75,030 and 4,537 bytes).
- The repeat `token_count` report survived the trim (cumulative 4,784,765
  repeats), so `naive_per_frame_sum` (6,832,295) differs from
  `final_cumulative_total_tokens` (6,694,163) — the inequality that makes this
  fixture usable.
- `cache_write_input_tokens` is present in `codex_rollout.jsonl` and absent from
  `codex_rollout_v0626.jsonl`.

## `codex_rollout.expected.json`

The machine-readable companion the extractor tests load. `expected_calls`
equals `distinct_cumulative_totals` (80). `naive_per_frame_sum` differs from
`final_cumulative_total_tokens`, which is what proves the fixture contains a
repeat report. `expected_tool_call_events` / `expected_tool_result_events` are
the summed counts of the call-family (`function_call` + `custom_tool_call`) and
output-family (`function_call_output` + `custom_tool_call_output`) frames kept.
