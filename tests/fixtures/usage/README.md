# Claude Code usage fixtures

## `claude_transcript.jsonl`

A **scrubbed, curated subset** of a real Claude Code 2.1.214 session transcript,
used to test the Claude usage extractor's per-call deduplication and inclusive
token arithmetic. It is not the whole transcript: the full session held hundreds
of frames and over a megabyte of prose. Shipping that crashed downstream tooling,
so only the frames needed to exercise the extractor's edge cases are retained,
with original line order preserved.

### Provenance

- **Source:** a Claude Code project transcript,
  `~/.claude/projects/scrubbed/16a1e984-1dbd-454b-b9f2-0a9e4b4683d3.jsonl`
  (project directory name scrubbed for privacy; this repository is public).
- **CLI version:** `2.1.214 (Claude Code)` (the `version` field carried on every frame).

### Scrubbing rules applied

Preserved exactly (assertions depend on them): `type`, `timestamp`, `uuid`,
`parentUuid`, `requestId`, `message.id`, `message.model`, `message.role`,
`message.usage` (every subfield), and the `type` of each `message.content[]`
item.

Replaced:

- every `message.content[].text` → `"[scrubbed]"`
- every `tool_use` item's `.input` → `{}`
- every `tool_result` item's `.content` → `"[scrubbed]"`
- top-level `toolUseResult` payloads → `"[scrubbed]"`
- `cwd` → `"/scrubbed"`, `gitBranch` → `"main"`
- `sessionId` → a fixed fake UUID (`00000000-0000-4000-8000-000000000000`)
- any absolute filesystem path anywhere → `/scrubbed`, and any residual
  personal-name token → `scrubbed`

Dropped entirely (the extractor ignores them and they carry the most content):
`attachment`, `file-history-snapshot`, `last-prompt`, and `queue-operation`
frames.

### Verified counts (measured on the shipped fixture)

| Property | Value |
|---|---|
| total lines | 15 |
| `type=="assistant"` frames | 13 |
| distinct `message.id` among assistant frames | 8 |
| assistant frames with `message.model == "<synthetic>"` | 1 |
| assistant frames with `requestId == null` | 1 |
| most-repeated `message.id` | `msg_011CdBpZs1PvZ3gsGPM8rXdf` |
| ...its frame count | 6 |
| expected de-duplicated calls (`distinct − synthetic`) | 7 |

### Sample call — `msg_011CdBpZs1PvZ3gsGPM8rXdf`

| Field | Value |
|---|---|
| `input_tokens` | 2 |
| `output_tokens` | 3195 |
| `cache_read_input_tokens` | 254643 |
| `cache_creation_input_tokens` | 504 |
| **computed inclusive input** (`input + cache_read + cache_creation`) | 255149 |

The inclusive-input total is what `gen_ai.usage.input_tokens` must equal for this
call: Anthropic reports `input_tokens` *excluding* cache, so the extractor adds
`cache_read_input_tokens` and `cache_creation_input_tokens` back in.

### Verification (real `jq` output)

```
$ wc -l -c tests/fixtures/usage/claude_transcript.jsonl
      15   17291 tests/fixtures/usage/claude_transcript.jsonl

$ jq -rc 'select(.type=="assistant")' … | wc -l          # assistant frames
13
$ jq -rc 'select(.type=="assistant").message.id' … | sort -u | wc -l   # distinct ids
8
$ jq -rc 'select(.type=="assistant" and .message.model=="<synthetic>")' … | wc -l
1
$ jq -rc 'select(.type=="assistant" and .requestId==null)' … | wc -l
1
$ jq -rc 'select(.type=="assistant").message.id' … | sort | uniq -c | sort -rn
   6 msg_011CdBpZs1PvZ3gsGPM8rXdf
   1 msg_011CdATG2uFx1p3cezx4pJh2
   1 msg_011CdATFYGKcS9wwCERfd2xZ
   1 msg_011CdATFPLJn9tySKE1hqKHk
   1 msg_011CdATFhedZkNVu3DiBHFaa
   1 msg_011CdATFBHktRHGavvpHssof
   1 msg_011CdATEmRK5u9k4QLvKAA2f
   1 1b3a9ee4-60ff-49bc-b876-99790a06f70f

$ jq -c 'select(.message.id=="msg_011CdBpZs1PvZ3gsGPM8rXdf").message.usage
         | {input_tokens,output_tokens,cache_read_input_tokens,cache_creation_input_tokens}' … | head -1
{"input_tokens":2,"output_tokens":3195,"cache_read_input_tokens":254643,"cache_creation_input_tokens":504}
```

Exactly one `message.id` (`msg_011CdBpZs1PvZ3gsGPM8rXdf`) appears on ≥4 frames —
six identical-usage frames, the per-call deduplication case. All six carry the
same `message.usage`, so a naive extractor would emit six rows for one API call.

## `claude_transcript.expected.json`

The machine-readable companion the extractor tests load. Every value is measured
from the shipped `claude_transcript.jsonl`, not copied from a specification.
