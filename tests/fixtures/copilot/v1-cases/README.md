# Copilot V1 synthetic contract cases

Every file in this directory is synthetic.  It documents capture contracts and
reader/archive edge cases; it is not evidence of Copilot CLI behavior.

`source-slice.json` and `source-batch.json` are small, JSON-round-trippable
values for independent reader, archive, and installer tests.  They include the
full source-record envelope with schema version `1` in each event payload.
`exhausted` belongs only to `SourceSlice`; `SourceBatch` must not carry it.

`trailing-json.jsonl` ends with an incomplete JSON record.  `trailing-utf8.hex`
is the bytes of a JSONL file whose final UTF-8 code point is incomplete; tests
that need raw bytes should decode the hex, rather than silently converting it
to replacement characters.  A reader must defer both tails until a later read.

`unknown-event-fields.jsonl` requires lossless retention of fields that V1 does
not interpret.  `database-row-revisions.json` represents a row-ID reuse and a
changed row revision: equal logical snapshots deduplicate, changed content is a
new evidence record.  `distinct-hook-observations.json` contains identical
payloads with distinct observation IDs; both records must survive.
`missing-event-id.jsonl` requires a locator built from file generation, byte
location, and content digest rather than an invented native event ID.
`source-key-prefix-collision.json` is the archive-reuse contract for two
homes whose SHA-256 digests share a 16-character display prefix.

The observed `cli-1.0.83` sibling fixture has two main prompts and one child
prompt.  The child prompt is retained as source evidence, not asserted to be a
third human request.  Its six `assistant_usage_events` rows remain raw database
evidence and must not create UsageStore rows.

## Identity and revision handoff

The selected source home is canonicalized and SHA-256 hashed in full.  Stored
session IDs use only the first 16 digest characters for readability.
`stored_session_id` checks that one `SourcePaths` value is internally
consistent (full digest matches that home).  Two legitimate homes whose
digests share a 16-character prefix still produce the same stored ID; see
`source-key-prefix-collision.json`.  On reuse the archive must compare the
full `source_key` retained in session metadata with the candidate home and
refuse to merge a prefix collision.  Distinct native IDs from different homes
must not merge either.

Transcript identity is source key + native session + native event ID.  Without
an event ID it is file generation + byte location + content digest and is
explicitly lower confidence.  Database identity is table + row primary key +
content revision; its generation remains separate to expose replacement or
row-ID reuse.  Hook identity is a generated observation ID: same-content hooks
are independent observations, while rereading one spool entry is idempotent.
