"""Normalize archived Copilot database usage without assigning it to messages.

The SQLite rows are the accounting authority.  Transcript checkpoints and
shutdown records can validate the result, but never create another charge.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime
from typing import Any
from urllib.parse import quote

from thirdeye.usage.types import UsageRow

from .types import (
    AccountingCandidate,
    AccountingProjection,
    AccountingProjectionState,
    DatabaseRevision,
    ProjectionDiagnostic,
    SourceRecord,
)

_USAGE_TABLE = "assistant_usage_events"
_METRIC_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "total_nano_aiu",
)
_SUPPLEMENTAL_FIELDS = (
    "total_nano_aiu",
    "request_multiplier",
    "duration_ms",
    "time_to_first_token_ms",
    "output_ttft_ms",
    "inter_token_latency_ms",
    "initiator",
    "api_endpoint",
    "reasoning_effort",
    "content_filter_triggered",
    "token_details_json",
)
_TOKEN_FIELDS = _METRIC_FIELDS[:-1]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _valid_timestamp(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _integer(value: object) -> int | None:
    """Return a non-negative integer without turning missing values into zero."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _source_key(record: SourceRecord) -> str | None:
    prefix = "copilot-db:"
    source_id = record["source_id"]
    if not source_id.startswith(prefix):
        return None
    key = source_id[len(prefix) :].split(":", 1)[0]
    return key if len(key) == 64 else None


def _revision(record: SourceRecord) -> DatabaseRevision | None:
    locator = record.get("locator")
    if not isinstance(locator, dict) or locator.get("table") != _USAGE_TABLE:
        return None
    generation = _string(locator.get("generation"))
    content_revision = _string(locator.get("content_revision"))
    if generation is None or content_revision is None or "primary_key" not in locator:
        return None
    return {
        "table": _USAGE_TABLE,
        "primary_key": quote(_canonical_json(locator["primary_key"]), safe=""),
        "generation": generation,
        "content_revision": content_revision,
    }


def _logical_call_id(record: SourceRecord, revision: DatabaseRevision) -> str | None:
    source_key = _source_key(record)
    if source_key is None:
        return None
    return (
        f"copilot:usage:{source_key}:{revision['table']}:"
        f"{quote(revision['generation'], safe='')}:{revision['primary_key']}"
    )


def _metrics_digest(row: dict[str, Any]) -> str:
    metrics = {field: row.get(field) for field in _METRIC_FIELDS}
    return "sha256:" + hashlib.sha256(_canonical_json(metrics).encode("utf-8")).hexdigest()


def _row_is_incompatible(row: dict[str, Any]) -> bool:
    """Reject revisions that cannot describe a single completed request."""

    input_tokens = _integer(row.get("input_tokens"))
    cache_read = _integer(row.get("cache_read_tokens"))
    cache_write = _integer(row.get("cache_write_tokens"))
    if input_tokens is None:
        return False
    return (cache_read is not None and cache_read > input_tokens) or (
        cache_write is not None and cache_write > input_tokens
    )


def _candidate(
    record: SourceRecord, revision: DatabaseRevision, logical_id: str
) -> AccountingCandidate:
    payload = record.get("payload")
    row = payload.get("row") if isinstance(payload, dict) else None
    assert isinstance(row, dict)
    timestamp = _valid_timestamp(record.get("ts")) or _valid_timestamp(row.get("created_at"))
    supplemental = {
        field: row[field]
        for field in _SUPPLEMENTAL_FIELDS
        if field in row and row[field] is not None
    }
    # Fully specified token usage belongs in UsageRow.  Retain token values in
    # the candidate only when a partial row cannot produce a UsageRow.
    if _integer(row.get("input_tokens")) is None or _integer(row.get("output_tokens")) is None:
        supplemental.update(
            {
                field: row[field]
                for field in _TOKEN_FIELDS
                if field in row and row[field] is not None
            }
        )
    turn_index = _integer(row.get("turn_index"))
    return {
        "usage_source_id": record["source_id"],
        "logical_call_id": logical_id,
        "turn_index": turn_index,
        "agent_id": _string(row.get("agent_id")),
        "parent_tool_call_id": _string(row.get("parent_tool_call_id")),
        "model": _string(row.get("model")),
        "provider": _string(row.get("provider")) or _string(row.get("provider_name")),
        "source_ids": [record["source_id"]],
        "source_references": [
            {"source_id": record["source_id"], "source_kind": "database", "role": "usage_row"}
        ],
        "revision": revision,
        "timestamp": timestamp,
        "finish_reason": _string(row.get("finish_reason")),
        "supplemental_metrics": supplemental,
    }


def _missing_fields(candidate: AccountingCandidate, row: dict[str, Any]) -> list[str]:
    required = {
        "timestamp": candidate["timestamp"],
        "model": candidate["model"],
        "input_tokens": _integer(row.get("input_tokens")),
        "output_tokens": _integer(row.get("output_tokens")),
    }
    return [name for name, value in required.items() if value is None]


def _usage_row(
    candidate: AccountingCandidate, session_id: str, row: dict[str, Any]
) -> UsageRow | None:
    missing = _missing_fields(candidate, row)
    if missing:
        return None
    input_tokens = _integer(row["input_tokens"])
    output_tokens = _integer(row["output_tokens"])
    assert input_tokens is not None and output_tokens is not None
    return UsageRow(
        session_id=session_id,
        seq=0,
        call_id=candidate["logical_call_id"],
        ts=candidate["timestamp"],  # guarded by _missing_fields
        platform="copilot",
        provider_name=candidate["provider"] or "unknown",
        response_model=candidate["model"],  # guarded by _missing_fields
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=_integer(row.get("cache_read_tokens")),
        cache_creation_input_tokens=_integer(row.get("cache_write_tokens")),
        reasoning_output_tokens=_integer(row.get("reasoning_tokens")),
    )


def _diagnostic(
    code: str, severity: str, message: str, source_ids: list[str], **details: Any
) -> ProjectionDiagnostic:
    return {
        "code": code,  # type: ignore[typeddict-item]
        "severity": severity,  # type: ignore[typeddict-item]
        "message": message,
        "source_ids": source_ids,
        "details": details,
    }


def _shutdown_totals(records: Iterable[SourceRecord]) -> list[tuple[str, dict[str, int]]]:
    totals: list[tuple[str, dict[str, int]]] = []
    for record in records:
        payload = record.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "session.shutdown":
            continue
        data = payload.get("data")
        if not isinstance(data, dict):
            continue
        model_metrics = data.get("modelMetrics")
        if not isinstance(model_metrics, dict):
            continue
        aggregate = {name: 0 for name in _METRIC_FIELDS}
        found = False
        for model in model_metrics.values():
            usage = model.get("usage") if isinstance(model, dict) else None
            if not isinstance(usage, dict):
                continue
            values = {
                "input_tokens": usage.get("inputTokens"),
                "output_tokens": usage.get("outputTokens"),
                "cache_read_tokens": usage.get("cacheReadTokens"),
                "cache_write_tokens": usage.get("cacheWriteTokens"),
                "reasoning_tokens": usage.get("reasoningTokens"),
                "total_nano_aiu": model.get("totalNanoAiu"),
            }
            if all(_integer(value) is not None for value in values.values()):
                found = True
                for name, value in values.items():
                    aggregate[name] += _integer(value) or 0
        if found:
            totals.append((record["source_id"], aggregate))
    return totals


def build_accounting(
    records: list[SourceRecord], prior_state: dict[str, Any]
) -> tuple[AccountingProjection, dict[str, Any]]:
    """Build independent database accounting from immutable V1 source records.

    Revisions are applied in archive order.  A missing row is intentionally not
    a deletion: only archived observations can replace an accounting result.
    """

    # The V1 archive preserves observation order, so a later archived revision
    # is authoritative.  Keep every source ID as conflict evidence even though
    # only the newest revision supplies the candidate.
    inherited_calls = prior_state.get("logical_calls")
    if not isinstance(inherited_calls, dict):
        accounting_state = prior_state.get("accounting_state")
        inherited_calls = (
            accounting_state.get("logical_calls") if isinstance(accounting_state, dict) else {}
        )
    selected: dict[str, tuple[SourceRecord, DatabaseRevision, AccountingCandidate]] = {}
    revision_sources: dict[str, list[str]] = {}
    diagnostics: list[ProjectionDiagnostic] = []
    primary_generations: dict[tuple[str, str], set[str]] = {}

    for record in records:
        payload = record.get("payload")
        if record.get("source_kind") != "database" or not isinstance(payload, dict):
            continue
        if payload.get("table") != _USAGE_TABLE or not isinstance(payload.get("row"), dict):
            continue
        revision = _revision(record)
        logical_id = _logical_call_id(record, revision) if revision is not None else None
        if revision is None or logical_id is None:
            continue
        candidate = _candidate(record, revision, logical_id)
        primary_generations.setdefault((revision["table"], revision["primary_key"]), set()).add(
            revision["generation"]
        )
        selected[logical_id] = (record, revision, candidate)
        revision_sources.setdefault(logical_id, []).append(record["source_id"])

    for (table, primary_key), generations in sorted(primary_generations.items()):
        if len(generations) > 1:
            diagnostics.append(
                _diagnostic(
                    "usage_row_id_reuse",
                    "warning",
                    "database row ID was reused by a different database generation",
                    [],
                    table=table,
                    primary_key=primary_key,
                    generations=sorted(generations),
                )
            )

    candidates: list[AccountingCandidate] = []
    usage_rows: list[UsageRow] = []
    accounted = {field: 0 for field in _METRIC_FIELDS}
    logical_calls = {
        key: value.copy()
        for key, value in inherited_calls.items()
        if isinstance(key, str) and isinstance(value, dict)
    }
    for logical_id, (record, revision, candidate) in selected.items():
        payload = record["payload"]
        row = payload["row"]
        assert isinstance(row, dict)
        if _row_is_incompatible(row):
            logical_calls.pop(logical_id, None)
            diagnostics.append(
                _diagnostic(
                    "usage_revision_conflict",
                    "error",
                    "incompatible metrics for one logical call; quarantined",
                    revision_sources[logical_id],
                    logical_call_id=logical_id,
                )
            )
            continue
        candidates.append(candidate)
        missing = _missing_fields(candidate, row)
        if missing:
            diagnostics.append(
                _diagnostic(
                    "missing_usage_fields",
                    "warning",
                    "database usage row is incomplete",
                    [record["source_id"]],
                    logical_call_id=logical_id,
                    missing_fields=missing,
                )
            )
        source_key = _source_key(record)
        assert source_key is not None
        session_id = f"copilot-{source_key[:16]}-{record['native_session_id']}"
        usage_row = _usage_row(candidate, session_id, row)
        if usage_row is not None:
            usage_rows.append(usage_row)
        for field in _METRIC_FIELDS:
            value = _integer(row.get(field))
            if value is not None:
                accounted[field] += value
        logical_calls[logical_id] = {
            "logical_call_id": logical_id,
            "generation": revision["generation"],
            "content_revision": revision["content_revision"],
            "metrics_digest": _metrics_digest(row),
            "usage_source_id": record["source_id"],
        }

    for record in records:
        payload = record.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "session.usage_checkpoint":
            continue
        diagnostics.append(
            _diagnostic(
                "checkpoint_not_additive",
                "info",
                "checkpoint snapshot validates totals and is not a seventh call",
                [record["source_id"]],
            )
        )

    for source_id, expected in _shutdown_totals(records):
        if any(accounted[field] != expected[field] for field in _METRIC_FIELDS):
            diagnostics.append(
                _diagnostic(
                    "shutdown_total_mismatch",
                    "warning",
                    "per-call totals do not equal session.shutdown usage",
                    [source_id],
                    **{
                        f"expected_{field}": expected[field]
                        for field in _METRIC_FIELDS
                    },
                    **{
                        f"accounted_{field}": accounted[field]
                        for field in _METRIC_FIELDS
                    },
                )
            )

    next_state: AccountingProjectionState = {"logical_calls": logical_calls}
    return {"usage_rows": usage_rows, "candidates": candidates, "diagnostics": diagnostics}, next_state
