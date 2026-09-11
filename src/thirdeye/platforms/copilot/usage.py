"""Normalize archived Copilot database usage without assigning it to messages.

The SQLite rows are the accounting authority.  Transcript checkpoints and
shutdown records can validate the result, but never create another charge.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any
from urllib.parse import quote

from thirdeye.usage.types import UsageRow

from .identity import SOURCE_KEY_DIGEST_LEN, SOURCE_KEY_PREFIX_LEN
from .types import (
    AccountingCandidate,
    AccountingProjection,
    DatabaseRevision,
    DiagnosticCode,
    DiagnosticSeverity,
    ProjectionDiagnostic,
    SourceRecord,
)

_USAGE_TABLE = "assistant_usage_events"
_MAIN_AGENT_KEY = "main"
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
_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
)


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


def _metric_number(value: object) -> int | None:
    """Accept whole-number floats so shutdown ``totalNanoAiu`` can validate."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value >= 0 and value.is_integer():
        return int(value)
    return None


def _source_key(record: SourceRecord) -> str | None:
    prefix = "copilot-db:"
    source_id = record["source_id"]
    if not source_id.startswith(prefix):
        return None
    key = source_id[len(prefix) :].split(":", 1)[0]
    return key if len(key) == SOURCE_KEY_DIGEST_LEN else None


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


def _row_metrics(row: dict[str, Any]) -> dict[str, int]:
    metrics: dict[str, int] = {}
    for field in _METRIC_FIELDS:
        value = _integer(row.get(field))
        if value is not None:
            metrics[field] = value
    return metrics


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


def _row_key(table: str, primary_key: str) -> str:
    return f"{table}\x1f{primary_key}"


def _agent_key(agent_id: str | None) -> str:
    return agent_id or _MAIN_AGENT_KEY


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


def _attach_revision_sources(candidate: AccountingCandidate, source_ids: list[str]) -> None:
    candidate["source_ids"] = list(source_ids)
    candidate["source_references"] = [
        {"source_id": source_id, "source_kind": "database", "role": "usage_row"}
        for source_id in source_ids
    ]


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
    code: DiagnosticCode,
    severity: DiagnosticSeverity,
    message: str,
    source_ids: list[str],
    **details: Any,
) -> ProjectionDiagnostic:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "source_ids": source_ids,
        "details": details,
    }


def _add_metrics(target: dict[str, int], metrics: dict[str, int], *, sign: int = 1) -> None:
    for field, value in metrics.items():
        target[field] = target.get(field, 0) + sign * value


def _zero_metrics() -> dict[str, int]:
    return {field: 0 for field in _METRIC_FIELDS}


def _metrics_from_call(call: dict[str, Any]) -> dict[str, int]:
    if call.get("quarantined"):
        return {}
    metrics = call.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    parsed: dict[str, int] = {}
    for field in _METRIC_FIELDS:
        value = _integer(metrics.get(field))
        if value is not None:
            parsed[field] = value
    return parsed


def _parse_usage_block(usage: dict[str, Any], total_nano_aiu: object) -> dict[str, int] | None:
    values = {
        "input_tokens": usage.get("inputTokens"),
        "output_tokens": usage.get("outputTokens"),
        "cache_read_tokens": usage.get("cacheReadTokens"),
        "cache_write_tokens": usage.get("cacheWriteTokens"),
        "reasoning_tokens": usage.get("reasoningTokens"),
        "total_nano_aiu": total_nano_aiu,
    }
    parsed = {name: _metric_number(value) for name, value in values.items()}
    if any(value is None for value in parsed.values()):
        return None
    return {name: value for name, value in parsed.items() if value is not None}


def _aggregate_model_metrics(model_metrics: object) -> dict[str, int] | None:
    if not isinstance(model_metrics, dict) or not model_metrics:
        return None
    aggregate = _zero_metrics()
    for model in model_metrics.values():
        if not isinstance(model, dict):
            return None
        usage = model.get("usage")
        if not isinstance(usage, dict):
            return None
        parsed = _parse_usage_block(usage, model.get("totalNanoAiu"))
        if parsed is None:
            return None
        _add_metrics(aggregate, parsed)
    return aggregate


def _agent_shutdown_totals(agent_metrics: object) -> dict[str, dict[str, int]] | None:
    if not isinstance(agent_metrics, dict) or not agent_metrics:
        return None
    totals: dict[str, dict[str, int]] = {}
    for agent_id, payload in agent_metrics.items():
        if not isinstance(agent_id, str) or not isinstance(payload, dict):
            return None
        parsed = _aggregate_model_metrics(payload.get("modelMetrics"))
        if parsed is None:
            return None
        totals[agent_id] = parsed
    return totals


def _unwrap_prior_state(prior_state: dict[str, Any]) -> dict[str, Any]:
    """Accept a flat accounting state or a nested ProjectionState envelope.

    ``prior_state`` may be ``{"logical_calls": ...}`` (and optional cumulative
    keys) or ``{"accounting_state": {"logical_calls": ...}}``.
    """

    inherited_calls = prior_state.get("logical_calls")
    if isinstance(inherited_calls, dict):
        return prior_state
    nested = prior_state.get("accounting_state")
    if isinstance(nested, dict):
        return nested
    return {}


def _hydrate_row_identity(
    source: dict[str, Any], inherited_calls: dict[str, Any]
) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    generations: dict[str, set[str]] = {}
    sources: dict[str, list[str]] = {}
    prior_generations = source.get("row_generations")
    if isinstance(prior_generations, dict):
        for key, values in prior_generations.items():
            if isinstance(key, str) and isinstance(values, list):
                generations[key] = {item for item in values if isinstance(item, str)}
    prior_sources = source.get("row_sources")
    if isinstance(prior_sources, dict):
        for key, values in prior_sources.items():
            if isinstance(key, str) and isinstance(values, list):
                sources[key] = [item for item in values if isinstance(item, str)]
    for call in inherited_calls.values():
        if not isinstance(call, dict):
            continue
        table = call.get("table") if isinstance(call.get("table"), str) else _USAGE_TABLE
        primary_key = call.get("primary_key")
        generation = call.get("generation")
        if not isinstance(primary_key, str) or not isinstance(generation, str):
            continue
        key = _row_key(table, primary_key)
        generations.setdefault(key, set()).add(generation)
        usage_source_id = call.get("usage_source_id")
        if isinstance(usage_source_id, str) and usage_source_id not in sources.setdefault(key, []):
            sources[key].append(usage_source_id)
    return generations, sources


def _hydrate_accounted(
    source: dict[str, Any], inherited_calls: dict[str, Any]
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    accounted = _zero_metrics()
    accounted_by_agent: dict[str, dict[str, int]] = {}
    prior_accounted = source.get("accounted_metrics")
    if isinstance(prior_accounted, dict) and any(
        _integer(prior_accounted.get(field)) is not None for field in _METRIC_FIELDS
    ):
        for field in _METRIC_FIELDS:
            value = _integer(prior_accounted.get(field))
            if value is not None:
                accounted[field] = value
        prior_agents = source.get("accounted_by_agent")
        if isinstance(prior_agents, dict):
            for agent_id, metrics in prior_agents.items():
                if isinstance(agent_id, str) and isinstance(metrics, dict):
                    bucket = _zero_metrics()
                    _add_metrics(bucket, _metrics_from_call({"metrics": metrics}))
                    accounted_by_agent[agent_id] = bucket
        return accounted, accounted_by_agent
    for call in inherited_calls.values():
        if not isinstance(call, dict):
            continue
        metrics = _metrics_from_call(call)
        _add_metrics(accounted, metrics)
        agent_id = call.get("agent_id")
        key = _agent_key(agent_id if isinstance(agent_id, str) else None)
        accounted_by_agent.setdefault(key, _zero_metrics())
        _add_metrics(accounted_by_agent[key], metrics)
    return accounted, accounted_by_agent


def _logical_call_entry(
    logical_id: str,
    revision: DatabaseRevision,
    record: SourceRecord,
    row: dict[str, Any],
    candidate: AccountingCandidate,
    *,
    quarantined: bool,
) -> dict[str, Any]:
    return {
        "logical_call_id": logical_id,
        "generation": revision["generation"],
        "content_revision": revision["content_revision"],
        "metrics_digest": _metrics_digest(row),
        "usage_source_id": record["source_id"],
        "table": revision["table"],
        "primary_key": revision["primary_key"],
        "agent_id": candidate["agent_id"],
        "metrics": {} if quarantined else _row_metrics(row),
        "quarantined": quarantined,
    }


def _mismatch_details(expected: dict[str, int], accounted: dict[str, int]) -> dict[str, int]:
    details: dict[str, int] = {}
    for field in _METRIC_FIELDS:
        details[f"expected_{field}"] = expected[field]
        details[f"accounted_{field}"] = accounted[field]
    return details


def _metrics_match(expected: dict[str, int], accounted: dict[str, int]) -> bool:
    return all(accounted.get(field, 0) == expected[field] for field in _METRIC_FIELDS)


def build_accounting(
    records: list[SourceRecord], prior_state: dict[str, Any]
) -> tuple[AccountingProjection, dict[str, Any]]:
    """Build independent database accounting from immutable V1 source records.

    Revisions are applied in archive order.  A missing row is intentionally not
    a deletion: only archived observations can replace an accounting result.

    ``prior_state`` may be a flat accounting document (``logical_calls`` plus
    optional cumulative keys) or a full projection state with nested
    ``accounting_state``.  Cumulative ``accounted_metrics``,
    ``accounted_by_agent``, ``row_generations``, and ``row_sources`` support
    disjoint archive partitions; prefix replay remains equivalent to a full
    pass.
    """

    source = _unwrap_prior_state(prior_state)
    inherited_raw = source.get("logical_calls")
    inherited_calls = inherited_raw if isinstance(inherited_raw, dict) else {}
    selected: dict[str, tuple[SourceRecord, DatabaseRevision, AccountingCandidate, str | None]] = {}
    revision_sources: dict[str, list[str]] = {}
    diagnostics: list[ProjectionDiagnostic] = []
    row_generations, row_sources = _hydrate_row_identity(source, inherited_calls)
    prior_digests = {
        logical_id: call["metrics_digest"]
        for logical_id, call in inherited_calls.items()
        if isinstance(logical_id, str)
        and isinstance(call, dict)
        and isinstance(call.get("metrics_digest"), str)
    }

    for record in records:
        payload = record.get("payload")
        if record.get("source_kind") != "database" or not isinstance(payload, dict):
            continue
        if payload.get("table") != _USAGE_TABLE or not isinstance(payload.get("row"), dict):
            continue
        revision = _revision(record)
        logical_id = _logical_call_id(record, revision) if revision is not None else None
        if revision is None or logical_id is None:
            reason = (
                "source_id is not a 64-character copilot-db identity"
                if revision is not None
                else "locator is missing generation, content_revision, or primary_key"
            )
            diagnostics.append(
                _diagnostic(
                    "capability_gap",
                    "warning",
                    "usage row cannot be accounted because archive identity is incomplete",
                    [record["source_id"]],
                    reason=reason,
                )
            )
            continue
        candidate = _candidate(record, revision, logical_id)
        key = _row_key(revision["table"], revision["primary_key"])
        row_generations.setdefault(key, set()).add(revision["generation"])
        if record["source_id"] not in row_sources.setdefault(key, []):
            row_sources[key].append(record["source_id"])
        previous_digest = prior_digests.get(logical_id)
        if logical_id in selected:
            previous_digest = _metrics_digest(selected[logical_id][0]["payload"]["row"])
        selected[logical_id] = (record, revision, candidate, previous_digest)
        revision_sources.setdefault(logical_id, []).append(record["source_id"])
        prior_digests[logical_id] = _metrics_digest(payload["row"])

    for key in sorted(row_generations):
        generations = row_generations[key]
        if len(generations) > 1:
            table, primary_key = key.split("\x1f", 1)
            diagnostics.append(
                _diagnostic(
                    "usage_row_id_reuse",
                    "warning",
                    "database row ID was reused by a different database generation",
                    list(row_sources.get(key, [])),
                    table=table,
                    primary_key=primary_key,
                    generations=sorted(generations),
                )
            )

    candidates: list[AccountingCandidate] = []
    usage_rows: list[UsageRow] = []
    accounted, accounted_by_agent = _hydrate_accounted(source, inherited_calls)
    logical_calls = {
        key: value.copy()
        for key, value in inherited_calls.items()
        if isinstance(key, str) and isinstance(value, dict)
    }
    unknown_provider_ids: list[str] = []
    for logical_id, (record, revision, candidate, previous_digest) in selected.items():
        payload = record["payload"]
        row = payload["row"]
        assert isinstance(row, dict)
        _attach_revision_sources(candidate, revision_sources[logical_id])
        previous = logical_calls.get(logical_id)
        if isinstance(previous, dict):
            old_metrics = _metrics_from_call(previous)
            _add_metrics(accounted, old_metrics, sign=-1)
            previous_agent = previous.get("agent_id")
            old_agent = _agent_key(previous_agent if isinstance(previous_agent, str) else None)
            if old_agent in accounted_by_agent:
                _add_metrics(accounted_by_agent[old_agent], old_metrics, sign=-1)
        digest = _metrics_digest(row)
        if _row_is_incompatible(row):
            details: dict[str, Any] = {
                "logical_call_id": logical_id,
                "metrics_digest": digest,
            }
            if previous_digest is not None:
                details["prior_metrics_digest"] = previous_digest
            diagnostics.append(
                _diagnostic(
                    "usage_revision_conflict",
                    "error",
                    "incompatible metrics for one logical call; quarantined",
                    revision_sources[logical_id],
                    **details,
                )
            )
            candidates.append(candidate)
            logical_calls[logical_id] = _logical_call_entry(
                logical_id, revision, record, row, candidate, quarantined=True
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
        session_id = f"copilot-{source_key[:SOURCE_KEY_PREFIX_LEN]}-{record['native_session_id']}"
        usage_row = _usage_row(candidate, session_id, row)
        if usage_row is not None:
            usage_rows.append(usage_row)
            if usage_row.provider_name == "unknown":
                unknown_provider_ids.append(record["source_id"])
        # Account every present metric field, including partial rows that cannot
        # emit a UsageRow.  Mismatch diagnostics therefore describe archived
        # observations, not only exported totals.
        metrics = _row_metrics(row)
        _add_metrics(accounted, metrics)
        agent_key = _agent_key(candidate["agent_id"])
        accounted_by_agent.setdefault(agent_key, _zero_metrics())
        _add_metrics(accounted_by_agent[agent_key], metrics)
        logical_calls[logical_id] = _logical_call_entry(
            logical_id, revision, record, row, candidate, quarantined=False
        )

    if unknown_provider_ids:
        diagnostics.append(
            _diagnostic(
                "unknown_provider",
                "info",
                "database usage rows do not name a provider; UsageRow uses the unknown sentinel",
                unknown_provider_ids,
            )
        )

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

    for record in records:
        payload = record.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "session.shutdown":
            continue
        data = payload.get("data")
        if not isinstance(data, dict):
            diagnostics.append(
                _diagnostic(
                    "capability_gap",
                    "warning",
                    "session.shutdown usage cannot be used to validate accounting",
                    [record["source_id"]],
                    reason="missing_shutdown_data",
                )
            )
            continue
        session_totals = _aggregate_model_metrics(data.get("modelMetrics"))
        if session_totals is None:
            diagnostics.append(
                _diagnostic(
                    "capability_gap",
                    "warning",
                    "session.shutdown usage cannot be used to validate accounting",
                    [record["source_id"]],
                    reason="unparseable_model_metrics",
                )
            )
            continue
        if not _metrics_match(session_totals, accounted):
            diagnostics.append(
                _diagnostic(
                    "shutdown_total_mismatch",
                    "warning",
                    "per-call totals do not equal session.shutdown usage",
                    [record["source_id"]],
                    **_mismatch_details(session_totals, accounted),
                )
            )
        agent_metrics = data.get("agentMetrics")
        if agent_metrics is None:
            continue
        agent_totals = _agent_shutdown_totals(agent_metrics)
        if agent_totals is None:
            diagnostics.append(
                _diagnostic(
                    "capability_gap",
                    "warning",
                    "session.shutdown agentMetrics cannot be used to validate per-agent accounting",
                    [record["source_id"]],
                    reason="unparseable_agent_metrics",
                )
            )
            continue
        for agent_id, expected in agent_totals.items():
            actual = accounted_by_agent.get(agent_id, _zero_metrics())
            if not _metrics_match(expected, actual):
                diagnostics.append(
                    _diagnostic(
                        "shutdown_total_mismatch",
                        "warning",
                        "per-agent totals do not equal session.shutdown agentMetrics",
                        [record["source_id"]],
                        agent_id=agent_id,
                        **_mismatch_details(expected, actual),
                    )
                )

    next_state: dict[str, Any] = {
        "logical_calls": logical_calls,
        "accounted_metrics": accounted,
        "accounted_by_agent": accounted_by_agent,
        "row_generations": {key: sorted(values) for key, values in row_generations.items()},
        "row_sources": row_sources,
    }
    return {
        "usage_rows": usage_rows,
        "candidates": candidates,
        "diagnostics": diagnostics,
    }, next_state
