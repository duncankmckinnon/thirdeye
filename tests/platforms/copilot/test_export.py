"""Behavioral tests for Copilot export eligibility, placement ledger, and queue_exports."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from thirdeye import otel_export
from thirdeye.config import Config, LogfireSettings
from thirdeye.paths import session_dir
from thirdeye.platforms.copilot.archive import commit_batch
from thirdeye.platforms.copilot.constants import PLATFORM_NAME
from thirdeye.platforms.copilot.export import queue_exports
from thirdeye.platforms.copilot.export_state import (
    empty_export_state,
    export_state_path,
    initialize_eligibility,
    load_export_state,
    mark_placement_delivered,
    record_placement,
    update_export_state,
    usage_digest,
)
from thirdeye.platforms.copilot.identity import resolve_sources, stored_session_id
from thirdeye.platforms.copilot.projection import build_projection
from thirdeye.platforms.copilot.transcript import read_transcript
from thirdeye.platforms.copilot.types import Projection, SourceBatch, SourcePaths, SourceRecord
from thirdeye.usage.types import UsageRow

FIXTURES = Path(__file__).parent / "fixtures"
RECON = FIXTURES / "reconciliation-cases"
NATIVE_SESSION_ID = "5a7e8e11-4a6b-49ff-a33e-95d411c4cdd6"
SOURCE_KEY = "a" * 64
GENERATION = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
OBSERVED_AT = "2026-09-10T17:09:00.000Z"
TURN_ONE = f"copilot:turn:{SOURCE_KEY}:{NATIVE_SESSION_ID}:interaction-main-1"
CALL_MATCHED = (
    f"copilot:call:{SOURCE_KEY}/{NATIVE_SESSION_ID}/a4a17e63-7ba5-422f-8ee9-b495be417328"
)
ACCOUNTING_MATCHED = (
    f"copilot:usage:{SOURCE_KEY}:assistant_usage_events:"
    f"sha256%3Abbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:13"
)
ACCOUNTING_UNMATCHED = (
    f"copilot:usage:{SOURCE_KEY}:assistant_usage_events:"
    f"sha256%3Abbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:14"
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _record(source_id: str, *, native_session_id: str = NATIVE_SESSION_ID) -> SourceRecord:
    return {
        "source_id": source_id,
        "source_kind": "transcript",
        "native_session_id": native_session_id,
        "ts": "2026-09-10T17:08:24.000Z",
        "observed_at": "2026-09-10T17:08:25.000Z",
        "payload": {"schema_version": 1, "type": "user.message"},
        "locator": {"file": "events.jsonl", "offset": 0},
    }


def _batch(paths: SourcePaths, records: list[SourceRecord]) -> SourceBatch:
    return {
        "source_key": paths["source_key"],
        "native_session_id": NATIVE_SESSION_ID,
        "cwd": "/proj",
        "records": records,
        "next_cursor": {"generation": 1},
        "diagnostics": [],
    }


def _usage_row(**overrides: Any) -> UsageRow:
    fields = dict(
        session_id="stored-session",
        seq=0,
        call_id=ACCOUNTING_MATCHED,
        ts="2026-09-10T17:08:24.498Z",
        platform=PLATFORM_NAME,
        provider_name="unknown",
        response_model="gpt-5.6-luna",
        input_tokens=6452,
        output_tokens=107,
        cache_creation_input_tokens=6449,
    )
    fields.update(overrides)
    return UsageRow(**fields)


def _main_turn(
    *,
    turn_id: str = TURN_ONE,
    status: str = "completed",
    accounting_calls: list[dict[str, Any]] | None = None,
    subagents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "turn_id": turn_id,
        "start_ts": "2026-09-10T17:08:24.000Z",
        "end_ts": "2026-09-10T17:08:28.000Z",
        "input_message": "hello",
        "output_message": "done",
        "status": status,
        "llm_calls": [
            {
                "call_id": CALL_MATCHED,
                "provider": "unknown",
                "model": "gpt-5.6-luna",
                "start_ts": "2026-09-10T17:08:24.503Z",
                "end_ts": "2026-09-10T17:08:24.593Z",
                "input_messages": [],
                "output_messages": [],
                "usage": {},
                "tool_calls": [],
            }
        ],
        "permission_requests": [],
        "subagents": subagents or [],
        "attributes": {"interaction_id": "interaction-main-1"},
        "accounting_calls": accounting_calls or [],
    }


def _accounting_call(
    *,
    accounting_id: str = ACCOUNTING_MATCHED,
    attribution_status: str = "matched",
    call_id: str | None = CALL_MATCHED,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "accounting_id": accounting_id,
        "usage": usage or _usage_row(call_id=accounting_id).to_dict(),
        "attribution_status": attribution_status,
        "agent_id": None,
        "call_id": call_id,
        "attributes": {"copilot.logical_call_id": accounting_id},
    }


def _attribution(
    *,
    logical_call_id: str,
    status: str = "matched",
    stored_turn_id: str | None = TURN_ONE,
    call_id: str | None = None,
    usage_source_id: str = "db-row-1",
) -> dict[str, Any]:
    return {
        "usage_source_id": usage_source_id,
        "logical_call_id": logical_call_id,
        "stored_turn_id": stored_turn_id,
        "agent_id": None,
        "call_id": call_id,
        "status": status,
        "join_kind": "inferred",
        "evidence": ["join_kind:inferred"],
    }


def _projection(**overrides: Any) -> Projection:
    base: Projection = {
        "normalized_events": [],
        "turns": [],
        "usage_rows": [],
        "attributions": [],
        "pending": [],
        "diagnostics": [],
    }
    base.update(overrides)
    return base


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(root=tmp_path / "thirdeye")


@pytest.fixture
def enabled_config(tmp_path: Path) -> Config:
    return Config(
        root=tmp_path / "thirdeye",
        logfire=LogfireSettings(enabled=True, token="fake-token"),
    )


@pytest.fixture
def paths(tmp_path: Path) -> SourcePaths:
    home = tmp_path / "copilot-home"
    home.mkdir()
    return resolve_sources(home)


def _seed_session(config: Config, paths: SourcePaths) -> str:
    commit_batch(config, paths, _batch(paths, [_record("key/a/event-1")]))
    return stored_session_id(paths, NATIVE_SESSION_ID)


def _directory(config: Config, stored: str) -> Path:
    return session_dir(config.root, PLATFORM_NAME, stored)


@pytest.fixture
def export_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {
        "turn": [],
        "turn_accounting": [],
        "session_accounting": [],
    }

    def _turn(*args: Any, **kwargs: Any) -> None:
        calls["turn"].append(args[5])

    def _turn_accounting(*args: Any, **kwargs: Any) -> bool:
        calls["turn_accounting"].append(args[6])
        return True

    def _session_accounting(*args: Any, **kwargs: Any) -> bool:
        calls["session_accounting"].append(args[5])
        return True

    monkeypatch.setattr(otel_export, "export_turn", _turn)
    monkeypatch.setattr(otel_export, "export_turn_accounting", _turn_accounting)
    monkeypatch.setattr(otel_export, "export_session_accounting", _session_accounting)
    return calls


class TestExportState:
    def test_initialize_eligibility_first_activation_excludes_terminal_history(self) -> None:
        state = initialize_eligibility(
            empty_export_state(),
            terminal_turn_ids=["turn-a", "turn-b"],
            accounting_ids=["acct-1"],
            include_history=False,
        )
        assert state["activated"] is True
        assert state["excluded_turn_ids"] == ["turn-a", "turn-b"]
        assert state["excluded_accounting_ids"] == ["acct-1"]

    def test_initialize_eligibility_include_history_keeps_terminal_eligible(self) -> None:
        state = initialize_eligibility(
            empty_export_state(),
            terminal_turn_ids=["turn-a"],
            accounting_ids=["acct-1"],
            include_history=True,
        )
        assert state["activated"] is True
        assert state["excluded_turn_ids"] == []
        assert state["excluded_accounting_ids"] == []

    def test_initialize_eligibility_later_include_history_opts_in(self) -> None:
        first = initialize_eligibility(
            empty_export_state(),
            terminal_turn_ids=["turn-a"],
            accounting_ids=["acct-1"],
            include_history=False,
        )
        second = initialize_eligibility(
            first,
            terminal_turn_ids=["turn-a"],
            accounting_ids=["acct-1"],
            include_history=True,
        )
        assert second["excluded_turn_ids"] == []
        assert second["excluded_accounting_ids"] == []

    def test_record_placement_is_idempotent_for_same_decision(self) -> None:
        usage = _usage_row().to_dict()
        state, entry, accepted = record_placement(
            empty_export_state(),
            accounting_id="acct-1",
            destination="chat-span",
            span_id="call-1",
            usage=usage,
        )
        assert accepted is True
        assert entry is not None
        again, same_entry, same_accepted = record_placement(
            state,
            accounting_id="acct-1",
            destination="chat-span",
            span_id="call-1",
            usage=usage,
        )
        assert same_accepted is True
        assert same_entry == entry

    def test_record_placement_conflicts_on_changed_destination(self) -> None:
        usage = _usage_row().to_dict()
        state, _, accepted = record_placement(
            empty_export_state(),
            accounting_id="acct-1",
            destination="turn-accounting-span",
            span_id="span-a",
            usage=usage,
        )
        assert accepted is True
        conflicted, existing, rejected = record_placement(
            state,
            accounting_id="acct-1",
            destination="chat-span",
            span_id="call-1",
            usage=usage,
        )
        assert rejected is False
        assert existing is not None
        assert existing["destination"] == "turn-accounting-span"
        assert "acct-1" in conflicted["conflicts"]

    def test_record_placement_conflicts_on_usage_correction(self) -> None:
        usage = _usage_row().to_dict()
        state, _, accepted = record_placement(
            empty_export_state(),
            accounting_id="acct-1",
            destination="chat-span",
            span_id="call-1",
            usage=usage,
        )
        assert accepted is True
        corrected = dict(usage)
        corrected["output_tokens"] = 999
        conflicted, _, rejected = record_placement(
            state,
            accounting_id="acct-1",
            destination="chat-span",
            span_id="call-1",
            usage=corrected,
        )
        assert rejected is False
        assert "acct-1" in conflicted["conflicts"]
        assert conflicted["conflicts"]["acct-1"]["candidate"]["usage_digest"] == usage_digest(
            corrected
        )


class TestQueueExports:
    def test_unknown_session_raises(self, enabled_config: Config) -> None:
        projection = _projection()
        with pytest.raises(ValueError, match="unknown Copilot session"):
            queue_exports(enabled_config, "missing-session", projection)

    def test_unconfigured_returns_zero_but_activates(
        self, config: Config, paths: SourcePaths
    ) -> None:
        stored = _seed_session(config, paths)
        projection = _projection(
            turns=[_main_turn(accounting_calls=[_accounting_call()])],
            usage_rows=[_usage_row()],
            attributions=[_attribution(logical_call_id=ACCOUNTING_MATCHED)],
        )
        queued = queue_exports(config, stored, projection)
        assert queued == 0
        state = load_export_state(config, stored)
        assert state["activated"] is True
        assert TURN_ONE in state["excluded_turn_ids"]
        assert ACCOUNTING_MATCHED in state["excluded_accounting_ids"]

    def test_first_activation_without_history_queues_nothing(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        stored = _seed_session(enabled_config, paths)
        projection = _projection(
            turns=[
                _main_turn(
                    accounting_calls=[
                        _accounting_call(),
                        _accounting_call(
                            accounting_id=ACCOUNTING_UNMATCHED,
                            attribution_status="ambiguous",
                            call_id=None,
                            usage=_usage_row(call_id=ACCOUNTING_UNMATCHED).to_dict(),
                        ),
                    ]
                )
            ],
            usage_rows=[
                _usage_row(),
                _usage_row(call_id=ACCOUNTING_UNMATCHED, input_tokens=6587, output_tokens=5),
            ],
            attributions=[
                _attribution(logical_call_id=ACCOUNTING_MATCHED),
                _attribution(
                    logical_call_id=ACCOUNTING_UNMATCHED,
                    status="ambiguous",
                    call_id=None,
                ),
            ],
        )
        queued = queue_exports(enabled_config, stored, projection)
        assert queued == 0
        assert export_calls["turn"] == []
        assert export_calls["turn_accounting"] == []
        assert export_calls["session_accounting"] == []

    def test_include_history_queues_terminal_turn_and_matched_accounting(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        stored = _seed_session(enabled_config, paths)
        projection = _projection(
            turns=[_main_turn(accounting_calls=[_accounting_call()])],
            usage_rows=[_usage_row()],
            attributions=[_attribution(logical_call_id=ACCOUNTING_MATCHED)],
        )
        queued = queue_exports(enabled_config, stored, projection, include_history=True)
        assert queued == 1
        assert len(export_calls["turn"]) == 1
        turn = export_calls["turn"][0]
        assert turn["turn_span_id"]
        assert len(turn["accounting_calls"]) == 1
        assert turn["accounting_calls"][0]["accounting_id"] == ACCOUNTING_MATCHED
        assert export_calls["turn_accounting"] == []
        state = load_export_state(enabled_config, stored)
        placement = state["placements"][ACCOUNTING_MATCHED]
        assert placement["destination"] == "chat-span"
        assert placement["span_id"] == CALL_MATCHED

    def test_ambiguous_terminal_usage_queues_turn_accounting_job(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        stored = _seed_session(enabled_config, paths)
        projection = _projection(
            turns=[
                _main_turn(
                    accounting_calls=[
                        _accounting_call(
                            accounting_id=ACCOUNTING_UNMATCHED,
                            attribution_status="ambiguous",
                            call_id=None,
                            usage=_usage_row(call_id=ACCOUNTING_UNMATCHED).to_dict(),
                        )
                    ]
                )
            ],
            usage_rows=[_usage_row(call_id=ACCOUNTING_UNMATCHED, input_tokens=6587, output_tokens=5)],
            attributions=[
                _attribution(
                    logical_call_id=ACCOUNTING_UNMATCHED,
                    status="ambiguous",
                    call_id=None,
                )
            ],
        )
        queued = queue_exports(enabled_config, stored, projection, include_history=True)
        assert queued == 2
        assert len(export_calls["turn_accounting"]) == 1
        assert export_calls["turn_accounting"][0]["accounting_id"] == ACCOUNTING_UNMATCHED
        assert len(export_calls["turn"]) == 1
        assert export_calls["turn"][0]["accounting_calls"] == []
        state = load_export_state(enabled_config, stored)
        assert state["placements"][ACCOUNTING_UNMATCHED]["destination"] == "turn-accounting-span"

    def test_session_accounting_for_unattached_ambiguous_usage(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        stored = _seed_session(enabled_config, paths)
        usage = _usage_row(call_id=ACCOUNTING_UNMATCHED, input_tokens=6587, output_tokens=5)
        projection = _projection(
            turns=[_main_turn()],
            usage_rows=[usage],
            attributions=[
                _attribution(
                    logical_call_id=ACCOUNTING_UNMATCHED,
                    status="ambiguous",
                    stored_turn_id=None,
                    call_id=None,
                )
            ],
        )
        queued = queue_exports(enabled_config, stored, projection, include_history=True)
        assert queued == 2
        assert len(export_calls["session_accounting"]) == 1
        assert export_calls["session_accounting"][0]["accounting_id"] == ACCOUNTING_UNMATCHED
        state = load_export_state(enabled_config, stored)
        assert state["placements"][ACCOUNTING_UNMATCHED]["destination"] == "session-accounting-span"

    def test_pending_and_conflicting_attributions_are_not_exported(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        stored = _seed_session(enabled_config, paths)
        pending_usage = _usage_row(call_id="pending-call", input_tokens=100, output_tokens=1)
        conflicting_usage = _usage_row(call_id="conflict-call", input_tokens=200, output_tokens=2)
        projection = _projection(
            turns=[
                _main_turn(
                    accounting_calls=[
                        _accounting_call(
                            accounting_id="pending-call",
                            attribution_status="pending",
                            call_id=None,
                            usage=pending_usage.to_dict(),
                        )
                    ]
                )
            ],
            usage_rows=[pending_usage, conflicting_usage],
            attributions=[
                _attribution(logical_call_id="pending-call", status="pending", call_id=None),
                _attribution(
                    logical_call_id="conflict-call",
                    status="conflicting",
                    stored_turn_id=None,
                    call_id=None,
                ),
            ],
        )
        queued = queue_exports(enabled_config, stored, projection, include_history=True)
        assert queued == 1
        assert export_calls["turn_accounting"] == []
        assert export_calls["session_accounting"] == []
        state = load_export_state(enabled_config, stored)
        assert "pending-call" not in state["placements"]
        assert "conflict-call" not in state["placements"]

    def test_fallback_placement_prevents_later_chat_relocation(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        stored = _seed_session(enabled_config, paths)
        ambiguous_projection = _projection(
            turns=[
                _main_turn(
                    accounting_calls=[
                        _accounting_call(
                            accounting_id=ACCOUNTING_UNMATCHED,
                            attribution_status="ambiguous",
                            call_id=None,
                            usage=_usage_row(call_id=ACCOUNTING_UNMATCHED).to_dict(),
                        )
                    ]
                )
            ],
            usage_rows=[_usage_row(call_id=ACCOUNTING_UNMATCHED, input_tokens=6587, output_tokens=5)],
            attributions=[
                _attribution(
                    logical_call_id=ACCOUNTING_UNMATCHED,
                    status="ambiguous",
                    call_id=None,
                )
            ],
        )
        queue_exports(enabled_config, stored, ambiguous_projection, include_history=True)
        export_calls["turn"].clear()
        export_calls["turn_accounting"].clear()

        matched_projection = _projection(
            turns=[
                _main_turn(
                    accounting_calls=[
                        _accounting_call(
                            accounting_id=ACCOUNTING_UNMATCHED,
                            attribution_status="matched",
                            call_id=CALL_MATCHED,
                            usage=_usage_row(call_id=ACCOUNTING_UNMATCHED).to_dict(),
                        )
                    ]
                )
            ],
            usage_rows=[_usage_row(call_id=ACCOUNTING_UNMATCHED, input_tokens=6587, output_tokens=5)],
            attributions=[_attribution(logical_call_id=ACCOUNTING_UNMATCHED, call_id=CALL_MATCHED)],
        )
        queue_exports(enabled_config, stored, matched_projection, include_history=True)

        state = load_export_state(enabled_config, stored)
        placement = state["placements"][ACCOUNTING_UNMATCHED]
        assert placement["destination"] == "turn-accounting-span"
        assert ACCOUNTING_UNMATCHED in state["conflicts"]
        assert export_calls["turn_accounting"] == []
        turn = export_calls["turn"][0]
        assert turn["accounting_calls"] == []

    def test_emitted_placement_skips_requeue(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        stored = _seed_session(enabled_config, paths)

        def _mark_delivered(state: dict[str, Any]) -> dict[str, Any]:
            placed, _, _ = record_placement(
                state,
                accounting_id=ACCOUNTING_UNMATCHED,
                destination="turn-accounting-span",
                span_id=f"accounting:{stored}:{TURN_ONE}:{ACCOUNTING_UNMATCHED}",
                usage=_usage_row(call_id=ACCOUNTING_UNMATCHED).to_dict(),
            )
            return mark_placement_delivered(placed, ACCOUNTING_UNMATCHED)

        update_export_state(enabled_config, stored, _mark_delivered)
        update_export_state(
            enabled_config,
            stored,
            lambda state: initialize_eligibility(
                state,
                terminal_turn_ids=[TURN_ONE],
                accounting_ids=[ACCOUNTING_UNMATCHED],
                include_history=True,
            ),
        )

        projection = _projection(
            turns=[
                _main_turn(
                    accounting_calls=[
                        _accounting_call(
                            accounting_id=ACCOUNTING_UNMATCHED,
                            attribution_status="ambiguous",
                            call_id=None,
                            usage=_usage_row(call_id=ACCOUNTING_UNMATCHED).to_dict(),
                        )
                    ]
                )
            ],
            usage_rows=[_usage_row(call_id=ACCOUNTING_UNMATCHED, input_tokens=6587, output_tokens=5)],
            attributions=[
                _attribution(
                    logical_call_id=ACCOUNTING_UNMATCHED,
                    status="ambiguous",
                    call_id=None,
                )
            ],
        )
        queued = queue_exports(enabled_config, stored, projection, include_history=True)
        assert queued == 1
        assert export_calls["turn_accounting"] == []
        assert len(export_calls["turn"]) == 1

    def test_late_terminal_turn_becomes_eligible_without_history_opt_in(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        stored = _seed_session(enabled_config, paths)
        first_turn = _main_turn(turn_id="turn-first")
        queue_exports(
            enabled_config,
            stored,
            _projection(turns=[first_turn], usage_rows=[], attributions=[]),
        )
        export_calls["turn"].clear()

        second_turn = _main_turn(turn_id="turn-second")
        queued = queue_exports(
            enabled_config,
            stored,
            _projection(turns=[first_turn, second_turn], usage_rows=[], attributions=[]),
        )
        assert queued == 1
        assert export_calls["turn"][0]["turn_id"] == "turn-second"

    def test_accounting_job_failure_records_error(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stored = _seed_session(enabled_config, paths)
        monkeypatch.setattr(otel_export, "export_turn", lambda *args, **kwargs: None)
        monkeypatch.setattr(otel_export, "export_turn_accounting", lambda *args, **kwargs: False)
        monkeypatch.setattr(otel_export, "export_session_accounting", lambda *args, **kwargs: False)

        projection = _projection(
            turns=[
                _main_turn(
                    accounting_calls=[
                        _accounting_call(
                            accounting_id=ACCOUNTING_UNMATCHED,
                            attribution_status="ambiguous",
                            call_id=None,
                            usage=_usage_row(call_id=ACCOUNTING_UNMATCHED).to_dict(),
                        )
                    ]
                )
            ],
            usage_rows=[_usage_row(call_id=ACCOUNTING_UNMATCHED, input_tokens=6587, output_tokens=5)],
            attributions=[
                _attribution(
                    logical_call_id=ACCOUNTING_UNMATCHED,
                    status="ambiguous",
                    call_id=None,
                )
            ],
        )
        queued = queue_exports(enabled_config, stored, projection, include_history=True)
        assert queued == 1
        state = load_export_state(enabled_config, stored)
        assert state["placements"][ACCOUNTING_UNMATCHED]["last_error"] == (
            "accounting job was not queued"
        )

    def test_restart_preserves_ledger_and_boundary(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        stored = _seed_session(enabled_config, paths)
        projection = _projection(
            turns=[_main_turn(accounting_calls=[_accounting_call()])],
            usage_rows=[_usage_row()],
            attributions=[_attribution(logical_call_id=ACCOUNTING_MATCHED)],
        )
        queue_exports(enabled_config, stored, projection, include_history=True)
        ledger_path = export_state_path(_directory(enabled_config, stored))
        saved = json.loads(ledger_path.read_text(encoding="utf-8"))

        export_calls["turn"].clear()
        reloaded = Config(
            root=enabled_config.root,
            logfire=LogfireSettings(enabled=True, token="fake-token"),
        )
        queued = queue_exports(reloaded, stored, projection, include_history=True)
        assert queued == 1
        assert json.loads(ledger_path.read_text(encoding="utf-8"))["placements"] == saved["placements"]

    def test_non_terminal_turns_are_not_exported(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        stored = _seed_session(enabled_config, paths)
        projection = _projection(
            turns=[_main_turn(status="in_progress", accounting_calls=[_accounting_call()])],
            usage_rows=[_usage_row()],
            attributions=[_attribution(logical_call_id=ACCOUNTING_MATCHED)],
        )
        queued = queue_exports(enabled_config, stored, projection, include_history=True)
        assert queued == 0
        assert export_calls["turn"] == []


def _drain_cli_transcript(home: Path) -> list[SourceRecord]:
    session_dir = home / "session-state" / NATIVE_SESSION_ID
    session_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "events.jsonl", session_dir / "events.jsonl")
    (session_dir / "workspace.yaml").write_text("cwd: /sanitized/workspace\n", encoding="utf-8")
    paths = resolve_sources(home)
    cursor: dict[str, Any] = {}
    records: list[SourceRecord] = []
    while True:
        slice_ = read_transcript(paths, NATIVE_SESSION_ID, cursor)
        records.extend(slice_["records"])
        cursor = slice_["next_cursor"]
        if slice_["exhausted"]:
            break
    return [record for record in records if record["source_kind"] == "transcript"]


def _source_key(records: list[SourceRecord]) -> str:
    for record in records:
        if record["source_kind"] == "transcript":
            return record["source_id"].split("/", 1)[0]
    pytest.fail("expected transcript record")


def _usage_record(
    row: dict[str, Any],
    *,
    content_revision: str,
    source_key: str = SOURCE_KEY,
) -> SourceRecord:
    primary_key = row["id"]
    source_id = (
        f"copilot-db:{source_key}:{NATIVE_SESSION_ID}:assistant_usage_events:"
        f"{primary_key}:{content_revision}"
    )
    return {
        "source_id": source_id,
        "source_kind": "database",
        "native_session_id": NATIVE_SESSION_ID,
        "ts": row.get("created_at"),
        "observed_at": OBSERVED_AT,
        "payload": {"table": "assistant_usage_events", "row": row},
        "locator": {
            "database": "/example/.copilot/session-store.db",
            "table": "assistant_usage_events",
            "primary_key": primary_key,
            "content_revision": content_revision,
            "generation": GENERATION,
        },
    }


def _six_call_records(*, source_key: str = SOURCE_KEY) -> list[SourceRecord]:
    rows = _load_json(FIXTURES / "assistant-usage-events.json")
    revisions = {
        call["row_id"]: call["usage_source_id"].rsplit(":", 1)[-1]
        for call in _load_json(RECON / "observed-six-calls.json")["calls"]
    }
    return [
        _usage_record(row, content_revision=revisions[row["id"]], source_key=source_key)
        for row in rows
    ]


def test_build_projection_fixture_export_with_history(
    tmp_path: Path,
    export_calls: dict[str, list[Any]],
) -> None:
    home = tmp_path / "copilot-home"
    home.mkdir()
    paths = resolve_sources(home)
    config = Config(
        root=tmp_path / "thirdeye",
        logfire=LogfireSettings(enabled=True, token="fake-token"),
    )
    commit_batch(config, paths, _batch(paths, _drain_cli_transcript(home)))
    stored = stored_session_id(paths, NATIVE_SESSION_ID)

    source_key = _source_key(_drain_cli_transcript(home))
    records = _drain_cli_transcript(home) + _six_call_records(source_key=source_key)
    projection, _ = build_projection(records, {})

    queued = queue_exports(config, stored, projection, include_history=True)
    assert queued >= len(projection["turns"])
    assert export_calls["turn"]
    matched = [
        item
        for turn in projection["turns"]
        for item in turn.get("accounting_calls") or []
        if item.get("attribution_status") == "matched"
    ]
    assert matched
    state = load_export_state(config, stored)
    for item in matched:
        assert item["accounting_id"] in state["placements"]
        assert state["placements"][item["accounting_id"]]["destination"] == "chat-span"
