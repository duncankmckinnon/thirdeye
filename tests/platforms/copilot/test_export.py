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
from thirdeye.platforms.copilot import export_transport
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
CALL_MATCHED = f"copilot:call:{SOURCE_KEY}/{NATIVE_SESSION_ID}/a4a17e63-7ba5-422f-8ee9-b495be417328"
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

    def _turn(*args: Any, **kwargs: Any) -> bool:
        calls["turn"].append(args[4])
        return True

    def _turn_accounting(*args: Any, **kwargs: Any) -> bool:
        calls["turn_accounting"].append(args[5])
        return True

    def _session_accounting(*args: Any, **kwargs: Any) -> bool:
        calls["session_accounting"].append(args[4])
        return True

    monkeypatch.setattr(export_transport, "queue_turn", _turn)
    monkeypatch.setattr(export_transport, "queue_turn_accounting", _turn_accounting)
    monkeypatch.setattr(export_transport, "queue_session_accounting", _session_accounting)
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

    def test_record_placement_replaces_undelivered_placement_on_changed_destination(
        self,
    ) -> None:
        """Nothing was ever confirmed delivered, so a corrected destination
        just replaces the queued-but-unconfirmed placement outright."""
        usage = _usage_row().to_dict()
        state, _, accepted = record_placement(
            empty_export_state(),
            accounting_id="acct-1",
            destination="turn-accounting-span",
            span_id="span-a",
            usage=usage,
        )
        assert accepted is True
        replaced, entry, accepted_again = record_placement(
            state,
            accounting_id="acct-1",
            destination="chat-span",
            span_id="call-1",
            usage=usage,
        )
        assert accepted_again is True
        assert entry is not None
        assert entry["destination"] == "chat-span"
        assert replaced["placements"]["acct-1"]["destination"] == "chat-span"
        assert "acct-1" not in replaced["conflicts"]

    def test_record_placement_replaces_undelivered_placement_on_usage_correction(
        self,
    ) -> None:
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
        replaced, entry, accepted_again = record_placement(
            state,
            accounting_id="acct-1",
            destination="chat-span",
            span_id="call-1",
            usage=corrected,
        )
        assert accepted_again is True
        assert entry is not None
        assert entry["usage_digest"] == usage_digest(corrected)
        assert "acct-1" not in replaced["conflicts"]

    def test_record_placement_conflicts_on_changed_destination_after_delivery(self) -> None:
        usage = _usage_row().to_dict()
        state, _, accepted = record_placement(
            empty_export_state(),
            accounting_id="acct-1",
            destination="turn-accounting-span",
            span_id="span-a",
            usage=usage,
            delivered=True,
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

    def test_record_placement_conflicts_on_usage_correction_after_delivery(self) -> None:
        usage = _usage_row().to_dict()
        state, _, accepted = record_placement(
            empty_export_state(),
            accounting_id="acct-1",
            destination="chat-span",
            span_id="call-1",
            usage=usage,
            delivered=True,
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

    def test_record_placement_self_heals_emitted_flag_when_delivery_confirmed(self) -> None:
        """Same destination/span/usage as before, but the caller now has a
        fresh durable-claim read showing delivery succeeded: the ledger's own
        ``emitted`` flag was never told directly, so this is the only place
        it catches up."""
        usage = _usage_row().to_dict()
        state, entry, _ = record_placement(
            empty_export_state(),
            accounting_id="acct-1",
            destination="chat-span",
            span_id="call-1",
            usage=usage,
        )
        assert entry is not None
        assert entry["emitted"] is False
        healed, healed_entry, accepted = record_placement(
            state,
            accounting_id="acct-1",
            destination="chat-span",
            span_id="call-1",
            usage=usage,
            delivered=True,
        )
        assert accepted is True
        assert healed_entry is not None
        assert healed_entry["emitted"] is True
        assert healed["placements"]["acct-1"]["emitted"] is True

    def test_record_placement_conflicts_when_old_job_is_claimed_in_flight(self) -> None:
        """The old job's own worker could deliver at any moment -- relocating
        while it is ``"claimed"`` is proven neither safe nor already known to
        be delivered, so it must be quarantined rather than guessed either
        way, exactly like a confirmed-delivered correction."""
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
            old_job_state="claimed",
        )
        assert rejected is False
        assert existing is not None
        assert existing["destination"] == "turn-accounting-span"
        assert "acct-1" in conflicted["conflicts"]
        assert "in flight" in conflicted["conflicts"]["acct-1"]["reason"]

    def test_record_placement_replaces_when_old_job_proven_inert(self) -> None:
        """Unlike ``"claimed"``, a ``None`` (never queued), ``"queued"``
        (untouched by any worker), or ``"failed"`` (permanently gave up, will
        never be retried by the transport) old job state proves nothing is
        in flight, so relocating is exactly as safe as it always was for an
        undelivered placement."""
        usage = _usage_row().to_dict()
        for old_state in (None, "queued", "failed"):
            state, _, accepted = record_placement(
                empty_export_state(),
                accounting_id="acct-1",
                destination="turn-accounting-span",
                span_id="span-a",
                usage=usage,
            )
            assert accepted is True
            replaced, entry, accepted_again = record_placement(
                state,
                accounting_id="acct-1",
                destination="chat-span",
                span_id="call-1",
                usage=usage,
                old_job_state=old_state,
            )
            assert accepted_again is True, old_state
            assert entry is not None
            assert entry["destination"] == "chat-span"
            assert "acct-1" not in replaced["conflicts"]


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

    def test_unsupported_ledger_schema_version_fails_closed(
        self, config: Config, paths: SourcePaths
    ) -> None:
        """A missing ledger file legitimately starts from empty state, but a
        present file with an unrecognized ``schema_version`` must never be
        silently treated as empty -- that would forget every recorded
        placement and risk emitting tokens at a second location."""
        stored = _seed_session(config, paths)
        directory = _directory(config, stored)
        directory.mkdir(parents=True, exist_ok=True)
        export_state_path(directory).write_text(
            json.dumps({"schema_version": 999, "placements": {"acct-1": {"emitted": True}}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unsupported Copilot export ledger schema_version"):
            load_export_state(config, stored)
        projection = _projection(turns=[_main_turn()])
        with pytest.raises(ValueError, match="unsupported Copilot export ledger schema_version"):
            queue_exports(config, stored, projection)

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
            usage_rows=[
                _usage_row(call_id=ACCOUNTING_UNMATCHED, input_tokens=6587, output_tokens=5)
            ],
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

    def _ambiguous_then_matched_projections(self) -> tuple[Projection, Projection]:
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
            usage_rows=[
                _usage_row(call_id=ACCOUNTING_UNMATCHED, input_tokens=6587, output_tokens=5)
            ],
            attributions=[
                _attribution(
                    logical_call_id=ACCOUNTING_UNMATCHED,
                    status="ambiguous",
                    call_id=None,
                )
            ],
        )
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
            usage_rows=[
                _usage_row(call_id=ACCOUNTING_UNMATCHED, input_tokens=6587, output_tokens=5)
            ],
            attributions=[_attribution(logical_call_id=ACCOUNTING_UNMATCHED, call_id=CALL_MATCHED)],
        )
        return ambiguous_projection, matched_projection

    def test_pre_delivery_correction_relocates_to_chat_without_conflict(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        """Nothing was ever confirmed delivered for the fallback placement
        (no real worker ran), so improved local matching is still free to
        relocate the tokens onto the now-known chat span."""
        stored = _seed_session(enabled_config, paths)
        ambiguous_projection, matched_projection = self._ambiguous_then_matched_projections()
        queue_exports(enabled_config, stored, ambiguous_projection, include_history=True)
        export_calls["turn"].clear()
        export_calls["turn_accounting"].clear()

        queue_exports(enabled_config, stored, matched_projection, include_history=True)

        state = load_export_state(enabled_config, stored)
        placement = state["placements"][ACCOUNTING_UNMATCHED]
        assert placement["destination"] == "chat-span"
        assert placement["span_id"] == CALL_MATCHED
        assert ACCOUNTING_UNMATCHED not in state["conflicts"]
        turn = export_calls["turn"][0]
        assert turn["accounting_calls"][0]["accounting_id"] == ACCOUNTING_UNMATCHED

    def test_fallback_placement_prevents_later_chat_relocation_after_confirmed_delivery(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        """Once the durable transport claim shows the fallback span was
        actually flushed, later local matching can no longer move those
        tokens onto the chat span — that would double the emitted total."""
        stored = _seed_session(enabled_config, paths)
        ambiguous_projection, matched_projection = self._ambiguous_then_matched_projections()
        queue_exports(enabled_config, stored, ambiguous_projection, include_history=True)

        directory = _directory(enabled_config, stored)
        export_transport._mark_delivered(directory, ACCOUNTING_UNMATCHED)

        export_calls["turn"].clear()
        export_calls["turn_accounting"].clear()

        queue_exports(enabled_config, stored, matched_projection, include_history=True)

        state = load_export_state(enabled_config, stored)
        placement = state["placements"][ACCOUNTING_UNMATCHED]
        assert placement["destination"] == "turn-accounting-span"
        assert placement["emitted"] is True
        assert ACCOUNTING_UNMATCHED in state["conflicts"]
        assert export_calls["turn_accounting"] == []
        turn = export_calls["turn"][0]
        assert turn["accounting_calls"] == []

    def test_confirmed_turn_delivery_routes_new_match_to_fallback(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        """Even with no prior accounting placement at all, a chat span that
        the durable turn claim shows was already flushed is immutable: a
        newly-matched usage row must land on a turn-accounting fallback
        span, never on that already-sent chat span."""
        stored = _seed_session(enabled_config, paths)
        turn_only_projection = _projection(turns=[_main_turn()])
        queue_exports(enabled_config, stored, turn_only_projection, include_history=True)

        directory = _directory(enabled_config, stored)
        claim_path = otel_export._turn_claim_path(directory, TURN_ONE)
        claim_path.parent.mkdir(parents=True, exist_ok=True)
        claim_path.write_text("sent", encoding="utf-8", newline="\n")

        export_calls["turn"].clear()
        _, matched_projection = self._ambiguous_then_matched_projections()
        queue_exports(enabled_config, stored, matched_projection, include_history=True)

        state = load_export_state(enabled_config, stored)
        placement = state["placements"][ACCOUNTING_UNMATCHED]
        assert placement["destination"] == "turn-accounting-span"
        assert len(export_calls["turn_accounting"]) == 1
        turn = export_calls["turn"][0]
        assert turn["accounting_calls"] == []

    def test_turn_claim_confirms_previously_embedded_chat_accounting(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        """The turn acknowledgement is sufficient for accounting that the
        Copilot ledger already placed inside that turn's chat span. This
        survives a crash before the shared exporter writes its secondary
        per-accounting acknowledgement."""
        stored = _seed_session(enabled_config, paths)
        _, matched_projection = self._ambiguous_then_matched_projections()
        queue_exports(enabled_config, stored, matched_projection, include_history=True)

        directory = _directory(enabled_config, stored)
        claim_path = otel_export._turn_claim_path(directory, TURN_ONE)
        claim_path.parent.mkdir(parents=True, exist_ok=True)
        claim_path.write_text("sent", encoding="utf-8", newline="\n")
        export_calls["turn"].clear()
        export_calls["turn_accounting"].clear()

        queue_exports(enabled_config, stored, matched_projection, include_history=True)

        placement = load_export_state(enabled_config, stored)["placements"][ACCOUNTING_UNMATCHED]
        assert placement["destination"] == "chat-span"
        assert placement["emitted"] is True
        assert export_calls["turn_accounting"] == []

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
            usage_rows=[
                _usage_row(call_id=ACCOUNTING_UNMATCHED, input_tokens=6587, output_tokens=5)
            ],
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
        monkeypatch.setattr(export_transport, "queue_turn", lambda *args, **kwargs: True)
        monkeypatch.setattr(
            export_transport, "queue_turn_accounting", lambda *args, **kwargs: False
        )
        monkeypatch.setattr(
            export_transport, "queue_session_accounting", lambda *args, **kwargs: False
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
            usage_rows=[
                _usage_row(call_id=ACCOUNTING_UNMATCHED, input_tokens=6587, output_tokens=5)
            ],
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

    def test_turn_job_failure_is_not_counted_and_is_recorded(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A Copilot turn spawn/write failure must not be counted as a
        successful queue."""
        stored = _seed_session(enabled_config, paths)
        monkeypatch.setattr(export_transport, "queue_turn", lambda *args, **kwargs: False)

        projection = _projection(turns=[_main_turn()])
        queued = queue_exports(enabled_config, stored, projection, include_history=True)
        assert queued == 0
        state = load_export_state(enabled_config, stored)
        assert state["turn_errors"][TURN_ONE] == "turn export job was not queued"

    def test_errors_clear_once_a_retry_succeeds(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stale `last_error`/turn error from an earlier failed attempt must
        not linger once a later reconciliation successfully queues the job."""
        stored = _seed_session(enabled_config, paths)
        monkeypatch.setattr(export_transport, "queue_turn", lambda *args, **kwargs: False)
        monkeypatch.setattr(
            export_transport, "queue_turn_accounting", lambda *args, **kwargs: False
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
            usage_rows=[
                _usage_row(call_id=ACCOUNTING_UNMATCHED, input_tokens=6587, output_tokens=5)
            ],
            attributions=[
                _attribution(logical_call_id=ACCOUNTING_UNMATCHED, status="ambiguous", call_id=None)
            ],
        )
        queue_exports(enabled_config, stored, projection, include_history=True)
        state = load_export_state(enabled_config, stored)
        assert state["turn_errors"][TURN_ONE] == "turn export job was not queued"
        assert state["placements"][ACCOUNTING_UNMATCHED]["last_error"] == (
            "accounting job was not queued"
        )

        monkeypatch.setattr(export_transport, "queue_turn", lambda *args, **kwargs: True)
        monkeypatch.setattr(export_transport, "queue_turn_accounting", lambda *args, **kwargs: True)
        queue_exports(enabled_config, stored, projection, include_history=True)

        state = load_export_state(enabled_config, stored)
        assert TURN_ONE not in state["turn_errors"]
        assert state["placements"][ACCOUNTING_UNMATCHED]["last_error"] is None

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
        assert (
            json.loads(ledger_path.read_text(encoding="utf-8"))["placements"] == saved["placements"]
        )

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

    def test_open_interaction_accounting_becomes_eligible_once_turn_completes(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        """First activation must only wall off *already terminal* history.

        A usage row owned by an interaction that is still open at first
        activation is not history yet — it must stay eligible once that
        interaction later completes, without an explicit ``--export``.
        """
        stored = _seed_session(enabled_config, paths)
        open_projection = _projection(
            turns=[
                _main_turn(
                    status="in_progress",
                    accounting_calls=[_accounting_call()],
                )
            ],
            usage_rows=[_usage_row()],
            attributions=[_attribution(logical_call_id=ACCOUNTING_MATCHED)],
        )
        queued = queue_exports(enabled_config, stored, open_projection)
        assert queued == 0

        completed_projection = _projection(
            turns=[_main_turn(status="completed", accounting_calls=[_accounting_call()])],
            usage_rows=[_usage_row()],
            attributions=[_attribution(logical_call_id=ACCOUNTING_MATCHED)],
        )
        queued = queue_exports(enabled_config, stored, completed_projection)
        assert queued == 1
        state = load_export_state(enabled_config, stored)
        assert ACCOUNTING_MATCHED in state["placements"]
        assert state["placements"][ACCOUNTING_MATCHED]["destination"] == "chat-span"

    def test_pending_attribution_on_open_turn_is_eligible_once_resolved(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        """A pending (unresolved) attribution owned by a still-open turn at
        first activation must not be forever excluded once the interaction
        finishes and the attribution later resolves to ambiguous/matched."""
        stored = _seed_session(enabled_config, paths)
        open_projection = _projection(
            turns=[
                _main_turn(
                    status="in_progress",
                    accounting_calls=[
                        _accounting_call(
                            accounting_id=ACCOUNTING_UNMATCHED,
                            attribution_status="pending",
                            call_id=None,
                            usage=_usage_row(call_id=ACCOUNTING_UNMATCHED).to_dict(),
                        )
                    ],
                )
            ],
            usage_rows=[_usage_row(call_id=ACCOUNTING_UNMATCHED)],
            attributions=[
                _attribution(logical_call_id=ACCOUNTING_UNMATCHED, status="pending", call_id=None)
            ],
        )
        queue_exports(enabled_config, stored, open_projection)

        resolved_projection = _projection(
            turns=[
                _main_turn(
                    status="completed",
                    accounting_calls=[
                        _accounting_call(
                            accounting_id=ACCOUNTING_UNMATCHED,
                            attribution_status="ambiguous",
                            call_id=None,
                            usage=_usage_row(call_id=ACCOUNTING_UNMATCHED).to_dict(),
                        )
                    ],
                )
            ],
            usage_rows=[_usage_row(call_id=ACCOUNTING_UNMATCHED)],
            attributions=[
                _attribution(logical_call_id=ACCOUNTING_UNMATCHED, status="ambiguous", call_id=None)
            ],
        )
        queued = queue_exports(enabled_config, stored, resolved_projection)
        assert queued == 2
        state = load_export_state(enabled_config, stored)
        assert ACCOUNTING_UNMATCHED in state["placements"]

    def test_child_turn_accounting_stays_excluded_with_its_main_interaction(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        """A nested child (subagent) turn is never exported as its own job,
        so its own id is never a member of the boundary lists built from main
        turns. Eligibility for its accounting must resolve to the owning main
        interaction, not the child's own id -- otherwise usage discovered
        later for a child of an already-excluded historical main turn would
        wrongly become eligible on its own."""
        stored = _seed_session(enabled_config, paths)
        child_turn_id = f"{TURN_ONE}:agent-1"
        first_projection = _projection(
            turns=[_main_turn(subagents=[_main_turn(turn_id=child_turn_id)])],
        )
        queued = queue_exports(enabled_config, stored, first_projection)
        assert queued == 0
        state = load_export_state(enabled_config, stored)
        assert TURN_ONE in state["excluded_turn_ids"]
        assert child_turn_id not in state["excluded_turn_ids"]

        late_child_accounting_projection = _projection(
            turns=[
                _main_turn(
                    subagents=[
                        _main_turn(
                            turn_id=child_turn_id,
                            accounting_calls=[_accounting_call()],
                        )
                    ]
                )
            ],
            usage_rows=[_usage_row()],
            attributions=[
                _attribution(logical_call_id=ACCOUNTING_MATCHED, stored_turn_id=child_turn_id)
            ],
        )
        queued = queue_exports(enabled_config, stored, late_child_accounting_projection)
        assert queued == 0
        assert export_calls["turn_accounting"] == []
        state = load_export_state(enabled_config, stored)
        assert ACCOUNTING_MATCHED not in state["placements"]

    def test_pending_ownerless_usage_is_eligible_once_it_resolves_to_a_completed_turn(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        """At first activation a usage row's attribution may still be
        ``"pending"`` with no ``stored_turn_id`` at all -- unresolved, not
        confirmed ownerless. That must not be locked into the historical
        boundary the way a *resolved* ownerless usage row would be: once it
        later resolves to a real, completed owning turn, it must still be
        exportable, exactly like any other accounting attached late to an
        interaction that was open (or simply not yet observed) at
        activation."""
        stored = _seed_session(enabled_config, paths)
        pending_projection = _projection(
            usage_rows=[_usage_row()],
            attributions=[
                _attribution(
                    logical_call_id=ACCOUNTING_MATCHED,
                    status="pending",
                    stored_turn_id=None,
                    call_id=None,
                )
            ],
        )
        queued = queue_exports(enabled_config, stored, pending_projection)
        assert queued == 0
        state = load_export_state(enabled_config, stored)
        assert ACCOUNTING_MATCHED not in state["excluded_accounting_ids"]

        resolved_projection = _projection(
            turns=[_main_turn(accounting_calls=[_accounting_call()])],
            usage_rows=[_usage_row()],
            attributions=[_attribution(logical_call_id=ACCOUNTING_MATCHED)],
        )
        queued = queue_exports(enabled_config, stored, resolved_projection)
        assert queued == 1  # the turn job; chat-span placement has no separate job
        state = load_export_state(enabled_config, stored)
        assert ACCOUNTING_MATCHED in state["placements"]
        assert state["placements"][ACCOUNTING_MATCHED]["destination"] == "chat-span"

    def test_relocation_cancels_stale_queued_job_at_old_span(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        """A prior reconciliation placed unmatched usage on a turn-accounting
        fallback span and dispatched its deterministic job, which is still
        sitting on disk untouched (``"queued"``) because no worker has
        claimed it yet. When a later reconciliation discovers a real match
        and relocates the tokens to the chat span, the stale fallback job
        must be cancelled -- otherwise the already-dispatched worker could
        still pick it up later and deliver the same tokens a second time."""
        stored = _seed_session(enabled_config, paths)
        old_span_id = f"accounting:{stored}:{TURN_ONE}:{ACCOUNTING_UNMATCHED}"
        job_path = export_transport.job_path(enabled_config.root, old_span_id)
        job_path.parent.mkdir(parents=True, exist_ok=True)
        job_path.write_text(json.dumps({"state": "queued", "attempt": 0}), encoding="utf-8")

        def _seed(state: dict[str, Any]) -> dict[str, Any]:
            placed, _, _ = record_placement(
                state,
                accounting_id=ACCOUNTING_UNMATCHED,
                destination="turn-accounting-span",
                span_id=old_span_id,
                usage=_usage_row(call_id=ACCOUNTING_UNMATCHED).to_dict(),
            )
            return initialize_eligibility(
                placed,
                terminal_turn_ids=[TURN_ONE],
                accounting_ids=[],
                include_history=True,
            )

        update_export_state(enabled_config, stored, _seed)
        assert job_path.exists()

        _, matched_projection = self._ambiguous_then_matched_projections()
        queue_exports(enabled_config, stored, matched_projection, include_history=True)

        assert not job_path.exists()
        state = load_export_state(enabled_config, stored)
        placement = state["placements"][ACCOUNTING_UNMATCHED]
        assert placement["destination"] == "chat-span"
        assert ACCOUNTING_UNMATCHED not in state["conflicts"]

    def test_relocation_is_quarantined_while_old_job_is_claimed(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        """Unlike the merely-``"queued"`` case, a job a worker has already
        ``"claimed"`` could deliver at any moment. Relocating anyway risks a
        duplicate if it does; leaving the old placement in place and
        recording it as an inconclusive conflict is the only safe response,
        the same way a confirmed-delivered correction is quarantined rather
        than guessed."""
        stored = _seed_session(enabled_config, paths)
        old_span_id = f"accounting:{stored}:{TURN_ONE}:{ACCOUNTING_UNMATCHED}"
        job_path = export_transport.job_path(enabled_config.root, old_span_id)
        job_path.parent.mkdir(parents=True, exist_ok=True)
        job_path.write_text(json.dumps({"state": "claimed", "attempt": 0}), encoding="utf-8")
        claim_path = export_transport.claim_path(job_path)
        claim_path.write_text("worker-42", encoding="utf-8")

        def _seed(state: dict[str, Any]) -> dict[str, Any]:
            placed, _, _ = record_placement(
                state,
                accounting_id=ACCOUNTING_UNMATCHED,
                destination="turn-accounting-span",
                span_id=old_span_id,
                usage=_usage_row(call_id=ACCOUNTING_UNMATCHED).to_dict(),
            )
            return initialize_eligibility(
                placed,
                terminal_turn_ids=[TURN_ONE],
                accounting_ids=[],
                include_history=True,
            )

        update_export_state(enabled_config, stored, _seed)

        _, matched_projection = self._ambiguous_then_matched_projections()
        queue_exports(enabled_config, stored, matched_projection, include_history=True)

        assert job_path.exists()
        assert claim_path.exists()
        assert json.loads(job_path.read_text(encoding="utf-8"))["state"] == "claimed"
        state = load_export_state(enabled_config, stored)
        placement = state["placements"][ACCOUNTING_UNMATCHED]
        assert placement["destination"] == "turn-accounting-span"
        assert ACCOUNTING_UNMATCHED in state["conflicts"]
        assert "in flight" in state["conflicts"][ACCOUNTING_UNMATCHED]["reason"]

    def test_relocation_is_quarantined_when_worker_claims_during_cancellation(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stored = _seed_session(enabled_config, paths)
        old_span_id = f"accounting:{stored}:{TURN_ONE}:{ACCOUNTING_UNMATCHED}"

        def _seed(state: dict[str, Any]) -> dict[str, Any]:
            placed, _, _ = record_placement(
                state,
                accounting_id=ACCOUNTING_UNMATCHED,
                destination="turn-accounting-span",
                span_id=old_span_id,
                usage=_usage_row(call_id=ACCOUNTING_UNMATCHED).to_dict(),
            )
            return initialize_eligibility(
                placed,
                terminal_turn_ids=[TURN_ONE],
                accounting_ids=[],
                include_history=True,
            )

        update_export_state(enabled_config, stored, _seed)
        monkeypatch.setattr(
            export_transport,
            "cancel",
            lambda *args: {"state": "claimed", "attempt": 0, "last_error": None},
        )

        _, matched_projection = self._ambiguous_then_matched_projections()
        queue_exports(enabled_config, stored, matched_projection, include_history=True)

        state = load_export_state(enabled_config, stored)
        assert state["placements"][ACCOUNTING_UNMATCHED]["span_id"] == old_span_id
        assert ACCOUNTING_UNMATCHED in state["conflicts"]

    def test_permanently_failed_accounting_job_is_reported_and_not_cleared(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
    ) -> None:
        """The generic transport gives up on a deterministic accounting job
        after exhausting its retries and marks it permanently ``"failed"`` on
        disk (see ``otel_worker._claim_job``) -- it will never retry that job
        again on its own. A local write+dispatch reporting success only means
        the job file exists and a worker was spawned at some point; it must
        not be conflated with actual delivery health, or a permanently stuck
        job would silently look fine and have its error cleared forever."""
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
            usage_rows=[
                _usage_row(call_id=ACCOUNTING_UNMATCHED, input_tokens=6587, output_tokens=5)
            ],
            attributions=[
                _attribution(logical_call_id=ACCOUNTING_UNMATCHED, status="ambiguous", call_id=None)
            ],
        )
        span_id = f"accounting:{stored}:{TURN_ONE}:{ACCOUNTING_UNMATCHED}"
        job_path = export_transport.job_path(enabled_config.root, span_id)
        job_path.parent.mkdir(parents=True, exist_ok=True)
        job_path.write_text(
            json.dumps(
                {"state": "failed", "attempt": 5, "last_error": "TimeoutError: collector stalled"}
            ),
            encoding="utf-8",
        )

        queued = queue_exports(enabled_config, stored, projection, include_history=True)
        assert (
            queued == 1
        )  # only the turn job; the permanently-failed accounting job does not count
        state = load_export_state(enabled_config, stored)
        placement = state["placements"][ACCOUNTING_UNMATCHED]
        assert placement["job_state"] == "failed"
        assert placement["job_attempt"] == 5
        assert placement["last_error"] == "TimeoutError: collector stalled"

    @pytest.mark.parametrize(
        ("job_state", "last_error"),
        [
            ("queued", None),
            ("claimed", None),
            ("retrying", "ConnectionError: collector unavailable"),
        ],
    )
    def test_accounting_worker_lifecycle_is_reflected_in_ledger(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        export_calls: dict[str, list[Any]],
        monkeypatch: pytest.MonkeyPatch,
        job_state: str,
        last_error: str | None,
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
            usage_rows=[_usage_row(call_id=ACCOUNTING_UNMATCHED)],
            attributions=[
                _attribution(logical_call_id=ACCOUNTING_UNMATCHED, status="ambiguous", call_id=None)
            ],
        )
        monkeypatch.setattr(
            export_transport,
            "status",
            lambda *args: {"state": job_state, "attempt": 2, "last_error": last_error},
        )

        queue_exports(enabled_config, stored, projection, include_history=True)

        placement = load_export_state(enabled_config, stored)["placements"][ACCOUNTING_UNMATCHED]
        assert placement["job_state"] == job_state
        assert placement["job_attempt"] == 2
        assert placement["last_error"] == last_error


class TestWorkerConfirmedDelivery:
    """`queue_exports` against a real (locally flushed, never remote) Logfire
    instance and the real detached worker, to verify the durable delivery
    claim actually prevents a second emission across a simulated restart."""

    @pytest.fixture(autouse=True)
    def _reset_otel_state(self):
        pytest.importorskip("logfire")
        from thirdeye import otel_export as _otel_export

        _otel_export._state["attempted"] = False
        _otel_export._state["instance"] = None
        _otel_export._state["id_generator"] = None
        yield
        _otel_export._state["attempted"] = False
        _otel_export._state["instance"] = None
        _otel_export._state["id_generator"] = None

    @pytest.fixture
    def exporter(self):
        from logfire.testing import TestExporter

        return TestExporter()

    @pytest.fixture
    def wired_instance(self, exporter, monkeypatch: pytest.MonkeyPatch):
        import logfire
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        instance = logfire.configure(
            send_to_logfire=False,
            console=False,
            additional_span_processors=[SimpleSpanProcessor(exporter)],
            advanced=logfire.AdvancedOptions(id_generator=otel_export._id_generator()),
        )
        monkeypatch.setattr(otel_export, "_get_instance", lambda config, platform: instance)
        return instance

    @pytest.fixture(autouse=True)
    def _synchronous_worker(self, monkeypatch: pytest.MonkeyPatch, enabled_config: Config):
        """Run the detached worker in-process instead of spawning a real
        child, same as the generic transport's own worker tests do."""
        from thirdeye import otel_worker

        monkeypatch.setattr(Config, "load", lambda: enabled_config)

        def _run(job_path: Path) -> None:
            otel_worker.main([str(job_path)])

        monkeypatch.setattr(otel_export, "_spawn", _run)

        def _run_accounting(job_path: Path) -> None:
            export_transport.main([str(job_path)])

        monkeypatch.setattr(export_transport, "_spawn", _run_accounting)

    def test_confirmed_accounting_delivery_survives_restart_without_double_emission(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        wired_instance,
        exporter,
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
            usage_rows=[
                _usage_row(call_id=ACCOUNTING_UNMATCHED, input_tokens=6587, output_tokens=5)
            ],
            attributions=[
                _attribution(logical_call_id=ACCOUNTING_UNMATCHED, status="ambiguous", call_id=None)
            ],
        )

        queue_exports(enabled_config, stored, projection, include_history=True)
        directory = _directory(enabled_config, stored)
        assert export_transport.delivery_sent(directory, ACCOUNTING_UNMATCHED) is True
        first_accounting_spans = [
            span for span in exporter.exported_spans_as_dict() if span["name"] == "accounting"
        ]
        assert len(first_accounting_spans) == 1

        # Simulate a restart: fresh Config/instance state, same durable
        # ledger and durable transport claim on disk.
        reloaded = Config(
            root=enabled_config.root,
            logfire=LogfireSettings(enabled=True, token="fake-token"),
        )
        queued = queue_exports(reloaded, stored, projection, include_history=True)
        assert queued == 1  # the turn job re-queues; harmless, first-wins claim there too

        second_accounting_spans = [
            span for span in exporter.exported_spans_as_dict() if span["name"] == "accounting"
        ]
        assert len(second_accounting_spans) == 1  # unchanged: no second accounting emission
        state = load_export_state(enabled_config, stored)
        assert state["placements"][ACCOUNTING_UNMATCHED]["emitted"] is True

    def test_chat_embedded_accounting_survives_restart_without_relocation_or_duplicate(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        wired_instance,
        exporter,
    ) -> None:
        """Matched usage placed on the chat span is delivered as part of the
        turn's own job -- there is no separate fallback accounting job for
        it. Copilot reconciles the durable turn claim with its chat placement;
        otherwise a later pass could relocate the accounting to a fallback
        span and duplicate tokens already flushed on the chat span."""
        stored = _seed_session(enabled_config, paths)
        projection = _projection(
            turns=[_main_turn(accounting_calls=[_accounting_call()])],
            usage_rows=[_usage_row()],
            attributions=[_attribution(logical_call_id=ACCOUNTING_MATCHED, call_id=CALL_MATCHED)],
        )

        queue_exports(enabled_config, stored, projection, include_history=True)
        directory = _directory(enabled_config, stored)
        assert otel_export.turn_export_sent(directory, TURN_ONE) is True
        chat_spans = [
            span for span in exporter.exported_spans_as_dict() if span["name"].startswith("chat")
        ]
        assert len(chat_spans) == 1
        accounting_spans = [
            span for span in exporter.exported_spans_as_dict() if span["name"] == "accounting"
        ]
        assert len(accounting_spans) == 0

        # Simulate a restart: fresh Config/instance state, same durable
        # ledger, turn claim, and accounting delivery claim on disk.
        reloaded = Config(
            root=enabled_config.root,
            logfire=LogfireSettings(enabled=True, token="fake-token"),
        )
        queue_exports(reloaded, stored, projection, include_history=True)

        chat_spans_after = [
            span for span in exporter.exported_spans_as_dict() if span["name"].startswith("chat")
        ]
        accounting_spans_after = [
            span for span in exporter.exported_spans_as_dict() if span["name"] == "accounting"
        ]
        assert len(chat_spans_after) == 1  # unchanged: the turn's own claim is first-wins
        assert len(accounting_spans_after) == 0  # never relocated to a fallback span
        state = load_export_state(enabled_config, stored)
        placement = state["placements"][ACCOUNTING_MATCHED]
        assert placement["destination"] == "chat-span"
        assert placement["emitted"] is True
        assert ACCOUNTING_MATCHED not in state["conflicts"]

    def test_undelivered_same_span_token_correction_rewrites_job_and_delivers_once(
        self,
        enabled_config: Config,
        paths: SourcePaths,
        wired_instance,
        exporter,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stored = _seed_session(enabled_config, paths)
        delayed: list[Path] = []

        def _hold(job_path: Path) -> None:
            delayed.append(job_path)

        monkeypatch.setattr(export_transport, "_spawn", _hold)
        monkeypatch.setattr(otel_export, "_spawn", lambda job_path: None)

        first = _projection(
            turns=[
                _main_turn(
                    accounting_calls=[
                        _accounting_call(
                            accounting_id=ACCOUNTING_UNMATCHED,
                            attribution_status="ambiguous",
                            call_id=None,
                            usage=_usage_row(
                                call_id=ACCOUNTING_UNMATCHED, input_tokens=6587, output_tokens=5
                            ).to_dict(),
                        )
                    ]
                )
            ],
            usage_rows=[
                _usage_row(call_id=ACCOUNTING_UNMATCHED, input_tokens=6587, output_tokens=5)
            ],
            attributions=[
                _attribution(logical_call_id=ACCOUNTING_UNMATCHED, status="ambiguous", call_id=None)
            ],
        )
        queue_exports(enabled_config, stored, first, include_history=True)
        span_id = f"accounting:{stored}:{TURN_ONE}:{ACCOUNTING_UNMATCHED}"
        job_path = export_transport.job_path(enabled_config.root, span_id)
        assert job_path.exists()
        first_job = json.loads(job_path.read_text(encoding="utf-8"))
        assert first_job["usage"]["gen_ai.usage.output_tokens"] == 5

        corrected_usage = _usage_row(
            call_id=ACCOUNTING_UNMATCHED, input_tokens=6587, output_tokens=999
        ).to_dict()
        second = _projection(
            turns=[
                _main_turn(
                    accounting_calls=[
                        _accounting_call(
                            accounting_id=ACCOUNTING_UNMATCHED,
                            attribution_status="ambiguous",
                            call_id=None,
                            usage=corrected_usage,
                        )
                    ]
                )
            ],
            usage_rows=[
                _usage_row(call_id=ACCOUNTING_UNMATCHED, input_tokens=6587, output_tokens=999)
            ],
            attributions=[
                _attribution(logical_call_id=ACCOUNTING_UNMATCHED, status="ambiguous", call_id=None)
            ],
        )
        queue_exports(enabled_config, stored, second, include_history=True)
        rewritten = json.loads(job_path.read_text(encoding="utf-8"))
        assert rewritten["usage"]["gen_ai.usage.output_tokens"] == 999
        assert rewritten["state"] == "queued"

        export_transport.main([str(job_path)])
        accounting_spans = [
            span for span in exporter.exported_spans_as_dict() if span["name"] == "accounting"
        ]
        assert len(accounting_spans) == 1
        assert accounting_spans[0]["attributes"]["gen_ai.usage.output_tokens"] == 999
        directory = _directory(enabled_config, stored)
        assert export_transport.delivery_sent(directory, ACCOUNTING_UNMATCHED) is True
        state = load_export_state(enabled_config, stored)
        assert ACCOUNTING_UNMATCHED not in state["conflicts"]

    def test_observed_six_call_queue_exports_labels_native_billing_not_usd(
        self,
        enabled_config: Config,
        wired_instance,
        exporter,
        tmp_path: Path,
    ) -> None:
        home = tmp_path / "copilot-home"
        home.mkdir()
        paths = resolve_sources(home)
        commit_batch(enabled_config, paths, _batch(paths, _drain_cli_transcript(home)))
        stored = stored_session_id(paths, NATIVE_SESSION_ID)
        source_key = _source_key(_drain_cli_transcript(home))
        records = _drain_cli_transcript(home) + _six_call_records(source_key=source_key)
        projection, _ = build_projection(records, {})

        queue_exports(enabled_config, stored, projection, include_history=True)

        billed = [
            span
            for span in exporter.exported_spans_as_dict()
            if span["attributes"].get("thirdeye.accounting.billing.kind") == "copilot-native-unit"
        ]
        assert billed
        for span in billed:
            assert span["attributes"]["thirdeye.accounting.billing.kind"] == "copilot-native-unit"
            assert "operation.cost" not in span["attributes"]
            assert span["attributes"]["copilot.billing.nano_aiu"]


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
