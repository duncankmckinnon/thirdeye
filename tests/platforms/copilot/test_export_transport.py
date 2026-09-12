"""Copilot accounting transport claim, cancellation, and retry behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from thirdeye import otel_export
from thirdeye.config import Config, LogfireSettings
from thirdeye.platforms.copilot import export_transport
from thirdeye.platforms.copilot.export_state import record_placement, update_export_state


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        root=tmp_path / "thirdeye",
        logfire=LogfireSettings(enabled=True, token="fake-token"),
    )


def _payload(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "job_id": "accounting:stored-session:acct-1",
        "kind": "session_accounting",
        "session_dir": "/traces/copilot/stored-session",
        "session_id": "stored-session",
        "cwd": "/proj",
        "accounting_id": "acct-1",
        "usage": {},
        "attribution_status": "ambiguous",
        "state": "queued",
        "attempt": 0,
    }
    value.update(overrides)
    return value


def _write(config: Config, payload: dict[str, Any]) -> Path:
    path = export_transport.job_path(config.root, str(payload["job_id"]))
    export_transport._write_job(path, payload)
    return path


def _seed_placement(
    config: Config, payload: dict[str, Any], *, span_id: str | None = None
) -> None:
    def _record(state: dict[str, Any]) -> dict[str, Any]:
        updated, _, _ = record_placement(
            state,
            accounting_id=str(payload["accounting_id"]),
            destination="session-accounting-span",
            span_id=span_id or str(payload["job_id"]),
            usage={},
        )
        return updated

    update_export_state(config, str(payload["session_id"]), _record)


def test_queue_is_deterministic(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[Path] = []
    monkeypatch.setattr(export_transport, "_spawn", spawned.append)
    accounting = {
        "accounting_id": "acct-1",
        "usage": {},
        "attribution_status": "ambiguous",
    }
    _seed_placement(config, _payload())

    assert export_transport.queue_session_accounting(
        config, Path("/traces/copilot/stored-session"), "stored-session", "/proj", accounting
    )
    assert export_transport.queue_session_accounting(
        config, Path("/traces/copilot/stored-session"), "stored-session", "/proj", accounting
    )

    jobs = list(export_transport.jobs_dir(config.root).glob("accounting-*.json"))
    assert len(jobs) == 1
    assert len(spawned) == 2


def test_queue_does_not_recreate_confirmed_delivery(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned: list[Path] = []
    monkeypatch.setattr(export_transport, "_spawn", spawned.append)
    session_dir = config.root / "traces" / "copilot" / "stored-session"
    otel_export._mark_accounting_sent(session_dir, "acct-1")
    accounting = {
        "accounting_id": "acct-1",
        "usage": {},
        "attribution_status": "ambiguous",
    }
    _seed_placement(config, _payload())

    assert export_transport.queue_session_accounting(
        config, session_dir, "stored-session", "/proj", accounting
    )

    assert list(export_transport.jobs_dir(config.root).glob("accounting-*.json")) == []
    assert spawned == []


def test_queue_does_not_publish_a_stale_placement(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned: list[Path] = []
    monkeypatch.setattr(export_transport, "_spawn", spawned.append)
    payload = _payload()
    _seed_placement(config, payload, span_id="accounting:stored-session:new-placement")

    assert export_transport._queue(config, payload)

    assert not export_transport.job_path(config.root, payload["job_id"]).exists()
    assert spawned == []


def test_queue_does_not_publish_while_another_owner_holds_the_claim(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned: list[Path] = []
    monkeypatch.setattr(export_transport, "_spawn", spawned.append)
    payload = _payload()
    path = export_transport.job_path(config.root, payload["job_id"])
    owner = export_transport.claim_path(path)
    owner.parent.mkdir(parents=True, exist_ok=True)
    owner.write_text("cancel-in-progress", encoding="utf-8")

    assert export_transport._queue(config, payload)

    assert not path.exists()
    assert spawned == []


def test_cancel_preserves_worker_owned_job(config: Config) -> None:
    payload = _payload()
    path = _write(config, payload)
    export_transport.claim_path(path).write_text("worker-42", encoding="utf-8")

    assert export_transport.status(config.root, payload["job_id"])["state"] == "claimed"
    result = export_transport.cancel(config.root, payload["job_id"])

    assert result["state"] == "claimed"
    assert path.exists()
    assert export_transport.claim_path(path).exists()


def test_cancelled_job_cannot_be_resurrected_by_a_stale_reader(config: Config) -> None:
    payload = _payload()
    path = _write(config, payload)

    result = export_transport.cancel(config.root, payload["job_id"])
    claimed = export_transport._acquire(path)

    assert result["state"] == "cancelled"
    assert claimed is None
    assert not path.exists()
    assert not export_transport.claim_path(path).exists()


@pytest.mark.parametrize(
    ("attempt", "expected_state"),
    [(0, "retrying"), (export_transport._MAX_ATTEMPTS - 1, "failed")],
)
def test_worker_failure_persists_state_and_error(
    config: Config,
    monkeypatch: pytest.MonkeyPatch,
    attempt: int,
    expected_state: str,
) -> None:
    payload = _payload(attempt=attempt)
    path = _write(config, payload)
    monkeypatch.setattr(Config, "load", lambda: config)

    def _fail(config: Config, payload: dict[str, Any]) -> None:
        raise TimeoutError("collector stalled")

    monkeypatch.setattr(export_transport, "_deliver", _fail)

    export_transport.run(path)

    status = export_transport.status(config.root, payload["job_id"])
    assert status == {
        "state": expected_state,
        "attempt": attempt + 1,
        "last_error": "TimeoutError: collector stalled",
    }
    assert not export_transport.claim_path(path).exists()
