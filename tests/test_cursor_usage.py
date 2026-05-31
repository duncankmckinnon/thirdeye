from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from thirdeye.paths import session_dir, usage_log_path
from thirdeye.platforms.cursor.usage import (
    _get_int,
    _get_str,
    capture_usage_cursor,
)
from thirdeye.usage.store import UsageStore


SESSION_ID = "test-conv-123"


def _store_for(tmp_path: Path) -> UsageStore:
    return UsageStore(session_dir(tmp_path, "cursor", SESSION_ID))


def _rows(tmp_path: Path) -> list:
    return list(_store_for(tmp_path).iter_rows())


def _complete_snake_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "model": "composer-1",
        "input_tokens": 100,
        "output_tokens": 50,
    }
    payload.update(overrides)
    return payload


def _complete_camel_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "model": "composer-1",
        "inputTokens": 100,
        "outputTokens": 50,
    }
    payload.update(overrides)
    return payload


class TestGetStr:
    def test_first_key_present_wins(self) -> None:
        assert _get_str({"model": "a", "model_name": "b"}, "model", "model_name") == "a"

    def test_falls_back_to_second_key(self) -> None:
        assert _get_str({"model_name": "b"}, "model", "model_name") == "b"

    def test_empty_string_falls_through(self) -> None:
        assert _get_str({"model": "", "model_name": "b"}, "model", "model_name") == "b"

    def test_returns_default_when_all_missing(self) -> None:
        assert _get_str({}, "model", "model_name", default="x") == "x"
        assert _get_str({}, "model", "model_name") == ""


class TestGetInt:
    def test_int_value(self) -> None:
        assert _get_int({"input_tokens": 5}, "input_tokens") == 5

    def test_string_int_value(self) -> None:
        assert _get_int({"input_tokens": "42"}, "input_tokens") == 42

    def test_zero_is_valid(self) -> None:
        assert _get_int({"input_tokens": 0}, "input_tokens") == 0
        assert _get_int({"input_tokens": "0"}, "input_tokens") == 0

    def test_empty_string_falls_through(self) -> None:
        assert _get_int({"input_tokens": "", "inputTokens": 7}, "input_tokens", "inputTokens") == 7

    def test_dash_dash_falls_through(self) -> None:
        assert _get_int({"input_tokens": "--", "inputTokens": 9}, "input_tokens", "inputTokens") == 9

    def test_returns_none_when_all_missing(self) -> None:
        assert _get_int({}, "input_tokens", "inputTokens") is None

    def test_unparseable_falls_through_then_returns_none(self) -> None:
        assert _get_int({"input_tokens": "abc"}, "input_tokens") is None
        assert _get_int({"input_tokens": "abc", "inputTokens": "xyz"}, "input_tokens", "inputTokens") is None


class TestCaptureFromCompletePayload:
    def test_writes_one_row_with_correct_fields(self, tmp_path: Path) -> None:
        result = capture_usage_cursor(
            thirdeye_home=tmp_path,
            session_id=SESSION_ID,
            payload=_complete_snake_payload(),
            triggering_seq=7,
        )
        assert result == 1
        rows = _rows(tmp_path)
        assert len(rows) == 1
        row = rows[0]
        assert row.session_id == SESSION_ID
        assert row.model == "composer-1"
        assert row.input_tokens == 100
        assert row.output_tokens == 50
        assert row.total_tokens == 150

    def test_camelcase_payload_works(self, tmp_path: Path) -> None:
        result = capture_usage_cursor(
            thirdeye_home=tmp_path,
            session_id=SESSION_ID,
            payload=_complete_camel_payload(),
            triggering_seq=3,
        )
        assert result == 1
        rows = _rows(tmp_path)
        assert len(rows) == 1
        assert rows[0].input_tokens == 100
        assert rows[0].output_tokens == 50
        assert rows[0].total_tokens == 150

    def test_zero_tokens_writes_row(self, tmp_path: Path) -> None:
        result = capture_usage_cursor(
            thirdeye_home=tmp_path,
            session_id=SESSION_ID,
            payload=_complete_snake_payload(input_tokens=0, output_tokens=0),
            triggering_seq=1,
        )
        assert result == 1
        rows = _rows(tmp_path)
        assert len(rows) == 1
        assert rows[0].input_tokens == 0
        assert rows[0].output_tokens == 0
        assert rows[0].total_tokens == 0

    def test_uses_triggering_seq(self, tmp_path: Path) -> None:
        capture_usage_cursor(
            thirdeye_home=tmp_path,
            session_id=SESSION_ID,
            payload=_complete_snake_payload(),
            triggering_seq=42,
        )
        rows = _rows(tmp_path)
        assert rows[0].seq == 42

    def test_platform_is_cursor(self, tmp_path: Path) -> None:
        capture_usage_cursor(
            thirdeye_home=tmp_path,
            session_id=SESSION_ID,
            payload=_complete_snake_payload(),
            triggering_seq=1,
        )
        rows = _rows(tmp_path)
        assert rows[0].platform == "cursor"


class TestCaptureSkipsIncomplete:
    def test_missing_model_returns_zero(self, tmp_path: Path) -> None:
        payload = _complete_snake_payload()
        payload.pop("model")
        result = capture_usage_cursor(
            thirdeye_home=tmp_path,
            session_id=SESSION_ID,
            payload=payload,
            triggering_seq=1,
        )
        assert result == 0

    def test_missing_input_tokens_returns_zero(self, tmp_path: Path) -> None:
        payload = _complete_snake_payload()
        payload.pop("input_tokens")
        result = capture_usage_cursor(
            thirdeye_home=tmp_path,
            session_id=SESSION_ID,
            payload=payload,
            triggering_seq=1,
        )
        assert result == 0

    def test_missing_output_tokens_returns_zero(self, tmp_path: Path) -> None:
        payload = _complete_snake_payload()
        payload.pop("output_tokens")
        result = capture_usage_cursor(
            thirdeye_home=tmp_path,
            session_id=SESSION_ID,
            payload=payload,
            triggering_seq=1,
        )
        assert result == 0

    def test_returns_zero_writes_no_row(self, tmp_path: Path) -> None:
        capture_usage_cursor(
            thirdeye_home=tmp_path,
            session_id=SESSION_ID,
            payload={"input_tokens": 10, "output_tokens": 5},
            triggering_seq=1,
        )
        assert _rows(tmp_path) == []


class TestCaptureIdempotent:
    def test_second_call_does_not_write_second_row(self, tmp_path: Path) -> None:
        capture_usage_cursor(
            thirdeye_home=tmp_path,
            session_id=SESSION_ID,
            payload=_complete_snake_payload(),
            triggering_seq=1,
        )
        capture_usage_cursor(
            thirdeye_home=tmp_path,
            session_id=SESSION_ID,
            payload=_complete_snake_payload(),
            triggering_seq=2,
        )
        rows = _rows(tmp_path)
        assert len(rows) == 1


class TestCaptureErrorsAreSwallowed:
    def test_broken_store_logs_error_and_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(self, rows):
            raise RuntimeError("disk full")

        monkeypatch.setattr(UsageStore, "append", boom)

        result = capture_usage_cursor(
            thirdeye_home=tmp_path,
            session_id=SESSION_ID,
            payload=_complete_snake_payload(),
            triggering_seq=1,
        )
        assert result is None

        log = usage_log_path(tmp_path)
        assert log.exists()
        entry = json.loads(log.read_text().strip().splitlines()[-1])
        assert entry["platform"] == "cursor"
        assert entry["phase"] == "extract_usage"
        assert entry["session_id"] == SESSION_ID
        assert entry["error_class"] == "RuntimeError"
