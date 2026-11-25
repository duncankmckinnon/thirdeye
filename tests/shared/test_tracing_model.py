from __future__ import annotations

from thirdeye.tracing.model import total_usage


def _call(call_id: str, **usage: int) -> dict:
    return {"call_id": call_id, "usage": usage}


def test_sums_each_reported_field():
    assert total_usage(
        [
            _call("a", input_tokens=10, output_tokens=2, cache_read_input_tokens=100),
            _call("b", input_tokens=5, output_tokens=1),
        ]
    ) == {"input_tokens": 15, "output_tokens": 3, "cache_read_input_tokens": 100}


def test_a_call_split_into_two_groups_counts_once():
    """A Claude message whose frames straddle a user frame parses as two
    groups sharing one call_id, each repeating the message's usage."""
    assert total_usage(
        [
            _call("a", input_tokens=10, output_tokens=1),
            _call("a", input_tokens=10, output_tokens=4),
            _call("b", input_tokens=5, output_tokens=1),
        ]
    ) == {"input_tokens": 15, "output_tokens": 5}


def test_no_calls_is_empty():
    assert total_usage([]) == {}
