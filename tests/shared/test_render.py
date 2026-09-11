"""Tests for thirdeye.render — terse, jsonl, and tree event renderers."""

from __future__ import annotations

import json

from thirdeye.render import render_event_jsonl, render_event_terse, render_event_tree

# -- helpers ------------------------------------------------------------------


def _ev(**kw):
    """Build a minimal event dict with sensible defaults."""
    base = {"t": "x", "ts": "2026-04-30T17:00:00.000Z", "seq": 0}
    base.update(kw)
    return base


# -- terse renderer -----------------------------------------------------------


class TestRenderEventTerse:
    # Basic format: "<seq> <t> <content>"

    def test_scalar_string_data(self):
        assert render_event_terse(_ev(t="user_message", seq=12, data="hi")) == "12 user_message hi"

    def test_scalar_int_data(self):
        assert render_event_terse(_ev(t="counter", seq=5, data=42)) == "5 counter 42"

    def test_scalar_float_data(self):
        assert render_event_terse(_ev(t="metric", seq=7, data=3.14)) == "7 metric 3.14"

    def test_scalar_bool_true(self):
        assert render_event_terse(_ev(t="flag", seq=1, data=True)) == "1 flag True"

    def test_scalar_bool_false(self):
        assert render_event_terse(_ev(t="flag", seq=2, data=False)) == "2 flag False"

    def test_no_data_key(self):
        """When 'data' key is absent, line is just '<seq> <t>'."""
        assert render_event_terse(_ev(t="session_start", seq=0)) == "0 session_start"

    def test_none_data(self):
        """Explicit None should behave like missing data."""
        assert render_event_terse(_ev(t="session_start", seq=0, data=None)) == "0 session_start"

    def test_flat_object_values_only(self):
        """Flat object renders space-separated VALUES, not keys."""
        line = render_event_terse(_ev(t="tool_call", seq=3, data={"name": "Read", "path": "x.py"}))
        assert line == "3 tool_call Read x.py"

    def test_flat_object_single_key(self):
        line = render_event_terse(_ev(t="note", seq=0, data={"msg": "hello"}))
        assert line == "0 note hello"

    def test_flat_object_numeric_values(self):
        line = render_event_terse(_ev(t="stats", seq=1, data={"count": 10, "avg": 2.5}))
        assert line == "1 stats 10 2.5"

    def test_flat_object_mixed_scalar_types(self):
        line = render_event_terse(_ev(t="info", seq=0, data={"name": "x", "count": 3, "ok": True}))
        assert line == "0 info x 3 True"

    def test_flat_object_empty_dict(self):
        """Empty dict is still a flat object with no values."""
        line = render_event_terse(_ev(t="empty", seq=0, data={}))
        assert line == "0 empty"

    def test_nested_object_uses_json(self):
        """Nested dicts collapse to compact JSON."""
        line = render_event_terse(_ev(t="weird", seq=1, data={"a": {"b": 1}}))
        assert line.startswith("1 weird ")
        assert "{" in line
        parsed = json.loads(line[len("1 weird ") :])
        assert parsed == {"a": {"b": 1}}

    def test_nested_object_with_list_value(self):
        """Dict containing a list is nested, not flat."""
        line = render_event_terse(_ev(t="cmd", seq=0, data={"args": [1, 2, 3]}))
        assert line.startswith("0 cmd ")
        assert "[" in line

    def test_list_data_uses_json(self):
        """Top-level list renders as compact JSON."""
        line = render_event_terse(_ev(t="items", seq=0, data=[1, 2, 3]))
        assert line == "0 items [1,2,3]"

    def test_list_of_strings(self):
        line = render_event_terse(_ev(t="tags", seq=0, data=["a", "b"]))
        assert line == '0 tags ["a","b"]'

    def test_empty_list(self):
        line = render_event_terse(_ev(t="items", seq=0, data=[]))
        assert line == "0 items []"

    # Truncation behavior

    def test_truncated_to_width(self):
        big = "x" * 500
        line = render_event_terse(_ev(t="user_message", seq=0, data=big), width=50)
        assert len(line) == 50
        assert line.endswith("\u2026")

    def test_exact_width_not_truncated(self):
        """Line exactly at width should NOT be truncated."""
        # "0 x " + data = width chars
        width = 20
        data = "a" * (width - len("0 x "))
        line = render_event_terse(_ev(t="x", seq=0, data=data), width=width)
        assert len(line) == width
        assert "\u2026" not in line

    def test_one_over_width_is_truncated(self):
        width = 20
        data = "a" * (width - len("0 x ") + 1)
        line = render_event_terse(_ev(t="x", seq=0, data=data), width=width)
        assert len(line) == width
        assert line.endswith("\u2026")

    def test_no_truncation_when_width_zero(self):
        big = "x" * 500
        line = render_event_terse(_ev(t="user_message", seq=0, data=big), width=0)
        assert "\u2026" not in line
        assert len(line) > 500

    def test_default_width_is_120(self):
        big = "x" * 200
        line = render_event_terse(_ev(t="user_message", seq=0, data=big))
        assert len(line) == 120
        assert line.endswith("\u2026")

    # Open vocabulary: t is opaque, never branched on

    def test_arbitrary_t_value(self):
        """Any string should work as t."""
        line = render_event_terse(_ev(t="my.custom.event_type-v2", seq=99, data="ok"))
        assert line == "99 my.custom.event_type-v2 ok"

    def test_empty_string_data(self):
        """Empty string data results in just '<seq> <t>'."""
        line = render_event_terse(_ev(t="x", seq=0, data=""))
        # str("") is "", which is falsy, so no content appended
        assert line == "0 x"


# -- jsonl renderer -----------------------------------------------------------


class TestRenderEventJsonl:
    def test_compact_separators(self):
        line = render_event_jsonl(_ev(t="x", seq=2, data={"a": 1}))
        assert line == '{"t":"x","ts":"2026-04-30T17:00:00.000Z","seq":2,"data":{"a":1}}'

    def test_no_data_key(self):
        ev = _ev(t="start", seq=0)
        line = render_event_jsonl(ev)
        parsed = json.loads(line)
        assert parsed["t"] == "start"
        assert parsed["seq"] == 0
        assert "data" not in parsed

    def test_none_data(self):
        line = render_event_jsonl(_ev(t="x", seq=0, data=None))
        parsed = json.loads(line)
        assert parsed["data"] is None

    def test_nested_data_preserved(self):
        data = {"a": {"b": [1, 2]}, "c": "d"}
        line = render_event_jsonl(_ev(data=data))
        parsed = json.loads(line)
        assert parsed["data"] == data

    def test_string_data(self):
        line = render_event_jsonl(_ev(data="hello"))
        parsed = json.loads(line)
        assert parsed["data"] == "hello"

    def test_unicode_preserved(self):
        """ensure_ascii=False means unicode passes through."""
        line = render_event_jsonl(_ev(data="cafe\u0301"))
        assert "caf\u00e9" in line or "cafe\u0301" in line
        parsed = json.loads(line)
        assert parsed["data"] == "caf\u00e9" or parsed["data"] == "cafe\u0301"

    def test_output_is_valid_json(self):
        ev = _ev(t="complex", seq=10, data={"nested": {"list": [1, "two", None, True]}})
        line = render_event_jsonl(ev)
        parsed = json.loads(line)
        assert parsed["t"] == "complex"
        assert parsed["seq"] == 10

    def test_default_str_for_non_serializable(self):
        """default=str should handle non-JSON-native types."""
        from datetime import datetime

        dt = datetime(2026, 4, 30, 12, 0, 0)
        line = render_event_jsonl(_ev(data={"when": dt}))
        parsed = json.loads(line)
        assert isinstance(parsed["data"]["when"], str)


# -- tree renderer ------------------------------------------------------------


class TestRenderEventTree:
    def test_header_format(self):
        tree = render_event_tree(_ev(t="tool_call", seq=4, ts="2026-04-30T17:00:00.000Z"))
        first_line = tree.split("\n")[0]
        assert first_line == "#4 tool_call  (2026-04-30T17:00:00.000Z)"

    def test_no_data(self):
        """No data means just the header, no body."""
        tree = render_event_tree(_ev(t="start", seq=0))
        assert "\n" not in tree
        assert tree.startswith("#0 start")

    def test_none_data(self):
        tree = render_event_tree(_ev(t="start", seq=0, data=None))
        assert "\n" not in tree

    def test_flat_dict_indented(self):
        tree = render_event_tree(_ev(t="tool_call", seq=4, data={"name": "Read", "path": "x.py"}))
        assert "tool_call" in tree
        lines = tree.split("\n")
        assert len(lines) >= 2
        # Data lines should be indented with 2 spaces (indent=1)
        assert "  name: Read" in tree
        assert "  path: x.py" in tree

    def test_nested_dict(self):
        data = {"outer": {"inner": "val"}}
        tree = render_event_tree(_ev(t="nested", seq=1, data=data))
        lines = tree.split("\n")
        # outer: (header for nested)
        assert any("  outer:" in l for l in lines)
        # inner indented deeper
        assert any("    inner: val" in l for l in lines)

    def test_list_renders_with_dashes(self):
        data = ["alpha", "beta", "gamma"]
        tree = render_event_tree(_ev(t="items", seq=0, data=data))
        lines = tree.split("\n")
        assert any("  - alpha" in l for l in lines)
        assert any("  - beta" in l for l in lines)
        assert any("  - gamma" in l for l in lines)

    def test_dict_with_list_value(self):
        data = {"files": ["a.py", "b.py"]}
        tree = render_event_tree(_ev(t="ev", seq=0, data=data))
        lines = tree.split("\n")
        assert any("  files:" in l for l in lines)
        assert any("    - a.py" in l for l in lines)
        assert any("    - b.py" in l for l in lines)

    def test_scalar_data(self):
        tree = render_event_tree(_ev(t="msg", seq=0, data="hello world"))
        lines = tree.split("\n")
        assert len(lines) >= 2
        assert "  hello world" in lines[1]

    def test_numeric_data(self):
        tree = render_event_tree(_ev(t="count", seq=0, data=42))
        lines = tree.split("\n")
        assert "  42" in lines[1]


# -- _is_flat_object (internal helper) ----------------------------------------


class TestIsFlatObject:
    """Test the internal helper to ensure correct flat/nested classification."""

    def test_flat_all_scalars(self):
        from thirdeye.render import _is_flat_object

        assert _is_flat_object({"a": 1, "b": "x", "c": True}) is True

    def test_not_flat_nested_dict(self):
        from thirdeye.render import _is_flat_object

        assert _is_flat_object({"a": {"b": 1}}) is False

    def test_not_flat_nested_list(self):
        from thirdeye.render import _is_flat_object

        assert _is_flat_object({"a": [1, 2]}) is False

    def test_empty_dict_is_flat(self):
        from thirdeye.render import _is_flat_object

        assert _is_flat_object({}) is True

    def test_none_value_is_flat(self):
        from thirdeye.render import _is_flat_object

        assert _is_flat_object({"a": None}) is True
