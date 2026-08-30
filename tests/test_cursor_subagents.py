import json
from pathlib import Path

from thirdeye.platforms.cursor.subagents import (
    CursorSubagentWindow,
    CursorTranscriptSummary,
    cursor_subagent_generation_id,
    cursor_subagent_windows,
    events_for_subagent,
    modern_subagent_stop_seqs,
    read_cursor_transcript,
    window_for_stop,
)

FIXTURES = Path(__file__).parent / "fixtures"


def event(seq, event_type, **data):
    return {"seq": seq, "t": event_type, "data": data}


def start(seq, subagent_id, tool_call_id=None, **data):
    if tool_call_id is not None:
        data["tool_call_id"] = tool_call_id
    return event(seq, "subagent_start", subagent_id=subagent_id, **data)


def stop(seq, subagent_id, **data):
    return event(seq, "subagent_message", subagent_id=subagent_id, **data)


def resume(seq, subagent_id, tool_call_id, **tool_input):
    return event(
        seq,
        "tool_call",
        tool_name="Task",
        tool_use_id=tool_call_id,
        tool_input={"resume": subagent_id, **tool_input},
    )


def write_jsonl(path, records):
    path.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")


def message(role, *blocks):
    return {"role": role, "message": {"content": list(blocks)}}


def text_block(text):
    return {"type": "text", "text": text}


class TestCursorSubagentGenerationId:
    def test_empty(self):
        assert cursor_subagent_generation_id("") == ""

    def test_fixed_vector(self):
        assert cursor_subagent_generation_id("call-123") == "f31a9706-039a-4013-baba-49d1b610a9ca"

    def test_is_lowercase_uuid4_with_rfc_variant(self):
        result = cursor_subagent_generation_id("Task-ABC")

        assert result == result.lower()
        assert [len(part) for part in result.split("-")] == [8, 4, 4, 4, 12]
        assert all(character in "0123456789abcdef-" for character in result)
        assert result[14] == "4"
        assert result[19] in "89ab"

    def test_same_input_is_stable(self):
        assert cursor_subagent_generation_id("call-stable") == cursor_subagent_generation_id(
            "call-stable"
        )

    def test_different_tool_ids_differ(self):
        assert cursor_subagent_generation_id("call-a") != cursor_subagent_generation_id("call-b")


class TestCursorSubagentWindows:
    def test_sequential(self):
        first_start = start(1, "child-a", "call-a")
        first_stop = stop(2, "child-a")
        second_start = start(3, "child-b", "call-b")
        second_stop = stop(4, "child-b")

        windows = cursor_subagent_windows([second_stop, first_start, second_start, first_stop])

        assert windows == [
            CursorSubagentWindow(
                first_start,
                first_stop,
                "child-a",
                "call-a",
                cursor_subagent_generation_id("call-a"),
            ),
            CursorSubagentWindow(
                second_start,
                second_stop,
                "child-b",
                "call-b",
                cursor_subagent_generation_id("call-b"),
            ),
        ]

    def test_parallel_reverse_stop(self):
        start_a = start(10, "child-a", "call-a")
        start_b = start(11, "child-b", "call-b")
        stop_b = stop(12, "child-b")
        stop_a = stop(13, "child-a")

        windows = cursor_subagent_windows([stop_a, start_b, start_a, stop_b])

        assert [
            (window.subagent_id, window.start_event, window.stop_event) for window in windows
        ] == [
            ("child-b", start_b, stop_b),
            ("child-a", start_a, stop_a),
        ]

    def test_duplicate_stop_is_ignored(self):
        start_event = start(1, "child", "call")
        first_stop = stop(2, "child")

        windows = cursor_subagent_windows([start_event, first_stop, stop(3, "child")])

        assert len(windows) == 1
        assert windows[0].stop_event is first_stop

    def test_resume_task_opens_new_window_for_reused_id(self):
        first_start = start(1, "child", "call-one")
        first_stop = stop(2, "child")
        resume_start = resume(3, "child", "call-two", prompt="Continue inspection")
        resumed_stop = stop(5, "child")

        windows = cursor_subagent_windows([resumed_stop, resume_start, first_stop, first_start])

        assert len(windows) == 2
        resumed = windows[1]
        assert resumed.start_event["seq"] == 3
        assert resumed.start_event["data"]["cursor_resume"] is True
        assert resumed.start_event["data"]["task"] == "Continue inspection"
        assert resumed.stop_event is resumed_stop
        assert resumed.tool_call_id == "call-two"
        assert resumed.generation_id == cursor_subagent_generation_id("call-two")

    def test_resume_task_and_cli_start_coalesce_by_tool_call_id(self):
        events = [
            start(1, "child", "call-one"),
            stop(2, "child"),
            resume(3, "child", "call-two"),
            start(4, "child", "call-two", task="CLI resume"),
            stop(5, "child"),
        ]

        windows = cursor_subagent_windows(events)

        assert len(windows) == 2
        assert windows[1].start_event is events[3]
        assert windows[1].tool_call_id == "call-two"

    def test_modern_stop_sequences_include_suppressed_duplicates(self):
        events = [start(1, "child", "call"), stop(2, "child"), stop(3, "child")]

        assert modern_subagent_stop_seqs(events) == {2, 3}

    def test_unmatched_start_omitted(self):
        assert cursor_subagent_windows([start(1, "child", "call")]) == []

    def test_unmatched_stop_legacy(self):
        stop_event = stop(1, "historical")

        assert cursor_subagent_windows([stop_event]) == [
            CursorSubagentWindow(None, stop_event, "historical", "", "")
        ]

    def test_reused_id_pairs_in_order(self):
        start_one = start(1, "reused", "call-one")
        stop_one = stop(2, "reused")
        start_two = start(3, "reused", "call-two")
        stop_two = stop(4, "reused")

        windows = cursor_subagent_windows([stop_two, start_two, stop_one, start_one])

        assert [
            (window.start_event, window.stop_event, window.tool_call_id) for window in windows
        ] == [
            (start_one, stop_one, "call-one"),
            (start_two, stop_two, "call-two"),
        ]

    def test_camel_case_fields(self):
        start_event = event(1, "subagent_start", subagentId="child", toolCallId="call")
        stop_event = event(2, "subagent_message", agentId="child", generationId="wrong")

        [window] = cursor_subagent_windows([stop_event, start_event])

        assert window.subagent_id == "child"
        assert window.tool_call_id == "call"
        assert window.generation_id == cursor_subagent_generation_id("call")

    def test_result_sorted_by_stop_sequence(self):
        # Equal start sequences preserve input order, while completed windows are
        # always returned according to their stop sequence.
        start_a = start(1, "child-a", "call-a")
        start_b = start(1, "child-b", "call-b")
        stop_b = stop(8, "child-b")
        stop_a = stop(9, "child-a")

        windows = cursor_subagent_windows([stop_a, start_a, stop_b, start_b])

        assert [window.stop_event["seq"] for window in windows] == [8, 9]


class TestWindowForStop:
    def test_lookup_by_sequence_after_json_round_trip(self):
        events = [start(1, "child", "call"), stop(2, "child")]
        round_tripped_stop = json.loads(json.dumps(events[1]))

        window = window_for_stop(events, round_tripped_stop)

        assert window is not None
        assert window.start_event == events[0]
        assert window.stop_event == events[1]

    def test_absent_sequence_returns_none(self):
        events = [start(1, "child", "call"), stop(2, "child")]

        assert window_for_stop(events, stop(99, "child")) is None


class TestEventsForSubagent:
    def test_exact_match(self):
        generation = cursor_subagent_generation_id("call-a")
        start_event = start(12, "child-a", "call-a")
        matching_call = event(13, "tool_call", generation_id=generation, tool_use_id="read-a")
        matching_result = event(14, "tool_result", generationId=generation, tool_use_id="read-a")
        stop_event = stop(20, "child-a")
        [window] = cursor_subagent_windows(
            [start_event, matching_call, matching_result, stop_event]
        )

        assert events_for_subagent(
            [matching_result, stop_event, start_event, matching_call], window
        ) == [matching_result, matching_call]

    def test_wrong_generation_inside_bounds(self):
        start_event = start(12, "child-a", "call-a")
        wrong = event(13, "tool_call", generation_id="parent-gen")
        stop_event = stop(20, "child-a")
        [window] = cursor_subagent_windows([start_event, wrong, stop_event])

        assert events_for_subagent([start_event, wrong, stop_event], window) == []

    def test_correct_generation_before_start(self):
        generation = cursor_subagent_generation_id("call-a")
        before = event(11, "tool_call", generation_id=generation)
        start_event = start(12, "child-a", "call-a")
        stop_event = stop(20, "child-a")
        [window] = cursor_subagent_windows([before, start_event, stop_event])

        assert events_for_subagent([before, start_event, stop_event], window) == []

    def test_correct_generation_after_stop(self):
        generation = cursor_subagent_generation_id("call-a")
        start_event = start(12, "child-a", "call-a")
        stop_event = stop(20, "child-a")
        after = event(21, "tool_result", generation_id=generation)
        [window] = cursor_subagent_windows([start_event, stop_event, after])

        assert events_for_subagent([start_event, stop_event, after], window) == []

    def test_absent_task_id(self):
        start_event = start(12, "child-a")
        candidate = event(13, "tool_call", generation_id="some-generation")
        stop_event = stop(20, "child-a")
        [window] = cursor_subagent_windows([start_event, candidate, stop_event])

        assert window.generation_id == ""
        assert events_for_subagent([start_event, candidate, stop_event], window) == []

    def test_legacy_window(self):
        candidate = event(13, "tool_call", generation_id="some-generation")
        stop_event = stop(20, "historical")
        [window] = cursor_subagent_windows([candidate, stop_event])

        assert window.start_event is None
        assert events_for_subagent([candidate, stop_event], window) == []


class TestCursorTranscript:
    def test_real_shape_fixture(self):
        summary = read_cursor_transcript(str(FIXTURES / "cursor-subagent-transcript.jsonl"))

        assert summary == CursorTranscriptSummary(
            "Inspect the sample module", "The sample module is valid."
        )

    def test_missing_path(self):
        assert read_cursor_transcript(None) == CursorTranscriptSummary("", "")
        assert read_cursor_transcript("") == CursorTranscriptSummary("", "")

    def test_directory_path(self, tmp_path):
        assert read_cursor_transcript(str(tmp_path)) == CursorTranscriptSummary("", "")

    def test_invalid_utf8(self, tmp_path):
        transcript = tmp_path / "invalid.jsonl"
        transcript.write_bytes(b"\xff\xfe\xfa")

        assert read_cursor_transcript(str(transcript)) == CursorTranscriptSummary("", "")

    def test_blank_file(self, tmp_path):
        transcript = tmp_path / "blank.jsonl"
        transcript.write_text("\n  \n", encoding="utf-8")

        assert read_cursor_transcript(str(transcript)) == CursorTranscriptSummary("", "")

    def test_malformed_line_surrounded_by_valid_messages(self, tmp_path):
        transcript = tmp_path / "malformed.jsonl"
        transcript.write_text(
            json.dumps(message("user", text_block("First question")))
            + "\n{not json}\n"
            + json.dumps(message("assistant", text_block("Final answer")))
            + "\n",
            encoding="utf-8",
        )

        assert read_cursor_transcript(str(transcript)) == CursorTranscriptSummary(
            "First question", "Final answer"
        )

    def test_unsupported_top_level_values(self, tmp_path):
        transcript = tmp_path / "unsupported.jsonl"
        write_jsonl(
            transcript,
            [
                None,
                [],
                "message",
                3,
                {"role": "system", "message": {"content": [text_block("ignore")]}},
                {"role": "user", "message": "not-a-mapping"},
                {"role": "assistant", "message": {"content": "not-a-list"}},
            ],
        )

        assert read_cursor_transcript(str(transcript)) == CursorTranscriptSummary("", "")

    def test_partial_records_are_ignored(self, tmp_path):
        transcript = tmp_path / "partial.jsonl"
        write_jsonl(
            transcript,
            [
                {},
                {"role": "user"},
                {"role": "assistant", "message": {}},
                message("user", {"type": "text"}),
                message("assistant", {"type": "text", "text": 42}),
            ],
        )

        assert read_cursor_transcript(str(transcript)) == CursorTranscriptSummary("", "")

    def test_text_block_joining(self, tmp_path):
        transcript = tmp_path / "multiple-text.jsonl"
        write_jsonl(
            transcript,
            [
                message("user", text_block("  first  "), text_block("second")),
                message("user", text_block("ignored later user")),
                message("assistant", text_block("draft")),
                message(
                    "assistant",
                    text_block(" final line one "),
                    {"type": "tool_use", "name": "read_file"},
                    text_block("line two"),
                ),
            ],
        )

        assert read_cursor_transcript(str(transcript)) == CursorTranscriptSummary(
            "first\nsecond", "final line one\nline two"
        )

    def test_tool_only_assistant(self, tmp_path):
        transcript = tmp_path / "tool-only.jsonl"
        write_jsonl(
            transcript,
            [
                message("user", text_block("Inspect")),
                message(
                    "assistant",
                    {"type": "tool_use", "name": "read_file", "input": {"path": "sample.py"}},
                ),
            ],
        )

        assert read_cursor_transcript(str(transcript)) == CursorTranscriptSummary("Inspect", "")

    def test_turn_ended_ignored(self, tmp_path):
        transcript = tmp_path / "turn-ended.jsonl"
        write_jsonl(
            transcript,
            [
                message("user", text_block("Question")),
                message("assistant", text_block("Answer")),
                {"type": "turn_ended", "status": "success"},
            ],
        )

        assert read_cursor_transcript(str(transcript)) == CursorTranscriptSummary(
            "Question", "Answer"
        )
