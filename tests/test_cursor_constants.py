from thirdeye.platforms.cursor.constants import TRACED_EVENTS


def test_traced_events_contains_subagent_stop_once():
    assert TRACED_EVENTS.count("subagentStop") == 1
