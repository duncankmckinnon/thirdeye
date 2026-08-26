from __future__ import annotations

import sys
from types import ModuleType

from thirdeye.config import Config
from thirdeye.logfire_dataset import export_sessions
from thirdeye.store import Store


class _Reader:
    def iter_events(self):
        yield {"seq": 0, "t": "user_message", "data": "hello"}


class _Store:
    def reader(self, session_id):
        assert session_id == "session-1"
        return _Reader()


def test_export_creates_named_dataset_and_cases(monkeypatch, tmp_path):
    store = Store(Config(root=tmp_path))
    with store.open_session("session-1", platform="claude", cwd="/project") as writer:
        writer.append("user_message", "ignored by stub")
    meta = store.get_meta("session-1")
    calls = []

    class Client:
        def __init__(self, api_key):
            calls.append(("init", api_key))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def create_dataset(self, **kwargs):
            calls.append(("create", kwargs))

        def add_cases(self, name, *, cases):
            calls.append(("cases", name, cases))

    package = ModuleType("logfire")
    experimental = ModuleType("logfire.experimental")
    api_client = ModuleType("logfire.experimental.api_client")
    api_client.LogfireAPIClient = Client
    monkeypatch.setitem(sys.modules, "logfire", package)
    monkeypatch.setitem(sys.modules, "logfire.experimental", experimental)
    monkeypatch.setitem(sys.modules, "logfire.experimental.api_client", api_client)

    count = export_sessions(api_key="secret", name="my-dataset", sessions=[meta], store=_Store())

    assert count == 1
    assert calls[0] == ("init", "secret")
    assert calls[1][0] == "create"
    assert calls[1][1]["name"] == "my-dataset"
    case = calls[2][2][0]
    assert case["name"] == "session-1"
    assert case["inputs"]["session"]["platform"] == "claude"
    assert case["inputs"]["events"][0]["data"] == "hello"


def test_export_turn_scope_creates_one_case_per_matching_turn(monkeypatch, tmp_path):
    store = Store(Config(root=tmp_path))
    with store.open_session("session-1", platform="claude", cwd="/project") as writer:
        writer.append("user_message", {"prompt": "first"})
        writer.append("assistant_message", {"text": "one"})
        writer.append("user_message", {"prompt": "second"})
        writer.append("assistant_message", {"text": "two"})
    meta = store.get_meta("session-1")
    added = []

    class Client:
        def __init__(self, api_key):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def create_dataset(self, **kwargs):
            pass

        def add_cases(self, name, *, cases):
            added.extend(cases)

    api_client = ModuleType("logfire.experimental.api_client")
    api_client.LogfireAPIClient = Client
    monkeypatch.setitem(sys.modules, "logfire", ModuleType("logfire"))
    monkeypatch.setitem(sys.modules, "logfire.experimental", ModuleType("logfire.experimental"))
    monkeypatch.setitem(sys.modules, "logfire.experimental.api_client", api_client)

    count = export_sessions(
        api_key="secret",
        name="turns",
        sessions=[meta],
        store=store,
        scope="turn",
        turn_id="2",
    )

    assert count == 1
    assert added[0]["name"] == "session-1:2"
    assert added[0]["inputs"]["turn"]["events"][0]["data"]["prompt"] == "second"
