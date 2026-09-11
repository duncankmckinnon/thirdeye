import re
import time

import pytest

from thirdeye.ids import new_ulid, resolve_prefix


def test_new_ulid_is_26_crockford_chars():
    ulid = new_ulid()
    assert len(ulid) == 26
    assert re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", ulid)


def test_new_ulid_is_unique():
    assert new_ulid() != new_ulid()


def test_new_ulid_is_time_sortable():
    a = new_ulid()
    time.sleep(0.002)
    b = new_ulid()
    assert a < b


def test_resolve_prefix_unique():
    assert resolve_prefix("01J", ["01J9G7XK4P", "02ABC"]) == "01J9G7XK4P"


def test_resolve_prefix_full_match():
    assert resolve_prefix("01J9G7XK4P", ["01J9G7XK4P", "02ABC"]) == "01J9G7XK4P"


def test_resolve_prefix_empty():
    with pytest.raises(ValueError, match="no session"):
        resolve_prefix("99", ["01J9G7XK4P"])


def test_resolve_prefix_ambiguous():
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_prefix("01", ["01ABC", "01XYZ"])
