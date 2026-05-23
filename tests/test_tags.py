"""Tests for thirdeye.tags — TagStore sidecar, validate_tag, extract_hashtags."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thirdeye.paths import tags_path
from thirdeye.tags import TagStore, extract_hashtags, validate_tag


class TestValidateTag:
    def test_simple_lowercase(self):
        assert validate_tag("foo") == "foo"

    def test_with_digits(self):
        assert validate_tag("bug-1") == "bug-1"

    def test_with_underscore(self):
        assert validate_tag("a_b-c") == "a_b-c"

    def test_strips_whitespace(self):
        assert validate_tag("  foo  ") == "foo"

    def test_min_length_1(self):
        assert validate_tag("a") == "a"

    def test_max_length_64(self):
        tag = "a" * 64
        assert validate_tag(tag) == tag

    def test_too_long(self):
        with pytest.raises(ValueError):
            validate_tag("a" * 65)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            validate_tag("")

    def test_capital_letter_raises(self):
        with pytest.raises(ValueError):
            validate_tag("Foo")

    def test_space_raises(self):
        with pytest.raises(ValueError):
            validate_tag("foo bar")


class TestExtractHashtags:
    def test_simple(self):
        assert extract_hashtags("hello #foo and #bar") == {"foo", "bar"}

    def test_lowercases(self):
        assert extract_hashtags("hello #Foo and #bar-1") == {"foo", "bar-1"}

    def test_lookbehind_blocks_inline(self):
        assert extract_hashtags("prefix#nope #ok") == {"ok"}

    def test_dedup(self):
        assert extract_hashtags("#a #a #A") == {"a"}

    def test_must_start_with_letter(self):
        assert extract_hashtags("#1abc") == set()

    def test_only_hashes(self):
        assert extract_hashtags("#" * 100) == set()

    def test_underscores_and_hyphens(self):
        assert extract_hashtags("#A_B-c") == {"a_b-c"}

    def test_empty_string(self):
        assert extract_hashtags("") == set()

    def test_none(self):
        assert extract_hashtags(None) == set()  # type: ignore[arg-type]

    def test_non_string(self):
        assert extract_hashtags(123) == set()  # type: ignore[arg-type]

    def test_no_hashtags(self):
        assert extract_hashtags("nothing here to see") == set()


class TestTagsPath:
    def test_ends_in_tags_jsonl(self, tmp_path: Path):
        p = tags_path(tmp_path)
        assert p.name == "tags.jsonl"
        assert p.parent == tmp_path


class TestTagStoreAdd:
    def test_file_created_on_first_write(self, tmp_path: Path):
        store = TagStore(tmp_path)
        assert not tags_path(tmp_path).exists()
        store.add(1, "foo")
        assert tags_path(tmp_path).exists()

    def test_one_line_per_call(self, tmp_path: Path):
        store = TagStore(tmp_path)
        store.add(1, "foo")
        store.add(1, "bar")
        store.add(2, "baz")
        with open(tags_path(tmp_path)) as f:
            lines = [ln for ln in f.read().splitlines() if ln]
        assert len(lines) == 3

    def test_duplicate_add_is_appended(self, tmp_path: Path):
        store = TagStore(tmp_path)
        store.add(1, "foo")
        store.add(1, "foo")
        with open(tags_path(tmp_path)) as f:
            lines = [ln for ln in f.read().splitlines() if ln]
        assert len(lines) == 2

    def test_tags_for_reflects_set(self, tmp_path: Path):
        store = TagStore(tmp_path)
        store.add(1, "foo")
        store.add(1, "bar")
        assert store.tags_for(1) == {"foo", "bar"}

    def test_line_shape(self, tmp_path: Path):
        store = TagStore(tmp_path)
        store.add(7, "foo", source="manual")
        with open(tags_path(tmp_path)) as f:
            line = f.readline().rstrip("\n")
        entry = json.loads(line)
        assert entry["seq"] == 7
        assert entry["tag"] == "foo"
        assert entry["op"] == "add"
        assert entry["source"] == "manual"
        assert isinstance(entry["at"], str) and entry["at"].endswith("Z")

    def test_auto_source(self, tmp_path: Path):
        store = TagStore(tmp_path)
        store.add(1, "foo", source="auto")
        with open(tags_path(tmp_path)) as f:
            entry = json.loads(f.readline())
        assert entry["source"] == "auto"

    def test_invalid_tag_raises(self, tmp_path: Path):
        store = TagStore(tmp_path)
        with pytest.raises(ValueError):
            store.add(1, "Bad Tag!")


class TestTagStoreRemove:
    def test_remove_flips_set(self, tmp_path: Path):
        store = TagStore(tmp_path)
        store.add(1, "foo")
        store.remove(1, "foo")
        assert store.tags_for(1) == set()

    def test_remove_absent_tag_appended_but_set_empty(self, tmp_path: Path):
        store = TagStore(tmp_path)
        store.remove(1, "foo")
        with open(tags_path(tmp_path)) as f:
            lines = [ln for ln in f.read().splitlines() if ln]
        assert len(lines) == 1
        assert store.tags_for(1) == set()

    def test_remove_then_add_brings_back(self, tmp_path: Path):
        store = TagStore(tmp_path)
        store.add(1, "foo")
        store.remove(1, "foo")
        store.add(1, "foo")
        assert store.tags_for(1) == {"foo"}


class TestTagStoreReplay:
    def test_full_sequence(self, tmp_path: Path):
        store = TagStore(tmp_path)
        store.add(42, "a")
        store.add(42, "b")
        store.remove(42, "a")
        store.add(42, "a")
        store.remove(42, "c")
        assert store.tags_for(42) == {"a", "b"}
        assert store.tagged_seq_count() == 1


class TestTagStoreCorruptLine:
    def test_skips_garbage_with_warning(self, tmp_path: Path, capsys):
        path = tags_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps({"seq": 1, "tag": "foo", "op": "add"}) + "\n")
            f.write("this is not json\n")
            f.write(json.dumps({"seq": 2, "tag": "bar", "op": "add"}) + "\n")
        store = TagStore(tmp_path)
        all_tags = store.all_tags()
        assert all_tags == {1: {"foo"}, 2: {"bar"}}
        err = capsys.readouterr().err
        assert "corrupt tag entry" in err

    def test_skips_missing_field_with_warning(self, tmp_path: Path, capsys):
        path = tags_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps({"seq": 1, "tag": "foo", "op": "add"}) + "\n")
            f.write(json.dumps({"seq": 2, "tag": "bar"}) + "\n")  # no op
            f.write(json.dumps({"seq": 3, "tag": "baz", "op": "add"}) + "\n")
        store = TagStore(tmp_path)
        assert store.all_tags() == {1: {"foo"}, 3: {"baz"}}
        err = capsys.readouterr().err
        assert "corrupt tag entry" in err


class TestTagStoreMultipleSeqs:
    def test_all_tags_keyed_by_seq(self, tmp_path: Path):
        store = TagStore(tmp_path)
        store.add(1, "foo")
        store.add(1, "bar")
        store.add(2, "baz")
        store.add(2, "foo")
        assert store.all_tags() == {1: {"foo", "bar"}, 2: {"baz", "foo"}}

    def test_unique_tags_returns_union(self, tmp_path: Path):
        store = TagStore(tmp_path)
        store.add(1, "foo")
        store.add(1, "bar")
        store.add(2, "baz")
        store.add(2, "foo")
        assert store.unique_tags() == {"foo", "bar", "baz"}

    def test_tagged_seq_count(self, tmp_path: Path):
        store = TagStore(tmp_path)
        store.add(1, "foo")
        store.add(2, "bar")
        store.add(3, "baz")
        store.remove(2, "bar")
        assert store.tagged_seq_count() == 2


class TestTagStoreFileMissing:
    def test_tags_for_empty_when_no_file(self, tmp_path: Path):
        store = TagStore(tmp_path)
        assert store.tags_for(1) == set()

    def test_all_tags_empty_when_no_file(self, tmp_path: Path):
        store = TagStore(tmp_path)
        assert store.all_tags() == {}

    def test_unique_tags_empty_when_no_file(self, tmp_path: Path):
        store = TagStore(tmp_path)
        assert store.unique_tags() == set()

    def test_tagged_seq_count_zero_when_no_file(self, tmp_path: Path):
        store = TagStore(tmp_path)
        assert store.tagged_seq_count() == 0


class TestValidateTagWidenedCharset:
    def test_allows_hash_in_value(self):
        assert validate_tag("step-test#1") == "step-test#1"

    def test_allows_dot(self):
        assert validate_tag("wb.plan") == "wb.plan"

    def test_alphanum_dash_still_valid(self):
        assert validate_tag("agent-tester") == "agent-tester"

    def test_length_cap_still_64(self):
        with pytest.raises(ValueError):
            validate_tag("a" * 65)

    def test_equals_still_rejected(self):
        with pytest.raises(ValueError):
            validate_tag("plan=foo")

    def test_comma_still_rejected(self):
        with pytest.raises(ValueError):
            validate_tag("plan,bar")

    def test_extractor_does_not_widen_to_dot(self):
        assert extract_hashtags("look at #tag-one and #tag.two") == {"tag-one", "tag"}
