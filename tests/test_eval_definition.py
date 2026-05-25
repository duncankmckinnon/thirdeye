from __future__ import annotations

from pathlib import Path

import pytest

from thirdeye.eval.definition import (
    SHIPPED_NAMES,
    EvalDefinition,
    delete_definition,
    list_definitions,
    load_definition,
    save_definition,
    validate_directive_yaml,
)
from thirdeye.paths import eval_def_path


def test_load_shipped_default_materializes(tmp_path: Path):
    """First read of a shipped default copies it into the user's home."""
    assert not eval_def_path(tmp_path, "default").exists()
    defn = load_definition(tmp_path, "default")
    assert defn.name == "default"
    assert defn.directive  # non-empty
    assert eval_def_path(tmp_path, "default").is_file()


def test_load_unknown_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no eval definition"):
        load_definition(tmp_path, "nonexistent")


def test_load_user_copy_takes_precedence(tmp_path: Path):
    """User-edited shipped definitions are not overwritten on subsequent loads."""
    load_definition(tmp_path, "default")  # materialize
    user_path = eval_def_path(tmp_path, "default")
    user_path.write_text("name: default\ndirective: my override\n")
    defn = load_definition(tmp_path, "default")
    assert defn.directive == "my override"


def test_save_user_definition(tmp_path: Path):
    defn = EvalDefinition(
        name="my-eval",
        description="custom",
        directive="evaluate X",
    )
    path = save_definition(tmp_path, defn)
    assert path == eval_def_path(tmp_path, "my-eval")
    loaded = load_definition(tmp_path, "my-eval")
    assert loaded == defn


def test_save_rejects_existing_without_force(tmp_path: Path):
    defn = EvalDefinition(name="x", description="", directive="d")
    save_definition(tmp_path, defn)
    with pytest.raises(FileExistsError):
        save_definition(tmp_path, defn)


def test_save_with_force_overwrites(tmp_path: Path):
    defn = EvalDefinition(name="x", description="", directive="v1")
    save_definition(tmp_path, defn)
    save_definition(tmp_path, EvalDefinition(name="x", description="", directive="v2"), force=True)
    assert load_definition(tmp_path, "x").directive == "v2"


def test_delete_existing(tmp_path: Path):
    save_definition(tmp_path, EvalDefinition(name="x", description="", directive="d"))
    assert delete_definition(tmp_path, "x") is True
    assert not eval_def_path(tmp_path, "x").exists()


def test_delete_missing_returns_false(tmp_path: Path):
    assert delete_definition(tmp_path, "ghost") is False


def test_delete_shipped_allows_restoration(tmp_path: Path):
    load_definition(tmp_path, "default")  # materialize
    assert delete_definition(tmp_path, "default") is True
    # Next load re-materializes from the shipped copy
    defn = load_definition(tmp_path, "default")
    assert defn.directive  # shipped content


def test_list_includes_shipped_and_user(tmp_path: Path):
    save_definition(tmp_path, EvalDefinition(name="my-custom", description="", directive="d"))
    names = list_definitions(tmp_path)
    for n in SHIPPED_NAMES:
        assert n in names
    assert "my-custom" in names


def test_definition_round_trip_via_yaml(tmp_path: Path):
    defn = EvalDefinition(
        name="rt",
        description="round trip",
        directive="evaluate",
        default_agent="codex",
        output_schema="v1",
    )
    save_definition(tmp_path, defn)
    assert load_definition(tmp_path, "rt") == defn


def test_atomic_save_no_tmp_leftover(tmp_path: Path):
    save_definition(tmp_path, EvalDefinition(name="x", description="", directive="d"))
    path = eval_def_path(tmp_path, "x")
    assert not path.with_suffix(".yaml.tmp").exists()


def test_validate_directive_yaml_ok():
    ok, err = validate_directive_yaml("name: foo\ndirective: |\n  do the thing\n")
    assert ok is True
    assert err == ""


def test_validate_directive_yaml_parse_error():
    ok, err = validate_directive_yaml("name: ::: bad")
    assert ok is False
    assert "YAML parse error" in err


def test_validate_directive_yaml_not_mapping():
    ok, err = validate_directive_yaml("- just\n- a\n- list\n")
    assert ok is False
    assert "mapping" in err


def test_validate_directive_yaml_missing_required():
    ok, err = validate_directive_yaml("description: only\n")
    assert ok is False
    assert "missing required keys" in err
    assert "name" in err
    assert "directive" in err


def test_validate_directive_yaml_empty_directive():
    ok, err = validate_directive_yaml("name: foo\ndirective: ''\n")
    assert ok is False
    assert "directive" in err


def test_validate_directive_yaml_non_string_directive():
    ok, err = validate_directive_yaml("name: foo\ndirective: 42\n")
    assert ok is False
    assert "directive" in err
