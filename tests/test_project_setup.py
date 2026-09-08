"""Tests for project setup: package layout, metadata, dependencies, and entrypoint."""

import importlib.metadata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


class TestPackageMetadata:
    def test_version_attribute(self):
        import thirdeye

        # version is derived from git tags via setuptools-scm; just check it's a non-empty
        # PEP 440-compatible string (e.g. "0.1.0" or "0.1.0.dev3+gabc1234")
        assert isinstance(thirdeye.__version__, str)
        assert thirdeye.__version__
        assert thirdeye.__version__[0].isdigit()

    def test_installed_metadata_name(self):
        meta = importlib.metadata.metadata("thrdi")
        assert meta["Name"] == "thrdi"

    def test_installed_metadata_version(self):
        meta = importlib.metadata.metadata("thrdi")
        # tag-derived; just confirm it's set and starts with a digit
        assert meta["Version"]
        assert meta["Version"][0].isdigit()

    def test_requires_python(self):
        meta = importlib.metadata.metadata("thrdi")
        assert meta["Requires-Python"] == ">=3.11"

    def test_license(self):
        meta = importlib.metadata.metadata("thrdi")
        assert "MIT" in (meta.get("License") or meta.get("License-Expression") or "")


class TestDependencies:
    @pytest.mark.parametrize("dep", ["click", "msgpack", "zstandard", "pyaml"])
    def test_runtime_dependency_declared(self, dep):
        requires = importlib.metadata.requires("thrdi") or []
        dep_names = [
            r.split()[0].split(">")[0].split("<")[0].split("=")[0].split("!")[0].split(";")[0]
            for r in requires
        ]
        assert dep in dep_names, f"{dep} not found in declared dependencies: {requires}"

    def test_click_importable(self):
        import click

        assert hasattr(click, "group")

    def test_msgpack_importable(self):
        import msgpack

        assert hasattr(msgpack, "packb")

    def test_zstandard_importable(self):
        import zstandard

        assert hasattr(zstandard, "ZstdCompressor")

    def test_pyaml_importable(self):
        import yaml

        assert hasattr(yaml, "safe_load")

    def test_pytest_importable(self):
        import pytest as _pytest

        assert hasattr(_pytest, "fixture")


class TestPackageLayout:
    def test_src_layout(self):
        src_dir = ROOT / "src" / "thirdeye"
        assert src_dir.is_dir()

    def test_init_exists(self):
        init = ROOT / "src" / "thirdeye" / "__init__.py"
        assert init.is_file()

    def test_main_exists(self):
        main = ROOT / "src" / "thirdeye" / "__main__.py"
        assert main.is_file()

    def test_tests_init_exists(self):
        tests_init = ROOT / "tests" / "__init__.py"
        assert tests_init.is_file()

    def test_pyproject_exists(self):
        assert (ROOT / "pyproject.toml").is_file()

    def test_uv_lock_exists(self):
        assert (ROOT / "uv.lock").is_file()


class TestMainModule:
    def test_main_imports_from_cli(self):
        main_path = ROOT / "src" / "thirdeye" / "__main__.py"
        content = main_path.read_text()
        assert "from thirdeye.cli import run" in content

    def test_main_has_name_guard(self):
        main_path = ROOT / "src" / "thirdeye" / "__main__.py"
        content = main_path.read_text()
        assert 'if __name__ == "__main__":' in content


class TestConsoleScript:
    def test_entrypoint_declared(self):
        eps = importlib.metadata.entry_points(group="console_scripts")
        thirdeye_eps = [ep for ep in eps if ep.name == "thirdeye"]
        assert len(thirdeye_eps) == 1
        assert thirdeye_eps[0].value == "thirdeye.cli:main"


class TestPyprojectContent:
    def setup_method(self):
        self.content = (ROOT / "pyproject.toml").read_text()

    def test_build_system(self):
        assert "setuptools" in self.content
        assert 'build-backend = "setuptools.build_meta"' in self.content

    def test_packages_find_where_src(self):
        assert 'where = ["src"]' in self.content

    def test_click_version_constraint(self):
        assert '"click>=8.1"' in self.content

    def test_msgpack_version_constraint(self):
        assert '"msgpack>=1.0"' in self.content

    def test_zstandard_version_constraint(self):
        assert '"zstandard>=0.22"' in self.content

    def test_pyaml_version_constraint(self):
        assert '"pyaml>=23.0"' in self.content

    def test_pytest_dev_dependency(self):
        assert '"pytest>=8.0"' in self.content

    def test_console_script_target(self):
        # `run`, not `main`: the console script has to set stdio to UTF-8
        # before Click writes anything. See thirdeye.cli.run.
        assert 'thirdeye = "thirdeye.cli:run"' in self.content
