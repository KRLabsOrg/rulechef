"""The version has one source of truth: pyproject.toml (see issue #36)."""

import re
from pathlib import Path

import rulechef


def test_module_version_matches_pyproject():
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    declared = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE).group(1)
    assert rulechef.__version__ == declared
