"""Actant ships the migrations for the tables it owns.

The point of shipping them is that upgrading the package brings its DDL
along. That only holds if the revision files are actually packaged and
locatable at runtime, so both are asserted here rather than assumed -- a
build config that quietly drops them would otherwise surface as a missing
column in somebody else's database.
"""

from __future__ import annotations

import re

from actant.migrations import versions_path
from actant.runtime.stores.postgres import ACTANT_RUNTIME_METADATA


def test_the_revision_directory_ships_and_is_locatable() -> None:
    path = versions_path()
    assert path.is_dir()


def test_there_is_an_initial_revision_on_the_actant_branch() -> None:
    initial = versions_path() / "0001_actant_runtime.py"
    source = initial.read_text()

    assert "down_revision: str | None = None" in source
    assert '("actant",)' in source


def test_every_runtime_table_is_created_by_a_revision() -> None:
    """A table in the metadata with no CREATE is a table nobody migrates."""
    created = set()
    for revision in versions_path().glob("*.py"):
        created.update(re.findall(r"op\.create_table\(\s*['\"]([^'\"]+)", revision.read_text()))

    assert set(ACTANT_RUNTIME_METADATA.tables) <= created
