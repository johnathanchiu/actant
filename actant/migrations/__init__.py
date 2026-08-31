"""Actant's own Alembic revisions, and where to find them.

Actant owns the tables its SQLAlchemy runtime stores read and write, so it
ships the migrations for them. An application embeds this directory as a
second ``version_locations`` entry and keeps its own revisions on its own
branch; upgrading the package then brings the DDL with it, instead of
leaving the application to discover that a dependency changed shape.

Alembic resolves ``version_locations`` while building its ``ScriptDirectory``,
which happens before ``env.py`` is imported -- so the path cannot be computed
there. Read it from here instead, before invoking a command:

    from alembic import command
    from alembic.config import Config
    from actant.migrations import versions_path

    config = Config("alembic.ini")
    config.set_main_option("version_locations", f"{local_versions} {versions_path()}")
    config.set_main_option("version_path_separator", "space")
    command.upgrade(config, "heads")

``heads``, plural: with Actant's branch alongside the application's there is
more than one, and ``head`` raises rather than guessing.

On a database that already has these tables, stamp instead of running::

    command.stamp(config, "actant@head")
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["versions_path"]


def versions_path() -> Path:
    """The directory holding Actant's revision files.

    Resolved from this module's own location, so it is correct whether Actant
    is installed into site-packages or checked out as an editable path
    dependency.
    """
    return Path(__file__).parent / "versions"
