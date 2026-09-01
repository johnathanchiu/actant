"""Alembic environment for authoring Actant's own migrations.

Not shipped and not what a consumer runs. An application embeds Actant's
revision files through ``version_locations`` and keeps its own ``env.py``;
this one exists so `just db-generate` can autogenerate against the runtime
metadata.
"""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

from actant.migrations import versions_path
from actant.runtime.stores.postgres import ACTANT_RUNTIME_METADATA

# The repo's own Postgres (`just demo-db-up`), on the uncommon port the
# compose file deliberately picks so it does not fight other projects'
# databases. A default of postgres:postgres on 5432 would do exactly that.
DEFAULT_DATABASE_URL = "postgresql+psycopg2://actant:actant@localhost:55435/actant_demo"

config = context.config
config.set_main_option(
    "sqlalchemy.url",
    os.environ.get("ACTANT_MIGRATIONS_DATABASE_URL", DEFAULT_DATABASE_URL),
)

target_metadata = ACTANT_RUNTIME_METADATA


def _sequential_revision_id(context, revision, directives) -> None:
    """Number revisions in order instead of by hash.

    Alembic's default is a random hex id. These revisions are read by people
    debugging someone else's deployment -- "which Actant revision is this
    database on" should be answerable at a glance, and hashes do not sort.

    Done here rather than by passing --rev-id so the convention holds without
    anyone remembering it.
    """
    del context, revision

    # Only numbering happens here. Emptying `directives` to suppress a no-op
    # revision also breaks `alembic check`, which runs this same hook and
    # then reads generated_revisions[-1].
    script = directives[0]
    existing = sorted(versions_path().glob("[0-9][0-9][0-9][0-9]_*.py"))
    nxt = int(existing[-1].name[:4]) + 1 if existing else 1
    script.rev_id = f"{nxt:04d}_{script.rev_id[:8]}"


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        # Off by default, and this is the only automated check that the
        # migrations still match the models: without it a changed
        # server_default writes no migration and every consumer keeps the
        # old default silently.
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_server_default=True,
            process_revision_directives=_sequential_revision_id,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
