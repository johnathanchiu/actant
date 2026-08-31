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

from actant.runtime.stores.postgres import ACTANT_RUNTIME_METADATA

config = context.config
config.set_main_option(
    "sqlalchemy.url",
    os.environ.get(
        "ACTANT_MIGRATIONS_DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/actant_scratch",
    ),
)

target_metadata = ACTANT_RUNTIME_METADATA


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
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
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
