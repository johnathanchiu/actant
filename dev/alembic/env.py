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
from actant.runtime.stores.postgres.models import BlocksJSONB

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


def _render_item(type_: str, obj: object, autogen_context) -> str | bool:
    """``BlocksJSONB`` renders as the JSONB it stores in.

    Autogenerate otherwise writes the decorator's import path into the revision
    (``actant.runtime.stores.postgres.models.BlocksJSONB``), tying a migration to
    application code that later changes. The database only ever sees JSONB. This is
    Alembic's documented hook for it ("Affecting the Rendering of Types Themselves").
    """
    if type_ == "type" and isinstance(obj, BlocksJSONB):
        autogen_context.imports.add("from sqlalchemy.dialects import postgresql")
        return "postgresql.JSONB(astext_type=sa.Text())"
    return False


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
        render_item=_render_item,
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
            render_item=_render_item,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
