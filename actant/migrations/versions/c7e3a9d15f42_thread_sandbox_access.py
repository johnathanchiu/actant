"""A thread records what it may write in its sandbox

Revision ID: c7e3a9d15f42
Revises: a4f2c8e61b07

``actant_threads.sandbox_access`` holds the ``SandboxAccess`` its starter gave the thread
(``spawn`` / ``send_message``), as its fields. NULL on existing rows: the agent definition's.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c7e3a9d15f42"
down_revision: str | None = "a4f2c8e61b07"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "actant_threads",
        sa.Column("sandbox_access", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("actant_threads", "sandbox_access")
