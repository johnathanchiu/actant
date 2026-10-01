"""Keyed sandboxes

Revision ID: 5b0e2f7c9a41
Revises: 1326e6b76924

A sandbox owned by a key (``actant.sandbox.registry``): its live id and the spec it is
reopened with, so every worker reaches it by its key. New and empty.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "5b0e2f7c9a41"
down_revision: str | None = "1326e6b76924"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "actant_sandboxes",
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("sandbox_id", sa.Text(), nullable=False),
        sa.Column("spec", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("key"),
    )


def downgrade() -> None:
    op.drop_table("actant_sandboxes")
