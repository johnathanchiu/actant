"""Sandboxes by id

Revision ID: 5b0e2f7c9a41
Revises: 1326e6b76924

A sandbox by the id its owner picked (``actant.sandbox.registry``): its backend's live id
and the spec it is reopened with, so every worker reaches it by its id. New and empty.
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
        sa.Column("sandbox_id", sa.Text(), nullable=False),
        sa.Column("provider_id", sa.Text(), nullable=False),
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
        sa.PrimaryKeyConstraint("sandbox_id"),
    )


def downgrade() -> None:
    op.drop_table("actant_sandboxes")
