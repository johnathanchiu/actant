"""Signed URLs shared across processes

Revision ID: 0003_signed_urls
Revises: 0002_sandbox_and_run_reason

One live signed URL per stored object. Every worker process reads it back
instead of signing its own, so a thread's image URL stays byte-identical
across processes and restarts until it nears expiry, and the provider's
prompt cache keeps matching.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_signed_urls"
down_revision: str | None = "0002_sandbox_and_run_reason"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "actant_signed_urls",
        sa.Column("location", sa.Text(), primary_key=True),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.Double(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("actant_signed_urls")
