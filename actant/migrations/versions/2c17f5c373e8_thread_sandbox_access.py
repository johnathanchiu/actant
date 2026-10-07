"""A thread records the sandbox access it was started with (all NULL: the definition's)

Revision ID: 2c17f5c373e8
Revises: a4f2c8e61b07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "2c17f5c373e8"
down_revision: str | None = "a4f2c8e61b07"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("actant_threads", sa.Column("sandbox_user", sa.Text(), nullable=True))
    op.add_column(
        "actant_threads", sa.Column("sandbox_writable", postgresql.ARRAY(sa.Text()), nullable=True)
    )
    op.add_column("actant_threads", sa.Column("sandbox_scratch", sa.Boolean(), nullable=True))


def downgrade() -> None:
    op.drop_column("actant_threads", "sandbox_scratch")
    op.drop_column("actant_threads", "sandbox_writable")
    op.drop_column("actant_threads", "sandbox_user")
