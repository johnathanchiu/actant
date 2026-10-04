"""Message cache tokens

Revision ID: a4f2c8e61b07
Revises: 8c1d4e2b7f30

The part of an assistant message's ``input_tokens`` read from, or written to, the
provider's prompt cache. NULL on existing rows and wherever the provider did not say.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a4f2c8e61b07"
down_revision: str | None = "8c1d4e2b7f30"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("actant_messages", sa.Column("cache_read_tokens", sa.Integer(), nullable=True))
    op.add_column("actant_messages", sa.Column("cache_write_tokens", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("actant_messages", "cache_write_tokens")
    op.drop_column("actant_messages", "cache_read_tokens")
