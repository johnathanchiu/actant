"""Sandbox id on threads, stop reason on runs

Revision ID: 0002_sandbox_and_run_reason
Revises: 0001_actant_runtime

A thread that opened a sandbox records its id, so any worker reattaches to
the same one. A run that ended for a reason the status does not carry (a
task agent that stopped without finishing) records it.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_sandbox_and_run_reason"
down_revision: str | None = "0001_actant_runtime"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("actant_threads", sa.Column("sandbox_id", sa.Text(), nullable=True))
    op.add_column("actant_runs", sa.Column("stop_reason", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("actant_runs", "stop_reason")
    op.drop_column("actant_threads", "sandbox_id")
