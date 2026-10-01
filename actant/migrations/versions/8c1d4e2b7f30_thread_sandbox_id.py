"""A thread names the sandbox it works in by its id

Revision ID: 8c1d4e2b7f30
Revises: 5b0e2f7c9a41

``actant_threads.sandbox_id`` held a backend's id for the sandbox the thread owned; it now
holds the id of a sandbox a product opened for the thread (``actant.sandbox.registry``),
and a thread's own sandbox is recorded in ``actant_sandboxes`` under the thread id. The old
backend ids are cleared: a thread whose sandbox was open across the upgrade opens a new one
over the same files.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "8c1d4e2b7f30"
down_revision: str | None = "5b0e2f7c9a41"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("UPDATE actant_threads SET sandbox_id = NULL")


def downgrade() -> None:
    op.execute("UPDATE actant_threads SET sandbox_id = NULL")
