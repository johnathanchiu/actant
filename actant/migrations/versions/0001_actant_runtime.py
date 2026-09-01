"""The Actant runtime schema

Revision ID: 0001_actant_runtime
Revises:

The five tables Actant's SQLAlchemy runtime stores read and write: threads,
runs, messages and their ordered parts, and tool calls.

This is a branch, labelled ``actant``. An application embeds it through
``version_locations`` and keeps its own revisions on its own branch, so
upgrading the package brings its DDL along instead of leaving the
application to notice that a dependency changed shape.

Adopting this on a database that already has the tables -- one where an
application created them itself before Actant shipped migrations -- means
stamping the branch rather than running it:

    alembic stamp actant@head

That records the branch as applied without re-issuing the DDL. Every later
Actant revision then applies normally.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_actant_runtime"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = ("actant",)
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "actant_messages",
        sa.Column("message_id", sa.Text(), nullable=False),
        sa.Column("agent_id", sa.Text(), nullable=False),
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("turn_id", sa.Text(), nullable=True),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("message_id"),
    )
    op.create_table(
        "actant_runs",
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("agent_id", sa.Text(), nullable=False),
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("turn_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("max_turns", sa.Integer(), server_default=sa.text("25"), nullable=False),
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
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_table(
        "actant_threads",
        sa.Column("agent_id", sa.Text(), nullable=False),
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("turn_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("active_run_id", sa.Text(), nullable=True),
        sa.Column("parent_thread_id", sa.Text(), nullable=True),
        sa.Column("parent_turn_id", sa.Text(), nullable=True),
        sa.Column("parent_tool_call_id", sa.Text(), nullable=True),
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
        sa.PrimaryKeyConstraint("agent_id", "thread_id"),
    )
    op.create_table(
        "actant_tool_calls",
        sa.Column("tool_call_id", sa.Text(), nullable=False),
        sa.Column("group_id", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("agent_id", sa.Text(), nullable=False),
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("turn_id", sa.Text(), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "args",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=True),
        sa.Column("wait_request", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
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
        sa.PrimaryKeyConstraint("tool_call_id"),
    )
    op.create_table(
        "actant_message_parts",
        sa.Column("message_id", sa.Text(), nullable=False),
        sa.Column("part_index", sa.Integer(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("content_blocks", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("signature", sa.Text(), nullable=True),
        sa.Column("reasoning_items", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("tool_call_id", sa.Text(), nullable=True),
        sa.Column("tool_name", sa.Text(), nullable=True),
        sa.Column("args", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("wait_status", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["message_id"], ["actant_messages.message_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("message_id", "part_index"),
    )
    op.create_index(
        "ix_actant_message_parts_tool_call_id",
        "actant_message_parts",
        ["tool_call_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_actant_message_parts_tool_call_id", table_name="actant_message_parts")
    op.drop_table("actant_message_parts")
    op.drop_table("actant_tool_calls")
    op.drop_table("actant_threads")
    op.drop_table("actant_runs")
    op.drop_table("actant_messages")
