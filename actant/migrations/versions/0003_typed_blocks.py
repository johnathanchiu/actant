"""Stored content blocks take the typed shapes

Revision ID: 0003_typed_blocks
Revises: 0002_sandbox_and_run_reason

A data migration: ``actant_message_parts.content_blocks`` is rewritten into the
shapes ``actant.blocks`` validates, so threads written before 0.20.0 still load.

- An image block with a URL source and a storage ``key`` (what the old
  ``prepare_messages`` read) becomes an ``AssetBlock`` of that key, its mime the
  source's ``media_type`` or ``image/png``.
- An image block with a URL source and no key becomes a text note. A signed URL
  expires, and history never stores one.
- An ``input_image`` block (the OpenAI Responses wire shape) with a ``data:``
  URL becomes an ``InlineImageBlock`` of its bytes; ``detail`` is dropped, the
  typed block has no field for it. With any other URL it becomes the note.

Every other block is left as it is, and every row is validated afterwards: a
shape this migration does not know fails the upgrade instead of being dropped.
Running it again changes nothing. It cannot be reversed: the URLs it replaces
are gone.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pydantic import ValidationError
from sqlalchemy.dialects.postgresql import JSONB

from actant.blocks import BLOCKS

revision: str = "0003_typed_blocks"
down_revision: str | None = "0002_sandbox_and_run_reason"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UNAVAILABLE = "[Image unavailable: stored before images were kept by key]"
_DATA_URL = re.compile(r"^data:(image/[\w.+-]+);base64,(.*)$", re.DOTALL)
_BATCH = 500
_log = logging.getLogger("alembic.runtime.migration")


def convert_block(block: object, counts: Counter[str]) -> object:
    """The typed shape of one legacy block; any other block unchanged."""
    if not isinstance(block, dict):
        return block
    kind = block.get("type")
    source = block.get("source")
    if kind == "image" and isinstance(source, dict) and source.get("type") == "url":
        key = source.get("key")
        if isinstance(key, str) and key:
            counts["url image with key -> asset"] += 1
            mime = source.get("media_type")
            return {
                "type": "asset",
                "storage_key": key,
                "mime": mime if isinstance(mime, str) and mime else "image/png",
            }
        counts["url image without key -> note"] += 1
        return {"type": "text", "text": UNAVAILABLE}
    if kind == "input_image":
        url = block.get("image_url")
        if isinstance(url, str) and url.startswith("data:"):
            match = _DATA_URL.match(url)
            if match is None:
                return block  # not a base64 image: fails validation below
            counts["input_image data url -> inline image"] += 1
            media_type, data = match.groups()
            source = {"type": "base64", "media_type": media_type, "data": data}
            return {"type": "image", "source": source}
        counts["input_image url -> note"] += 1
        return {"type": "text", "text": UNAVAILABLE}
    return block


def upgrade() -> None:
    connection = op.get_bind()
    parts = sa.table(
        "actant_message_parts",
        sa.column("message_id", sa.Text()),
        sa.column("part_index", sa.Integer()),
        sa.column("content_blocks", JSONB()),
    )
    update = (
        parts.update()
        .where(parts.c.message_id == sa.bindparam("m"), parts.c.part_index == sa.bindparam("p"))
        .values(content_blocks=sa.bindparam("blocks", type_=JSONB()))
    )
    # A server-side cursor, fetched _BATCH rows at a time.
    rows = connection.execute(
        sa.select(parts.c.message_id, parts.c.part_index, parts.c.content_blocks).where(
            parts.c.content_blocks.isnot(None)
        ),
        execution_options={"stream_results": True, "yield_per": _BATCH},
    )
    counts: Counter[str] = Counter()
    invalid: list[str] = []
    changed = 0
    for batch in rows.partitions():
        updates: list[dict[str, object]] = []
        for message_id, part_index, blocks in batch:
            if blocks is None:  # JSON null: a part with no blocks
                continue
            converted = (
                [convert_block(block, counts) for block in blocks]
                if isinstance(blocks, list)
                else blocks
            )
            try:
                BLOCKS.validate_python(converted)
            except ValidationError as exc:
                invalid.append(f"{message_id} part {part_index}: {exc.errors()[0]['msg']}")
                continue
            if converted != blocks:
                updates.append({"m": message_id, "p": part_index, "blocks": converted})
        if updates:
            connection.execute(update, updates)
            changed += len(updates)
    if invalid:
        raise RuntimeError(
            f"{len(invalid)} content_blocks rows do not validate after conversion; "
            "the upgrade rolls back. First: " + "; ".join(invalid[:5])
        )
    _log.info("0003_typed_blocks: %d rows rewritten; blocks %s", changed, dict(counts))


def downgrade() -> None:
    raise NotImplementedError(
        "0003_typed_blocks is irreversible: the legacy URL and input_image blocks it replaced "
        "are gone. Restore a backup taken before the upgrade instead."
    )
