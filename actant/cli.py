"""Command-line utilities shipped with Actant."""

from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import subprocess
import sys
from importlib import metadata, resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError
from sqlalchemy import select, type_coerce
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from actant.blocks import BLOCKS
from actant.runtime.stores.postgres.models import ActantMessagePartModel

_DEFAULT_COMPOSE_PROJECT = "actant-local"


def _add_server_overrides(parser: argparse.ArgumentParser, *, inherited: bool) -> None:
    default = argparse.SUPPRESS if inherited else None
    parser.add_argument(
        "--compose-file",
        default=(default if inherited else os.getenv("ACTANT_SERVER_COMPOSE_FILE")),
        help="use a custom Docker Compose file (ACTANT_SERVER_COMPOSE_FILE)",
    )
    parser.add_argument(
        "--project-name",
        default=(
            default if inherited else os.getenv("ACTANT_SERVER_PROJECT", _DEFAULT_COMPOSE_PROJECT)
        ),
        help="override the Compose project name (ACTANT_SERVER_PROJECT)",
    )
    parser.add_argument(
        "--compose-command",
        default=(
            default if inherited else os.getenv("ACTANT_SERVER_COMPOSE_COMMAND", "docker compose")
        ),
        help=(
            "override the Compose command, such as 'podman compose' "
            "(ACTANT_SERVER_COMPOSE_COMMAND)"
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="actant")
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    commands = parser.add_subparsers(dest="command", required=True)

    server = commands.add_parser(
        "server",
        help="manage a local Temporal development server",
        description=(
            "Manage the Docker-backed Temporal development server used by Actant. "
            "Production deployments should connect to an independently managed "
            "Temporal service."
        ),
    )
    _add_server_overrides(server, inherited=False)
    actions = server.add_subparsers(dest="server_command", required=True)

    start = actions.add_parser("start", help="start Temporal and its local UI")
    start.add_argument(
        "--detach",
        "-d",
        action="store_true",
        help="run the Docker services in the background",
    )
    start.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("ACTANT_TEMPORAL_PORT", "7233")),
        help="Temporal gRPC port (ACTANT_TEMPORAL_PORT)",
    )
    start.add_argument(
        "--ui-port",
        type=int,
        default=int(os.getenv("ACTANT_TEMPORAL_UI_PORT", "8233")),
        help="Temporal UI port (ACTANT_TEMPORAL_UI_PORT)",
    )
    start.add_argument("--no-ui", action="store_true", help="start Temporal without its UI")

    stop = actions.add_parser("stop", help="stop the server and keep its data")
    status = actions.add_parser("status", help="show server container status")
    reset = actions.add_parser("reset", help="stop the server and delete its data")

    logs = actions.add_parser("logs", help="show Temporal server logs")
    logs.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="continue following new log output",
    )
    for action in (start, stop, status, reset, logs):
        _add_server_overrides(action, inherited=True)

    validate = commands.add_parser(
        "validate-blocks",
        help="report stored content blocks that do not validate",
        description=(
            "Read every actant_message_parts.content_blocks row and print the ones "
            "actant.blocks rejects. Exits 1 if any do. Run it before deploying a "
            "release that changes the block types."
        ),
    )
    validate.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="SQLAlchemy async URL, e.g. postgresql+asyncpg://... (DATABASE_URL)",
    )
    return parser


def _version() -> str:
    try:
        return metadata.version("actant")
    except metadata.PackageNotFoundError:  # pragma: no cover - source checkout fallback
        return "unknown"


def _compose_file() -> Traversable:
    return resources.files("actant").joinpath("_assets", "temporal-compose.yml")


def _run_compose(
    args: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    compose_file: str | None = None,
    project_name: str = _DEFAULT_COMPOSE_PROJECT,
    compose_command: str = "docker compose",
) -> int:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    resource: Traversable | Path = Path(compose_file) if compose_file else _compose_file()
    with resources.as_file(resource) as compose_path:
        command = [
            *shlex.split(compose_command),
            "--project-name",
            project_name,
            "--file",
            str(Path(compose_path)),
            *args,
        ]
        try:
            return subprocess.run(command, env=merged_env, check=False).returncode
        except FileNotFoundError:
            print(
                "Actant's local server requires Docker with the Compose plugin. "
                "Install Docker, then retry this command.",
                file=sys.stderr,
            )
            return 127


def _server(args: argparse.Namespace) -> int:
    def run(compose_args: Sequence[str], *, env: dict[str, str] | None = None) -> int:
        return _run_compose(
            compose_args,
            env=env,
            compose_file=args.compose_file,
            project_name=args.project_name,
            compose_command=args.compose_command,
        )

    if args.server_command == "start":
        compose_args = [
            "up",
            "--remove-orphans",
            "temporal-postgres",
            "temporal",
        ]
        if not args.no_ui:
            compose_args.append("temporal-ui")
        if args.detach:
            compose_args[1:1] = ["--detach", "--wait", "--wait-timeout", "60"]
        result = run(
            compose_args,
            env={
                "ACTANT_TEMPORAL_PORT": str(args.port),
                "ACTANT_TEMPORAL_UI_PORT": str(args.ui_port),
            },
        )
        if result == 0:
            print(f"Temporal: localhost:{args.port}")
            if not args.no_ui:
                print(f"Temporal UI: http://localhost:{args.ui_port}")
        return result

    if args.server_command == "stop":
        return run(["down"])
    if args.server_command == "status":
        return run(["ps"])
    if args.server_command == "reset":
        return run(["down", "--volumes"])
    if args.server_command == "logs":
        compose_args = ["logs"]
        if args.follow:
            compose_args.append("--follow")
        compose_args.append("temporal")
        return run(compose_args)
    raise AssertionError(f"unknown server command: {args.server_command}")


async def invalid_block_rows(engine: AsyncEngine) -> list[tuple[str, int, str]]:
    """``(message_id, part_index, error)`` for each stored block list that does not validate."""
    parts = ActantMessagePartModel.__table__
    # JSONB, not the model's validating type: a bad row is reported, not raised.
    raw = type_coerce(parts.c.content_blocks, JSONB)
    query = select(parts.c.message_id, parts.c.part_index, raw)
    invalid: list[tuple[str, int, str]] = []
    async with engine.connect() as connection:
        async for message_id, part_index, blocks in await connection.stream(query):
            if blocks is None:  # SQL NULL or JSON null: a part with no blocks
                continue
            try:
                BLOCKS.validate_python(blocks)
            except ValidationError as exc:
                errors = "; ".join(
                    f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
                )
                invalid.append((message_id, part_index, errors))
    return invalid


def _validate_blocks(database_url: str | None) -> int:
    if not database_url:
        print("validate-blocks needs --database-url or DATABASE_URL", file=sys.stderr)
        return 2

    async def run() -> list[tuple[str, int, str]]:
        engine = create_async_engine(database_url)
        try:
            return await invalid_block_rows(engine)
        finally:
            await engine.dispose()

    invalid = asyncio.run(run())
    for message_id, part_index, errors in invalid:
        print(f"{message_id} part {part_index}: {errors}")
    print(f"{len(invalid)} invalid content_blocks rows", file=sys.stderr)
    return 1 if invalid else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "server":
        return _server(args)
    if args.command == "validate-blocks":
        return _validate_blocks(args.database_url)
    raise AssertionError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
