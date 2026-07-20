"""Keep fenced Python examples syntactically valid.

The documentation contains both complete examples and fragments that rely on
application-owned objects. Wrapping each block in an async function lets us
compile both ordinary and top-level-await snippets without pretending those
external services exist in the unit-test process.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest

from actant.tools import InMemorySubagentRegistry, TaskTool, ToolRegistry

ROOT = Path(__file__).parents[1]
DOCUMENTS = [
    ROOT / "README.md",
    *sorted((ROOT / "docs").glob("*.md")),
    *[
        path
        for path in sorted((ROOT / "examples").rglob("*.md"))
        if not {".venv", "node_modules"}.intersection(path.parts)
    ],
]
PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.DOTALL)


def _snippets() -> list[object]:
    snippets: list[object] = []
    for document in DOCUMENTS:
        for index, match in enumerate(PYTHON_BLOCK.finditer(document.read_text()), start=1):
            snippets.append(
                pytest.param(
                    document,
                    index,
                    match.group(1),
                    id=f"{document.relative_to(ROOT)}:{index}",
                )
            )
    return snippets


@pytest.mark.parametrize(("document", "index", "source"), _snippets())
def test_python_snippet_compiles(document: Path, index: int, source: str) -> None:
    wrapped = "async def _example():\n" + textwrap.indent(source, "    ")
    ast.parse(wrapped, filename=f"{document}#python-{index}")


def test_python_snippets_are_ruff_formatted(tmp_path: Path) -> None:
    snippet_paths: list[str] = []
    for document, index, source in (
        (document, index, source)
        for document in DOCUMENTS
        for index, match in enumerate(PYTHON_BLOCK.finditer(document.read_text()), start=1)
        for source in [match.group(1)]
    ):
        path = tmp_path / f"{document.relative_to(ROOT).as_posix().replace('/', '__')}-{index}.py"
        path.write_text(source)
        snippet_paths.append(str(path))

    subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", *snippet_paths],
        check=True,
    )


@pytest.mark.parametrize(
    ("document", "index", "source"),
    [
        pytest.param(
            document,
            index,
            match.group(1),
            id=f"{document.relative_to(ROOT)}:{index}",
        )
        for document in DOCUMENTS
        for index, match in enumerate(BASH_BLOCK.finditer(document.read_text()), start=1)
    ],
)
def test_bash_snippet_parses(document: Path, index: int, source: str) -> None:
    subprocess.run(
        ["bash", "-n"],
        input=source,
        text=True,
        check=True,
    )


def test_readme_uuid_thread_id_round_trips() -> None:
    thread_id = uuid4().hex

    assert UUID(thread_id).hex == thread_id


def test_readme_subagent_registry_constructs_task_tool() -> None:
    registry = InMemorySubagentRegistry({})
    tools = ToolRegistry([TaskTool(invoker=registry)])

    schema = tools.schemas_for()[0]
    function = cast(dict[str, object], schema["function"])
    assert function["name"] == "task"
