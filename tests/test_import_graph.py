"""Keep internal package dependencies acyclic."""

from __future__ import annotations

import ast
from pathlib import Path


def test_actant_module_imports_are_acyclic() -> None:
    package = Path(__file__).parents[1] / "actant"
    module_by_path = {path: _module_name(path, package.parent) for path in package.rglob("*.py")}
    known_modules = set(module_by_path.values())
    graph = {module: set[str]() for module in known_modules}

    for path, module in module_by_path.items():
        tree = ast.parse(path.read_text(), filename=str(path))
        for imported in _absolute_imports(tree):
            dependency = _nearest_module(imported, known_modules)
            if dependency is not None and dependency != module:
                graph[module].add(dependency)

    assert _find_cycle(graph) is None


def _module_name(path: Path, package_parent: Path) -> str:
    module = ".".join(path.relative_to(package_parent).with_suffix("").parts)
    return module.removesuffix(".__init__")


def _absolute_imports(tree: ast.AST) -> list[str]:
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.append(node.module)
    return imports


def _nearest_module(imported: str, known_modules: set[str]) -> str | None:
    candidate = imported
    while candidate:
        if candidate in known_modules:
            return candidate
        candidate = candidate.rpartition(".")[0]
    return None


def _find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    visited: set[str] = set()
    active: list[str] = []
    active_set: set[str] = set()

    def visit(module: str) -> list[str] | None:
        if module in active_set:
            start = active.index(module)
            return [*active[start:], module]
        if module in visited:
            return None

        active.append(module)
        active_set.add(module)
        for dependency in graph[module]:
            cycle = visit(dependency)
            if cycle is not None:
                return cycle
        active.pop()
        active_set.remove(module)
        visited.add(module)
        return None

    for module in graph:
        cycle = visit(module)
        if cycle is not None:
            return cycle
    return None
