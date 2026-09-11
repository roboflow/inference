"""Bind `ModelManagerModelsProvider(model_manager)` at the four composition roots (Task 11.7).

AST-located, one value substitution per root: every expression bound to the
`"workflows_core.model_manager"` key - a dict-literal entry or a
`params["workflows_core.model_manager"] = ...` assignment - whose value is the
bare name `model_manager` becomes `ModelManagerModelsProvider(model_manager)`,
and the import is added once after the last top-level import. Nothing else on
the line changes, so the edit composes with whatever wrapper another phase
put around the dictionary (Phase 9's `install_workflows_platform_bindings({...})`,
Phase 5/6/10's extra keys). Re-parses before writing; idempotent (a file whose
bindings are already wrapped reports 0 and is left alone).

Run:  python scripts/phase11_bind_models_provider.py <root.py> [<root.py> ...]
"""

import ast
import sys
from pathlib import Path

KEY = "workflows_core.model_manager"
IMPORT = (
    "from inference.core.interfaces.workflows_models_provider import (\n"
    "    ModelManagerModelsProvider,\n"
    ")\n"
)


def _bound_values(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == KEY:
                    yield value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == KEY
                ):
                    yield node.value


def rewrite(path: Path) -> int:
    source = path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in source else "\n"
    lines = source.split(newline)
    tree = ast.parse(source)
    targets = [
        v
        for v in _bound_values(tree)
        if isinstance(v, ast.Name) and v.id == "model_manager"
    ]
    if not targets:
        print(f"{path}: 0 bindings rewritten (already wrapped or absent)")
        return 0
    for value in sorted(targets, key=lambda n: (n.lineno, n.col_offset), reverse=True):
        assert value.lineno == value.end_lineno
        line = lines[value.lineno - 1]
        assert line[value.col_offset : value.end_col_offset] == "model_manager"
        lines[value.lineno - 1] = (
            line[: value.col_offset]
            + "ModelManagerModelsProvider(model_manager)"
            + line[value.end_col_offset :]
        )
    if "ModelManagerModelsProvider" not in source:
        last_import = max(
            n.end_lineno
            for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom))
        )
        lines[last_import:last_import] = IMPORT.rstrip("\n").split("\n")
    updated = newline.join(lines)
    ast.parse(updated)
    remaining = [
        v
        for v in _bound_values(ast.parse(updated))
        if isinstance(v, ast.Name) and v.id == "model_manager"
    ]
    assert not remaining, "post-state: a raw binding survived"
    path.write_text(updated, encoding="utf-8")
    print(f"{path}: {len(targets)} bindings rewritten")
    return len(targets)


if __name__ == "__main__":
    total = sum(rewrite(Path(p)) for p in sys.argv[1:])
    print(f"TOTAL {total}")
