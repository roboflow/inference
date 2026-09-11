#!/usr/bin/env python
"""Repoint `from inference.core.utils.image_utils import ...` inside
`inference/core/workflows` to `inference.core.workflows.utils.images`.

Only the module path changes; every imported name already exists under the new
path (`inference/core/workflows/utils/images.py`), so no body edit is needed.

Located by AST, so function-local imports are caught (`modal_executor.py` has one
at line 860). Applied by substring replacement on the statement's FIRST line,
which is where the module path always is for both the single-line and the
parenthesised forms. Files are read and written as BYTES and split with
`splitlines(keepends=True)`, so the one CRLF file in the set
(`core_steps/models/foundation/llama_vision/v1.py`) keeps its line endings.

The script is order-independent with respect to Phase 9: it rewrites whatever it
finds and prints the counts, so it works whether or not the six relocating files
are still in the tree.

Usage:
    PYTHONPATH=inference_models python scripts/repoint_image_utils_imports.py [--check]
"""

import argparse
import ast
import sys
from pathlib import Path

OLD_MODULE = "inference.core.utils.image_utils"
NEW_MODULE = "inference.core.workflows.utils.images"
WORKFLOWS_ROOT = Path("inference/core/workflows")


def statements_to_rewrite(path: Path) -> list:
    source = path.read_bytes().decode("utf-8")
    tree = ast.parse(source)
    return sorted(
        {
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == OLD_MODULE
        }
    )


def rewrite(path: Path, check_only: bool) -> int:
    linenos = statements_to_rewrite(path)
    if not linenos:
        return 0
    raw = path.read_bytes().decode("utf-8")
    lines = raw.splitlines(keepends=True)
    for lineno in linenos:
        index = lineno - 1
        if OLD_MODULE not in lines[index]:
            raise SystemExit(
                f"{path}:{lineno}: expected the module path on the statement's "
                f"first line, found: {lines[index]!r}"
            )
        lines[index] = lines[index].replace(OLD_MODULE, NEW_MODULE)
    if not check_only:
        path.write_bytes("".join(lines).encode("utf-8"))
    return len(linenos)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="report, do not write")
    args = parser.parse_args()

    if not WORKFLOWS_ROOT.is_dir():
        raise SystemExit("run me from the repository root")

    files = 0
    statements = 0
    for path in sorted(WORKFLOWS_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rewritten = rewrite(path, check_only=args.check)
        if rewritten:
            files += 1
            statements += rewritten
            print(f"{path}: {rewritten} statement(s)")
    print(f"TOTAL files={files} statements={statements}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
