"""Replace `ModelEndpointType.CORE_MODEL` with the workflows-local constant.

BSD `sed` here does not honour `\\b`, so this is a Python codemod: it locates
the attribute accesses through `ast`, counts them, re-parses its own output and
verifies the post-state (no `ModelEndpointType` reference left in the file).
Idempotent: a file with no matches contributes zero and still passes.
"""

import argparse
import ast
import pathlib
import sys

NEW_IMPORT = (
    "from inference.core.workflows.prototypes.models_provider import (\n"
    "    CORE_MODEL_ENDPOINT_TYPE,\n"
    ")"
)


def patch(path: pathlib.Path) -> tuple[int, int]:
    source = path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in source else "\n"
    lines = source.split(newline)
    tree = ast.parse(source)

    usages = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "CORE_MODEL"
            and isinstance(node.value, ast.Name)
            and node.value.id == "ModelEndpointType"
        ):
            usages.append(node)
    import_span = None
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "inference.core.roboflow_api"
            and [a.name for a in node.names] == ["ModelEndpointType"]
        ):
            import_span = (node.lineno, node.end_lineno)
    if not usages and import_span is None:
        print(f"SKIP (already repointed): {path}")
        return 0, 0

    for node in sorted(usages, key=lambda n: (n.lineno, n.col_offset), reverse=True):
        index = node.lineno - 1
        line = lines[index]
        assert line[node.col_offset : node.end_col_offset] == "ModelEndpointType.CORE_MODEL"
        lines[index] = (
            line[: node.col_offset]
            + "CORE_MODEL_ENDPOINT_TYPE"
            + line[node.end_col_offset :]
        )
    if import_span is not None:
        start, end = import_span
        lines[start - 1 : end] = NEW_IMPORT.split("\n")
    updated = newline.join(lines)
    if "ModelEndpointType" in updated:
        print(f"FAIL: {path} still mentions ModelEndpointType", file=sys.stderr)
        raise SystemExit(2)
    ast.parse(updated)
    path.write_text(updated, encoding="utf-8")
    return (1 if import_span else 0), len(usages)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--expected-imports", type=int, required=True)
    parser.add_argument("--expected-usages", type=int, required=True)
    args = parser.parse_args()
    total_i = total_u = 0
    for name in args.files:
        i, u = patch(pathlib.Path(name))
        total_i += i
        total_u += u
        print(f"{i:3d} imports {u:3d} usages  {name}")
    print(f"TOTAL {total_i} imports, {total_u} usages")
    print("POST-STATE verified: no ModelEndpointType reference remains")
    if total_i != args.expected_imports or total_u != args.expected_usages:
        print("FAIL: counts do not match", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
