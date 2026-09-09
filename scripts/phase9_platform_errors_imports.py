"""Repoint `from inference.core.exceptions import …` at the relocated module.

Names are unchanged, so this rewrites only the module part of the ImportFrom
node - which is why the function-local import in email_notification/v2.py keeps
its indentation. Refuses any name that did not move.
"""

import argparse
import ast
import pathlib
import sys

MOVED = {
    "RoboflowAPIRequestError",
    "RoboflowAPIUnsuccessfulRequestError",
    "RoboflowAPIForbiddenError",
    "FeatureDeprecatedError",
}
NEW_MODULE = "inference.core.workflows.prototypes.platform_errors"
OLD_MODULE = "inference.core.exceptions"


def patch(path: pathlib.Path) -> int:
    source = path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in source else "\n"
    lines = source.split(newline)
    nodes = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module == OLD_MODULE
    ]
    if not nodes:
        print(f"SKIP (already repointed): {path}")
        return 0
    for node in nodes:
        unknown = [a.name for a in node.names if a.name not in MOVED]
        if unknown:
            print(f"FAIL: {path}:{node.lineno} imports unmoved names {unknown}", file=sys.stderr)
            raise SystemExit(2)
    for node in sorted(nodes, key=lambda n: n.lineno, reverse=True):
        # Replace only the module text; the `import ...` clause and the
        # statement's indentation are preserved verbatim.
        first = lines[node.lineno - 1]
        assert OLD_MODULE in first, (path, node.lineno, first)
        lines[node.lineno - 1] = first.replace(OLD_MODULE, NEW_MODULE, 1)
    updated = newline.join(lines)
    ast.parse(updated)
    if OLD_MODULE in updated:
        print(f"FAIL: {path} still imports {OLD_MODULE}", file=sys.stderr)
        raise SystemExit(2)
    path.write_text(updated, encoding="utf-8")
    return len(nodes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    total = 0
    for name in args.files:
        count = patch(pathlib.Path(name))
        total += count
        print(f"{count:3d}  {name}")
    print(f"TOTAL {total} import statements repointed")
    print(f"POST-STATE verified: no {OLD_MODULE} import remains in these files")
    if total != args.expected:
        print("FAIL: count does not match", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
