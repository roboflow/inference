"""Rewrite module paths after Phase 9's `git mv` of the Roboflow-platform blocks.

Two prefix rewrites, applied to the moved sources (they import each other) and
to every test that names them. Order-independent: if another phase already
repointed some references the count is lower, and `--expected` is the number
the caller measured immediately before. Every rewritten file is re-parsed with
`ast` before it is written, and the post-state (no stale prefix anywhere) is
verified after every write.
"""

import argparse
import ast
import pathlib
import sys

REPLACEMENTS = (
    (
        "inference.core.workflows.core_steps.sinks.roboflow",
        "inference.roboflow_workflows_plugin.sinks",
    ),
    (
        "inference.core.workflows.core_steps.integrations.roboflow",
        "inference.roboflow_workflows_plugin.integrations",
    ),
)


def rewrite(path: pathlib.Path) -> int:
    original = path.read_text(encoding="utf-8")
    updated = original
    hits = 0
    for old, new in REPLACEMENTS:
        hits += updated.count(old)
        updated = updated.replace(old, new)
    if not hits:
        return 0
    try:
        ast.parse(updated)
    except SyntaxError as error:
        print(f"FAIL: {path} would not parse: {error}", file=sys.stderr)
        raise SystemExit(2)
    path.write_text(updated, encoding="utf-8")
    return hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    total, touched, remaining = 0, [], []
    for root in args.roots:
        base = pathlib.Path(root)
        files = [base] if base.is_file() else sorted(base.rglob("*.py"))
        for path in files:
            if "__pycache__" in path.parts:
                continue
            hits = rewrite(path)
            if hits:
                total += hits
                touched.append((str(path), hits))
            text = path.read_text(encoding="utf-8")
            if any(old in text for old, _ in REPLACEMENTS):
                remaining.append(str(path))
    for name, hits in touched:
        print(f"{hits:4d}  {name}")
    print(f"TOTAL {total} replacements in {len(touched)} files")
    print(f"POST-STATE stale references remaining: {len(remaining)} {remaining}")
    if total != args.expected or remaining:
        print("FAIL", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
