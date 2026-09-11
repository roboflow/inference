"""Rewrite module paths after Phase 9's `git mv` of the Roboflow-platform blocks.

Two prefix rewrites, applied to the moved sources (they import each other) and
to every test that names them. Order-independent: if another phase already
repointed some references the count is lower, and `--expected` is the number
the caller measured immediately before.

Every rewrite is computed fully in memory first: the original source is
parsed with `ast` (to catch a file that was already broken before we touched
it), the rewritten source is parsed too, and the aggregate `--expected` count
plus the post-state (no stale prefix anywhere) are both validated against the
in-memory results. Only once all of that passes does the script write
anything to disk - a count mismatch or a parse failure never leaves any file
half-rewritten.
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


def compute_rewrite(path: pathlib.Path):
    """Compute the rewritten text for `path` without writing anything.

    Returns `(hits, updated)`. `updated` is `None` when `hits == 0` (nothing
    to do). Both the original and the rewritten source are parsed with `ast`
    so a syntax problem - pre-existing or introduced by the rewrite - is
    caught before any file on disk is touched.
    """
    original = path.read_text(encoding="utf-8")
    updated = original
    hits = 0
    for old, new in REPLACEMENTS:
        hits += updated.count(old)
        updated = updated.replace(old, new)
    if not hits:
        return 0, None
    try:
        ast.parse(original)
    except SyntaxError as error:
        print(f"FAIL: {path} original source would not parse: {error}", file=sys.stderr)
        raise SystemExit(2)
    try:
        ast.parse(updated)
    except SyntaxError as error:
        print(f"FAIL: {path} would not parse after rewrite: {error}", file=sys.stderr)
        raise SystemExit(2)
    return hits, updated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()

    total, touched, remaining, pending_writes = 0, [], [], []
    for root in args.roots:
        base = pathlib.Path(root)
        files = [base] if base.is_file() else sorted(base.rglob("*.py"))
        for path in files:
            if "__pycache__" in path.parts:
                continue
            hits, updated = compute_rewrite(path)
            if not hits:
                continue
            total += hits
            touched.append((str(path), hits))
            pending_writes.append((path, updated))
            if any(old in updated for old, _ in REPLACEMENTS):
                remaining.append(str(path))

    for name, hits in touched:
        print(f"{hits:4d}  {name}")
    print(f"TOTAL {total} replacements in {len(touched)} files")
    print(f"POST-STATE stale references remaining: {len(remaining)} {remaining}")

    if total != args.expected or remaining:
        print("FAIL", file=sys.stderr)
        return 1

    if not pending_writes:
        print("SKIP: no-op, nothing to write")
        return 0

    # Validation above passed for the aggregate and every individual file -
    # only now do we write, and only the files that actually changed.
    for path, updated in pending_writes:
        path.write_text(updated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
