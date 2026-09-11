"""Repoint `from inference.core.env import ...` at the Workflows configuration facade.

AST-located, byte-safe, idempotent. Only the module token on the statement's
FIRST physical line is rewritten - no name list, no expression and no body is
touched - so the codemod is order-independent with respect to Phases 9, 10 and
11, which rewrite different import statements in many of the same files.

    python scripts/repoint_env_imports.py            # rewrite
    python scripts/repoint_env_imports.py --check    # exit 1 if anything remains
    python scripts/repoint_env_imports.py --list     # print the files it would touch
"""

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_ROOT = REPO_ROOT / "inference" / "core" / "workflows"
OLD_MODULE = "inference.core.env"
NEW_MODULE = "inference.core.workflows.environment"

# Phase 9 relocates these two trees wholesale (controller ruling R-S); they
# keep their `env` imports until then, and cease to exist afterwards.
SKIP_PREFIXES = (
    WORKFLOWS_ROOT / "core_steps" / "sinks" / "roboflow",
    WORKFLOWS_ROOT / "core_steps" / "integrations" / "roboflow",
)


def _skipped(path: Path) -> bool:
    return any(prefix in path.parents for prefix in SKIP_PREFIXES)


def _statement_lines(source: str, path: Path) -> list:
    tree = ast.parse(source, filename=str(path))
    # ast.walk, not tree.body: `modal_executor.py` has four function-local
    # statements on top of its module-level one.
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == OLD_MODULE
        and node.level == 0
    )


def targets() -> list:
    found = []
    for path in sorted(WORKFLOWS_ROOT.rglob("*.py")):
        if _skipped(path) or "__pycache__" in str(path):
            continue
        source = path.read_bytes().decode("utf-8")
        lines = _statement_lines(source, path)
        if lines:
            found.append((path, lines))
    return found


def rewrite(path: Path, linenos: list) -> None:
    # Bytes + keepends: `core_steps/models/foundation/llama_vision/v1.py` is
    # CRLF (663 CRLF endings), and text mode with default newline translation
    # would rewrite every line ending in the file.
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines(keepends=True)
    for lineno in linenos:
        index = lineno - 1
        line = lines[index]
        if OLD_MODULE not in line:
            raise SystemExit(
                f"{path}:{lineno}: AST reported an `{OLD_MODULE}` import but the "
                f"first physical line does not contain the token: {line!r}"
            )
        lines[index] = line.replace(OLD_MODULE, NEW_MODULE, 1)
    path.write_bytes("".join(lines).encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    found = targets()
    if args.list or args.check:
        for path, linenos in found:
            print(f"{path.relative_to(REPO_ROOT)}: {linenos}")
        print(f"{len(found)} files, {sum(len(l) for _, l in found)} statements")
        return 1 if (args.check and found) else 0
    for path, linenos in found:
        rewrite(path, linenos)
    print(f"rewrote {len(found)} files, {sum(len(l) for _, l in found)} statements")
    return 0


if __name__ == "__main__":
    sys.exit(main())
