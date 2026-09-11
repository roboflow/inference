"""Emit the `inference.core.env` symbol -> call-site manifest for the workflows tree.

Every count in DECONTAMINATION.PLAN.PHASE-5.MD comes from this script. Phase 9
owns the rows under `core_steps/sinks/roboflow/**` and
`core_steps/integrations/roboflow/**` (controller ruling R-S), so they are
reported separately and excluded from the "owned" figures.

    python scripts/generate_env_import_manifest.py --summary
    python scripts/generate_env_import_manifest.py --json out.json
"""

import argparse
import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_ROOT = REPO_ROOT / "inference" / "core" / "workflows"
ENV_MODULE = "inference.core.env"
PHASE_9_PREFIXES = (
    WORKFLOWS_ROOT / "core_steps" / "sinks" / "roboflow",
    WORKFLOWS_ROOT / "core_steps" / "integrations" / "roboflow",
)


def phase_9_owned(path: Path) -> bool:
    return any(prefix in path.parents for prefix in PHASE_9_PREFIXES)


def _function_local_nodes(tree: ast.AST) -> set:
    local = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.ImportFrom):
                    local.add(id(sub))
    return local


def build() -> dict:
    symbols, owned_files, owned_statements = {}, set(), 0
    phase_9_files, phase_9_statements = set(), 0
    for path in sorted(WORKFLOWS_ROOT.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        tree = ast.parse(path.read_bytes().decode("utf-8"), filename=str(path))
        local = _function_local_nodes(tree)
        relative = path.relative_to(REPO_ROOT).as_posix()
        deferred = phase_9_owned(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != ENV_MODULE or node.level != 0:
                continue
            if deferred:
                phase_9_files.add(relative)
                phase_9_statements += 1
                continue
            owned_files.add(relative)
            owned_statements += 1
            for alias in node.names:
                symbols.setdefault(alias.name, []).append(
                    {
                        "file": relative,
                        "line": node.lineno,
                        "function_local": id(node) in local,
                    }
                )
    return {
        "symbols": {k: symbols[k] for k in sorted(symbols)},
        "owned_symbols": sorted(symbols),
        "owned_files": sorted(owned_files),
        "owned_statements": owned_statements,
        "phase_9_files": sorted(phase_9_files),
        "phase_9_statements": phase_9_statements,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", metavar="PATH")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    manifest = build()
    if args.json:
        Path(args.json).write_text(json.dumps(manifest, indent=1))
    if args.summary or not args.json:
        print(f"owned symbols:    {len(manifest['owned_symbols'])}")
        print(f"owned files:      {len(manifest['owned_files'])}")
        print(f"owned statements: {manifest['owned_statements']}")
        print(f"phase 9 files:    {len(manifest['phase_9_files'])}")
        print(f"phase 9 stmts:    {manifest['phase_9_statements']}")
        multi = [
            f
            for f in manifest["owned_files"]
            if sum(
                1
                for sites in manifest["symbols"].values()
                for s in sites
                if s["file"] == f
            )
            and len(
                {
                    s["line"]
                    for sites in manifest["symbols"].values()
                    for s in sites
                    if s["file"] == f
                }
            )
            > 1
        ]
        print(f"files with >1 statement: {multi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
