"""Repoint header building and `wrap_url` at the injected platform client.

All 24 call sites are inside methods, so `self._platform_client` is in scope.
Replacements are made on AST-located `Name` nodes in call position, so a
comment or a docstring mentioning the name is untouched. Every file is
transformed, re-parsed and post-state-verified BEFORE any file is written.
"""

import argparse
import ast
import pathlib
import re
import sys

RENAMES = {
    "build_roboflow_api_headers": "self._platform_client.build_api_headers",
    "get_extra_weights_provider_headers": "self._platform_client.build_weights_provider_headers",
    "wrap_url": "self._platform_client.wrap_url",
}
IMPORTS = re.compile(
    r"^from inference\.core\.(?:roboflow_api import (?:build_roboflow_api_headers"
    r"|get_extra_weights_provider_headers)|utils\.url_utils import wrap_url)[ \t]*\r?\n",
    re.MULTILINE,
)


def _call_sites(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in RENAMES:
                yield node.func


def transform(source: str, path: str):
    newline = "\r\n" if "\r\n" in source else "\n"
    lines = source.split(newline)
    sites = sorted(_call_sites(ast.parse(source)),
                   key=lambda n: (n.lineno, n.col_offset), reverse=True)
    for func in sites:
        index = func.lineno - 1
        line = lines[index]
        assert line[func.col_offset : func.end_col_offset] == func.id, (path, func.lineno)
        lines[index] = line[: func.col_offset] + RENAMES[func.id] + line[func.end_col_offset :]
    updated = newline.join(lines)
    removed = len(IMPORTS.findall(updated))
    updated = IMPORTS.sub("", updated)
    tree = ast.parse(updated)
    leftovers = [f.id for f in _call_sites(tree)]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {
            "inference.core.roboflow_api", "inference.core.utils.url_utils"}:
            leftovers.extend(a.name for a in node.names if a.name in RENAMES)
    if leftovers:
        print(f"FAIL: {path} still references {leftovers}", file=sys.stderr)
        raise SystemExit(2)
    return updated, len(sites), removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--expected-calls", type=int, required=True)
    parser.add_argument("--expected-imports", type=int, required=True)
    args = parser.parse_args()
    total_c = total_i = 0
    outputs = {}
    for name in args.files:
        path = pathlib.Path(name)
        updated, calls, imports = transform(path.read_text(encoding="utf-8"), name)
        again, _, _ = transform(updated, name)
        if again != updated:
            print(f"FAIL: {name} not idempotent", file=sys.stderr)
            return 2
        outputs[path] = updated
        total_c += calls
        total_i += imports
        print(f"{calls:3d} calls {imports:3d} imports  {name}")
    print(f"TOTAL {total_c} calls, {total_i} imports")
    if total_c != args.expected_calls or total_i != args.expected_imports:
        print("FAIL: counts do not match", file=sys.stderr)
        return 1
    for path, updated in outputs.items():
        path.write_text(updated, encoding="utf-8")
    print("POST-STATE verified: no bare header/wrap_url call or import remains; idempotent; written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
