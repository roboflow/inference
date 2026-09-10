"""Add `platform_client=` where a test reaches a signature Task 9.4 changed.

Two edits, both origin-resolved so a same-named helper in an untouched module
(`anthropic_claude/v2`, `google_gemini/v2`, `lmm/v1`, `openai/v1`) is left alone:

  1. a call whose callee resolves - through `from X import name [as alias]`,
     through `module.name` where `module` was imported from a touched package,
     or through a `pytest.mark.parametrize` argument whose values are such
     imported names - to one of the changed modules gains
     `platform_client=platform_client`;
  2. a keyword-form construction of one of the named block classes gains the
     same keyword (positional constructions are refused: none exist).

The file gains, once, after its last top-level import:

    from tests.workflows.unit_tests.prototypes.platform_client_double import (
        RecordingPlatformClient,
    )

    platform_client = RecordingPlatformClient()


    @pytest.fixture(autouse=True)
    def _reset_platform_client():
        platform_client.reset()

so no test signature or decorator order changes and no return value leaks
between tests. Tests that seed a response or assert on the call use
`platform_client.post_mock` (a `unittest.mock.Mock`), which keeps every
existing `mock_post.*` assertion verbatim - see Task 9.4 Step 8.

Every file's rewrite is computed fully in memory first (`patch()` never
writes); `main()` validates the aggregate `--expected-calls` /
`--expected-constructions` counts against the in-memory results and only
then writes - a count mismatch never leaves any file half-rewritten.
"""

import argparse
import ast
import collections
import pathlib
import sys
from typing import Optional

CHAIN = {
    "run_gpt_4v_llm_prompting",
    "execute_gpt_4v_requests",
    "execute_gpt_4v_request",
    "_execute_proxied_openai_request",
    "run_openai_prompting",
    "execute_openai_requests",
    "execute_openai_request",
    "run_gemini_prompting",
    "execute_gemini_requests",
    "execute_gemini_request",
    "_execute_proxied_gemini_request",
    "run_claude_prompting",
    "execute_claude_requests",
    "execute_claude_request",
    "_execute_proxied_claude_request",
    "run_spacexai_prompting",
    "execute_spacexai_requests",
    "execute_spacexai_request",
    "_execute_proxied_spacexai_request",
    "_execute_proxied_google_vision_request",
    "send_email_via_roboflow_proxy",
    "send_sms_via_roboflow_proxy",
    "_execute_proxied_openrouter_request",
}
PREAMBLE = (
    "from tests.workflows.unit_tests.prototypes.platform_client_double import (\n"
    "    RecordingPlatformClient,\n"
    ")\n"
    "\n"
    "platform_client = RecordingPlatformClient()\n"
    "\n"
    "\n"
    "@pytest.fixture(autouse=True)\n"
    "def _reset_platform_client():\n"
    "    platform_client.reset()"
)


def _touched_modules(list_file: str):
    return {
        line.strip().replace("/", ".")[:-3]
        for line in open(list_file, encoding="utf-8")
        if line.strip()
    }


def _affected_calls(tree, touched, classes):
    origin, alias = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                origin[a.asname or a.name] = (node.module, a.name)
                alias[a.asname or a.name] = f"{node.module}.{a.name}"
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    param_origins = collections.defaultdict(set)
    for fn in functions:
        for dec in fn.decorator_list:
            if not (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Attribute)
                and dec.func.attr == "parametrize"
            ):
                continue
            if len(dec.args) < 2 or not isinstance(dec.args[0], ast.Constant):
                continue
            names = [s.strip() for s in dec.args[0].value.split(",")]
            if not isinstance(dec.args[1], (ast.List, ast.Tuple)):
                continue
            for elt in dec.args[1].elts:
                items = elt.elts if isinstance(elt, (ast.Tuple, ast.List)) else [elt]
                for pname, item in zip(names, items):
                    if isinstance(item, ast.Name) and item.id in origin:
                        param_origins[(fn.name, pname)].add(origin[item.id])
    helper_calls, constructions = {}, {}
    for fn in functions:
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            mods = set()
            if isinstance(f, ast.Name):
                if f.id in origin and origin[f.id][1] in CHAIN:
                    mods.add(origin[f.id][0])
                elif (fn.name, f.id) in param_origins:
                    for mod, name in param_origins[(fn.name, f.id)]:
                        if name in CHAIN:
                            mods.add(mod)
                elif (
                    f.id in origin
                    and origin[f.id][1] in classes
                    and origin[f.id][0] in touched
                ):
                    if node.args:
                        raise SystemExit(
                            f"positional construction at line {node.lineno}"
                        )
                    constructions[(node.lineno, node.col_offset)] = node
                    continue
            elif (
                isinstance(f, ast.Attribute)
                and f.attr in CHAIN
                and isinstance(f.value, ast.Name)
            ):
                if f.value.id in alias:
                    mods.add(alias[f.value.id])
            if mods and any(m in touched for m in mods):
                helper_calls[(node.lineno, node.col_offset)] = node
    return list(helper_calls.values()), list(constructions.values())


def patch(path: pathlib.Path, touched, classes) -> tuple[int, int, Optional[str]]:
    """Compute the rewrite for `path` without writing anything.

    Returns `(edited_calls, edited_ctors, updated)`. `updated` is `None` when
    nothing needed editing (nothing to write).
    """
    source = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in source else "\n"
    lines = source.split(newline)
    tree = ast.parse(source)
    calls, ctors = _affected_calls(tree, touched, classes)
    edits = []
    edited_calls = edited_ctors = 0
    for node in calls + ctors:
        keys = {k.arg for k in node.keywords}
        if "platform_client" in keys:
            continue
        if node in calls:
            edited_calls += 1
        else:
            edited_ctors += 1
        if "roboflow_api_key" in keys:
            kw = next(k for k in node.keywords if k.arg == "roboflow_api_key")
            edits.append((kw.value.end_lineno, kw.value.end_col_offset))
        elif node.keywords:
            last = node.keywords[-1].value
            edits.append((last.end_lineno, last.end_col_offset))
        else:
            raise SystemExit(f"{path}:{node.lineno}: call without keywords")
    for lineno, col in sorted(set(edits), reverse=True):
        line = lines[lineno - 1]
        lines[lineno - 1] = (
            line[:col] + ", platform_client=platform_client" + line[col:]
        )
    updated = newline.join(lines)
    if edits and "RecordingPlatformClient" not in updated:
        out = updated.split(newline)
        new_tree = ast.parse(updated)
        anchor = None
        has_pytest = False
        for node in new_tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                anchor = node
                if isinstance(node, ast.Import) and any(
                    a.name == "pytest" for a in node.names
                ):
                    has_pytest = True
        if anchor is None:
            raise SystemExit(f"{path}: no import anchor")
        preamble = PREAMBLE if has_pytest else "import pytest\n\n" + PREAMBLE
        out.insert(anchor.end_lineno, preamble.replace("\n", newline))
        updated = newline.join(out)
    ast.parse(updated)
    if not edits:
        return edited_calls, edited_ctors, None
    return edited_calls, edited_ctors, updated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument(
        "--touched", required=True, help="file listing the changed modules"
    )
    parser.add_argument(
        "--classes", required=True, help="comma-separated block class names"
    )
    parser.add_argument("--expected-calls", type=int, required=True)
    parser.add_argument("--expected-constructions", type=int, required=True)
    args = parser.parse_args()
    touched = _touched_modules(args.touched)
    classes = set(args.classes.split(","))
    total_calls = total_ctors = 0
    pending_writes = []
    for name in args.files:
        path = pathlib.Path(name)
        calls, ctors, updated = patch(path, touched, classes)
        total_calls += calls
        total_ctors += ctors
        print(f"{calls:3d} calls {ctors:3d} constructions edited  {name}")
        if updated is not None:
            pending_writes.append((path, updated))
    print(f"TOTAL {total_calls} helper calls, {total_ctors} block constructions edited")
    if total_calls != args.expected_calls or total_ctors != args.expected_constructions:
        print("FAIL: counts do not match", file=sys.stderr)
        return 1

    # Validation above passed for the aggregate - only now do we write.
    for path, updated in pending_writes:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
