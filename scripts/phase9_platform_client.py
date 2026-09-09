"""Phase 9: give a Workflow block the `platform_client` port, in one pass.

Three transformations computed from ONE parse and applied as (line, col) spans
bottom-up, so they cannot reject each other's output:

  A  every top-level `def` with a `roboflow_api_key` parameter gains
     `platform_client: RoboflowPlatformClient` right after it, and every call
     passing `roboflow_api_key=<expr>` gains `platform_client=<mapped expr>`;
  B  the named block class gains the `platform_client` init parameter, the
     `self._platform_client` assignment (or `platform_client=platform_client`
     forwarded to `super().__init__`), and `"platform_client"` in
     `get_init_parameters()`;
  C  `from inference.core.roboflow_api import post_to_roboflow_api` is dropped,
     `post_to_roboflow_api(` becomes `platform_client.post(`, and the port
     import is added.

Guards are SPECIFIC - a def's argument list, a call's keyword set, the
`get_init_parameters` literal, the constructor's argument list - never
file-wide or class-wide substring presence. That makes the script idempotent
and order-independent, and it is why Task 9.5 can reuse it on files that still
import OTHER names from `inference.core.roboflow_api`: the verifier rejects a
surviving `post_to_roboflow_api` import, not the module.

The constructor parameter is appended AFTER the last default value when the
signature has one (`arg.end_col_offset` stops after the annotation, before
` = False`), which is what `email_notification/v2.py` needs.
"""

import argparse
import ast
import pathlib
import sys

PARAM_ANNOTATION = "platform_client: RoboflowPlatformClient"
CTOR_PARAM = "platform_client: RoboflowPlatformClient = OFFLINE_PLATFORM_CLIENT"
MAPPING = {
    "roboflow_api_key": "platform_client",
    "self._api_key": "self._platform_client",
    "self._roboflow_api_key": "self._platform_client",
}
IMPORT_BLOCK = (
    "from inference.core.workflows.prototypes.platform_client import (\n"
    "    OFFLINE_PLATFORM_CLIENT,\n"
    "    RoboflowPlatformClient,\n"
    ")"
)
OLD_IMPORT_MODULE = "inference.core.roboflow_api"
PROXY_HELPER = "post_to_roboflow_api"


def _apply(lines, edits):
    """edits: (lineno, col, end_lineno, end_col, text), 1-based lines."""
    for lineno, col, end_lineno, end_col, text in sorted(
        edits, key=lambda e: (e[0], e[1]), reverse=True
    ):
        first, last = lines[lineno - 1], lines[end_lineno - 1]
        lines[lineno - 1 : end_lineno] = [first[:col] + text + last[end_col:]]
    return lines


def transform(source: str, class_name: str, path: str):
    newline = "\r\n" if "\r\n" in source else "\n"
    lines = source.split(newline)
    tree = ast.parse(source)
    edits = []
    stats = {"defs": 0, "calls": 0, "ctor": 0, "gip": 0, "super": 0,
             "post_calls": 0, "old_imports": 0}

    for node in tree.body:                                    # A: defs
        if not isinstance(node, ast.FunctionDef):
            continue
        names = [a.arg for a in node.args.args]
        if "roboflow_api_key" not in names or "platform_client" in names:
            continue
        if (node.args.vararg or node.args.kwarg or node.args.kwonlyargs
                or node.args.posonlyargs):
            raise SystemExit(f"{path}: unsupported signature {node.name}")
        arg = node.args.args[names.index("roboflow_api_key")]
        edits.append((arg.end_lineno, arg.end_col_offset, arg.end_lineno,
                      arg.end_col_offset, f", {PARAM_ANNOTATION}"))
        stats["defs"] += 1

    for node in ast.walk(tree):                               # A: calls
        if not isinstance(node, ast.Call):
            continue
        keys = {k.arg for k in node.keywords}
        if "roboflow_api_key" not in keys or "platform_client" in keys:
            continue
        kw = next(k for k in node.keywords if k.arg == "roboflow_api_key")
        expr = ast.unparse(kw.value)
        if expr not in MAPPING:
            raise SystemExit(
                f"{path}: unexpected roboflow_api_key expression {expr!r} "
                f"at line {kw.value.lineno}"
            )
        edits.append((kw.value.end_lineno, kw.value.end_col_offset,
                      kw.value.end_lineno, kw.value.end_col_offset,
                      f", platform_client={MAPPING[expr]}"))
        stats["calls"] += 1

    target = next((n for n in ast.walk(tree)                  # B: the class
                   if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    if target is None:
        raise SystemExit(f"{path}: no class {class_name}")
    init = next((n for n in target.body
                 if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    gip = next((n for n in target.body
                if isinstance(n, ast.FunctionDef) and n.name == "get_init_parameters"), None)
    if init is None or gip is None:
        raise SystemExit(f"{path}: {class_name} needs __init__ and get_init_parameters")

    ret = next((n for n in ast.walk(gip) if isinstance(n, ast.Return)), None)
    if ret is None or not isinstance(ret.value, ast.List):
        raise SystemExit(f"{path}: {class_name}.get_init_parameters must return a list")
    declared = [e.value for e in ret.value.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    if len(declared) != len(ret.value.elts):
        raise SystemExit(f"{path}: non-literal in get_init_parameters")
    if "platform_client" not in declared:
        rendered = "        return [" + ", ".join(
            f'"{n}"' for n in declared + ["platform_client"]) + "]"
        edits.append((ret.lineno, 0, ret.end_lineno,
                      len(lines[ret.end_lineno - 1]), rendered))
        stats["gip"] += 1

    if "platform_client" not in [a.arg for a in init.args.args]:
        last_arg = init.args.args[-1]
        if last_arg.arg == "self":
            raise SystemExit(f"{path}: {class_name}.__init__ takes no parameters")
        anchor = init.args.defaults[-1] if init.args.defaults else last_arg
        edits.append((anchor.end_lineno, anchor.end_col_offset, anchor.end_lineno,
                      anchor.end_col_offset, f", {CTOR_PARAM}"))
        stats["ctor"] += 1
        super_call = None
        for node in ast.walk(init):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "__init__"
                    and isinstance(node.func.value, ast.Call)
                    and isinstance(node.func.value.func, ast.Name)
                    and node.func.value.func.id == "super"):
                super_call = node
        if super_call is not None:
            last_kw = super_call.keywords[-1]
            edits.append((last_kw.value.end_lineno, last_kw.value.end_col_offset,
                          last_kw.value.end_lineno, last_kw.value.end_col_offset,
                          ", platform_client=platform_client"))
            stats["super"] += 1
        else:
            end = init.body[-1].end_lineno
            edits.append((end, len(lines[end - 1]), end, len(lines[end - 1]),
                          newline + "        self._platform_client = platform_client"))

    for node in ast.walk(tree):                               # C: the proxy call
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == PROXY_HELPER):
            edits.append((node.func.lineno, node.func.col_offset,
                          node.func.end_lineno, node.func.end_col_offset,
                          "platform_client.post"))
            stats["post_calls"] += 1
    old_import_span = None
    for node in tree.body:
        if (isinstance(node, ast.ImportFrom) and node.module == OLD_IMPORT_MODULE
                and [a.name for a in node.names] == [PROXY_HELPER]):
            old_import_span = (node.lineno, node.end_lineno)
            stats["old_imports"] += 1

    updated = newline.join(_apply(lines, edits))
    if old_import_span is not None:
        out = updated.split(newline)
        start, end = old_import_span
        del out[start - 1 : end]
        updated = newline.join(out)
    if "prototypes.platform_client" not in updated:
        out = updated.split(newline)
        anchor = None
        for node in ast.parse(updated).body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                anchor = node
        if anchor is None:
            raise SystemExit(f"{path}: no import anchor")
        out.insert(anchor.end_lineno, IMPORT_BLOCK)
        updated = newline.join(out)
    ast.parse(updated)
    return updated, stats


def verify(source: str, class_name: str, path: str):
    """Post-state, SPECIFIC to what this script removes: the proxy helper.

    Other `inference.core.roboflow_api` names (header builders in the Task 9.5
    inputs) legitimately survive this script; `scripts/phase9_platform_headers.py`
    removes and verifies those.
    """
    tree = ast.parse(source)
    problems = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            names = [a.arg for a in node.args.args]
            if "roboflow_api_key" in names and "platform_client" not in names:
                problems.append(f"def {node.name} lacks platform_client")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            keys = {k.arg for k in node.keywords}
            if "roboflow_api_key" in keys and "platform_client" not in keys:
                problems.append(f"call at line {node.lineno} lacks platform_client")
        if isinstance(node, ast.Name) and node.id == PROXY_HELPER:
            problems.append(f"{PROXY_HELPER} referenced at line {node.lineno}")
        if (isinstance(node, ast.ImportFrom) and node.module == OLD_IMPORT_MODULE
                and any(a.name == PROXY_HELPER for a in node.names)):
            problems.append(f"{PROXY_HELPER} import survives at line {node.lineno}")
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == class_name)
    gip = next(n for n in cls.body
               if isinstance(n, ast.FunctionDef) and n.name == "get_init_parameters")
    ret = next(n for n in ast.walk(gip) if isinstance(n, ast.Return))
    if "platform_client" not in [e.value for e in ret.value.elts]:
        problems.append("get_init_parameters lacks platform_client")
    init = next(n for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    if "platform_client" not in [a.arg for a in init.args.args]:
        problems.append("__init__ lacks platform_client")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", nargs="+", help="path=ClassName")
    parser.add_argument("--expected-defs", type=int, required=True)
    parser.add_argument("--expected-calls", type=int, required=True)
    parser.add_argument("--expected-post-calls", type=int, required=True)
    parser.add_argument("--expected-ctor", type=int, required=True)
    args = parser.parse_args()
    totals = dict(defs=0, calls=0, ctor=0, gip=0, super=0, post_calls=0, old_imports=0)
    outputs = {}
    for pair in args.pairs:
        path_text, _, class_name = pair.partition("=")
        path = pathlib.Path(path_text)
        updated, stats = transform(path.read_text(encoding="utf-8"), class_name, path_text)
        problems = verify(updated, class_name, path_text)
        if problems:
            print(f"FAIL {path_text}: {problems}", file=sys.stderr)
            return 2
        again, _ = transform(updated, class_name, path_text)
        if again != updated:
            print(f"FAIL {path_text}: not idempotent", file=sys.stderr)
            return 2
        for key in totals:
            totals[key] += stats[key]
        outputs[path_text] = updated
        print(f"  {path_text}: {stats}")
    print("TOTALS", totals)
    for name, expected in (("defs", args.expected_defs), ("calls", args.expected_calls),
                           ("post_calls", args.expected_post_calls),
                           ("ctor", args.expected_ctor)):
        if totals[name] != expected:
            print(f"FAIL: {name} {totals[name]} != {expected}", file=sys.stderr)
            return 1
    for path_text, updated in outputs.items():
        pathlib.Path(path_text).write_text(updated, encoding="utf-8")
    print("POST-STATE verified for every file; idempotent; written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
