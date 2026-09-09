"""AST check: mock targets (string, f-string, patch.object) and `from` imports naming
the given symbols under a test root. Prints file:line per hit and a TOTAL; exits 1 on any hit
outside --allow files."""

import argparse
import ast
import pathlib
import sys

p = argparse.ArgumentParser()
p.add_argument("root")
p.add_argument("--symbols", required=True)
p.add_argument("--allow", default="")
a = p.parse_args()
symbols = set(a.symbols.split(","))
allow = {x for x in a.allow.split(",") if x}
hits = []
for path in sorted(pathlib.Path(a.root).rglob("*.py")):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        print(f"FAIL: {path}: {e}", file=sys.stderr)
        sys.exit(2)
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module in {
            "inference.core.roboflow_api",
            "inference.core.utils.url_utils",
        }:
            for al in n.names:
                if al.name in symbols:
                    hits.append((str(path), n.lineno, f"import {al.name}"))
        if isinstance(n, ast.Call):
            f = n.func
            is_patch = (isinstance(f, ast.Name) and f.id == "patch") or (
                isinstance(f, ast.Attribute)
                and f.attr in {"patch", "object", "setattr"}
            )
            if not is_patch:
                continue
            for arg in list(n.args) + [k.value for k in n.keywords]:
                text = None
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    text = arg.value
                elif isinstance(arg, ast.JoinedStr):
                    text = "".join(
                        v.value for v in arg.values if isinstance(v, ast.Constant)
                    )
                if text and (text in symbols or text.rsplit(".", 1)[-1] in symbols):
                    hits.append(
                        (str(path), n.lineno, f"mock target {text.rsplit('.', 1)[-1]}")
                    )
                    break
bad = [h for h in hits if h[0] not in allow]
for h in hits:
    print(("allowed " if h[0] in allow else "") + f"{h[0]}:{h[1]} {h[2]}")
print(f"TOTAL {len(hits)} references, {len(bad)} outside the allow-list")
sys.exit(1 if bad else 0)
