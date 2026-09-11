"""Lists every annotation in a file that references one of the given names.

`compileall` does not evaluate annotations, so a deleted import that survives
only in a `-> List[Point]` return type compiles and then fails at import time.

Run:  python scripts/phase11_annotation_scan.py <file.py> Name1,Name2,...
Exit code 1 (and a printed table) when any annotation still references one.
"""

import ast
import sys


def scan(path: str, names: set) -> list:
    tree = ast.parse(open(path, encoding="utf-8").read())
    hits = []
    for node in ast.walk(tree):
        annotations = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                annotations.append(("return", node.returns))
            arguments = (
                list(node.args.posonlyargs)
                + list(node.args.args)
                + list(node.args.kwonlyargs)
            )
            for argument in arguments:
                if argument.annotation is not None:
                    annotations.append((f"arg {argument.arg}", argument.annotation))
        elif isinstance(node, ast.AnnAssign):
            annotations.append(("annotated assignment", node.annotation))
        for label, annotation in annotations:
            referenced = sorted(
                {
                    inner.id
                    for inner in ast.walk(annotation)
                    if isinstance(inner, ast.Name) and inner.id in names
                }
            )
            if referenced:
                hits.append((node.lineno, label, referenced, ast.unparse(annotation)))
    return hits


if __name__ == "__main__":
    path, names = sys.argv[1], set(sys.argv[2].split(","))
    hits = scan(path, names)
    for lineno, label, referenced, text in hits:
        print(f"{path}:{lineno}: {label}: {text}   -> {referenced}")
    sys.exit(1 if hits else 0)
