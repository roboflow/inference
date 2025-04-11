"""Generate the code reference pages."""
import ast
import os
from pathlib import Path
from typing import Union

import mkdocs_gen_files

SKIP_MODULES = [
    "inference.enterprise.device_manager.command_handler",
    "inference.enterprise.parallel.celeryconfig",
]


def module_has_docstrings(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=path)
        except SyntaxError:
            return False  # skip broken files

    if has_docstring(tree):
        return True

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if has_docstring(node):
                return True
    return False


def has_docstring(node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module]):
    return ast.get_docstring(node) is not None


if not os.environ.get("SKIP_CODEGEN"):
    for package in ["inference", "inference_sdk", "inference_cli"]:
        nav = mkdocs_gen_files.Nav()
        src = Path(__file__).parent.parent.parent / package

        for path in sorted(p for p in src.rglob("*.py") if "landing" not in p.parts):
            if not module_has_docstrings(path=path.as_posix()):
                continue
            module_path = path.relative_to(src.parent).with_suffix("")
            doc_path = path.relative_to(src.parent).with_suffix(".md")
            full_doc_path = Path("reference", doc_path)

            parts = list(module_path.parts)
            identifier = ".".join(parts)
            if parts[-1] == "__main__" or parts[-1] == "__init__" or identifier in SKIP_MODULES:
                # print("SKIPPING", identifier)
                continue

            nav[parts] = f"/reference/{module_path.as_posix()}.md"

            with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                fd.write(f"::: {identifier}")

            edit_path = f"https://github.com/roboflow/inference/tree/main/{module_path.as_posix()}.py"
            # print("Edit path:", edit_path)
            mkdocs_gen_files.set_edit_path(full_doc_path, edit_path)

        with mkdocs_gen_files.open(f"reference/{package}/index.md", "w") as nav_file:
            generator = nav.build_literate_nav()
            lines = list(generator)
            # print("GENERATING NAVIGATION")
            # print("\n".join(lines))
            nav_file.writelines(lines)


