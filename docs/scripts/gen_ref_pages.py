"""Generate the code reference pages."""

import os
from pathlib import Path

import mkdocs_gen_files

SKIP_MODULES = [
    "inference.enterprise.device_manager.command_handler",
    "inference.enterprise.parallel.celeryconfig",
]

if not os.environ.get("SKIP_CODEGEN"):
    for package in ["inference", "inference_sdk", "inference_cli"]:
        nav = mkdocs_gen_files.Nav()
        src = Path(__file__).parent.parent.parent / package

        for path in sorted(p for p in src.rglob("*.py") if "landing" not in p.parts):
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
