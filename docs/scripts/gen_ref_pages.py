"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

src = Path(__file__).parent.parent.parent / "inference"
SKIP_MODULES = [
    "inference.enterprise.device_manager.command_handler",
    "inference.enterprise.parallel.celeryconfig",
]

for path in sorted(p for p in src.rglob("*.py") if "landing" not in p.parts):
    module_path = path.relative_to(src.parent).with_suffix("")
    doc_path = path.relative_to(src.parent).with_suffix(".md")
    full_doc_path = Path("docs", "reference", doc_path)

    parts = list(module_path.parts)
    identifier = ".".join(parts)
    if parts[-1] == "__main__" or parts[-1] == "__init__" or identifier in SKIP_MODULES:
        print("SKIPPING", identifier)
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: {identifier}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("docs/reference/nav.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
