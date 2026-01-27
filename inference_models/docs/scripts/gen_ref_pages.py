"""Generate API reference pages automatically from docstrings."""

from pathlib import Path

import mkdocs_gen_files

# Modules to skip
SKIP_MODULES = {
    "inference_models.tests",
    "inference_models.development",
}


def module_has_docstrings(path: str) -> bool:
    """Check if a Python file has any docstrings."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            # Simple heuristic: check for triple quotes
            return '"""' in content or "'''" in content
    except Exception:
        return False


# Get the source directory - go up from docs/scripts/ to inference_models/
# Path structure: inference_models/docs/scripts/gen_ref_pages.py
# We want: inference_models/inference_models/
src = Path(__file__).parent.parent.parent / "inference_models"

# Create navigation structure
nav = mkdocs_gen_files.Nav()

# Iterate through all Python files
for path in sorted(src.rglob("*.py")):
    # Skip if in development or tests
    if any(skip in str(path) for skip in ["development", "tests", "__pycache__"]):
        continue

    # Skip if no docstrings
    if not module_has_docstrings(path.as_posix()):
        continue

    # Get module path relative to src
    module_path = path.relative_to(src.parent).with_suffix("")
    doc_path = path.relative_to(src.parent).with_suffix(".md")
    full_doc_path = Path("api-reference", doc_path)

    parts = list(module_path.parts)
    identifier = ".".join(parts)

    # Skip __init__ and __main__
    if parts[-1] in ("__main__", "__init__") or identifier in SKIP_MODULES:
        continue

    # Add to navigation
    nav[parts] = f"/api-reference/{module_path.as_posix()}.md"

    # Generate the documentation page
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: {identifier}")

    # Set edit path
    edit_path = (
        f"https://github.com/roboflow/inference/tree/main/{module_path.as_posix()}.py"
    )
    mkdocs_gen_files.set_edit_path(full_doc_path, edit_path)

# Write navigation file
with mkdocs_gen_files.open("api-reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
