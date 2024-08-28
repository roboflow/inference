import json
import os
from collections import defaultdict
from typing import List, Dict, Optional

import pytest

from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import GALLERY_ENTRIES, \
    WorkflowGalleryEntry

INTEGRATION_TESTS_DIRECTORY = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "tests/workflows/integration_tests/execution"
))
DOCS_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "docs"
))
WORKFLOWS_DOCS_ROOT_PATH = os.path.join(DOCS_ROOT, "workflows")

GALLERY_INDEX_PATH = os.path.join(WORKFLOWS_DOCS_ROOT_PATH, "gallery_index.md")
GALLERY_DIR_PATH = os.path.join(WORKFLOWS_DOCS_ROOT_PATH, "gallery")


def generate_gallery() -> None:
    run_pytest_collection_to_fill_workflows_gallery()
    categorised_gallery = categorise_gallery(gallery=GALLERY_ENTRIES)
    generate_gallery_index(categories=list(categorised_gallery.keys()))
    for category, entries in categorised_gallery.items():
        generate_gallery_page_for_category(category=category, entries=entries)


def run_pytest_collection_to_fill_workflows_gallery() -> None:
    pytest.main(["--collect-only", INTEGRATION_TESTS_DIRECTORY])


def categorise_gallery(gallery: List[WorkflowGalleryEntry]) -> Dict[str, List[WorkflowGalleryEntry]]:
    result = defaultdict(list)
    for item in gallery:
        result[item.category].append(item)
    return result


def generate_gallery_index(categories: List[str]) -> None:
    index = read_file(path=GALLERY_INDEX_PATH)
    index_lines = index.split("\n")
    list_start_index = find_line_with_marker(lines=index_lines, marker='<ul id="workflows-gallery">')
    list_end_index = find_line_with_marker(lines=index_lines, marker="</ul>")
    if list_start_index is None or list_end_index is None:
        raise RuntimeError("Could not find expected <ul> markers in gallery index file")
    categories_entries = [
        f'\t<li><a href="{generate_gallery_page_link(category=category)}">{category}</a></li>'
        for category in categories
    ]
    new_index = index_lines[:list_start_index + 1]
    new_index.extend(categories_entries)
    new_index.extend(index_lines[list_end_index:])
    new_index_content = "\n".join(new_index)
    write_file(path=GALLERY_INDEX_PATH, content=new_index_content)


GALLERY_PAGE_TEMPLATE = """
# Example Workflows - {category}

Below you can find example workflows you can use as inspiration to build your apps.

{examples}
""".strip()


def generate_gallery_page_for_category(
    category: str,
    entries: List[WorkflowGalleryEntry],
) -> None:
    examples = [
        generate_gallery_entry_docs(entry=entry)
        for entry in entries
    ]
    page_content = GALLERY_PAGE_TEMPLATE.format(
        category=category,
        examples="\n\n".join(examples)
    )
    file_path = generate_gallery_page_file_path(category=category)
    write_file(path=file_path, content=page_content)


GALLERY_ENTRY_TEMPLATE = """
## {title}

{description}

??? tip "Workflow definition"

    ```json
    {workflow_definition}
    ```
""".strip()


def generate_gallery_entry_docs(entry: WorkflowGalleryEntry) -> str:
    return GALLERY_ENTRY_TEMPLATE.format(
        title=entry.use_case_title,
        description=entry.use_case_description,
        workflow_definition="\n\t".join(json.dumps(entry.workflow_definition, indent=4).split("\n")),
    )


def read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    path = os.path.abspath(path)
    parent_dir = os.path.dirname(path)
    os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def find_line_with_marker(lines: List[str], marker: str) -> Optional[int]:
    for i, line in enumerate(lines):
        if marker in line:
            return i
    return None


def generate_gallery_page_link(category: str) -> str:
    file_path = generate_gallery_page_file_path(category=category)
    return file_path[len(DOCS_ROOT):-3]


def generate_gallery_page_file_path(category: str) -> str:
    category_slug = slugify_category(category=category)
    return os.path.join(GALLERY_DIR_PATH, f"{category_slug}.md")


def slugify_category(category: str) -> str:
    return category.lower().replace(" ", "_").replace("/", "_")


if __name__ == "__main__":
    generate_gallery()
