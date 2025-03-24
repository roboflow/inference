import json
import os
import re
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional

import pytest
import requests
from requests import Response

from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    GALLERY_ENTRIES,
    WorkflowGalleryEntry,
)

API_URL = "https://api.roboflow.com"
API_KEY_PATTERN = re.compile(r"api_key=(.[^&]*)")
KEY_VALUE_GROUP = 1
MIN_KEY_LENGTH_TO_REVEAL_PREFIX = 8
INLINE_UQL_PARAMETER_PATTERN = re.compile(r"({{\s*\$parameters\.(\w+)\s*}})")
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

GALLERY_DIR = os.path.join(WORKFLOWS_DOCS_ROOT_PATH, "gallery")
GALLERY_INDEX_TEMPLATE = os.path.join(WORKFLOWS_DOCS_ROOT_PATH, "gallery_index_template.md")
GALLERY_INDEX_PATH = os.path.join(WORKFLOWS_DOCS_ROOT_PATH, "gallery", "index.md")
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
    # move the Basic Workflows category to the start
    basic_workflows_index = categories.index("Basic Workflows")
    categories.insert(0, categories.pop(basic_workflows_index))
    
    index = read_file(path=GALLERY_INDEX_TEMPLATE)
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

    write_gallery_summary_md(categories)

def to_title_case(s: str) -> str:
    """
    Convert e.g. 'object_detection' -> 'Object Detection'
    """
    words = re.split(r'[_\s]+', s.lower())
    return " ".join(w.capitalize() for w in words if w)

def write_gallery_summary_md(categories: List[str]) -> None:
    """
    Creates docs/workflows/gallery/SUMMARY.md for mkdocs-literate-nav.
    """
    lines = []

    for category in categories:
        url = generate_gallery_page_link(category=category)

        # relative links (remove `/workflows/gallery/` prefix)
        url = url.replace("/workflows/gallery/", "")

        url = f"{url}.md"

        category = category.replace("Workflows with ", "")
        category = category.replace("Workflows for ", "")
        category = category.replace("Workflows ", "")
        category = category.replace(" in Workflows", "")

        if category == "Filtering resulting data based on value delta change":
            category = "Filtering Data"

        if category == "Enhanced by Roboflow Platform":
            category = "Enhanced by Roboflow"
        
        if category == "Advanced Inference Techniques":
            category = "Advanced Techniques"
        
        if category == "Integration With External Apps":
            category = "External Integrations"

        category = to_title_case(category)

        if category == "Ocr":
            category = "OCR"

        lines.append(f"* [{category}]({url})")

    summary_path = os.path.join(GALLERY_DIR, "SUMMARY.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

GALLERY_PAGE_TEMPLATE = """
# {category}

Below you can find example workflows you can use as inspiration to build your apps.

{examples}

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {{
    display: none;
}}
</style>
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

{preview_iframe}

??? tip "Workflow definition"

    ```json
    {workflow_definition}
    ```
""".strip()


def generate_gallery_entry_docs(entry: WorkflowGalleryEntry) -> str:
    preview_iframe = ""
    if entry.workflow_name_in_app:
        preview_iframe = generate_preview_iframe(
            workflow_name_in_app=entry.workflow_name_in_app,
            workflow_definition=entry.workflow_definition,
        )
    workflow_definition = _dump_workflow_definition(workflow_definition=entry.workflow_definition)
    return GALLERY_ENTRY_TEMPLATE.format(
        title=entry.use_case_title,
        description=entry.use_case_description,
        preview_iframe=preview_iframe,
        workflow_definition=workflow_definition,
    )


def _dump_workflow_definition(workflow_definition: dict) -> str:
    definition_stringified = "\n\t".join(json.dumps(workflow_definition, indent=4).split("\n"))
    return INLINE_UQL_PARAMETER_PATTERN.sub(_escape_uql_brackets, definition_stringified)


def _escape_uql_brackets(match: re.Match) -> str:
    content = match.group(0)
    return "{{ '{{' }}" + content[2:-2] + "{{ '}}' }}"


def generate_preview_iframe(workflow_name_in_app: str, workflow_definition: dict) -> str:
    api_key = os.environ["WORKFLOWS_GALLERY_API_KEY"]
    workspace_name = retrieve_workspace_name_from_api(api_key=api_key)
    workflow_id = get_workflow_id(
        workspace_name=workspace_name,
        workflow_name=workflow_name_in_app,
        api_key=api_key,
    )
    if workflow_id is None:
        workflow_id = create_workflow(
            workspace_name=workspace_name,
            workflow_name=workflow_name_in_app,
            workflow_definition=workflow_definition,
            api_key=api_key,
        )
    else:
        update_workflow(
            workspace_name=workspace_name,
            workflow_name=workflow_name_in_app,
            workflow_id=workflow_id,
            workflow_definition=workflow_definition,
            api_key=api_key,
        )
    iframe_token = generate_workflow_token(
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        api_key=api_key
    )
    human_readable_name = workflow_name_in_app.replace("-", " ")
    return (
        f'<div style="height: 768px;">'
        f'<iframe src="https://app.roboflow.com/workflows/embed/{iframe_token}?showGraph=true" '
        f'loading="lazy" title="Roboflow Workflow for {human_readable_name}"'
        f' style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>'
    )


@lru_cache(maxsize=16)
def retrieve_workspace_name_from_api(api_key: str) -> str:
    response = requests.post(API_URL, params={"api_key": api_key})
    api_key_safe_raise_for_status(response=response)
    return response.json()["workspace"]


def get_workflow_id(workspace_name: str, workflow_name: str, api_key: str) -> Optional[str]:
    response = requests.get(
        f"{API_URL}/{workspace_name}/workflows/{workflow_name}",
        params={"api_key": api_key}
    )
    print(f"{API_URL}/{workspace_name}/workflows/{workflow_name} - {response.status_code}")
    if response.status_code == 404:
        return None
    api_key_safe_raise_for_status(response=response)
    return response.json()["workflow"]["id"]


def create_workflow(
    workspace_name: str,
    workflow_name: str,
    workflow_definition: dict,
    api_key: str,
) -> str:
    response = requests.post(
        f"{API_URL}/{workspace_name}/createWorkflow",
        params={
            "api_key": api_key,
            "name": workflow_name,
            "url": workflow_name,
            "template": "custom",
            "config": json.dumps({"specification": workflow_definition}),
        },
    )
    api_key_safe_raise_for_status(response=response)
    return response.json()["workflows"]["id"]


def update_workflow(
    workspace_name: str,
    workflow_name: str,
    workflow_id: str,
    workflow_definition: dict,
    api_key: str,
) -> None:
    response = requests.post(
        f"{API_URL}/{workspace_name}/updateWorkflow",
        json={
            "api_key": api_key,
            "id": workflow_id,
            "name": workflow_name,
            "url": workflow_name,
            "template": "workflow-from-inference-tests",
            "config": json.dumps({"specification": workflow_definition}),
        }
    )
    api_key_safe_raise_for_status(response=response)


def generate_workflow_token(workspace_name: str, workflow_id: str, api_key: str) -> str:
    response = requests.post(
        f"{API_URL}/{workspace_name}/workflowToken",
        json={
            "api_key": api_key,
            "id": workflow_id,
        }
    )
    api_key_safe_raise_for_status(response=response)
    return response.json()["token"]


def api_key_safe_raise_for_status(response: Response) -> None:
    request_is_successful = response.status_code < 400
    if request_is_successful:
        return None
    response.url = API_KEY_PATTERN.sub(deduct_api_key, response.url)
    response.raise_for_status()


def deduct_api_key(match: re.Match) -> str:
    key_value = match.group(KEY_VALUE_GROUP)
    if len(key_value) < MIN_KEY_LENGTH_TO_REVEAL_PREFIX:
        return f"api_key=***"
    key_prefix = key_value[:2]
    key_postfix = key_value[-2:]
    return f"api_key={key_prefix}***{key_postfix}"


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
