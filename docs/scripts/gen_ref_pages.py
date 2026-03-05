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

# Section descriptions for package index pages.
# Keys are (package, section_path) tuples where section_path uses "/" separators.
# A section_path of "" means the top-level modules in that package.
SECTION_DESCRIPTIONS = {
    ("inference_sdk", ""): (
        "inference_sdk",
        "Top-level SDK configuration: API URLs, timeouts, environment variable loading, and remote execution settings.",
    ),
    ("inference_sdk", "http"): (
        "http",
        "Core HTTP client for making inference requests. `InferenceHTTPClient` supports object detection, classification, segmentation, keypoint detection, OCR, CLIP embeddings, and workflow execution.",
    ),
    ("inference_sdk", "http/utils"): (
        "http/utils",
        "Internal utilities for request building, image encoding/decoding, response post-processing, retries, and API key handling.",
    ),
    ("inference_sdk", "utils"): (
        "utils",
        "General-purpose helpers: lifecycle decorators (`@deprecated`, `@experimental`), environment variable parsing, and SDK logging.",
    ),
    ("inference_sdk", "webrtc"): (
        "webrtc",
        "WebRTC streaming client for real-time video inference over peer connections. Supports webcam, RTSP, MJPEG, and video file sources with configurable output routing.",
    ),
    ("inference_cli", ""): (
        "inference_cli",
        "CLI entry points for server management, inference, benchmarking, cloud deployment, and workflow execution.",
    ),
    ("inference_cli", "lib"): (
        "lib",
        "Internal adapters for Docker container management, benchmarking, cloud deployment, and inference execution.",
    ),
    ("inference_cli", "lib/benchmark"): (
        "lib/benchmark",
        "Benchmarking utilities for measuring API speed, model inference speed, and platform performance.",
    ),
    ("inference_cli", "lib/roboflow_cloud"): (
        "lib/roboflow_cloud",
        "Roboflow cloud integration: configuration, API operations, and error handling.",
    ),
    ("inference_cli", "lib/roboflow_cloud/batch_processing"): (
        "lib/roboflow_cloud/batch_processing",
        "Batch processing for running workflows on large datasets via the Roboflow cloud.",
    ),
    ("inference_cli", "lib/roboflow_cloud/data_staging"): (
        "lib/roboflow_cloud/data_staging",
        "Data staging operations for uploading and managing data in the Roboflow cloud.",
    ),
    ("inference_cli", "lib/workflows"): (
        "lib/workflows",
        "Workflow execution adapters for local images, remote images, and video sources.",
    ),
}


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


def _section_key(parts, package):
    """Get the section path for a module's parts (everything between package and module name)."""
    # parts = ["inference_sdk", "http", "utils", "encoding"]
    # section = "http/utils"
    if len(parts) <= 2:
        return ""
    return "/".join(parts[1:-1])


def _build_structured_index(package, collected_modules):
    """Build a structured index page with section headers and descriptions."""
    lines = []
    lines.append(f"# `{package}` API Reference\n\n")

    # Group modules by section
    sections = {}
    for parts, doc_path in collected_modules:
        section = _section_key(parts, package)
        if section not in sections:
            sections[section] = []
        sections[section].append((parts, doc_path))

    for section_path, modules in sections.items():
        desc_key = (package, section_path)
        if desc_key in SECTION_DESCRIPTIONS:
            title, description = SECTION_DESCRIPTIONS[desc_key]
            if section_path == "":
                lines.append(f"## Top-level\n\n")
            else:
                lines.append(f"## `{title}`\n\n")
            lines.append(f"{description}\n\n")
        else:
            if section_path:
                lines.append(f"## `{section_path}`\n\n")

        for parts, doc_path in modules:
            module_name = parts[-1]
            lines.append(f"- [`{module_name}`]({doc_path})\n")
        lines.append("\n")

    return "".join(lines)


if not os.environ.get("SKIP_CODEGEN"):
    for package in ["inference", "inference_sdk", "inference_cli"]:
        nav = mkdocs_gen_files.Nav()
        src = Path(__file__).parent.parent.parent / package

        collected_modules = []

        for path in sorted(p for p in src.rglob("*.py") if "landing" not in p.parts):
            if not module_has_docstrings(path=path.as_posix()):
                continue
            module_path = path.relative_to(src.parent).with_suffix("")
            doc_path = path.relative_to(src.parent).with_suffix(".md")
            full_doc_path = Path("reference", doc_path)

            parts = list(module_path.parts)
            identifier = ".".join(parts)
            if parts[-1] == "__main__" or parts[-1] == "__init__" or identifier in SKIP_MODULES:
                continue

            nav[parts] = f"/reference/{module_path.as_posix()}.md"
            collected_modules.append((parts, f"/reference/{module_path.as_posix()}.md"))

            with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                fd.write(f"::: {identifier}")

            edit_path = f"https://github.com/roboflow/inference/tree/main/{module_path.as_posix()}.py"
            mkdocs_gen_files.set_edit_path(full_doc_path, edit_path)

        # Write literate-nav SUMMARY for sidebar navigation
        with mkdocs_gen_files.open(f"reference/{package}/SUMMARY.md", "w") as nav_file:
            generator = nav.build_literate_nav()
            lines = list(generator)
            nav_file.writelines(lines)

        # Write a structured index page for SDK and CLI packages
        if package in ("inference_sdk", "inference_cli"):
            with mkdocs_gen_files.open(f"reference/{package}/index.md", "w") as index_file:
                index_file.write(_build_structured_index(package, collected_modules))


