"""Generate the code reference pages.

Can run as a mkdocs gen-files plugin script, or standalone via:
    python -m docs.scripts.gen_ref_pages
When run standalone, writes directly to docs/reference/.
"""
import ast
import os
from pathlib import Path
from typing import Union

try:
    import mkdocs_gen_files
except ImportError:
    mkdocs_gen_files = None

SKIP_MODULES = [
    "inference.enterprise.device_manager.command_handler",
    "inference.enterprise.parallel.celeryconfig",
]

# Section descriptions for package index pages.
# Keys are (package, section_path) tuples where section_path uses "/" separators.
# A section_path of "" means the top-level modules in that package.
SECTION_DESCRIPTIONS = {
    # ── inference ──
    ("inference", ""): (
        "inference",
        "Top-level inference package: version info, configuration, and convenience imports.",
    ),
    ("inference", "core"): (
        "core",
        "Core framework internals: environment config, data entities, and shared utilities.",
    ),
    ("inference", "core/active_learning"): (
        "core/active_learning",
        "Active learning loop: sampling strategies, data collection middleware, and configuration.",
    ),
    ("inference", "core/cache"): (
        "core/cache",
        "Caching backends (in-memory, Redis) used for model artefacts and inference results.",
    ),
    ("inference", "core/devices"): (
        "core/devices",
        "Hardware device detection and selection helpers.",
    ),
    ("inference", "core/entities"): (
        "core/entities",
        "Shared data classes and request/response entities.",
    ),
    ("inference", "core/interfaces"): (
        "core/interfaces",
        "High-level inference interfaces: camera, HTTP, and stream processing.",
    ),
    ("inference", "core/managers"): (
        "core/managers",
        "Model lifecycle managers: loading, unloading, registry, and resolution.",
    ),
    ("inference", "core/models"): (
        "core/models",
        "Base model classes and common prediction logic shared across model types.",
    ),
    ("inference", "core/registries"): (
        "core/registries",
        "Model and block registries for dynamic lookup and plugin discovery.",
    ),
    ("inference", "core/utils"): (
        "core/utils",
        "General-purpose utilities: image encoding, file I/O, hashing, URL handling, and more.",
    ),
    ("inference", "core/workflows"): (
        "core/workflows",
        "Workflow execution engine entry points and helpers.",
    ),
    ("inference", "enterprise"): (
        "enterprise",
        "Enterprise-only features: device management and licensing.",
    ),
    ("inference", "enterprise/parallel"): (
        "enterprise/parallel",
        "Parallel HTTP inference via Celery workers for high-throughput deployments.",
    ),
    ("inference", "enterprise/stream_management"): (
        "enterprise/stream_management",
        "Stream Management API for controlling long-running video inference pipelines.",
    ),
    ("inference", "enterprise/workflows"): (
        "enterprise/workflows",
        "Enterprise workflow extensions and enterprise-only blocks.",
    ),
    ("inference", "models"): (
        "models",
        "Model implementations. Each sub-package wraps a specific architecture.",
    ),
    ("inference", "usage_tracking"): (
        "usage_tracking",
        "Anonymous usage and telemetry reporting.",
    ),
    # ── inference_sdk ──
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
    # ── inference_cli ──
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
    if len(parts) <= 2:
        return ""
    return "/".join(parts[1:-1])


def _build_single_page_index(package, collected_modules):
    """Build a single-page API reference with inline ::: directives for every module."""
    lines = []
    lines.append(f"# `{package}` API Reference\n\n")

    # Group modules by section
    sections = {}
    for parts, identifier in collected_modules:
        section = _section_key(parts, package)
        if section not in sections:
            sections[section] = []
        sections[section].append((parts, identifier))

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

        for parts, identifier in modules:
            lines.append(f"::: {identifier}\n\n")

    return "".join(lines)


def _open_output(rel_path):
    """Open an output file via gen-files plugin or directly to docs/."""
    if mkdocs_gen_files is not None:
        return mkdocs_gen_files.open(rel_path, "w")
    # Standalone mode: write directly to docs/
    docs_dir = Path(__file__).parent.parent
    out = docs_dir / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    return open(out, "w")


if not os.environ.get("SKIP_CODEGEN"):
    for package in ["inference", "inference_sdk", "inference_cli"]:
        src = Path(__file__).parent.parent.parent / package

        collected_modules = []

        for path in sorted(p for p in src.rglob("*.py") if "landing" not in p.parts):
            if not module_has_docstrings(path=path.as_posix()):
                continue
            module_path = path.relative_to(src.parent).with_suffix("")

            parts = list(module_path.parts)
            identifier = ".".join(parts)
            if parts[-1] == "__main__" or parts[-1] == "__init__" or identifier in SKIP_MODULES:
                continue

            collected_modules.append((parts, identifier))

        # Write single-page index with all ::: directives
        with _open_output(f"reference/{package}/index.md") as index_file:
            index_file.write(_build_single_page_index(package, collected_modules))
