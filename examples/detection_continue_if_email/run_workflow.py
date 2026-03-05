#!/usr/bin/env python3
"""
Run the workflow: image → detection → continue_if → email_notification.

Workflows are loaded from JSON files in the workflows/ directory by name.
Supports single image (--image-path) or batch of images (--image-dir).
"""

import json
import os
import sys
from pathlib import Path

import click

# Add repo root so we can import inference
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.core.env import MAX_ACTIVE_MODELS
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.utils.image_utils import load_image
from inference.models.utils import ROBOFLOW_MODEL_TYPES


WORKFLOWS_DIR = Path(__file__).resolve().parent / "workflows"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_workflow(name: str) -> dict:
    """Load workflow definition from workflows/<name>.json."""
    path = WORKFLOWS_DIR / f"{name}.json" if not name.endswith(".json") else WORKFLOWS_DIR / name
    if not path.exists():
        available = [f.stem for f in WORKFLOWS_DIR.glob("*.json")]
        raise FileNotFoundError(
            f"Workflow file not found: {path}. Available: {available}"
        )
    with open(path) as f:
        return json.load(f)


def collect_image_paths_from_dir(image_dir: Path) -> list[Path]:
    """Return sorted list of image file paths in the given directory."""
    paths = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda p: p.name)


def main(
    workflow_name: str,
    image_path: str | None,
    image_dir: str | None,
    send_email: bool,
) -> None:
    """Run detection → continue_if → email_notification workflow."""
    if not os.environ.get("ROBOFLOW_API_KEY"):
        click.echo("Warning: ROBOFLOW_API_KEY not set. Detection may fail.", err=True)

    image_paths_for_display = None
    if image_dir is not None:
        dir_path = Path(image_dir).expanduser().resolve()
        if not dir_path.is_dir():
            raise click.BadParameter(f"Not a directory: {dir_path}", param_hint="--image-dir")
        image_paths = collect_image_paths_from_dir(dir_path)
        if not image_paths:
            raise click.BadParameter(f"No image files found in {dir_path}", param_hint="--image-dir")
        # Pass paths as list; execution engine will load each (batch mode)
        image_input = [str(p) for p in image_paths]
        image_paths_for_display = image_paths
        batch_size = len(image_input)
        click.echo(f"Batch mode: {batch_size} images from {dir_path}")
    else:
        if image_path is None:
            raise click.UsageError(
                "Provide either --image-path (single image) or --image-dir (batch of images)."
            )
        runtime_image, _ = load_image(image_path)
        image_input = runtime_image
        batch_size = 1

    workflow_definition = load_workflow(workflow_name)

    model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    manager = ModelManager(model_registry=model_registry)
    manager = WithFixedSizeCache(manager, max_size=MAX_ACTIVE_MODELS)

    init_params = {
        "workflows_core.model_manager": manager,
        "workflows_core.api_key": os.environ.get("ROBOFLOW_API_KEY"),
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters=init_params,
        max_concurrent_steps=4,
    )

    runtime_parameters = {
        "image": image_input,
        "dry_run": not send_email,
    }
    input_names = {inp.get("name") for inp in workflow_definition.get("inputs", [])}
    if "image_names" in input_names:
        if image_paths_for_display is not None:
            image_names_input = [p.name for p in image_paths_for_display]
        else:
            image_names_input = (
                [Path(image_path).name] if image_path else ["image"]
            )
        runtime_parameters["image_names"] = image_names_input
    print(
        f"Running workflow: {workflow_name} "
        f"(image → detection → continue_if → email_notification)"
    )
    if batch_size > 1:
        print(f"Batch: {batch_size} images.")
    if not send_email:
        print("Email step in dry-run mode (output only, no mail sent). Use --send-email to send.")
    result = engine.run(runtime_parameters=runtime_parameters)

    print("\nWorkflow output:")
    for i, out in enumerate(result):
        print("-" * 100)
        label = f"batch index {i}"
        if image_paths_for_display is not None and i < len(image_paths_for_display):
            label = f"{image_paths_for_display[i].name} ({label})"
        detections = out["detections"]
        if hasattr(detections, "xyxy"):
            n = len(detections.xyxy) if detections.xyxy is not None else 0
            print(f"  [{label}] {n} detection(s); xyxy: {detections.xyxy}")
        else:
            print(f"  [{label}] {detections}")


@click.command()
@click.option(
    "--workflow-name",
    default="workflow_with_workaround",
    show_default=True,
    help="Workflow name (filename without .json in workflows/).",
)
@click.option(
    "--image-path",
    default=None,
    help="Path or URL to a single input image (use --image-dir for batch).",
)
@click.option(
    "--image-dir",
    default=None,
    help="Path to directory of images; run workflow on all images in the directory (batch mode).",
)
@click.option(
    "--send-email",
    is_flag=True,
    help="Actually send the email. By default dry_run is true (email step runs but does not send).",
)
def run(
    workflow_name: str,
    image_path: str | None,
    image_dir: str | None,
    send_email: bool,
) -> None:
    main(
        workflow_name=workflow_name,
        image_path=image_path,
        image_dir=image_dir,
        send_email=send_email,
    )


if __name__ == "__main__":
    run()
