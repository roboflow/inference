#!/usr/bin/env python3
"""
Run the workflow: image → detection → continue_if → email_notification.

Workflows are loaded from JSON files in the workflows/ directory by name.
Both workflows hit ExecutionEngine.init (the problematic one will raise
ControlFlowDefinitionError during init). Use --workflow to select which file to run.
"""

import json
import os
import sys
from pathlib import Path

import click
import numpy as np

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
    help="Path or URL to input image. If omitted, a small placeholder image is used.",
)
@click.option(
    "--send-email",
    is_flag=True,
    help="Actually send the email. By default dry_run is true (email step runs but does not send).",
)
def main(workflow_name: str, image_path: str | None, send_email: bool) -> None:
    """Run detection → continue_if → email_notification workflow."""
    if not os.environ.get("ROBOFLOW_API_KEY"):
        click.echo("Warning: ROBOFLOW_API_KEY not set. Detection may fail.", err=True)

    runtime_image, _ = load_image(image_path)
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

    # dry_run=True: email step runs and returns output but does not send (disable_sink)
    runtime_parameters = {
        "image": runtime_image,
        "dry_run": not send_email,
    }
    print(f"Running workflow: {workflow_name} (image → detection → continue_if → email_notification)")
    if not send_email:
        print("Email step in dry-run mode (output only, no mail sent). Use --send-email to send.")
    result = engine.run(runtime_parameters=runtime_parameters)

    print("\nWorkflow output:")
    for i, out in enumerate(result):
        print('-' * 100)
        print(out['detections'].xyxy)


if __name__ == "__main__":
    main()
