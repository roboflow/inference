#!/usr/bin/env python3
"""
Run the workflow: image -> detections -> email_notification.

Workflows are loaded from JSON file by path.
Supports single image (--image-path).
"""

import json
import os
from pathlib import Path

import click

from inference.core.managers.base import ModelManager
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.utils.image_utils import load_image
from inference.models.utils import ROBOFLOW_MODEL_TYPES

DEFAULT_WORKFLOW_PATH = Path(__file__).resolve().parent / "workflow.json"


def main(
    workflow_path: Path,
    image_path: Path,
) -> None:
    if not os.environ.get("ROBOFLOW_API_KEY"):
        click.echo("Warning: ROBOFLOW_API_KEY not set. Email may fail.", err=True)

    image_input, _ = load_image(str(image_path))

    with open(workflow_path) as f:
        workflow_definition = json.load(f)

    model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    manager = ModelManager(model_registry=model_registry)

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
    }

    _ = engine.run(runtime_parameters=runtime_parameters)


@click.command()
@click.option(
    "--workflow-path",
    default=DEFAULT_WORKFLOW_PATH,
    type=click.Path(exists=True),
    show_default=True,
    help="Path to workflow JSON file.",
)
@click.option(
    "--image-path",
    type=click.Path(exists=True),
    help="Path to a single input image.",
)
def run(
    workflow_path: Path,
    image_path: Path,
) -> None:
    main(
        workflow_path=workflow_path,
        image_path=image_path,
    )


if __name__ == "__main__":
    run()
