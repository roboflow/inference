#!/usr/bin/env python3
"""
Plane detection example: run a workflow that performs depth estimation using
DepthAnything V2 and saves the results locally to disk.

Workflow: image -> DepthAnything V2 (depth map) -> save metadata to disk.

Outputs saved to disk:
- depth_{i}.png: colored depth visualization
- depth_{i}.npy: normalized depth array (numpy format)
- depth_metadata_*.json: workflow metadata (from local_file_sink step)
"""

import base64
import json
import os
from pathlib import Path
from typing import Any, List

import cv2
import click
import numpy as np

from inference.core.env import MAX_ACTIVE_MODELS, WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.models.utils import ROBOFLOW_MODEL_TYPES


def load_workflow_definition(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def run_workflow(
    image_path: Path,
    workflow_path: Path,
    output_dir: Path,
) -> List[dict]:
    workflow_definition = load_workflow_definition(workflow_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    model_manager = ModelManager(model_registry=model_registry)
    model_manager = WithFixedSizeCache(model_manager, max_size=MAX_ACTIVE_MODELS)

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY environment variable is required. "
            "Set it with: export ROBOFLOW_API_KEY=your_key"
        )
    init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters=init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    runtime_params: dict = {
        "image": [image_rgb],
        "output_dir": str(output_dir),
    }
    result = execution_engine.run(runtime_parameters=runtime_params)
    return result


def _save_depth_image(image_data: Any, path: Path) -> None:
    """Save depth visualization image from WorkflowImageData or serialized dict."""
    if hasattr(image_data, "base64_image"):
        img_bytes = base64.b64decode(image_data.base64_image)
    elif isinstance(image_data, dict) and image_data.get("type") == "base64":
        img_bytes = base64.b64decode(image_data.get("value", ""))
    elif isinstance(image_data, np.ndarray):
        img_rgb = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
        success, buf = cv2.imencode(".png", img_rgb)
        if not success:
            raise RuntimeError("Failed to encode depth image as PNG")
        img_bytes = buf.tobytes()
    else:
        raise ValueError(f"Unexpected image data type: {type(image_data)}")
    path.write_bytes(img_bytes)


@click.command(
    help="Run plane detection workflow: image -> DepthAnything V2 -> save depth map and metadata to disk."
)
@click.argument("image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--workflow",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path(__file__).resolve().parent / "default.json",
    show_default=True,
    help="Path to workflow definition JSON.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Write depth images (.png), depth arrays (.npy), and metadata (.json) into this directory.",
)
def main(
    image: Path,
    workflow: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_workflow(
        image_path=image,
        workflow_path=workflow,
        output_dir=output_dir,
    )

    for i, batch_item in enumerate(result):
        print(f"--- Result for image index {i} ---")
        depth_results = batch_item.get("depth_results")
        if not depth_results:
            print("No depth results in output")
            continue

        for j, depth_item in enumerate(depth_results):
            prefix = f"depth_{i}_{j}" if len(depth_results) > 1 else f"depth_{i}"
            image_data = depth_item.get("image")
            normalized_depth = depth_item.get("normalized_depth")

            if image_data is not None:
                depth_img_path = output_dir / f"{prefix}.png"
                _save_depth_image(image_data, depth_img_path)
                print(f"depth image written: {depth_img_path}")

            if normalized_depth is not None:
                depth_array_path = output_dir / f"{prefix}.npy"
                np.save(depth_array_path, np.asarray(normalized_depth))
                print(f"depth array written: {depth_array_path}")


if __name__ == "__main__":
    main()
