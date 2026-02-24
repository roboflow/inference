#!/usr/bin/env python3
"""
Indoor design example: run a workflow that detects an object (YOLO World),
segments it (SAM2), and produces 3D mesh + Gaussian splat (SAM3 3D).

Workflow: image + text prompt -> YOLO World (bbox) -> SAM2 (mask) -> SAM3 3D (mesh, PLY, metadata).

Running on Mac (Apple Silicon): SAM2 and SAM3 3D will use MPS (Metal) when available
if the DEVICE env var is not set. To force MPS: DEVICE=mps python main.py ...
If the 3D step fails on MPS (e.g. unsupported ops in the pipeline), run with DEVICE=cpu.
"""

import os

# Enable SAM3 3D block so the workflow definition (segment_anything3_3d_objects@v1) is accepted
os.environ.setdefault("SAM3_3D_OBJECTS_ENABLED", "True")

import base64
import json
from pathlib import Path
from typing import List

import cv2
import click

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
    object_prompt: str,
    workflow_path: Path,
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

    # object_prompt: YOLO World expects a list of class names (e.g. ["sofa"])
    result = execution_engine.run(
        runtime_parameters={
            "image": [image_rgb],
            "object_prompt": [object_prompt],
        },
    )
    return result


@click.command(
    help="Run indoor design workflow: image + prompt -> YOLO World -> SAM2 -> SAM3 3D (mesh, PLY, metadata)."
)
@click.argument("image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--prompt",
    type=str,
    default="object",
    show_default=True,
    help="Text prompt for the object to detect (e.g. 'sofa', 'chair'). Passed to YOLO World.",
)
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
    help="Write mesh_glb and gaussian_ply as files into this directory.",
)
def main(
    image: Path,
    prompt: str,
    workflow: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_workflow(
        image_path=image,
        object_prompt=prompt,
        workflow_path=workflow,
    )

    # Result is a list of dicts (one per input image)
    for i, batch_item in enumerate(result):
        print(f"--- Result for image index {i} ---")
        mesh_b64 = batch_item.get("mesh_glb")
        ply_b64 = batch_item.get("gaussian_ply")
        objects_meta = batch_item.get("objects", [])
        inference_time = batch_item.get("inference_time")

        if inference_time is not None:
            print(f"inference_time: {inference_time:.2f}s")

        print(f"objects (metadata count): {len(objects_meta)}")


        if mesh_b64:
            mesh_path = output_dir / f"mesh_{i}.glb"
            mesh_bytes = base64.b64decode(mesh_b64)
            mesh_path.write_bytes(mesh_bytes)
            print(f"mesh_glb written: {mesh_path}")
        if ply_b64:
            ply_path = output_dir / f"gaussian_{i}.ply"
            ply_bytes = base64.b64decode(ply_b64)
            ply_path.write_bytes(ply_bytes)
            print(f"gaussian_ply written: {ply_path}")


if __name__ == "__main__":
    main()
