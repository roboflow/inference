#!/usr/bin/env python3
"""
Run the workflow: image → detection → continue_if → email_notification.

Uses the lineage workaround (message_parameters referencing detection) so the
workflow compiles. Use --demonstrate-error to see ControlFlowDefinitionError
when the email step has no data input from detection.
"""

import argparse
import os
import sys
from pathlib import Path

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
from inference.core.workflows.errors import ControlFlowDefinitionError
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.v1.compiler.core import compile_workflow
from inference.models.utils import ROBOFLOW_MODEL_TYPES

# Workflow with workaround: email has message_parameters referencing detection,
# so input lineage matches control-flow lineage and compilation succeeds.
WORKFLOW_WITH_WORKAROUND = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["person"],
        },
        {
            "type": "ContinueIf",
            "name": "continue_if",
            "condition_statement": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "prediction",
                            "operations": [{"type": "SequenceLength"}],
                        },
                        "comparator": {"type": "(Number) >="},
                        "right_operand": {
                            "type": "StaticOperand",
                            "value": 1,
                        },
                    }
                ],
            },
            "evaluation_parameters": {"prediction": "$steps.detection.predictions"},
            "next_steps": ["$steps.email_notification"],
        },
        {
            "type": "roboflow_core/email_notification@v2",
            "name": "email_notification",
            "subject": "Workflow: detections found",
            "receiver_email": os.environ.get("RECEIVER_EMAIL", "noreply@example.com"),
            "message": "At least one detection found for this image. (message_parameters.predictions links lineage so the workflow compiles.)",
            "message_parameters": {
                "predictions": "$steps.detection.predictions",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "detections",
            "selector": "$steps.detection.predictions",
        },
    ],
}

# Problematic workflow: email has no input from detection → ControlFlowDefinitionError.
WORKFLOW_PROBLEMATIC = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["person"],
        },
        {
            "type": "ContinueIf",
            "name": "continue_if",
            "condition_statement": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "prediction",
                            "operations": [{"type": "SequenceLength"}],
                        },
                        "comparator": {"type": "(Number) >="},
                        "right_operand": {
                            "type": "StaticOperand",
                            "value": 1,
                        },
                    }
                ],
            },
            "evaluation_parameters": {"prediction": "$steps.detection.predictions"},
            "next_steps": ["$steps.email_notification"],
        },
        {
            "type": "roboflow_core/email_notification@v2",
            "name": "email_notification",
            "subject": "Workflow: detections found",
            "receiver_email": "noreply@example.com",
            "message": "At least one detection found for this image.",
            # No message_parameters → no data edge detection → email → ControlFlowDefinitionError
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "detections",
            "selector": "$steps.detection.predictions",
        },
    ],
}


def make_model_manager() -> ModelManager:
    registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    manager = ModelManager(model_registry=registry)
    return WithFixedSizeCache(manager, max_size=MAX_ACTIVE_MODELS)


def load_image(path_or_url: str | None) -> np.ndarray:
    """Load image as numpy array from path or URL; or return a small placeholder."""
    if path_or_url is None or path_or_url == "":
        # Minimal 64x64 RGB image so the pipeline runs without external file
        return np.zeros((64, 64, 3), dtype=np.uint8) + 128

    path = Path(path_or_url)
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        from inference.core.utils.image_utils import load_image_from_url

        return load_image_from_url(path_or_url)
    if path.exists():
        from inference.core.utils.image_utils import load_image

        out, _ = load_image(path)
        return out
    raise FileNotFoundError(f"Image not found: {path_or_url}")


def run(runtime_image: np.ndarray, demonstrate_error: bool) -> None:
    init_params = {
        "workflows_core.model_manager": make_model_manager(),
        "workflows_core.api_key": os.environ.get("ROBOFLOW_API_KEY"),
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    if demonstrate_error:
        print("Attempting to compile the problematic workflow (email has no data from detection)...")
        try:
            compile_workflow(
                workflow_definition=WORKFLOW_PROBLEMATIC,
                init_parameters=init_params,
            )
            print("Unexpected: compilation succeeded.", file=sys.stderr)
            sys.exit(1)
        except ControlFlowDefinitionError as e:
            print("ControlFlowDefinitionError (expected):")
            print(e.public_message)
            print("\nThis is the error described in docs/workflow-lineage-control-flow-summary.md")
            return

    engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_WORKAROUND,
        init_parameters=init_params,
        max_concurrent_steps=4,
    )

    print("Running workflow: image → detection → continue_if → email_notification")
    result = engine.run(runtime_parameters={"image": runtime_image})

    print("\nWorkflow output:")
    for i, out in enumerate(result):
        print(f"  Output[{i}]: keys = {list(out.keys())}")
        if "detections" in out:
            preds = out["detections"]
            n = len(preds) if isinstance(preds, list) else 0
            print(f"    detections: {n} prediction(s)")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run detection → continue_if → email_notification workflow."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path or URL to input image. If omitted, a small placeholder image is used.",
    )
    parser.add_argument(
        "--demonstrate-error",
        action="store_true",
        help="Only try to compile the problematic workflow (no message_parameters on email) and print ControlFlowDefinitionError.",
    )
    args = parser.parse_args()

    if not args.demonstrate_error and not os.environ.get("ROBOFLOW_API_KEY"):
        print("Warning: ROBOFLOW_API_KEY not set. Detection may fail.", file=sys.stderr)

    image = load_image(args.image)
    run(runtime_image=image, demonstrate_error=args.demonstrate_error)


if __name__ == "__main__":
    main()
