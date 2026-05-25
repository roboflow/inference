from unittest.mock import MagicMock, patch

import numpy as np
import torch

from inference_models.models.base.object_detection import Detections as TensorDetections

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.roboflow.object_detection.v1_tensor import (
    RoboflowObjectDetectionModelBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    OriginCoordinatesSystem,
    WorkflowImageData,
)


def _make_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(
            parent_id="p",
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=0, left_top_y=0, origin_width=64, origin_height=64
            ),
        ),
        numpy_image=np.zeros((64, 64, 3), dtype=np.uint8),
    )


def _make_detections(n: int = 1) -> TensorDetections:
    return TensorDetections(
        xyxy=torch.zeros((n, 4), dtype=torch.float32),
        class_id=torch.zeros((n,), dtype=torch.int64),
        confidence=torch.zeros((n,), dtype=torch.float32),
    )


def _kwargs(**over):
    base = {
        "model_id": "m/1",
        "class_agnostic_nms": False,
        "class_filter": None,
        "confidence": 0.5,
        "iou_threshold": 0.3,
        "max_detections": 300,
        "max_candidates": 3000,
        "disable_active_learning": True,
        "active_learning_target_dataset": None,
    }
    base.update(over)
    return base


def test_run_locally_returns_inference_models_detections_without_model_id_output() -> None:
    # given
    images = Batch(content=[_make_image()], indices=[(0,)])
    model_manager = MagicMock()
    model_manager.run_tensor_native_inference.return_value = [_make_detections(n=1)]
    model_manager.get_class_names.return_value = ["a"]
    block = RoboflowObjectDetectionModelBlockV1(
        model_manager=model_manager,
        api_key="key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # when
    result = block.run_locally(images=images, **_kwargs())

    # then
    assert len(result) == 1
    # v1 manifest declares only inference_id + predictions (no model_id output).
    assert set(result[0].keys()) == {"inference_id", "predictions"}
    assert isinstance(result[0]["predictions"], TensorDetections)


def test_run_remotely_returns_inference_models_detections_without_model_id_output() -> None:
    # given
    images = Batch(content=[_make_image()], indices=[(0,)])
    http_client = MagicMock()
    http_client.infer.return_value = [
        {"image": {"width": 64, "height": 64}, "predictions": []},
    ]
    model_manager = MagicMock()
    block = RoboflowObjectDetectionModelBlockV1(
        model_manager=model_manager,
        api_key="key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    # when
    with patch(
        "inference.core.workflows.core_steps.models.roboflow.object_detection.v1_tensor.InferenceHTTPClient",
        return_value=http_client,
    ):
        result = block.run_remotely(images=images, **_kwargs())

    # then
    assert set(result[0].keys()) == {"inference_id", "predictions"}
    assert isinstance(result[0]["predictions"], TensorDetections)
