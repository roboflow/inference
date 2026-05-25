from unittest.mock import MagicMock, patch

import numpy as np
import torch

from inference_models.models.base.keypoints_detection import KeyPoints

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.roboflow.keypoint_detection.v3_tensor import (
    RoboflowKeypointDetectionModelBlockV3,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    OriginCoordinatesSystem,
    WorkflowImageData,
)


def _image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(
            parent_id="p",
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=0, left_top_y=0, origin_width=64, origin_height=64
            ),
        ),
        numpy_image=np.zeros((64, 64, 3), dtype=np.uint8),
    )


def _kp(n: int = 1, k: int = 3) -> KeyPoints:
    return KeyPoints(
        xy=torch.zeros((n, k, 2), dtype=torch.float32),
        class_id=torch.zeros((n,), dtype=torch.int64),
        confidence=torch.zeros((n, k), dtype=torch.float32),
    )


def _kwargs(**over):
    base = {
        "model_id": "m/1",
        "class_agnostic_nms": False,
        "class_filter": None,
        "confidence": 0.5,
        "keypoint_confidence": 0.0,
        "iou_threshold": 0.3,
        "max_detections": 300,
        "max_candidates": 3000,
        "disable_active_learning": True,
        "active_learning_target_dataset": None,
    }
    base.update(over)
    return base


def test_run_locally_unpacks_keypoints_tuple_and_returns_KeyPoints() -> None:
    images = Batch(content=[_image()], indices=[(0,)])
    model_manager = MagicMock()
    # Adapter returns Tuple[List[KeyPoints], Optional[List[Detections]]]
    model_manager.run_tensor_native_inference.return_value = ([_kp()], None)
    model_manager.get_class_names.return_value = ["person"]
    block = RoboflowKeypointDetectionModelBlockV3(
        model_manager=model_manager, api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    result = block.run_locally(images=images, **_kwargs())
    assert isinstance(result[0]["predictions"], KeyPoints)
    assert result[0]["model_id"] == "m/1"


def test_run_remotely_builds_KeyPoints_from_response() -> None:
    images = Batch(content=[_image()], indices=[(0,)])
    http_client = MagicMock()
    http_client.infer.return_value = [{
        "image": {"width": 64, "height": 64},
        "predictions": [{
            "x": 30.0, "y": 30.0, "width": 10.0, "height": 10.0,
            "class": "person", "class_id": 0, "confidence": 0.9,
            "keypoints": [
                {"x": 1.0, "y": 1.0, "confidence": 0.8, "class_id": 0, "class": "nose"},
                {"x": 2.0, "y": 2.0, "confidence": 0.7, "class_id": 1, "class": "eye"},
            ],
        }],
    }]
    model_manager = MagicMock()
    block = RoboflowKeypointDetectionModelBlockV3(
        model_manager=model_manager, api_key="k",
        step_execution_mode=StepExecutionMode.REMOTE,
    )
    with patch(
        "inference.core.workflows.core_steps.models.roboflow.keypoint_detection.v3_tensor.InferenceHTTPClient",
        return_value=http_client,
    ):
        result = block.run_remotely(images=images, **_kwargs())
    pred = result[0]["predictions"]
    assert isinstance(pred, KeyPoints)
    assert pred.xy.shape == (1, 2, 2)
    assert pred.confidence.shape == (1, 2)
    assert pred.key_points_metadata[0]["bbox_xyxy"] == [25.0, 25.0, 35.0, 35.0]
