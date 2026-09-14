from unittest.mock import MagicMock

import numpy as np

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.roboflow.object_detection.v1 import (
    RoboflowObjectDetectionModelBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)

RAW_PREDICTION = {
    "inference_id": "inf-1",
    "image": {"width": 100, "height": 200},
    "predictions": [
        {
            "x": 50.0,
            "y": 100.0,
            "width": 20.0,
            "height": 40.0,
            "confidence": 0.9,
            "class": "cat",
            "class_id": 1,
            "detection_id": "d1",
            "parent_id": "p",
        }
    ],
}


def test_object_detection_v1_local_registers_then_infers_and_post_processes() -> None:
    model_manager = MagicMock()
    model_manager.run_object_detection.return_value = [RAW_PREDICTION]
    block = RoboflowObjectDetectionModelBlockV1(
        model_manager=model_manager,
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    images = Batch(
        content=[
            WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="p"),
                numpy_image=np.zeros((200, 100, 3), dtype=np.uint8),
            )
        ],
        indices=[(0,)],
    )
    result = block.run(
        images=images,
        model_id="m/1",
        class_agnostic_nms=False,
        class_filter=None,
        confidence=0.4,
        iou_threshold=0.3,
        max_detections=300,
        max_candidates=3000,
        disable_active_learning=False,
        active_learning_target_dataset=None,
    )
    model_manager.add_model.assert_called_once_with(model_id="m/1", api_key="k")
    call = model_manager.run_object_detection.call_args.kwargs
    assert call["model_id"] == "m/1" and call["api_key"] == "k"
    assert call["confidence"] == 0.4 and call["iou_threshold"] == 0.3
    assert result[0]["inference_id"] == "inf-1"
    detections = result[0]["predictions"]
    assert len(detections) == 1
    assert detections.xyxy.tolist() == [[40.0, 80.0, 60.0, 120.0]]
    assert detections.confidence.tolist() == [0.9]
