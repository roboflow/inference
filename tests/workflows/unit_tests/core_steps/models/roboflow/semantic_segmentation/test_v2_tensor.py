import base64
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch

from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.roboflow.semantic_segmentation.v2_tensor import (
    RoboflowSemanticSegmentationModelBlockV2,
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


def _make_result() -> SemanticSegmentationResult:
    return SemanticSegmentationResult(
        segmentation_map=torch.zeros((64, 64), dtype=torch.int64),
        confidence=torch.zeros((64, 64), dtype=torch.float32),
    )


def _encode_mask_png(arr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def test_run_locally_returns_semantic_segmentation_result() -> None:
    images = Batch(content=[_image()], indices=[(0,)])
    model_manager = MagicMock()
    model_manager.run_tensor_native_inference.return_value = [_make_result()]
    model_manager.get_class_names.return_value = ["background", "person"]
    block = RoboflowSemanticSegmentationModelBlockV2(
        model_manager=model_manager,
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    result = block.run_locally(
        images, "m/1", confidence="default",
    )
    assert isinstance(result[0]["predictions"], SemanticSegmentationResult)
    assert result[0]["model_id"] == "m/1"


def test_run_remotely_decodes_segmentation_mask_from_base64_png() -> None:
    images = Batch(content=[_image()], indices=[(0,)])
    # Build a (32, 32) mask where pixel value 2 fills the array.
    mask_arr = np.full((32, 32), 2, dtype=np.uint8)
    mask_b64 = _encode_mask_png(mask_arr)
    http_client = MagicMock()
    http_client.infer.return_value = [{
        "predictions": {
            "segmentation_mask": mask_b64,
            "class_map": {"0": "background", "2": "person"},
        },
    }]
    model_manager = MagicMock()
    block = RoboflowSemanticSegmentationModelBlockV2(
        model_manager=model_manager,
        api_key="k",
        step_execution_mode=StepExecutionMode.REMOTE,
    )
    with patch(
        "inference.core.workflows.core_steps.models.roboflow.semantic_segmentation.v2_tensor.InferenceHTTPClient",
        return_value=http_client,
    ):
        result = block.run_remotely(
            images, "m/1", confidence="default",
        )
    pred = result[0]["predictions"]
    assert isinstance(pred, SemanticSegmentationResult)
    assert pred.segmentation_map.shape == (32, 32)
    assert int(pred.segmentation_map.unique().tolist()[0]) == 2
