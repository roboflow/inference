from unittest.mock import MagicMock, patch

import numpy as np
import torch

from inference_models.models.base.classification import ClassificationPrediction

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.roboflow.multi_class_classification.v3_tensor import (
    RoboflowClassificationModelBlockV3,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    OriginCoordinatesSystem,
    WorkflowImageData,
)


def _image(parent_id: str = "p") -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(
            parent_id=parent_id,
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=0, left_top_y=0, origin_width=64, origin_height=64
            ),
        ),
        numpy_image=np.zeros((64, 64, 3), dtype=np.uint8),
    )


def test_run_locally_slices_batch_classification_prediction_per_image() -> None:
    images = Batch(content=[_image("p0"), _image("p1")], indices=[(0,), (1,)])
    batch_pred = ClassificationPrediction(
        class_id=torch.tensor([3, 7], dtype=torch.int64),
        confidence=torch.tensor([0.9, 0.8], dtype=torch.float32),
    )
    model_manager = MagicMock()
    model_manager.run_tensor_native_inference.return_value = batch_pred
    model_manager.get_class_names.return_value = ["a", "b", "c", "d", "e", "f", "g", "h"]
    block = RoboflowClassificationModelBlockV3(
        model_manager=model_manager, api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run_locally(
        images=images, model_id="m/1", confidence="default",
        disable_active_learning=True, active_learning_target_dataset=None,
    )

    assert len(result) == 2
    # Each row carries its own sliced ClassificationPrediction
    assert isinstance(result[0]["predictions"], ClassificationPrediction)
    assert result[0]["predictions"].class_id.tolist() == [3]
    assert result[1]["predictions"].class_id.tolist() == [7]
    # images_metadata is per-image plural list of length 1 in each slice
    assert len(result[0]["predictions"].images_metadata) == 1


def test_run_remotely_builds_classification_prediction_from_top1_per_response() -> None:
    images = Batch(content=[_image()], indices=[(0,)])
    http_client = MagicMock()
    http_client.infer.return_value = [{
        "top": "dog",
        "confidence": 0.85,
        "predictions": [
            {"class": "cat", "class_id": 0, "confidence": 0.1},
            {"class": "dog", "class_id": 1, "confidence": 0.85},
        ],
    }]
    model_manager = MagicMock()
    block = RoboflowClassificationModelBlockV3(
        model_manager=model_manager, api_key="k",
        step_execution_mode=StepExecutionMode.REMOTE,
    )
    with patch(
        "inference.core.workflows.core_steps.models.roboflow.multi_class_classification.v3_tensor.InferenceHTTPClient",
        return_value=http_client,
    ):
        result = block.run_remotely(
            images=images, model_id="m/1", confidence="default",
            disable_active_learning=True, active_learning_target_dataset=None,
        )

    pred = result[0]["predictions"]
    assert isinstance(pred, ClassificationPrediction)
    assert pred.class_id.tolist() == [1]
    assert abs(pred.confidence.item() - 0.85) < 1e-5
