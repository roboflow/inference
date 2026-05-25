from unittest.mock import MagicMock, patch

import numpy as np
import torch

from inference_models.models.base.classification import (
    MultiLabelClassificationPrediction,
)

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.roboflow.multi_label_classification.v3_tensor import (
    RoboflowMultiLabelClassificationModelBlockV3,
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


def _ml_pred(n: int = 2) -> MultiLabelClassificationPrediction:
    return MultiLabelClassificationPrediction(
        class_ids=torch.arange(n, dtype=torch.int64),
        confidence=torch.full((n,), 0.7, dtype=torch.float32),
    )


def test_run_locally_returns_one_multi_label_pred_per_image() -> None:
    images = Batch(content=[_image(), _image()], indices=[(0,), (1,)])
    model_manager = MagicMock()
    model_manager.run_tensor_native_inference.return_value = [_ml_pred(2), _ml_pred(1)]
    model_manager.get_class_names.return_value = ["a", "b", "c"]
    block = RoboflowMultiLabelClassificationModelBlockV3(
        model_manager=model_manager, api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    result = block.run_locally(
        images, "m/1", "default", True, None,
    )
    assert len(result) == 2
    assert isinstance(result[0]["predictions"], MultiLabelClassificationPrediction)
    assert result[0]["predictions"].class_ids.shape == (2,)
    assert result[1]["predictions"].class_ids.shape == (1,)


def test_run_remotely_builds_multi_label_predictions_from_dict_response() -> None:
    images = Batch(content=[_image()], indices=[(0,)])
    http_client = MagicMock()
    http_client.infer.return_value = [{
        "predictions": {
            "cat": {"class_id": 0, "confidence": 0.9},
            "dog": {"class_id": 1, "confidence": 0.85},
            "fish": {"class_id": 2, "confidence": 0.05},
        },
        "predicted_classes": ["cat", "dog"],
    }]
    model_manager = MagicMock()
    block = RoboflowMultiLabelClassificationModelBlockV3(
        model_manager=model_manager, api_key="k",
        step_execution_mode=StepExecutionMode.REMOTE,
    )
    with patch(
        "inference.core.workflows.core_steps.models.roboflow.multi_label_classification.v3_tensor.InferenceHTTPClient",
        return_value=http_client,
    ):
        result = block.run_remotely(images, "m/1", "default", True, None)
    pred = result[0]["predictions"]
    assert isinstance(pred, MultiLabelClassificationPrediction)
    # Only "cat" and "dog" were over threshold — class_ids should be [0, 1].
    assert pred.class_ids.tolist() == [0, 1]
