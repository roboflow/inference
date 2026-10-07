from unittest.mock import patch

import numpy as np
import pytest
import supervision as sv
from pycocotools import mask as mask_utils
from roboflow_workflows.execution_engine.constants import RLE_MASK_KEY_IN_SV_DETECTIONS

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    USE_INFERENCE_MODELS,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
)
from inference.core.interfaces.workflows_models_provider import (
    ModelManagerModelsProvider,
)
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine


@pytest.mark.skipif(
    not USE_INFERENCE_MODELS or ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="Reduced v5 masks require the non-tensor inference_models path",
)
@pytest.mark.parametrize(
    "mode,factor", [("accurate", 1.0), ("tradeoff", 0.5), ("fast", 0.0)]
)
def test_instance_segmentation_v5_with_visualization_blocks(
    model_manager: ModelManagerModelsProvider,
    dogs_image: np.ndarray,
    mode: str,
    factor: float,
) -> None:
    # given
    visualizations = ("mask", "polygon", "bounding_box")
    specification = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_instance_segmentation_model@v5",
                "name": "segment",
                "image": "$inputs.image",
                "model_id": "yolov8n-seg-640",
                "mask_decode_mode": mode,
                "tradeoff_factor": factor,
            },
            *[
                {
                    "type": f"roboflow_core/{name}_visualization@v1",
                    "name": name,
                    "image": "$inputs.image",
                    "predictions": "$steps.segment.predictions",
                    "copy_image": True,
                }
                for name in visualizations
            ],
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.segment.predictions",
            },
            *[
                {
                    "type": "JsonField",
                    "name": name,
                    "selector": f"$steps.{name}.image",
                }
                for name in visualizations
            ],
        ],
    }
    engine = ExecutionEngine.init(
        workflow_definition=specification,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    original_image = dogs_image.copy()
    raw_predictions = []
    run_segmentation = model_manager.run_instance_segmentation

    def capture_predictions(**kwargs):
        predictions = run_segmentation(**kwargs)
        raw_predictions.extend(predictions)
        return predictions

    # when
    with patch.object(
        model_manager, "run_instance_segmentation", side_effect=capture_predictions
    ):
        results = engine.run(runtime_parameters={"image": [dogs_image]})

    # then
    assert len(results) == 1
    assert len(raw_predictions) == 1
    raw_masks = [p["rle"] for p in raw_predictions[0]["predictions"]]
    assert len(raw_masks) == 2
    image_shape = dogs_image.shape[:2]
    for rle in raw_masks:
        if mode == "accurate":
            assert tuple(rle["size"]) == image_shape
        else:
            assert all(size < full for size, full in zip(rle["size"], image_shape))

    result = results[0]
    detections = result["predictions"]
    assert isinstance(detections, sv.Detections)
    assert len(detections) == 2
    assert detections.mask.shape == (2, *image_shape)
    assert all(mask.any() for mask in detections.mask)
    assert detections.data["class_name"].tolist() == ["dog", "dog"]
    for rle, dense in zip(
        detections.data[RLE_MASK_KEY_IN_SV_DETECTIONS], detections.mask
    ):
        assert tuple(rle["size"]) == image_shape
        np.testing.assert_array_equal(mask_utils.decode(rle).astype(bool), dense)

    for name in visualizations:
        rendered = result[name].numpy_image
        assert rendered.shape == dogs_image.shape
        assert rendered.dtype == dogs_image.dtype
        assert np.any(rendered != original_image)
    changed_pixels = np.any(result["mask"].numpy_image != original_image, axis=2)
    mask_union = detections.mask.any(axis=0)
    assert changed_pixels[mask_union].any()
    assert not changed_pixels[~mask_union].any()
    np.testing.assert_array_equal(dogs_image, original_image)
