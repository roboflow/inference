from typing import List, Union

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.formatters.vlm_as_detector.v1 import (
    BlockManifest,
    VLMAsDetectorBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("image", ["$inputs.image", "$steps.some.image"])
@pytest.mark.parametrize(
    "classes", ["$inputs.classes", "$steps.some.classes", ["a", "b"]]
)
def test_manifest_parsing_when_input_valid(
    image: str, classes: Union[str, List[str]]
) -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/vlm_as_detector@v1",
        "name": "parser",
        "image": image,
        "vlm_output": "$steps.vlm.output",
        "classes": classes,
        "model_type": "google-gemini",
        "task_type": "object-detection",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result == BlockManifest(
        type="roboflow_core/vlm_as_detector@v1",
        name="parser",
        image=image,
        vlm_output="$steps.vlm.output",
        classes=classes,
        model_type="google-gemini",
        task_type="object-detection",
    )


def test_run_method_for_claude_and_gemini_output() -> None:
    # given
    block = VLMAsDetectorBlockV1()
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
{"detections": [
  {"x_min": 0.01, "y_min": 0.15, "x_max": 0.15, "y_max": 0.85, "class_name": "cat", "confidence": 1.98},
  {"x_min": 0.17, "y_min": 0.25, "x_max": 0.32, "y_max": 0.85, "class_name": "dog", "confidence": 0.97},
  {"x_min": 0.33, "y_min": 0.15, "x_max": 0.47, "y_max": 0.85, "class_name": "cat", "confidence": 0.99},
  {"x_min": 0.49, "y_min": 0.30, "x_max": 0.65, "y_max": 0.85, "class_name": "dog", "confidence": 0.98},
  {"x_min": 0.67, "y_min": 0.20, "x_max": 0.82, "y_max": 0.85, "class_name": "cat", "confidence": 0.99},
  {"x_min": 0.84, "y_min": 0.25, "x_max": 0.99, "y_max": 0.85, "class_name": "unknown", "confidence": 0.97}
]}
    """

    # when
    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog", "lion"],
        model_type="google-gemini",
        task_type="object-detection",
    )

    # then
    assert result["error_status"] is False
    assert isinstance(result["predictions"], sv.Detections)
    assert len(result["inference_id"]) > 0
    assert np.allclose(
        result["predictions"].xyxy,
        np.array(
            [
                [2, 29, 25, 163],
                [29, 48, 54, 163],
                [55, 29, 79, 163],
                [82, 58, 109, 163],
                [113, 38, 138, 163],
                [141, 48, 166, 163],
            ]
        ),
        atol=1.0,
    )
    assert np.allclose(result["predictions"].class_id, np.array([0, 1, 0, 1, 0, -1]))
    assert np.allclose(
        result["predictions"].confidence, np.array([1.0, 0.97, 0.99, 0.98, 0.99, 0.97])
    )
    assert "class_name" in result["predictions"].data
    assert "image_dimensions" in result["predictions"].data
    assert "prediction_type" in result["predictions"].data
    assert "parent_coordinates" in result["predictions"].data
    assert "parent_dimensions" in result["predictions"].data
    assert "root_parent_coordinates" in result["predictions"].data
    assert "root_parent_dimensions" in result["predictions"].data
    assert "parent_id" in result["predictions"].data
    assert "root_parent_id" in result["predictions"].data


def test_run_method_for_invalid_claude_and_gemini_output() -> None:
    # given
    block = VLMAsDetectorBlockV1()
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
    {"detections": [
      {"x_min": 0.01, "y_min": 0.15, "x_max": 0.15, "y_max": 0.85, "confidence": 1.98},
      {"x_min": 0.17, "y_min": 0.25, "x_max": 0.32, "y_max": 0.85, "class_name": "dog", "confidence": 0.97},
      {"x_min": 0.33, "y_min": 0.15, "x_max": 0.47, "y_max": 0.85, "class_name": "cat", "confidence": 0.99},
      {"x_min": 0.49, "x_max": 0.65, "y_max": 0.85, "class_name": "dog", "confidence": 0.98},
      {"x_min": 0.67, "y_min": 0.20, "x_max": 0.82, "y_max": 0.85, "class_name": "cat", "confidence": 0.99},
      {"x_min": 0.84, "y_min": 0.25, "x_max": 0.99, "y_max": 0.85, "class_name": "unknown", "confidence": 0.97}
    ]}
        """

    # when
    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog", "lion"],
        model_type="google-gemini",
        task_type="object-detection",
    )

    # then
    assert result["error_status"] is True
    assert result["predictions"] is None
    assert len(result["inference_id"]) > 0


def test_run_method_for_invalid_json() -> None:
    # given
    block = VLMAsDetectorBlockV1()
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )

    # when
    result = block.run(
        image=image,
        vlm_output="invalid",
        classes=["cat", "dog", "lion"],
        model_type="google-gemini",
        task_type="object-detection",
    )

    # then
    assert result["error_status"] is True
    assert result["predictions"] is None
    assert len(result["inference_id"]) > 0
