"""Tensor-native parity tests for `vlm_as_detector` `v2_tensor` — mirrors the
numpy `test_v2.py` OpenAI object-detection coverage (`box_2d` absolute style,
structured wrapper, structured-empty, and the v1-v4 normalized-legacy style)
against the native `inference_models.Detections` carrier."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("inference_models")

from inference.core.workflows.core_steps.formatters.vlm_as_detector.v2_tensor import (
    VLMAsDetectorBlockV2,
)
from inference.core.workflows.execution_engine.constants import CLASS_NAME_KEY
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference_models.models.base.object_detection import Detections


def _class_names(detections: Detections) -> list:
    return [data[CLASS_NAME_KEY] for data in detections.bboxes_metadata]


def test_run_method_for_openai_box_2d_output_with_resize_tensor_native() -> None:
    # given - original 4096x2048 image is uploaded at 2048x1024 (max edge 2048),
    # so absolute pixel coordinates must be scaled up by 2x on both axes
    block = VLMAsDetectorBlockV2()
    image = WorkflowImageData(
        numpy_image=np.zeros((2048, 4096, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
[
  {"box_2d": [100, 200, 300, 400], "label": "cat", "confidence": 0.9},
  {"box_2d": [500, 600, 2100, 1100], "label": "dog"}
]
    """

    # when
    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="openai",
        task_type="object-detection",
    )

    # then - second box exceeds the 2048x1024 upload and is clamped before scaling
    assert result["error_status"] is False
    assert isinstance(result["predictions"], Detections)
    assert _class_names(result["predictions"]) == ["cat", "dog"]
    assert np.allclose(result["predictions"].class_id.cpu().numpy(), np.array([0, 1]))
    assert np.allclose(
        result["predictions"].xyxy.cpu().numpy(),
        np.array(
            [
                [200, 400, 600, 800],
                [1000, 1200, 4096, 2048],
            ]
        ),
        atol=1.0,
    )
    assert np.allclose(
        result["predictions"].confidence.cpu().numpy(), np.array([0.9, 1.0])
    )


def test_run_method_for_openai_structured_detections_output_with_resize_tensor_native() -> (
    None
):
    # given - format produced by roboflow_core/open_ai@v5 structured-absolute
    # style: box_2d entries wrapped in a "detections" object; original
    # 4096x2048 image uploads at 2048x1024, so coordinates scale up by 2x
    block = VLMAsDetectorBlockV2()
    image = WorkflowImageData(
        numpy_image=np.zeros((2048, 4096, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
{"detections": [
  {"box_2d": [100, 200, 300, 400], "label": "cat"},
  {"box_2d": [500, 600, 1000, 900], "label": "dog"}
]}
    """

    # when
    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="openai",
        task_type="object-detection",
    )

    # then
    assert result["error_status"] is False
    assert isinstance(result["predictions"], Detections)
    assert _class_names(result["predictions"]) == ["cat", "dog"]
    assert np.allclose(result["predictions"].class_id.cpu().numpy(), np.array([0, 1]))
    assert np.allclose(
        result["predictions"].xyxy.cpu().numpy(),
        np.array(
            [
                [200, 400, 600, 800],
                [1000, 1200, 2000, 1800],
            ]
        ),
        atol=1.0,
    )


def test_run_method_for_openai_structured_empty_detections_output_tensor_native() -> (
    None
):
    # given - structured outputs guarantee the wrapper even with no objects
    block = VLMAsDetectorBlockV2()
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = '{"detections": []}'

    # when
    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="openai",
        task_type="object-detection",
    )

    # then
    assert result["error_status"] is False
    assert isinstance(result["predictions"], Detections)
    assert len(result["predictions"]) == 0


def test_run_method_for_openai_legacy_detections_output_tensor_native() -> None:
    # given - format produced by roboflow_core/open_ai@v1-v4 object-detection task
    block = VLMAsDetectorBlockV2()
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
{"detections": [
  {"x_min": 0.01, "y_min": 0.15, "x_max": 0.15, "y_max": 0.85, "class_name": "cat", "confidence": 0.98},
  {"x_min": 0.17, "y_min": 0.25, "x_max": 0.32, "y_max": 0.85, "class_name": "dog", "confidence": 0.97}
]}
    """

    # when
    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="openai",
        task_type="object-detection",
    )

    # then
    assert result["error_status"] is False
    assert isinstance(result["predictions"], Detections)
    assert _class_names(result["predictions"]) == ["cat", "dog"]
    assert np.allclose(result["predictions"].class_id.cpu().numpy(), np.array([0, 1]))
    assert np.allclose(
        result["predictions"].xyxy.cpu().numpy(),
        np.array(
            [
                [2, 29, 25, 163],
                [29, 48, 54, 163],
            ]
        ),
        atol=1.0,
    )
    assert np.allclose(
        result["predictions"].confidence.cpu().numpy(), np.array([0.98, 0.97])
    )


def test_run_method_for_spacexai_percent_box_2d_output_tensor_native() -> None:
    block = VLMAsDetectorBlockV2()
    image = WorkflowImageData(
        numpy_image=np.zeros((200, 100, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
[
  {"box_2d": [10.0, 20.0, 50.0, 80.0], "label": "cat", "confidence": 0.9},
  {"box_2d": [60.0, 10.0, 110.0, 40.0], "label": "dog"}
]
    """

    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="spacexai",
        task_type="object-detection",
    )

    assert result["error_status"] is False
    assert isinstance(result["predictions"], Detections)
    assert _class_names(result["predictions"]) == ["cat", "dog"]
    assert np.allclose(result["predictions"].class_id.cpu().numpy(), np.array([0, 1]))
    assert np.allclose(
        result["predictions"].xyxy.cpu().numpy(),
        np.array(
            [
                [10, 40, 50, 160],
                [60, 20, 100, 80],
            ]
        ),
        atol=1.0,
    )
    assert np.allclose(
        result["predictions"].confidence.cpu().numpy(), np.array([0.9, 1.0])
    )


def test_run_method_for_qwen_box_2d_output_tensor_native() -> None:
    # given - qwen coordinates are normalized to 0-1000 on both axes; the
    # bbox_2d alias and label alias keys are accepted, and model-provided
    # confidence is ignored (hardcoded 1.0)
    block = VLMAsDetectorBlockV2()
    image = WorkflowImageData(
        numpy_image=np.zeros((480, 640, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
[
  {"box_2d": [100, 200, 500, 1000], "label": "cat", "confidence": 0.75},
  {"bbox_2d": [0, 0, 500, 500], "description": "unicorn"}
]
    """

    # when
    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="qwen",
        task_type="object-detection",
    )

    # then
    assert result["error_status"] is False
    assert isinstance(result["predictions"], Detections)
    assert _class_names(result["predictions"]) == ["cat", "unicorn"]
    assert np.allclose(result["predictions"].class_id.cpu().numpy(), np.array([0, -1]))
    assert np.allclose(
        result["predictions"].xyxy.cpu().numpy(),
        np.array(
            [
                [64, 96, 320, 480],
                [0, 0, 320, 240],
            ]
        ),
        atol=1.0,
    )
    assert np.allclose(
        result["predictions"].confidence.cpu().numpy(), np.array([1.0, 1.0])
    )


def test_run_method_for_qwen_unexpected_shape_sets_error_status_tensor_native() -> None:
    # given - neither a JSON list nor a {"detections": [...]} object
    block = VLMAsDetectorBlockV2()
    image = WorkflowImageData(
        numpy_image=np.zeros((480, 640, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )

    result = block.run(
        image=image,
        vlm_output='{"objects": []}',
        classes=["cat"],
        model_type="qwen",
        task_type="object-detection",
    )

    assert result["error_status"] is True
    assert result["predictions"] is None


def test_run_method_for_muse_named_fields_output_tensor_native() -> None:
    # given - muse coordinates are named x_min/y_min/x_max/y_max fields
    # normalized to 0-1000 on both axes; out-of-range values are clamped
    block = VLMAsDetectorBlockV2()
    image = WorkflowImageData(
        numpy_image=np.zeros((480, 640, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
[
  {"label": "cat", "x_min": 100, "y_min": 200, "x_max": 500, "y_max": 1000},
  {"label": "unicorn", "x_min": -50, "y_min": 0, "x_max": 1200, "y_max": 500}
]
    """

    # when
    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="muse",
        task_type="object-detection",
    )

    # then
    assert result["error_status"] is False
    assert isinstance(result["predictions"], Detections)
    assert _class_names(result["predictions"]) == ["cat", "unicorn"]
    assert np.allclose(result["predictions"].class_id.cpu().numpy(), np.array([0, -1]))
    assert np.allclose(
        result["predictions"].xyxy.cpu().numpy(),
        np.array(
            [
                [64, 96, 320, 480],
                [0, 0, 640, 240],
            ]
        ),
        atol=1.0,
    )
    assert np.allclose(
        result["predictions"].confidence.cpu().numpy(), np.array([1.0, 1.0])
    )


def test_run_method_for_muse_recovers_loose_objects_tensor_native() -> None:
    # given - Glimmer-style `{...}, {...}` output without array brackets,
    # which string2json rejects; the muse fallback must recover it
    block = VLMAsDetectorBlockV2()
    image = WorkflowImageData(
        numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = (
        '{"label": "cat", "x_min": 100, "y_min": 200, "x_max": 300, "y_max": 400}, '
        '{"label": "dog", "x_min": 10, "y_min": 20, "x_max": 30, "y_max": 40}'
    )

    # when
    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="muse",
        task_type="object-detection",
    )

    # then
    assert result["error_status"] is False
    assert isinstance(result["predictions"], Detections)
    assert _class_names(result["predictions"]) == ["cat", "dog"]
    assert np.allclose(
        result["predictions"].xyxy.cpu().numpy(),
        np.array(
            [
                [100, 200, 300, 400],
                [10, 20, 30, 40],
            ]
        ),
        atol=1.0,
    )
