from typing import List, Union

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.formatters.vlm_as_detector.v2 import (
    BlockManifest,
    VLMAsDetectorBlockV2,
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
        "type": "roboflow_core/vlm_as_detector@v2",
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
        type="roboflow_core/vlm_as_detector@v2",
        name="parser",
        image=image,
        vlm_output="$steps.vlm.output",
        classes=classes,
        model_type="google-gemini",
        task_type="object-detection",
    )


def test_run_method_for_claude_and_gemini_output() -> None:
    # given
    block = VLMAsDetectorBlockV2()
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


def test_run_method_for_gemini_native_box_2d_output() -> None:
    # given
    block = VLMAsDetectorBlockV2()
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
[
  {"box_2d": [29, 17, 163, 54], "label": "dog"},
  {"box_2d": [58, 82, 163, 109], "label": "dog"}
]
    """

    # when
    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="google-gemini",
        task_type="object-detection",
    )

    # then
    assert result["error_status"] is False
    assert isinstance(result["predictions"], sv.Detections)
    assert result["predictions"].data["class_name"].tolist() == ["dog", "dog"]
    assert np.allclose(result["predictions"].class_id, np.array([1, 1]))
    assert np.allclose(
        result["predictions"].xyxy,
        np.array(
            [
                [2.856, 5.568, 9.072, 31.296],
                [13.776, 11.136, 18.312, 31.296],
            ]
        ),
        atol=1.0,
    )


def test_run_method_for_openai_box_2d_output_with_resize() -> None:
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
    assert isinstance(result["predictions"], sv.Detections)
    assert result["predictions"].data["class_name"].tolist() == ["cat", "dog"]
    assert np.allclose(result["predictions"].class_id, np.array([0, 1]))
    assert np.allclose(
        result["predictions"].xyxy,
        np.array(
            [
                [200, 400, 600, 800],
                [1000, 1200, 4096, 2048],
            ]
        ),
        atol=1.0,
    )
    assert np.allclose(result["predictions"].confidence, np.array([0.9, 1.0]))


def test_run_method_for_openai_structured_detections_output_with_resize() -> None:
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
    assert isinstance(result["predictions"], sv.Detections)
    assert result["predictions"].data["class_name"].tolist() == ["cat", "dog"]
    assert np.allclose(result["predictions"].class_id, np.array([0, 1]))
    assert np.allclose(
        result["predictions"].xyxy,
        np.array(
            [
                [200, 400, 600, 800],
                [1000, 1200, 2000, 1800],
            ]
        ),
        atol=1.0,
    )


def test_run_method_for_openai_structured_empty_detections_output() -> None:
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
    assert isinstance(result["predictions"], sv.Detections)
    assert len(result["predictions"]) == 0


def test_run_method_for_openai_legacy_detections_output() -> None:
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
    assert isinstance(result["predictions"], sv.Detections)
    assert result["predictions"].data["class_name"].tolist() == ["cat", "dog"]
    assert np.allclose(result["predictions"].class_id, np.array([0, 1]))
    assert np.allclose(
        result["predictions"].xyxy,
        np.array(
            [
                [2, 29, 25, 163],
                [29, 48, 54, 163],
            ]
        ),
        atol=1.0,
    )
    assert np.allclose(result["predictions"].confidence, np.array([0.98, 0.97]))


def test_run_method_for_spacexai_percent_box_2d_output() -> None:
    # given - SpaceXAI Grok returns [x_min, y_min, x_max, y_max] as percentages 0-100
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

    # when
    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="spacexai",
        task_type="object-detection",
    )

    # then - second box x_max clamped to 100 before scaling to width=100
    assert result["error_status"] is False
    assert isinstance(result["predictions"], sv.Detections)
    assert result["predictions"].data["class_name"].tolist() == ["cat", "dog"]
    assert np.allclose(result["predictions"].class_id, np.array([0, 1]))
    assert np.allclose(
        result["predictions"].xyxy,
        np.array(
            [
                [10, 40, 50, 160],
                [60, 20, 100, 80],
            ]
        ),
        atol=1.0,
    )
    assert np.allclose(result["predictions"].confidence, np.array([0.9, 1.0]))


def test_run_method_for_spacexai_unknown_label_gets_class_id_minus_one() -> None:
    block = VLMAsDetectorBlockV2()
    image = WorkflowImageData(
        numpy_image=np.zeros((100, 100, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = '[{"box_2d": [0, 0, 50, 50], "label": "bird"}]'

    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="spacexai",
        task_type="object-detection",
    )

    assert result["error_status"] is False
    assert np.allclose(result["predictions"].class_id, np.array([-1]))


def test_run_method_for_anthropic_claude_box_2d_output_with_resize() -> None:
    # given - original 4000x3000 image is uploaded at 2212x1659 (Claude's
    # high-resolution tier resize), so absolute pixel coordinates must be
    # rescaled back onto the original image
    block = VLMAsDetectorBlockV2()
    image = WorkflowImageData(
        numpy_image=np.zeros((3000, 4000, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
[
  {"box_2d": [553, 415, 1106, 830], "label": "cat"},
  {"box_2d": [1106, 830, 2500, 1800], "label": "dog"}
]
    """

    # when
    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="anthropic-claude",
        task_type="object-detection",
    )

    # then - second box exceeds the 2212x1659 upload and is clamped before scaling
    assert result["error_status"] is False
    assert isinstance(result["predictions"], sv.Detections)
    assert result["predictions"].data["class_name"].tolist() == ["cat", "dog"]
    assert np.allclose(result["predictions"].class_id, np.array([0, 1]))
    assert np.allclose(
        result["predictions"].xyxy,
        np.array(
            [
                [
                    553 * 4000 / 2212,
                    415 * 3000 / 1659,
                    1106 * 4000 / 2212,
                    830 * 3000 / 1659,
                ],
                [1106 * 4000 / 2212, 830 * 3000 / 1659, 4000, 3000],
            ]
        ),
        atol=1.0,
    )
    assert np.allclose(result["predictions"].confidence, np.array([1.0, 1.0]))


def test_run_method_for_anthropic_claude_legacy_detections_output() -> None:
    # given - v1-v4 Claude blocks emit the normalized {"detections": [...]}
    # contract, which must keep parsing through the legacy path
    block = VLMAsDetectorBlockV2()
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
{"detections": [
  {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.6, "class_name": "cat", "confidence": 0.9}
]}
    """

    # when
    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="anthropic-claude",
        task_type="object-detection",
    )

    # then
    assert result["error_status"] is False
    assert isinstance(result["predictions"], sv.Detections)
    assert result["predictions"].data["class_name"].tolist() == ["cat"]
    assert np.allclose(result["predictions"].class_id, np.array([0]))
    assert np.allclose(
        result["predictions"].xyxy,
        np.array([[17, 38, 84, 115]]),
        atol=1.0,
    )
    assert np.allclose(result["predictions"].confidence, np.array([0.9]))


def test_manifest_parsing_for_zai_model_type() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/vlm_as_detector@v2",
        "name": "parser",
        "image": "$inputs.image",
        "vlm_output": "$steps.vlm.output",
        "classes": ["cat", "dog"],
        "model_type": "zai",
        "task_type": "object-detection",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.model_type == "zai"


def test_run_method_for_zai_box_2d_output() -> None:
    # given - the Z.ai GLM block prompts for the same box_2d contract as
    # Qwen: [x_min, y_min, x_max, y_max] integers normalized to 0-1000
    block = VLMAsDetectorBlockV2()
    image = WorkflowImageData(
        numpy_image=np.zeros((480, 640, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
[
  {"box_2d": [100, 200, 500, 1000], "label": "cat"},
  {"box_2d": [0, 0, 500, 500], "label": "unicorn"}
]
    """

    # when
    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="zai",
        task_type="object-detection",
    )

    # then
    assert result["error_status"] is False
    assert isinstance(result["predictions"], sv.Detections)
    assert result["predictions"].data["class_name"].tolist() == ["cat", "unicorn"]
    assert np.allclose(result["predictions"].class_id, np.array([0, -1]))
    assert np.allclose(
        result["predictions"].xyxy,
        np.array(
            [
                [64, 96, 320, 480],
                [0, 0, 320, 240],
            ]
        ),
        atol=1.0,
    )
    assert np.allclose(result["predictions"].confidence, np.array([1.0, 1.0]))


def test_run_method_for_zai_unexpected_shape_sets_error_status() -> None:
    # given - neither a JSON list nor a {"detections": [...]} object
    block = VLMAsDetectorBlockV2()
    image = WorkflowImageData(
        numpy_image=np.zeros((480, 640, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )

    # when
    result = block.run(
        image=image,
        vlm_output='{"objects": []}',
        classes=["cat"],
        model_type="zai",
        task_type="object-detection",
    )

    # then
    assert result["error_status"] is True
    assert result["predictions"] is None


def test_run_method_for_invalid_claude_and_gemini_output() -> None:
    # given
    block = VLMAsDetectorBlockV2()
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
    block = VLMAsDetectorBlockV2()
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


def test_formatter_for_florence2_object_detection() -> None:
    # given
    block = VLMAsDetectorBlockV2()
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
{"bboxes": [[434.0, 30.848499298095703, 760.4000244140625, 530.4144897460938], [0.4000000059604645, 96.13949584960938, 528.4000244140625, 564.5574951171875]], "labels": ["cat", "dog"]}
"""

    # when
    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="florence-2",
        task_type="object-detection",
    )

    # then
    assert result["error_status"] is False
    assert isinstance(result["predictions"], sv.Detections)
    assert len(result["inference_id"]) > 0
    assert np.allclose(
        result["predictions"].xyxy,
        np.array([[434, 30.848, 760.4, 530.41], [0.4, 96.139, 528.4, 564.56]]),
        atol=1e-1,
    ), "Expected coordinates to be the same as given in raw input"
    assert result["predictions"].class_id.tolist() == [7725, 5324]
    assert np.allclose(result["predictions"].confidence, np.array([1.0, 1.0]))
    assert result["predictions"].data["class_name"].tolist() == ["cat", "dog"]
    assert "class_name" in result["predictions"].data
    assert "image_dimensions" in result["predictions"].data
    assert "prediction_type" in result["predictions"].data
    assert "parent_coordinates" in result["predictions"].data
    assert "parent_dimensions" in result["predictions"].data
    assert "root_parent_coordinates" in result["predictions"].data
    assert "root_parent_dimensions" in result["predictions"].data
    assert "parent_id" in result["predictions"].data
    assert "root_parent_id" in result["predictions"].data


def test_formatter_for_florence2_open_vocabulary_object_detection() -> None:
    # given
    block = VLMAsDetectorBlockV2()
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
{"bboxes": [[434.0, 30.848499298095703, 760.4000244140625, 530.4144897460938], [0.4000000059604645, 96.13949584960938, 528.4000244140625, 564.5574951171875]], "bboxes_labels": ["cat", "dog"]}
"""

    # when
    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="florence-2",
        task_type="open-vocabulary-object-detection",
    )

    # then
    assert result["error_status"] is False
    assert isinstance(result["predictions"], sv.Detections)
    assert len(result["inference_id"]) > 0
    assert np.allclose(
        result["predictions"].xyxy,
        np.array([[434, 30.848, 760.4, 530.41], [0.4, 96.139, 528.4, 564.56]]),
        atol=1e-1,
    ), "Expected coordinates to be the same as given in raw input"
    assert result["predictions"].class_id.tolist() == [0, 1]
    assert np.allclose(result["predictions"].confidence, np.array([1.0, 1.0]))
    assert result["predictions"].data["class_name"].tolist() == ["cat", "dog"]
    assert "class_name" in result["predictions"].data
    assert "image_dimensions" in result["predictions"].data
    assert "prediction_type" in result["predictions"].data
    assert "parent_coordinates" in result["predictions"].data
    assert "parent_dimensions" in result["predictions"].data
    assert "root_parent_coordinates" in result["predictions"].data
    assert "root_parent_dimensions" in result["predictions"].data
    assert "parent_id" in result["predictions"].data
    assert "root_parent_id" in result["predictions"].data


def test_formatter_for_florence2_phase_grounded_detection() -> None:
    # given
    block = VLMAsDetectorBlockV2()
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
{"bboxes": [[434.0, 30.848499298095703, 760.4000244140625, 530.4144897460938], [0.4000000059604645, 96.13949584960938, 528.4000244140625, 564.5574951171875]], "labels": ["cat", "dog"]}
"""

    # when
    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="florence-2",
        task_type="phrase-grounded-object-detection",
    )

    # then
    assert result["error_status"] is False
    assert isinstance(result["predictions"], sv.Detections)
    assert len(result["inference_id"]) > 0
    assert np.allclose(
        result["predictions"].xyxy,
        np.array([[434, 30.848, 760.4, 530.41], [0.4, 96.139, 528.4, 564.56]]),
        atol=1e-1,
    ), "Expected coordinates to be the same as given in raw input"
    assert result["predictions"].class_id.tolist() == [7725, 5324]
    assert np.allclose(result["predictions"].confidence, np.array([1.0, 1.0]))
    assert result["predictions"].data["class_name"].tolist() == ["cat", "dog"]
    assert "class_name" in result["predictions"].data
    assert "image_dimensions" in result["predictions"].data
    assert "prediction_type" in result["predictions"].data
    assert "parent_coordinates" in result["predictions"].data
    assert "parent_dimensions" in result["predictions"].data
    assert "root_parent_coordinates" in result["predictions"].data
    assert "root_parent_dimensions" in result["predictions"].data
    assert "parent_id" in result["predictions"].data
    assert "root_parent_id" in result["predictions"].data


def test_formatter_for_florence2_region_proposal() -> None:
    # given
    block = VLMAsDetectorBlockV2()
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
{"bboxes": [[434.0, 30.848499298095703, 760.4000244140625, 530.4144897460938], [0.4000000059604645, 96.13949584960938, 528.4000244140625, 564.5574951171875]], "labels": ["", ""]}
"""

    # when
    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=[],
        model_type="florence-2",
        task_type="region-proposal",
    )

    # then
    assert result["error_status"] is False
    assert isinstance(result["predictions"], sv.Detections)
    assert len(result["inference_id"]) > 0
    assert np.allclose(
        result["predictions"].xyxy,
        np.array([[434, 30.848, 760.4, 530.41], [0.4, 96.139, 528.4, 564.56]]),
        atol=1e-1,
    ), "Expected coordinates to be the same as given in raw input"
    assert result["predictions"].class_id.tolist() == [0, 0]
    assert np.allclose(result["predictions"].confidence, np.array([1.0, 1.0]))
    assert result["predictions"].data["class_name"].tolist() == ["roi", "roi"]
    assert "class_name" in result["predictions"].data
    assert "image_dimensions" in result["predictions"].data
    assert "prediction_type" in result["predictions"].data
    assert "parent_coordinates" in result["predictions"].data
    assert "parent_dimensions" in result["predictions"].data
    assert "root_parent_coordinates" in result["predictions"].data
    assert "root_parent_dimensions" in result["predictions"].data
    assert "parent_id" in result["predictions"].data
    assert "root_parent_id" in result["predictions"].data


def test_formatter_for_florence2_ocr() -> None:
    # given
    block = VLMAsDetectorBlockV2()
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
{"quad_boxes": [[336.9599914550781, 77.22000122070312, 770.8800048828125, 77.22000122070312, 770.8800048828125, 144.1800079345703, 336.9599914550781, 144.1800079345703], [1273.919921875, 77.22000122070312, 1473.5999755859375, 77.22000122070312, 1473.5999755859375, 109.62000274658203, 1273.919921875, 109.62000274658203], [1652.159912109375, 72.9000015258789, 1828.7999267578125, 70.74000549316406, 1828.7999267578125, 129.05999755859375, 1652.159912109375, 131.22000122070312], [1273.919921875, 126.9000015258789, 1467.8399658203125, 126.9000015258789, 1467.8399658203125, 160.3800048828125, 1273.919921875, 160.3800048828125], [340.79998779296875, 173.3400115966797, 964.7999877929688, 173.3400115966797, 964.7999877929688, 250.02000427246094, 340.79998779296875, 251.10000610351562], [1273.919921875, 177.66000366210938, 1473.5999755859375, 177.66000366210938, 1473.5999755859375, 208.98001098632812, 1273.919921875, 208.98001098632812], [1272.0, 226.260009765625, 1467.8399658203125, 226.260009765625, 1467.8399658203125, 259.7400207519531, 1272.0, 259.7400207519531], [340.79998779296875, 264.05999755859375, 801.5999755859375, 264.05999755859375, 801.5999755859375, 345.0600280761719, 340.79998779296875, 345.0600280761719], [1273.919921875, 277.02001953125, 1471.679931640625, 277.02001953125, 1471.679931640625, 309.4200134277344, 1273.919921875, 309.4200134277344], [1273.919921875, 326.70001220703125, 1467.8399658203125, 326.70001220703125, 1467.8399658203125, 359.1000061035156, 1273.919921875, 359.1000061035156], [336.9599914550781, 376.3800048828125, 980.1599731445312, 376.3800048828125, 980.1599731445312, 417.4200134277344, 336.9599914550781, 417.4200134277344]], "labels": ["</s>What is OCR", "01010110", "veryfi", "010100101", "(Optical Character", "01010010", "011100101", "Recognition?", "0101010", "01010001", "A Friendly Introduction to OCR Software"]}    
"""

    # when
    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=[],
        model_type="florence-2",
        task_type="ocr-with-text-detection",
    )

    # then
    assert result["error_status"] is False
    assert isinstance(result["predictions"], sv.Detections)
    assert len(result["inference_id"]) > 0
    assert np.allclose(
        result["predictions"].xyxy,
        np.array(
            [
                [336.96, 77.22, 770.88, 144.18],
                [1273.9, 77.22, 1473.6, 109.62],
                [1652.2, 70.74, 1828.8, 131.22],
                [1273.9, 126.9, 1467.8, 160.38],
                [340.8, 173.34, 964.8, 251.1],
                [1273.9, 177.66, 1473.6, 208.98],
                [1272, 226.26, 1467.8, 259.74],
                [340.8, 264.06, 801.6, 345.06],
                [1273.9, 277.02, 1471.7, 309.42],
                [1273.9, 326.7, 1467.8, 359.1],
                [336.96, 376.38, 980.16, 417.42],
            ]
        ),
        atol=1e-1,
    ), "Expected coordinates to be the same as given in raw input"
    assert result["predictions"].class_id.tolist() == [0] * 11
    assert np.allclose(result["predictions"].confidence, np.array([1.0] * 11))
    assert result["predictions"].data["class_name"].tolist() == [
        "</s>What is OCR",
        "01010110",
        "veryfi",
        "010100101",
        "(Optical Character",
        "01010010",
        "011100101",
        "Recognition?",
        "0101010",
        "01010001",
        "A Friendly Introduction to OCR Software",
    ]
    assert "class_name" in result["predictions"].data
    assert "image_dimensions" in result["predictions"].data
    assert "prediction_type" in result["predictions"].data
    assert "parent_coordinates" in result["predictions"].data
    assert "parent_dimensions" in result["predictions"].data
    assert "root_parent_coordinates" in result["predictions"].data
    assert "root_parent_dimensions" in result["predictions"].data
    assert "parent_id" in result["predictions"].data
    assert "root_parent_id" in result["predictions"].data
