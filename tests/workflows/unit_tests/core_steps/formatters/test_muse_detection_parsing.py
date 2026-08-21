import numpy as np
import pytest

from inference.core.workflows.core_steps.formatters.vlm_as_detector.muse_detection_parsing import (
    convert_muse_detection_to_pixel_xyxy,
    parse_muse_object_detection_response,
)
from inference.core.workflows.core_steps.formatters.vlm_as_detector.v2 import (
    VLMAsDetectorBlockV2,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def _build_image(height: int, width: int) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((height, width, 3), dtype=np.uint8),
    )


def test_parse_scales_0_to_1000_fields_onto_non_square_image():
    result = parse_muse_object_detection_response(
        image=_build_image(480, 640),
        parsed_data=[
            {"label": "cat", "x_min": 100, "y_min": 200, "x_max": 300, "y_max": 400}
        ],
        classes=["cat", "dog"],
        inference_id="inf",
    )
    assert len(result) == 1
    np.testing.assert_allclose(result.xyxy[0], [64, 96, 192, 192])


def test_convert_box_clamps_out_of_range_coordinates():
    result = convert_muse_detection_to_pixel_xyxy(
        box=[-50.0, 0.0, 1200.0, 1000.0],
        image_height=480,
        image_width=640,
    )
    assert result == [0.0, 0.0, 640.0, 480.0]


def test_parse_accepts_detections_wrapper_and_single_object():
    wrapped = parse_muse_object_detection_response(
        image=_build_image(1000, 1000),
        parsed_data={
            "detections": [
                {"label": "cat", "x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4}
            ]
        },
        classes=["cat"],
        inference_id="inf",
    )
    single = parse_muse_object_detection_response(
        image=_build_image(1000, 1000),
        parsed_data={
            "label": "dog",
            "x_min": 10,
            "y_min": 20,
            "x_max": 30,
            "y_max": 40,
        },
        classes=["cat", "dog"],
        inference_id="inf",
    )
    assert len(wrapped) == 1
    assert len(single) == 1
    assert single.data["class_name"][0] == "dog"


def test_parse_skips_malformed_entries_but_keeps_valid_ones():
    result = parse_muse_object_detection_response(
        image=_build_image(1000, 1000),
        parsed_data=[
            {"label": "no-box"},
            {"label": "bool", "x_min": True, "y_min": 2, "x_max": 3, "y_max": 4},
            {"label": "str", "x_min": "1", "y_min": 2, "x_max": 3, "y_max": 4},
            {
                "label": "nan",
                "x_min": float("nan"),
                "y_min": 2,
                "x_max": 3,
                "y_max": 4,
            },
            "not-a-dict",
            {"label": "cat", "x_min": 100, "y_min": 200, "x_max": 300, "y_max": 400},
        ],
        classes=["cat"],
        inference_id="inf",
    )
    assert len(result) == 1
    assert result.data["class_name"][0] == "cat"


def test_parse_raises_on_unexpected_shape():
    with pytest.raises(ValueError):
        parse_muse_object_detection_response(
            image=_build_image(1000, 1000),
            parsed_data={"unrelated": "object"},
            classes=["cat"],
            inference_id="inf",
        )


def test_detector_run_recovers_glimmer_loose_objects():
    block = VLMAsDetectorBlockV2()
    result = block.run(
        image=_build_image(1000, 1000),
        vlm_output=(
            '{"label": "cat", "x_min": 100, "y_min": 200, "x_max": 300, "y_max": 400}, '
            '{"label": "dog", "x_min": 10, "y_min": 20, "x_max": 30, "y_max": 40}'
        ),
        classes=["cat", "dog"],
        model_type="muse",
        task_type="object-detection",
    )
    assert result["error_status"] is False
    assert len(result["predictions"]) == 2
    np.testing.assert_allclose(result["predictions"].xyxy[0], [100, 200, 300, 400])
    np.testing.assert_allclose(result["predictions"].xyxy[1], [10, 20, 30, 40])
    assert list(result["predictions"].data["class_name"]) == ["cat", "dog"]


def test_detector_run_keeps_error_status_for_garbage_with_unrelated_json():
    block = VLMAsDetectorBlockV2()
    result = block.run(
        image=_build_image(1000, 1000),
        vlm_output='I could not detect anything {"note": "sorry"} in the image',
        classes=["cat"],
        model_type="muse",
        task_type="object-detection",
    )
    assert result["error_status"] is True
    assert result["predictions"] is None
