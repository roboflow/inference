import numpy as np
import pytest

from inference.core.workflows.core_steps.formatters.vlm_as_detector.qwen_detection_parsing import (
    convert_qwen_detection_to_pixel_xyxy,
    get_qwen_detection_box,
    get_qwen_detection_class_name,
    parse_qwen_object_detection_response,
)
from inference.core.workflows.core_steps.common.serializers import serialise_sv_detections
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
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


# ---------------------------------------------------------------------------
# Box / label extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "detection",
    [
        pytest.param({}, id="no-box-key"),
        pytest.param({"box_2d": [1, 2, 3]}, id="too-few-coordinates"),
        pytest.param({"box_2d": [1, 2, 3, 4, 5]}, id="too-many-coordinates"),
        pytest.param({"box_2d": [1, 2, 3, "x"]}, id="non-numeric-coordinate"),
        pytest.param({"box_2d": [1, 2, 3, True]}, id="boolean-coordinate"),
        pytest.param({"box_2d": "1,2,3,4"}, id="box-not-a-list"),
    ],
)
def test_get_box_returns_none_for_malformed_boxes(detection) -> None:
    assert get_qwen_detection_box(detection) is None


def test_get_class_name_reads_label_aliases_in_priority_order() -> None:
    assert get_qwen_detection_class_name({"label": "cat"}) == "cat"
    assert get_qwen_detection_class_name({"description": "dog"}) == "dog"
    assert get_qwen_detection_class_name({"class_name": "bird"}) == "bird"
    assert get_qwen_detection_class_name({"class": "fish"}) == "fish"
    assert (
        get_qwen_detection_class_name({"label": "cat", "description": "dog"}) == "cat"
    )
    assert get_qwen_detection_class_name({}) == "unknown"


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


def test_convert_box_clamps_out_of_range_coordinates() -> None:
    result = convert_qwen_detection_to_pixel_xyxy(
        box=[-50.0, 0.0, 1200.0, 1000.0],
        image_height=480,
        image_width=640,
    )

    assert result == pytest.approx([0.0, 0.0, 640.0, 480.0])


# ---------------------------------------------------------------------------
# Full parsing
# ---------------------------------------------------------------------------


def test_parse_response_assembles_detections() -> None:
    parsed_data = [
        {"box_2d": [100, 200, 500, 1000], "label": "cat", "confidence": 0.75},
        {"bbox_2d": [0, 0, 500, 500], "description": "unicorn"},
    ]

    result = parse_qwen_object_detection_response(
        image=_build_image(height=480, width=640),
        parsed_data=parsed_data,
        classes=["cat", "dog"],
        inference_id="inference-id",
    )

    assert np.allclose(result.xyxy, [[64, 96, 320, 480], [0, 0, 320, 240]])
    assert result.class_id.tolist() == [0, -1]
    assert result["class_name"].tolist() == ["cat", "unicorn"]
    # Confidence is hardcoded to 1.0; the model-provided value is ignored.
    assert np.allclose(result.confidence, [1.0, 1.0])
    assert result.mask is None
    assert result.tracker_id is None
    assert result[INFERENCE_ID_KEY].tolist() == ["inference-id"] * 2
    assert result[PREDICTION_TYPE_KEY].tolist() == ["object-detection"] * 2
    assert result[IMAGE_DIMENSIONS_KEY].tolist() == [[480, 640]] * 2
    detection_ids = result[DETECTION_ID_KEY].tolist()
    assert len(set(detection_ids)) == 2
    assert all(detection_id for detection_id in detection_ids)
    assert result[PARENT_ID_KEY].tolist() == ["parent"] * 2
    assert result[PARENT_COORDINATES_KEY].tolist() == [[0, 0]] * 2
    assert result[PARENT_DIMENSIONS_KEY].tolist() == [[480, 640]] * 2
    assert result[ROOT_PARENT_ID_KEY].tolist() == ["parent"] * 2
    assert result[ROOT_PARENT_COORDINATES_KEY].tolist() == [[0, 0]] * 2
    assert result[ROOT_PARENT_DIMENSIONS_KEY].tolist() == [[480, 640]] * 2


def test_parse_response_skips_malformed_entries() -> None:
    parsed_data = [
        "not-a-dict",
        {"label": "cat"},
        {"box_2d": [1, 2, 3], "label": "cat"},
        {"box_2d": [100, 200, 500, 1000], "label": "cat"},
    ]

    result = parse_qwen_object_detection_response(
        image=_build_image(height=480, width=640),
        parsed_data=parsed_data,
        classes=["cat", "dog"],
        inference_id="inference-id",
    )

    assert len(result) == 1
    assert result["class_name"].tolist() == ["cat"]


def test_parse_response_for_empty_list() -> None:
    result = parse_qwen_object_detection_response(
        image=_build_image(height=480, width=640),
        parsed_data=[],
        classes=["cat", "dog"],
        inference_id="inference-id",
    )

    assert len(result) == 0
    # Empty detections must still carry image dimensions in metadata so the
    # numpy serialiser emits real width/height (matching the tensor-native path).
    assert result.metadata[IMAGE_DIMENSIONS_KEY] == [480, 640]
    serialized = serialise_sv_detections(result)
    assert serialized["image"] == {"width": 640, "height": 480}
    assert serialized["predictions"] == []


def test_parse_response_for_detections_wrapper() -> None:
    result = parse_qwen_object_detection_response(
        image=_build_image(height=480, width=640),
        parsed_data={"detections": [{"box_2d": [0, 0, 1000, 1000], "label": "cat"}]},
        classes=["cat"],
        inference_id="inference-id",
    )

    assert len(result) == 1
    assert np.allclose(result.xyxy, [[0, 0, 640, 480]])


def test_parse_response_raises_on_unexpected_shape() -> None:
    with pytest.raises(ValueError):
        parse_qwen_object_detection_response(
            image=_build_image(height=480, width=640),
            parsed_data={"objects": []},
            classes=["cat"],
            inference_id="inference-id",
        )
