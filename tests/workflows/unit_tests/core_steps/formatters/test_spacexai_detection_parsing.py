import numpy as np
import pytest

from inference.core.workflows.core_steps.common.serializers import serialise_sv_detections
from inference.core.workflows.core_steps.formatters.vlm_as_detector.spacexai_detection_parsing import (
    convert_spacexai_detection_to_pixel_xyxy,
    extract_spacexai_detection_entries,
    parse_spacexai_object_detection_response,
)
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


# ---------------------------------------------------------------------------
# extract_spacexai_detection_entries
# ---------------------------------------------------------------------------


def test_extract_entries_from_list() -> None:
    parsed_data = [{"box_2d": [10, 20, 50, 80], "label": "cat"}]
    result = extract_spacexai_detection_entries(parsed_data=parsed_data)
    assert result == parsed_data


def test_extract_entries_from_detections_wrapper() -> None:
    parsed_data = {"detections": [{"box_2d": [10, 20, 50, 80], "label": "cat"}]}
    result = extract_spacexai_detection_entries(parsed_data=parsed_data)
    assert result == [{"box_2d": [10, 20, 50, 80], "label": "cat"}]


def test_extract_entries_from_empty_list() -> None:
    assert extract_spacexai_detection_entries(parsed_data=[]) == []


def test_extract_entries_from_empty_detections_wrapper() -> None:
    assert extract_spacexai_detection_entries(parsed_data={"detections": []}) == []


@pytest.mark.parametrize(
    "parsed_data",
    [
        pytest.param("not-a-list-or-dict", id="string"),
        pytest.param(None, id="none"),
        pytest.param({"unrelated": "object"}, id="dict-without-detections-key"),
        pytest.param(42, id="int"),
    ],
)
def test_extract_entries_raises_on_unexpected_format(parsed_data) -> None:
    with pytest.raises(ValueError, match="Unexpected SpaceXAI"):
        extract_spacexai_detection_entries(parsed_data=parsed_data)


# ---------------------------------------------------------------------------
# convert_spacexai_detection_to_pixel_xyxy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "box_2d,image_height,image_width,expected",
    [
        pytest.param(
            [10, 20, 50, 80],
            480,
            640,
            [64.0, 96.0, 320.0, 384.0],
            id="basic-scaling",
        ),
        pytest.param(
            [0, 0, 100, 100],
            1080,
            1920,
            [0.0, 0.0, 1920.0, 1080.0],
            id="full-extent",
        ),
        pytest.param(
            [25, 25, 75, 75],
            1000,
            1000,
            [250.0, 250.0, 750.0, 750.0],
            id="square-image",
        ),
    ],
)
def test_box_2d_is_converted_to_pixel_xyxy(
    box_2d: list,
    image_height: int,
    image_width: int,
    expected: list,
) -> None:
    result = convert_spacexai_detection_to_pixel_xyxy(
        detection={"box_2d": box_2d},
        image_height=image_height,
        image_width=image_width,
    )
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "box_2d,image_height,image_width,expected",
    [
        pytest.param(
            [-10, -20, 50, 80],
            480,
            640,
            [0.0, 0.0, 320.0, 384.0],
            id="negative-coordinates-clamp-to-zero",
        ),
        pytest.param(
            [10, 20, 150, 120],
            480,
            640,
            [64.0, 96.0, 640.0, 480.0],
            id="coordinates-beyond-100-clamp-to-100",
        ),
        pytest.param(
            [-50, -50, 200, 200],
            1000,
            1000,
            [0.0, 0.0, 1000.0, 1000.0],
            id="both-sides-clamped",
        ),
    ],
)
def test_out_of_bounds_coordinates_are_clamped(
    box_2d: list,
    image_height: int,
    image_width: int,
    expected: list,
) -> None:
    result = convert_spacexai_detection_to_pixel_xyxy(
        detection={"box_2d": box_2d},
        image_height=image_height,
        image_width=image_width,
    )
    assert result == pytest.approx(expected)


def test_non_integer_scale_factors_are_applied_exactly() -> None:
    result = convert_spacexai_detection_to_pixel_xyxy(
        detection={"box_2d": [33, 67, 89, 12]},
        image_height=1080,
        image_width=1920,
    )
    assert result == pytest.approx(
        [33 / 100 * 1920, 67 / 100 * 1080, 89 / 100 * 1920, 12 / 100 * 1080]
    )


# ---------------------------------------------------------------------------
# parse_spacexai_object_detection_response
# ---------------------------------------------------------------------------


def _build_image(height: int, width: int) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((height, width, 3), dtype=np.uint8),
    )


@pytest.mark.parametrize(
    "parsed_data",
    [
        pytest.param({"unrelated": "object"}, id="dict-without-detections"),
        pytest.param("not-a-list", id="string"),
        pytest.param(None, id="none"),
    ],
)
def test_parse_raises_on_unexpected_format(parsed_data) -> None:
    with pytest.raises(ValueError, match="Unexpected SpaceXAI"):
        parse_spacexai_object_detection_response(
            image=_build_image(height=480, width=640),
            parsed_data=parsed_data,
            classes=["cat", "dog"],
            inference_id="inference-id",
        )


def test_parse_empty_list_carries_image_dimensions() -> None:
    result = parse_spacexai_object_detection_response(
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


def test_parse_empty_detections_wrapper_carries_image_dimensions() -> None:
    result = parse_spacexai_object_detection_response(
        image=_build_image(height=480, width=640),
        parsed_data={"detections": []},
        classes=["cat", "dog"],
        inference_id="inference-id",
    )

    assert len(result) == 0
    assert result.metadata[IMAGE_DIMENSIONS_KEY] == [480, 640]
    serialized = serialise_sv_detections(result)
    assert serialized["image"] == {"width": 640, "height": 480}


def test_parse_assembles_detections() -> None:
    parsed_data = [
        {"box_2d": [10, 20, 50, 80], "label": "cat", "confidence": 0.9},
        {"box_2d": [0, 0, 100, 100], "label": "unicorn"},
    ]

    result = parse_spacexai_object_detection_response(
        image=_build_image(height=480, width=640),
        parsed_data=parsed_data,
        classes=["cat", "dog"],
        inference_id="inference-id",
    )

    assert np.allclose(
        result.xyxy,
        [
            [10 / 100 * 640, 20 / 100 * 480, 50 / 100 * 640, 80 / 100 * 480],
            [0.0, 0.0, 640.0, 480.0],
        ],
    )
    assert result.class_id.tolist() == [0, -1]
    assert result["class_name"].tolist() == ["cat", "unicorn"]
    assert np.allclose(result.confidence, [0.9, 1.0])
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


def test_parse_clamps_out_of_bounds_coordinates() -> None:
    parsed_data = [
        {"box_2d": [-10, -20, 150, 120], "label": "cat"},
    ]

    result = parse_spacexai_object_detection_response(
        image=_build_image(height=480, width=640),
        parsed_data=parsed_data,
        classes=["cat", "dog"],
        inference_id="inference-id",
    )

    assert np.allclose(result.xyxy, [[0.0, 0.0, 640.0, 480.0]])
    assert result["class_name"].tolist() == ["cat"]


def test_parse_accepts_detections_wrapper() -> None:
    parsed_data = {
        "detections": [
            {"box_2d": [10, 20, 50, 80], "label": "cat", "confidence": 0.75},
        ]
    }

    result = parse_spacexai_object_detection_response(
        image=_build_image(height=480, width=640),
        parsed_data=parsed_data,
        classes=["cat", "dog"],
        inference_id="inference-id",
    )

    assert len(result) == 1
    assert result["class_name"].tolist() == ["cat"]
    assert np.allclose(result.confidence, [0.75])
