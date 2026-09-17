import numpy as np
import pytest

from inference.core.workflows.core_steps.common.serializers import serialise_sv_detections
from inference.core.workflows.core_steps.formatters.vlm_as_detector.gemini_detection_parsing import (
    convert_gemini_detection_to_pixel_xyxy,
    parse_gemini_object_detection_response,
)
from inference.core.workflows.execution_engine.constants import IMAGE_DIMENSIONS_KEY
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def _build_image(height: int, width: int) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((height, width, 3), dtype=np.uint8),
    )


def test_parse_empty_response_carries_image_dimensions() -> None:
    result = parse_gemini_object_detection_response(
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


@pytest.mark.parametrize(
    "box_2d,image_height,image_width,expected",
    [
        ([100, 200, 300, 400], 480, 640, [128.0, 48.0, 256.0, 144.0]),
        ([0, 0, 1000, 1000], 1080, 1920, [0.0, 0.0, 1920.0, 1080.0]),
        ([-50, 200, 1200, 400], 1200, 800, [160.0, -60.0, 320.0, 1440.0]),
    ],
)
def test_native_box_2d_is_converted_to_pixel_xyxy(
    box_2d: list,
    image_height: int,
    image_width: int,
    expected: list,
) -> None:
    result = convert_gemini_detection_to_pixel_xyxy(
        detection={"box_2d": box_2d},
        image_height=image_height,
        image_width=image_width,
    )

    assert result == expected


@pytest.mark.parametrize(
    "detection,image_height,image_width,expected",
    [
        (
            {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
            480,
            640,
            [64.0, 96.0, 192.0, 192.0],
        ),
        (
            {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0},
            1080,
            1920,
            [0.0, 0.0, 1920.0, 1080.0],
        ),
        (
            {"x_min": -0.1, "y_min": 0.2, "x_max": 1.2, "y_max": 0.9},
            1200,
            800,
            [-80.0, 240.0, 960.0, 1080.0],
        ),
    ],
)
def test_legacy_box_is_converted_to_pixel_xyxy(
    detection: dict,
    image_height: int,
    image_width: int,
    expected: list,
) -> None:
    result = convert_gemini_detection_to_pixel_xyxy(
        detection=detection,
        image_height=image_height,
        image_width=image_width,
    )

    assert result == expected


def test_box_2d_takes_precedence() -> None:
    detection = {
        "box_2d": [0, 0, 1000, 1000],
        "x_min": 0.5,
        "y_min": 0.5,
        "x_max": 0.6,
        "y_max": 0.6,
    }

    result = convert_gemini_detection_to_pixel_xyxy(
        detection=detection,
        image_height=1200,
        image_width=800,
    )

    assert result == [0.0, 0.0, 800.0, 1200.0]
