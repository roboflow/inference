import numpy as np
import pytest

from inference.core.workflows.core_steps.formatters.vlm_as_detector.openai_detection_parsing import (
    convert_openai_detection_to_pixel_xyxy,
    parse_openai_object_detection_response,
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


@pytest.mark.parametrize(
    "box_2d,image_height,image_width,expected",
    [
        pytest.param(
            [10, 20, 100, 200],
            480,
            640,
            [10.0, 20.0, 100.0, 200.0],
            id="within-max-edge-passes-through",
        ),
        pytest.param(
            [100, 50, 500, 400],
            2048,
            4096,
            [200.0, 100.0, 1000.0, 800.0],
            id="landscape-downscale-2x",
        ),
        pytest.param(
            [256, 1024, 512, 2048],
            4000,
            1000,
            [500.0, 2000.0, 1000.0, 4000.0],
            id="portrait-downscale",
        ),
        pytest.param(
            [0, 0, 2048, 1365],
            2000,
            3000,
            [0.0, 0.0, 3000.0, 2000.0],
            id="uploaded-extent-maps-to-original-extent",
        ),
    ],
)
def test_box_2d_is_converted_to_pixel_xyxy(
    box_2d: list,
    image_height: int,
    image_width: int,
    expected: list,
) -> None:
    result = convert_openai_detection_to_pixel_xyxy(
        detection={"box_2d": box_2d},
        image_height=image_height,
        image_width=image_width,
    )

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "box_2d,image_height,image_width,expected",
    [
        pytest.param(
            [-50, -10, 100, 200],
            480,
            640,
            [0.0, 0.0, 100.0, 200.0],
            id="negative-coordinates-clamp-to-zero",
        ),
        pytest.param(
            [10, 20, 700, 500],
            480,
            640,
            [10.0, 20.0, 640.0, 480.0],
            id="coordinates-beyond-image-clamp-to-edges",
        ),
        pytest.param(
            [-100, 0, 3000, 3000],
            2048,
            4096,
            [0.0, 0.0, 4096.0, 2048.0],
            id="clamp-against-uploaded-dims-before-rescaling",
        ),
    ],
)
def test_out_of_bounds_coordinates_are_clamped(
    box_2d: list,
    image_height: int,
    image_width: int,
    expected: list,
) -> None:
    result = convert_openai_detection_to_pixel_xyxy(
        detection={"box_2d": box_2d},
        image_height=image_height,
        image_width=image_width,
    )

    assert result == expected


def test_non_integer_scale_factors_are_applied_exactly() -> None:
    result = convert_openai_detection_to_pixel_xyxy(
        detection={"box_2d": [1024, 683, 1536, 1024]},
        image_height=2000,
        image_width=3000,
    )

    assert result == pytest.approx(
        [1500.0, 683 * 2000 / 1365, 2250.0, 1024 * 2000 / 1365]
    )


def _build_image(height: int, width: int) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((height, width, 3), dtype=np.uint8),
    )


@pytest.mark.parametrize(
    "parsed_data",
    [
        pytest.param({"detections": []}, id="dict"),
        pytest.param("not-a-list", id="string"),
        pytest.param(None, id="none"),
    ],
)
def test_parse_openai_object_detection_response_raises_on_non_list(
    parsed_data,
) -> None:
    with pytest.raises(ValueError):
        parse_openai_object_detection_response(
            image=_build_image(height=480, width=640),
            parsed_data=parsed_data,
            classes=["cat", "dog"],
            inference_id="inference-id",
        )


def test_parse_openai_object_detection_response_for_empty_list() -> None:
    result = parse_openai_object_detection_response(
        image=_build_image(height=480, width=640),
        parsed_data=[],
        classes=["cat", "dog"],
        inference_id="inference-id",
    )

    assert len(result) == 0


def test_parse_openai_object_detection_response_assembles_detections() -> None:
    parsed_data = [
        {"box_2d": [10, 20, 100, 200], "label": "cat", "confidence": 0.75},
        {"box_2d": [0, 0, 50, 50], "label": "unicorn"},
    ]

    result = parse_openai_object_detection_response(
        image=_build_image(height=480, width=640),
        parsed_data=parsed_data,
        classes=["cat", "dog"],
        inference_id="inference-id",
    )

    assert np.allclose(result.xyxy, [[10, 20, 100, 200], [0, 0, 50, 50]])
    assert result.class_id.tolist() == [0, -1]
    assert result["class_name"].tolist() == ["cat", "unicorn"]
    assert np.allclose(result.confidence, [0.75, 1.0])
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
