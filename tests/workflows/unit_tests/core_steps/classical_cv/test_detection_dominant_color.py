import re

import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.detection_dominant_color.v1 import (
    DetectionDominantColorBlockV1,
    DetectionDominantColorManifest,
    _process_single_image,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OutputDefinition,
    WorkflowImageData,
)


HEX_PATTERN = re.compile(r"^#[0-9a-f]{6}$")


# ──────────────────────────────────────────────────────────
# Manifest validation tests
# ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("image_field_alias", ["image", "images"])
def test_manifest_valid_with_defaults(image_field_alias: str) -> None:
    data = {
        "type": "roboflow_core/detection_dominant_color@v1",
        "name": "det_color1",
        image_field_alias: "$inputs.image",
        "predictions": "$steps.model.predictions",
    }

    result = DetectionDominantColorManifest.model_validate(data)

    assert result.color_clusters == 4
    assert result.max_iterations == 100
    assert result.target_size == 100


@pytest.mark.parametrize("predictions_field_alias", ["predictions", "detections"])
def test_manifest_valid_with_custom_params(predictions_field_alias: str) -> None:
    data = {
        "type": "roboflow_core/detection_dominant_color@v1",
        "name": "det_color2",
        "image": "$inputs.image",
        predictions_field_alias: "$steps.model.predictions",
        "color_clusters": 8,
        "max_iterations": 200,
        "target_size": 50,
    }

    result = DetectionDominantColorManifest.model_validate(data)

    assert result.color_clusters == 8
    assert result.max_iterations == 200
    assert result.target_size == 50


def test_manifest_describe_outputs() -> None:
    outputs = DetectionDominantColorManifest.describe_outputs()

    assert len(outputs) == 1
    assert outputs[0].name == "dominant_color_hex"


def test_manifest_get_output_dimensionality_offset() -> None:
    assert DetectionDominantColorManifest.get_output_dimensionality_offset() == 1


def test_manifest_get_parameters_accepting_batches() -> None:
    assert set(
        DetectionDominantColorManifest.get_parameters_accepting_batches()
    ) == {"image", "predictions"}


@pytest.mark.parametrize("invalid_clusters", [0, 11, -1])
def test_manifest_invalid_color_clusters(invalid_clusters: int) -> None:
    data = {
        "type": "roboflow_core/detection_dominant_color@v1",
        "name": "det_color_bad",
        "image": "$inputs.image",
        "predictions": "$steps.model.predictions",
        "color_clusters": invalid_clusters,
    }

    with pytest.raises(ValidationError):
        DetectionDominantColorManifest.model_validate(data)


@pytest.mark.parametrize("invalid_iterations", [0, 501, -5])
def test_manifest_invalid_max_iterations(invalid_iterations: int) -> None:
    data = {
        "type": "roboflow_core/detection_dominant_color@v1",
        "name": "det_color_bad",
        "image": "$inputs.image",
        "predictions": "$steps.model.predictions",
        "max_iterations": invalid_iterations,
    }

    with pytest.raises(ValidationError):
        DetectionDominantColorManifest.model_validate(data)


@pytest.mark.parametrize("invalid_target_size", [0, 251, -1])
def test_manifest_invalid_target_size(invalid_target_size: int) -> None:
    data = {
        "type": "roboflow_core/detection_dominant_color@v1",
        "name": "det_color_bad",
        "image": "$inputs.image",
        "predictions": "$steps.model.predictions",
        "target_size": invalid_target_size,
    }

    with pytest.raises(ValidationError):
        DetectionDominantColorManifest.model_validate(data)


# ──────────────────────────────────────────────────────────
# Helper to create WorkflowImageData
# ──────────────────────────────────────────────────────────


def _make_image(numpy_image: np.ndarray) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="test"),
        numpy_image=numpy_image,
    )


# ──────────────────────────────────────────────────────────
# Block.run() tests — object detection (bbox only, no mask)
# ──────────────────────────────────────────────────────────


def test_run_single_solid_red_bbox() -> None:
    # Solid red image (BGR: 0, 0, 255)
    image_np = np.zeros((100, 100, 3), dtype=np.uint8)
    image_np[:, :, 2] = 255

    detections = sv.Detections(
        xyxy=np.array([[0, 0, 100, 100]], dtype=np.float32),
    )

    block = DetectionDominantColorBlockV1()
    result = block.run(
        image=[_make_image(image_np)],
        predictions=[detections],
        color_clusters=4,
        max_iterations=100,
        target_size=100,
    )

    assert len(result) == 1  # one image
    assert len(result[0]) == 1  # one detection
    assert result[0][0]["dominant_color_hex"] == "#ff0000"


def test_run_single_solid_green_bbox() -> None:
    # Solid green image (BGR: 0, 255, 0)
    image_np = np.zeros((100, 100, 3), dtype=np.uint8)
    image_np[:, :, 1] = 255

    detections = sv.Detections(
        xyxy=np.array([[0, 0, 100, 100]], dtype=np.float32),
    )

    block = DetectionDominantColorBlockV1()
    result = block.run(
        image=[_make_image(image_np)],
        predictions=[detections],
        color_clusters=4,
        max_iterations=100,
        target_size=100,
    )

    assert len(result) == 1
    assert result[0][0]["dominant_color_hex"] == "#00ff00"


def test_run_multi_detection_two_colored_halves() -> None:
    # Left half red (BGR 0,0,255), right half blue (BGR 255,0,0)
    image_np = np.zeros((100, 200, 3), dtype=np.uint8)
    image_np[:, :100, 2] = 255   # left = red
    image_np[:, 100:, 0] = 255   # right = blue

    detections = sv.Detections(
        xyxy=np.array(
            [
                [0, 0, 100, 100],    # left half (red)
                [100, 0, 200, 100],  # right half (blue)
            ],
            dtype=np.float32,
        ),
    )

    block = DetectionDominantColorBlockV1()
    result = block.run(
        image=[_make_image(image_np)],
        predictions=[detections],
        color_clusters=4,
        max_iterations=100,
        target_size=100,
    )

    assert len(result) == 1
    assert len(result[0]) == 2
    assert result[0][0]["dominant_color_hex"] == "#ff0000"
    assert result[0][1]["dominant_color_hex"] == "#0000ff"


# ──────────────────────────────────────────────────────────
# Block.run() tests — instance segmentation (with mask)
# ──────────────────────────────────────────────────────────


def test_run_with_mask_selects_masked_region_only() -> None:
    # Image: left half red (BGR 0,0,255), right half blue (BGR 255,0,0)
    image_np = np.zeros((100, 100, 3), dtype=np.uint8)
    image_np[:, :50, 2] = 255   # left = red
    image_np[:, 50:, 0] = 255   # right = blue

    # Bbox covers whole image, mask covers only the left half (red)
    mask = np.zeros((1, 100, 100), dtype=bool)
    mask[0, :, :50] = True

    detections = sv.Detections(
        xyxy=np.array([[0, 0, 100, 100]], dtype=np.float32),
        mask=mask,
    )

    block = DetectionDominantColorBlockV1()
    result = block.run(
        image=[_make_image(image_np)],
        predictions=[detections],
        color_clusters=4,
        max_iterations=100,
        target_size=100,
    )

    assert len(result) == 1
    assert len(result[0]) == 1
    assert result[0][0]["dominant_color_hex"] == "#ff0000"


def test_run_with_mask_selects_blue_half() -> None:
    # Image: left half red (BGR 0,0,255), right half blue (BGR 255,0,0)
    image_np = np.zeros((100, 100, 3), dtype=np.uint8)
    image_np[:, :50, 2] = 255   # left = red
    image_np[:, 50:, 0] = 255   # right = blue

    # Bbox covers whole image, mask covers only the right half (blue)
    mask = np.zeros((1, 100, 100), dtype=bool)
    mask[0, :, 50:] = True

    detections = sv.Detections(
        xyxy=np.array([[0, 0, 100, 100]], dtype=np.float32),
        mask=mask,
    )

    block = DetectionDominantColorBlockV1()
    result = block.run(
        image=[_make_image(image_np)],
        predictions=[detections],
        color_clusters=4,
        max_iterations=100,
        target_size=100,
    )

    assert len(result) == 1
    assert result[0][0]["dominant_color_hex"] == "#0000ff"


# ──────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────


def test_run_empty_detections_returns_empty_list() -> None:
    image_np = np.zeros((100, 100, 3), dtype=np.uint8)

    detections = sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
    )

    block = DetectionDominantColorBlockV1()
    result = block.run(
        image=[_make_image(image_np)],
        predictions=[detections],
        color_clusters=4,
        max_iterations=100,
        target_size=100,
    )

    assert len(result) == 1
    assert result[0] == []


def test_run_very_small_detection_1x1_pixel() -> None:
    image_np = np.zeros((10, 10, 3), dtype=np.uint8)
    image_np[5, 5] = [128, 64, 32]  # BGR

    detections = sv.Detections(
        xyxy=np.array([[5, 5, 6, 6]], dtype=np.float32),
    )

    block = DetectionDominantColorBlockV1()
    result = block.run(
        image=[_make_image(image_np)],
        predictions=[detections],
        color_clusters=4,
        max_iterations=100,
        target_size=100,
    )

    assert len(result) == 1
    assert len(result[0]) == 1
    hex_val = result[0][0]["dominant_color_hex"]
    assert HEX_PATTERN.match(hex_val), f"Expected valid hex, got {hex_val}"
    # BGR (128, 64, 32) → RGB (32, 64, 128) → #204080
    assert hex_val == "#204080"


def test_run_zero_size_bbox_returns_none() -> None:
    image_np = np.zeros((100, 100, 3), dtype=np.uint8)

    # x1 == x2, so region has zero width
    detections = sv.Detections(
        xyxy=np.array([[50, 0, 50, 100]], dtype=np.float32),
    )

    block = DetectionDominantColorBlockV1()
    result = block.run(
        image=[_make_image(image_np)],
        predictions=[detections],
        color_clusters=4,
        max_iterations=100,
        target_size=100,
    )

    assert len(result) == 1
    assert len(result[0]) == 1
    assert result[0][0]["dominant_color_hex"] is None


def test_run_zero_height_bbox_returns_none() -> None:
    image_np = np.zeros((100, 100, 3), dtype=np.uint8)

    # y1 == y2, so region has zero height
    detections = sv.Detections(
        xyxy=np.array([[0, 50, 100, 50]], dtype=np.float32),
    )

    block = DetectionDominantColorBlockV1()
    result = block.run(
        image=[_make_image(image_np)],
        predictions=[detections],
        color_clusters=4,
        max_iterations=100,
        target_size=100,
    )

    assert len(result) == 1
    assert len(result[0]) == 1
    assert result[0][0]["dominant_color_hex"] is None


# ──────────────────────────────────────────────────────────
# Hex format validation
# ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bgr_color",
    [
        (0, 0, 255),      # red
        (0, 255, 0),      # green
        (255, 0, 0),      # blue
        (255, 255, 255),  # white
        (0, 0, 0),        # black
        (128, 128, 128),  # gray
        (0, 165, 255),    # orange (BGR)
    ],
)
def test_hex_output_format_matches_pattern(bgr_color: tuple) -> None:
    image_np = np.full((50, 50, 3), bgr_color, dtype=np.uint8)

    detections = sv.Detections(
        xyxy=np.array([[0, 0, 50, 50]], dtype=np.float32),
    )

    block = DetectionDominantColorBlockV1()
    result = block.run(
        image=[_make_image(image_np)],
        predictions=[detections],
        color_clusters=4,
        max_iterations=100,
        target_size=100,
    )

    hex_val = result[0][0]["dominant_color_hex"]
    assert HEX_PATTERN.match(hex_val), f"Expected hex format #rrggbb, got {hex_val}"


# ──────────────────────────────────────────────────────────
# _process_single_image unit tests
# ──────────────────────────────────────────────────────────


def test_process_single_image_solid_white() -> None:
    image_np = np.full((50, 50, 3), 255, dtype=np.uint8)
    image_data = _make_image(image_np)

    detections = sv.Detections(
        xyxy=np.array([[0, 0, 50, 50]], dtype=np.float32),
    )

    result = _process_single_image(
        image=image_data,
        detections=detections,
        color_clusters=4,
        max_iterations=100,
        target_size=100,
    )

    assert len(result) == 1
    assert result[0]["dominant_color_hex"] == "#ffffff"


def test_process_single_image_solid_black() -> None:
    image_np = np.zeros((50, 50, 3), dtype=np.uint8)
    image_data = _make_image(image_np)

    detections = sv.Detections(
        xyxy=np.array([[0, 0, 50, 50]], dtype=np.float32),
    )

    result = _process_single_image(
        image=image_data,
        detections=detections,
        color_clusters=4,
        max_iterations=100,
        target_size=100,
    )

    assert len(result) == 1
    assert result[0]["dominant_color_hex"] == "#000000"


def test_process_single_image_empty_detections() -> None:
    image_np = np.zeros((50, 50, 3), dtype=np.uint8)
    image_data = _make_image(image_np)

    detections = sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
    )

    result = _process_single_image(
        image=image_data,
        detections=detections,
        color_clusters=4,
        max_iterations=100,
        target_size=100,
    )

    assert result == []


# ──────────────────────────────────────────────────────────
# Multiple images in batch
# ──────────────────────────────────────────────────────────


def test_run_batch_of_two_images() -> None:
    # Image 1: solid red
    img1 = np.zeros((50, 50, 3), dtype=np.uint8)
    img1[:, :, 2] = 255

    # Image 2: solid blue
    img2 = np.zeros((50, 50, 3), dtype=np.uint8)
    img2[:, :, 0] = 255

    det1 = sv.Detections(xyxy=np.array([[0, 0, 50, 50]], dtype=np.float32))
    det2 = sv.Detections(xyxy=np.array([[0, 0, 50, 50]], dtype=np.float32))

    block = DetectionDominantColorBlockV1()
    result = block.run(
        image=[_make_image(img1), _make_image(img2)],
        predictions=[det1, det2],
        color_clusters=4,
        max_iterations=100,
        target_size=100,
    )

    assert len(result) == 2
    assert result[0][0]["dominant_color_hex"] == "#ff0000"
    assert result[1][0]["dominant_color_hex"] == "#0000ff"


# ──────────────────────────────────────────────────────────
# Mask with empty region (all-False mask)
# ──────────────────────────────────────────────────────────


def test_run_mask_all_false_returns_none() -> None:
    image_np = np.full((50, 50, 3), 128, dtype=np.uint8)

    mask = np.zeros((1, 50, 50), dtype=bool)  # no pixels selected

    detections = sv.Detections(
        xyxy=np.array([[0, 0, 50, 50]], dtype=np.float32),
        mask=mask,
    )

    block = DetectionDominantColorBlockV1()
    result = block.run(
        image=[_make_image(image_np)],
        predictions=[detections],
        color_clusters=4,
        max_iterations=100,
        target_size=100,
    )

    assert len(result) == 1
    assert len(result[0]) == 1
    assert result[0][0]["dominant_color_hex"] is None
