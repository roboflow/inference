import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.classical_cv.mask_area_measurement.v1 import (
    OUTPUT_KEY,
    MaskAreaMeasurementBlockV1,
    MaskAreaMeasurementManifest,
    get_detection_area,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition


# ---------------------------------------------------------------------------
# Tests for MaskAreaMeasurementManifest
# ---------------------------------------------------------------------------


def test_manifest_describe_outputs_returns_areas_output():
    # when
    outputs = MaskAreaMeasurementManifest.describe_outputs()

    # then
    assert len(outputs) == 1
    assert isinstance(outputs[0], OutputDefinition)
    assert outputs[0].name == OUTPUT_KEY


def test_manifest_valid_creation():
    # when
    manifest = MaskAreaMeasurementManifest(
        type="roboflow_core/mask_area_measurement@v1",
        name="mask_area_measurement_1",
        predictions="$steps.model.predictions",
    )

    # then
    assert manifest.type == "roboflow_core/mask_area_measurement@v1"
    assert manifest.predictions == "$steps.model.predictions"


def test_manifest_get_execution_engine_compatibility():
    # when
    compat = MaskAreaMeasurementManifest.get_execution_engine_compatibility()

    # then
    assert compat == ">=1.3.0,<2.0.0"


# ---------------------------------------------------------------------------
# Tests for get_detection_area — with segmentation mask
# ---------------------------------------------------------------------------


def test_get_detection_area_with_mask_computes_contour_area():
    # given — a filled rectangle mask (50x30 pixels)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:50, 10:60] = 1  # 30 rows x 50 cols

    detections = sv.Detections(
        xyxy=np.array([[10, 20, 60, 50]], dtype=np.float32),
        mask=np.array([mask]),
    )

    # when
    area = get_detection_area(detections, 0)

    # then — contour area of a 50x30 rectangle should be close to 1500
    # (cv2.contourArea may differ by a small amount due to contour tracing)
    assert area > 0
    assert abs(area - 1500.0) < 100  # contour tracing underestimates slightly


def test_get_detection_area_with_circular_mask():
    # given — a filled circle mask
    import cv2 as cv

    mask = np.zeros((200, 200), dtype=np.uint8)
    cv.circle(mask, (100, 100), 50, 1, -1)

    detections = sv.Detections(
        xyxy=np.array([[50, 50, 150, 150]], dtype=np.float32),
        mask=np.array([mask]),
    )

    # when
    area = get_detection_area(detections, 0)

    # then — area of circle r=50: pi*50^2 ~ 7854
    expected_area = np.pi * 50 * 50
    assert abs(area - expected_area) < 200  # contour tracing has some error


# ---------------------------------------------------------------------------
# Tests for get_detection_area — without mask (bbox fallback)
# ---------------------------------------------------------------------------


def test_get_detection_area_without_mask_computes_bbox_area():
    # given — bbox only (no mask)
    detections = sv.Detections(
        xyxy=np.array([[10, 20, 110, 70]], dtype=np.float32),
        mask=None,
    )

    # when
    area = get_detection_area(detections, 0)

    # then — width=100, height=50, area=5000
    assert area == 5000.0


def test_get_detection_area_without_mask_multiple_detections():
    # given
    detections = sv.Detections(
        xyxy=np.array(
            [
                [0, 0, 10, 10],
                [0, 0, 20, 30],
                [5, 5, 55, 25],
            ],
            dtype=np.float32,
        ),
        mask=None,
    )

    # when / then
    assert get_detection_area(detections, 0) == 100.0  # 10*10
    assert get_detection_area(detections, 1) == 600.0  # 20*30
    assert get_detection_area(detections, 2) == 1000.0  # 50*20


# ---------------------------------------------------------------------------
# Tests for get_detection_area — zero-area contour fallback
# ---------------------------------------------------------------------------


def test_get_detection_area_with_zero_area_mask_falls_back_to_bbox():
    # given — mask with a single pixel (contour area = 0)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50, 50] = 1  # single pixel — contour area is 0

    detections = sv.Detections(
        xyxy=np.array([[0, 0, 40, 25]], dtype=np.float32),
        mask=np.array([mask]),
    )

    # when
    area = get_detection_area(detections, 0)

    # then — should fall back to bbox: 40*25 = 1000
    assert area == 1000.0


def test_get_detection_area_with_empty_mask_falls_back_to_bbox():
    # given — all-zeros mask (no contours at all)
    mask = np.zeros((100, 100), dtype=np.uint8)

    detections = sv.Detections(
        xyxy=np.array([[10, 10, 50, 30]], dtype=np.float32),
        mask=np.array([mask]),
    )

    # when
    area = get_detection_area(detections, 0)

    # then — should fall back to bbox: 40*20 = 800
    assert area == 800.0


def test_get_detection_area_with_thin_line_mask_falls_back_to_bbox():
    # given — a single-pixel-wide horizontal line (contour area = 0)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50, 10:60] = 1  # horizontal line

    detections = sv.Detections(
        xyxy=np.array([[0, 0, 80, 60]], dtype=np.float32),
        mask=np.array([mask]),
    )

    # when
    area = get_detection_area(detections, 0)

    # then — line contour has 0 area, falls back to bbox: 80*60 = 4800
    assert area == 4800.0


# ---------------------------------------------------------------------------
# Tests for MaskAreaMeasurementBlockV1.run()
# ---------------------------------------------------------------------------


def test_run_with_multiple_detections_returns_list_of_areas():
    # given
    detections = sv.Detections(
        xyxy=np.array(
            [
                [0, 0, 100, 50],
                [10, 10, 30, 40],
                [0, 0, 200, 200],
            ],
            dtype=np.float32,
        ),
        mask=None,
    )

    # when
    block = MaskAreaMeasurementBlockV1()
    result = block.run(predictions=detections)

    # then
    assert OUTPUT_KEY in result
    areas = result[OUTPUT_KEY]
    assert len(areas) == 3
    assert areas[0] == 5000.0  # 100*50
    assert areas[1] == 600.0  # 20*30
    assert areas[2] == 40000.0  # 200*200


def test_run_with_empty_detections_returns_empty_list():
    # given
    detections = sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        mask=None,
    )

    # when
    block = MaskAreaMeasurementBlockV1()
    result = block.run(predictions=detections)

    # then
    assert result == {OUTPUT_KEY: []}


def test_run_with_mixed_mask_and_no_mask_detections():
    # given — two detections, one with mask, one effectively falling back
    mask_0 = np.zeros((100, 200), dtype=np.uint8)
    mask_0[10:40, 20:80] = 1  # 30x60 filled rectangle

    mask_1 = np.zeros((100, 200), dtype=np.uint8)  # empty mask -> bbox fallback

    detections = sv.Detections(
        xyxy=np.array(
            [
                [20, 10, 80, 40],
                [0, 0, 50, 50],
            ],
            dtype=np.float32,
        ),
        mask=np.array([mask_0, mask_1]),
    )

    # when
    block = MaskAreaMeasurementBlockV1()
    result = block.run(predictions=detections)

    # then
    areas = result[OUTPUT_KEY]
    assert len(areas) == 2
    # first detection: mask contour area (~1800, close to 30*60)
    assert areas[0] > 0
    assert abs(areas[0] - 1800.0) < 100
    # second detection: empty mask falls back to bbox 50*50=2500
    assert areas[1] == 2500.0


def test_run_with_single_detection_with_mask():
    # given
    import cv2 as cv

    mask = np.zeros((200, 200), dtype=np.uint8)
    cv.rectangle(mask, (25, 25), (175, 175), 1, -1)  # 150x150 filled rect

    detections = sv.Detections(
        xyxy=np.array([[25, 25, 175, 175]], dtype=np.float32),
        mask=np.array([mask]),
    )

    # when
    block = MaskAreaMeasurementBlockV1()
    result = block.run(predictions=detections)

    # then
    areas = result[OUTPUT_KEY]
    assert len(areas) == 1
    expected_area = 150.0 * 150.0
    assert abs(areas[0] - expected_area) < 50


def test_run_result_areas_are_floats():
    # given
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
        mask=None,
    )

    # when
    block = MaskAreaMeasurementBlockV1()
    result = block.run(predictions=detections)

    # then
    for area in result[OUTPUT_KEY]:
        assert isinstance(area, float)
