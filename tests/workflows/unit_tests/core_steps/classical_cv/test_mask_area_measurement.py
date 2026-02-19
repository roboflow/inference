import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.classical_cv.mask_area_measurement.v1 import (
    OUTPUT_KEY,
    MaskAreaMeasurementBlockV1,
    MaskAreaMeasurementManifest,
    compute_detection_areas,
)
from inference.core.workflows.execution_engine.constants import (
    AREA_KEY_IN_SV_DETECTIONS,
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
# Tests for compute_detection_areas — with segmentation mask
# ---------------------------------------------------------------------------


def test_compute_detection_areas_with_mask_counts_pixels():
    # given — a filled rectangle mask (50x30 pixels)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:50, 10:60] = 1  # 30 rows x 50 cols = 1500 pixels

    detections = sv.Detections(
        xyxy=np.array([[10, 20, 60, 50]], dtype=np.float32),
        mask=np.array([mask]),
    )

    # when
    areas = compute_detection_areas(detections)

    # then — pixel count of a 50x30 rectangle is exactly 1500
    assert len(areas) == 1
    assert areas[0] == 1500.0


def test_compute_detection_areas_with_circular_mask():
    # given — a filled circle mask
    import cv2 as cv

    mask = np.zeros((200, 200), dtype=np.uint8)
    cv.circle(mask, (100, 100), 50, 1, -1)

    detections = sv.Detections(
        xyxy=np.array([[50, 50, 150, 150]], dtype=np.float32),
        mask=np.array([mask]),
    )

    # when
    areas = compute_detection_areas(detections)

    # then — pixel count of circle r=50 is close to pi*50^2 ~ 7854
    expected_area = np.pi * 50 * 50
    assert len(areas) == 1
    assert abs(areas[0] - expected_area) < 100


def test_compute_detection_areas_with_mask_with_holes():
    # given — a filled rectangle with a hole in the center
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 1  # 80x80 filled region = 6400 pixels
    mask[30:70, 30:70] = 0  # 40x40 hole = 1600 pixels removed

    detections = sv.Detections(
        xyxy=np.array([[10, 10, 90, 90]], dtype=np.float32),
        mask=np.array([mask]),
    )

    # when
    areas = compute_detection_areas(detections)

    # then — 6400 - 1600 = 4800 pixels (hole is excluded)
    assert len(areas) == 1
    assert areas[0] == 4800.0


# ---------------------------------------------------------------------------
# Tests for compute_detection_areas — without mask (bbox fallback)
# ---------------------------------------------------------------------------


def test_compute_detection_areas_without_mask_computes_bbox_area():
    # given — bbox only (no mask)
    detections = sv.Detections(
        xyxy=np.array([[10, 20, 110, 70]], dtype=np.float32),
        mask=None,
    )

    # when
    areas = compute_detection_areas(detections)

    # then — width=100, height=50, area=5000
    assert len(areas) == 1
    assert areas[0] == 5000.0


def test_compute_detection_areas_without_mask_multiple_detections():
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

    # when
    areas = compute_detection_areas(detections)

    # then
    assert areas[0] == 100.0  # 10*10
    assert areas[1] == 600.0  # 20*30
    assert areas[2] == 1000.0  # 50*20


# ---------------------------------------------------------------------------
# Tests for compute_detection_areas — empty mask fallback and small masks
# ---------------------------------------------------------------------------


def test_compute_detection_areas_with_single_pixel_mask():
    # given — mask with a single pixel
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50, 50] = 1  # single pixel

    detections = sv.Detections(
        xyxy=np.array([[0, 0, 40, 25]], dtype=np.float32),
        mask=np.array([mask]),
    )

    # when
    areas = compute_detection_areas(detections)

    # then — single non-zero pixel = area of 1
    assert len(areas) == 1
    assert areas[0] == 1.0


def test_compute_detection_areas_with_empty_mask_falls_back_to_bbox():
    # given — all-zeros mask (no non-zero pixels)
    mask = np.zeros((100, 100), dtype=np.uint8)

    detections = sv.Detections(
        xyxy=np.array([[10, 10, 50, 30]], dtype=np.float32),
        mask=np.array([mask]),
    )

    # when
    areas = compute_detection_areas(detections)

    # then — should fall back to bbox: 40*20 = 800
    assert len(areas) == 1
    assert areas[0] == 800.0


def test_compute_detection_areas_with_thin_line_mask():
    # given — a single-pixel-wide horizontal line
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50, 10:60] = 1  # 50 pixels

    detections = sv.Detections(
        xyxy=np.array([[0, 0, 80, 60]], dtype=np.float32),
        mask=np.array([mask]),
    )

    # when
    areas = compute_detection_areas(detections)

    # then — pixel count = 50 (the line length)
    assert len(areas) == 1
    assert areas[0] == 50.0


# ---------------------------------------------------------------------------
# Tests for MaskAreaMeasurementBlockV1.run()
# ---------------------------------------------------------------------------


def test_run_returns_detections_with_area_in_data():
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

    # then — result is detections with area stored in .data
    assert OUTPUT_KEY in result
    out_detections = result[OUTPUT_KEY]
    assert isinstance(out_detections, sv.Detections)
    assert AREA_KEY_IN_SV_DETECTIONS in out_detections.data
    areas = out_detections.data[AREA_KEY_IN_SV_DETECTIONS]
    assert len(areas) == 3
    assert areas[0] == 5000.0  # 100*50
    assert areas[1] == 600.0  # 20*30
    assert areas[2] == 40000.0  # 200*200


def test_run_with_empty_detections():
    # given
    detections = sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        mask=None,
    )

    # when
    block = MaskAreaMeasurementBlockV1()
    result = block.run(predictions=detections)

    # then
    out_detections = result[OUTPUT_KEY]
    assert isinstance(out_detections, sv.Detections)
    assert len(out_detections.data[AREA_KEY_IN_SV_DETECTIONS]) == 0


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
    areas = result[OUTPUT_KEY].data[AREA_KEY_IN_SV_DETECTIONS]
    assert len(areas) == 2
    # first detection: mask pixel count = 30*60 = 1800
    assert areas[0] == 1800.0
    # second detection: empty mask falls back to bbox 50*50=2500
    assert areas[1] == 2500.0


def test_run_with_single_detection_with_mask():
    # given — 151x151 filled rect (cv2.rectangle includes both endpoints)
    import cv2 as cv

    mask = np.zeros((200, 200), dtype=np.uint8)
    cv.rectangle(mask, (25, 25), (175, 175), 1, -1)

    detections = sv.Detections(
        xyxy=np.array([[25, 25, 175, 175]], dtype=np.float32),
        mask=np.array([mask]),
    )

    # when
    block = MaskAreaMeasurementBlockV1()
    result = block.run(predictions=detections)

    # then — cv2.rectangle includes both endpoints: 151x151 = 22801
    areas = result[OUTPUT_KEY].data[AREA_KEY_IN_SV_DETECTIONS]
    assert len(areas) == 1
    expected_area = 151.0 * 151.0  # cv2.rectangle is endpoint-inclusive
    assert areas[0] == expected_area


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
    for area in result[OUTPUT_KEY].data[AREA_KEY_IN_SV_DETECTIONS]:
        assert isinstance(area, (float, np.floating))
