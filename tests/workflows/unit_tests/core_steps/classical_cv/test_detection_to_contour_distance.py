import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.detection_to_contour_distance.v1 import (
    DetectionToContourDistanceBlockV1,
    DetectionToContourDistanceManifest,
)


# ---------------------------------------------------------------------------
# Helper: build a square contour in the format returned by cv2.findContours
# ---------------------------------------------------------------------------


def _square_contour(x_min, y_min, x_max, y_max):
    """Return a contour as (N, 1, 2) int32 array – same shape as cv2.findContours."""
    return np.array(
        [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]],
        dtype=np.int32,
    ).reshape(-1, 1, 2)


# ===================================================================
# 1. Manifest validation
# ===================================================================


def test_manifest_valid_with_minimal_fields() -> None:
    # given
    data = {
        "type": "roboflow_core/detection_to_contour_distance@v1",
        "name": "distance_step",
        "predictions": "$steps.model.predictions",
        "contours": "$steps.contours.contours",
    }

    # when
    result = DetectionToContourDistanceManifest.model_validate(data)

    # then
    assert result.type == "roboflow_core/detection_to_contour_distance@v1"
    assert result.distance_threshold == 50  # default


def test_manifest_valid_with_all_fields() -> None:
    # given
    data = {
        "type": "roboflow_core/detection_to_contour_distance@v1",
        "name": "distance_step",
        "predictions": "$steps.model.predictions",
        "contours": "$steps.contours.contours",
        "distance_threshold": 100,
    }

    # when
    result = DetectionToContourDistanceManifest.model_validate(data)

    # then
    assert result.distance_threshold == 100


def test_manifest_accepts_selector_for_distance_threshold() -> None:
    # given
    data = {
        "type": "roboflow_core/detection_to_contour_distance@v1",
        "name": "distance_step",
        "predictions": "$steps.model.predictions",
        "contours": "$steps.contours.contours",
        "distance_threshold": "$inputs.distance_threshold",
    }

    # when
    result = DetectionToContourDistanceManifest.model_validate(data)

    # then
    assert result.distance_threshold == "$inputs.distance_threshold"


# ===================================================================
# 2. describe_outputs
# ===================================================================


def test_describe_outputs_returns_correct_definitions() -> None:
    # when
    outputs = DetectionToContourDistanceManifest.describe_outputs()

    # then
    output_names = {o.name for o in outputs}
    assert output_names == {"close_to_edge", "distances", "all_detections_with_flag"}


# ===================================================================
# 3. Empty detections
# ===================================================================


def test_run_with_empty_detections() -> None:
    # given
    block = DetectionToContourDistanceBlockV1()
    predictions = sv.Detections.empty()
    contours = [_square_contour(10, 10, 90, 90)]

    # when
    result = block.run(
        predictions=predictions,
        contours=contours,
        distance_threshold=50,
    )

    # then
    assert len(result["close_to_edge"]) == 0
    assert result["distances"] == []
    assert len(result["all_detections_with_flag"]) == 0


# ===================================================================
# 4. Empty contours
# ===================================================================


def test_run_with_empty_contours() -> None:
    # given
    block = DetectionToContourDistanceBlockV1()
    predictions = sv.Detections(
        xyxy=np.array([[10, 10, 30, 30]], dtype=np.float32),
    )
    contours = []

    # when
    result = block.run(
        predictions=predictions,
        contours=contours,
        distance_threshold=50,
    )

    # then
    assert len(result["close_to_edge"]) == 0
    assert result["distances"] == []
    # all_detections_with_flag should still contain the original detection
    assert len(result["all_detections_with_flag"]) == 1


# ===================================================================
# 5. Detection far from contour
# ===================================================================


def test_detection_far_from_contour() -> None:
    # given – contour is a 10x10 square at origin area, detection is far away
    block = DetectionToContourDistanceBlockV1()
    contours = [_square_contour(10, 10, 20, 20)]
    # Detection centered at (500, 500) – far from contour edges
    predictions = sv.Detections(
        xyxy=np.array([[490, 490, 510, 510]], dtype=np.float32),
    )

    # when
    result = block.run(
        predictions=predictions,
        contours=contours,
        distance_threshold=50,
    )

    # then
    assert len(result["close_to_edge"]) == 0
    assert len(result["distances"]) == 1
    assert result["distances"][0] > 50
    assert result["all_detections_with_flag"]["close_to_edge"][0] == False


# ===================================================================
# 6. Detection close to contour
# ===================================================================


def test_detection_close_to_contour() -> None:
    # given – contour is a square from (0,0) to (100,100)
    # detection centered at (105, 50) – 5px outside the right edge
    block = DetectionToContourDistanceBlockV1()
    contours = [_square_contour(0, 0, 100, 100)]
    predictions = sv.Detections(
        xyxy=np.array([[100, 45, 110, 55]], dtype=np.float32),
    )

    # when
    result = block.run(
        predictions=predictions,
        contours=contours,
        distance_threshold=50,
    )

    # then
    assert len(result["close_to_edge"]) == 1
    assert len(result["distances"]) == 1
    assert result["distances"][0] <= 50
    assert result["all_detections_with_flag"]["close_to_edge"][0] == True


# ===================================================================
# 7. Detection inside contour
# ===================================================================


def test_detection_inside_contour() -> None:
    # given – contour is a large square, detection center is inside
    # cv2.pointPolygonTest returns positive when inside, abs makes it distance to edge
    block = DetectionToContourDistanceBlockV1()
    contours = [_square_contour(0, 0, 200, 200)]
    # Detection centered at (100, 100) – center of the contour
    predictions = sv.Detections(
        xyxy=np.array([[90, 90, 110, 110]], dtype=np.float32),
    )

    # when
    result = block.run(
        predictions=predictions,
        contours=contours,
        distance_threshold=200,
    )

    # then – inside detection should have distance > 0 (distance to nearest edge)
    assert len(result["distances"]) == 1
    assert result["distances"][0] > 0
    # Center is at (100,100) in a 200x200 square, distance to nearest edge should be ~100
    assert result["distances"][0] == pytest.approx(100.0, abs=1.0)
    assert len(result["close_to_edge"]) == 1


# ===================================================================
# 8. Multiple detections mixed
# ===================================================================


def test_multiple_detections_mixed() -> None:
    # given – contour is square from (0,0) to (100,100)
    block = DetectionToContourDistanceBlockV1()
    contours = [_square_contour(0, 0, 100, 100)]

    # Detection 0: center (50, 50) – inside, ~50px from edge => close
    # Detection 1: center (500, 500) – far outside => NOT close
    # Detection 2: center (105, 50) – 5px outside right edge => close
    predictions = sv.Detections(
        xyxy=np.array(
            [
                [40, 40, 60, 60],
                [490, 490, 510, 510],
                [100, 45, 110, 55],
            ],
            dtype=np.float32,
        ),
    )

    # when
    result = block.run(
        predictions=predictions,
        contours=contours,
        distance_threshold=50,
    )

    # then
    assert len(result["distances"]) == 3
    assert len(result["close_to_edge"]) == 2  # detections 0 and 2
    flags = result["all_detections_with_flag"]["close_to_edge"]
    assert flags[0] == True   # inside, dist ~50
    assert flags[1] == False  # far away
    assert flags[2] == True   # close outside


# ===================================================================
# 9. Detection exactly on contour edge
# ===================================================================


def test_detection_exactly_on_contour_edge() -> None:
    # given – detection center exactly on the contour boundary
    block = DetectionToContourDistanceBlockV1()
    contours = [_square_contour(0, 0, 100, 100)]
    # Detection centered at (100, 50) – exactly on the right edge
    predictions = sv.Detections(
        xyxy=np.array([[95, 45, 105, 55]], dtype=np.float32),
    )

    # when
    result = block.run(
        predictions=predictions,
        contours=contours,
        distance_threshold=50,
    )

    # then – distance should be 0 (on the edge), should be close_to_edge
    assert len(result["distances"]) == 1
    assert result["distances"][0] == pytest.approx(0.0, abs=1.0)
    assert len(result["close_to_edge"]) == 1
    assert result["all_detections_with_flag"]["close_to_edge"][0] == True


# ===================================================================
# 10. Custom distance threshold
# ===================================================================


def test_custom_distance_threshold_excludes_when_tight() -> None:
    # given – detection is 5px outside the contour edge
    block = DetectionToContourDistanceBlockV1()
    contours = [_square_contour(0, 0, 100, 100)]
    # Detection centered at (105, 50) – 5px from right edge
    predictions = sv.Detections(
        xyxy=np.array([[100, 45, 110, 55]], dtype=np.float32),
    )

    # when – threshold of 3px is too tight to catch this detection
    result = block.run(
        predictions=predictions,
        contours=contours,
        distance_threshold=3,
    )

    # then – detection should NOT be flagged as close_to_edge
    assert len(result["close_to_edge"]) == 0
    assert result["all_detections_with_flag"]["close_to_edge"][0] == False


def test_custom_distance_threshold_includes_when_generous() -> None:
    # given – same detection 5px from edge, but threshold is 10
    block = DetectionToContourDistanceBlockV1()
    contours = [_square_contour(0, 0, 100, 100)]
    predictions = sv.Detections(
        xyxy=np.array([[100, 45, 110, 55]], dtype=np.float32),
    )

    # when – threshold of 10px should catch it
    result = block.run(
        predictions=predictions,
        contours=contours,
        distance_threshold=10,
    )

    # then
    assert len(result["close_to_edge"]) == 1
    assert result["all_detections_with_flag"]["close_to_edge"][0] == True
