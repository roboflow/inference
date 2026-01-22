import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.transformations.stitch_ocr_detections.v1 import (
    BlockManifest,
    stitch_ocr_detections,
)


def test_stitch_ocr_detections_when_valid_manifest_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/stitch_ocr_detections@v1",
        "name": "some",
        "predictions": "$steps.detection.predictions",
        "reading_direction": "left_to_right",
        "tolerance": "$inputs.tolerance",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type="roboflow_core/stitch_ocr_detections@v1",
        name="some",
        predictions="$steps.detection.predictions",
        reading_direction="left_to_right",
        tolerance="$inputs.tolerance",
    )


def test_stitch_ocr_detections_when_invalid_tolerance_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/stitch_ocr_detections@v1",
        "name": "some",
        "predictions": "$steps.detection.predictions",
        "reading_direction": "left_to_right",
        "tolerance": 0,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def create_test_detections(xyxy: np.ndarray, class_names: list) -> sv.Detections:
    """Helper function to create test detection objects."""
    return sv.Detections(
        xyxy=np.array(xyxy), data={"class_name": np.array(class_names)}
    )


def test_empty_detections():
    """Test handling of empty detections."""
    detections = create_test_detections(xyxy=np.array([]).reshape(0, 4), class_names=[])
    result = stitch_ocr_detections(detections)
    assert result == {"ocr_text": ""}


def test_left_to_right_single_line():
    """Test basic left-to-right reading of a single line."""
    detections = create_test_detections(
        xyxy=np.array(
            [
                [10, 0, 20, 10],  # "H"
                [30, 0, 40, 10],  # "E"
                [50, 0, 60, 10],  # "L"
                [70, 0, 80, 10],  # "L"
                [90, 0, 100, 10],  # "O"
            ]
        ),
        class_names=["H", "E", "L", "L", "O"],
    )
    result = stitch_ocr_detections(detections, reading_direction="left_to_right")
    assert result == {"ocr_text": "HELLO"}


def test_left_to_right_multiple_lines():
    """Test left-to-right reading with multiple lines."""
    detections = create_test_detections(
        xyxy=np.array(
            [
                [10, 0, 20, 10],  # "H"
                [30, 0, 40, 10],  # "I"
                [10, 20, 20, 30],  # "B"
                [30, 20, 40, 30],  # "Y"
                [50, 20, 60, 30],  # "E"
            ]
        ),
        class_names=["H", "I", "B", "Y", "E"],
    )
    result = stitch_ocr_detections(detections, reading_direction="left_to_right")
    assert result == {"ocr_text": "HI\nBYE"}


def test_right_to_left_single_line():
    """Test right-to-left reading of a single line."""
    detections = create_test_detections(
        xyxy=np.array(
            [
                [90, 0, 100, 10],  # "م"
                [70, 0, 80, 10],  # "ر"
                [50, 0, 60, 10],  # "ح"
                [30, 0, 40, 10],  # "ب"
                [10, 0, 20, 10],  # "ا"
            ]
        ),
        class_names=["م", "ر", "ح", "ب", "ا"],
    )
    result = stitch_ocr_detections(detections, reading_direction="right_to_left")
    assert result == {"ocr_text": "مرحبا"}


def test_vertical_top_to_bottom():
    """Test vertical reading from top to bottom."""
    detections = create_test_detections(
        xyxy=np.array(
            [
                # First column (rightmost)
                [20, 10, 30, 20],  # "上"
                [20, 30, 30, 40],  # "下"
                # Second column (leftmost)
                [0, 10, 10, 20],  # "左"
                [0, 30, 10, 40],  # "右"
            ]
        ),
        class_names=["上", "下", "左", "右"],
    )
    # With current logic, we'll group by original x-coord and sort by y
    result = stitch_ocr_detections(
        detections, reading_direction="vertical_top_to_bottom"
    )
    assert result == {"ocr_text": "左右 上下"}


def test_tolerance_grouping():
    """Test that tolerance parameter correctly groups lines."""
    detections = create_test_detections(
        xyxy=np.array(
            [
                [10, 0, 20, 10],  # "A"
                [30, 2, 40, 12],  # "B" (slightly offset)
                [10, 20, 20, 30],  # "C" (closer to D)
                [30, 22, 40, 32],  # "D" (slightly offset from C)
            ]
        ),
        class_names=["A", "B", "C", "D"],
    )

    # With small tolerance, should treat as 4 separate lines
    result_small = stitch_ocr_detections(detections, tolerance=1)
    assert result_small == {"ocr_text": "A\nB\nC\nD"}

    # With larger tolerance, should group into 2 lines
    result_large = stitch_ocr_detections(detections, tolerance=5)
    assert result_large == {"ocr_text": "AB\nCD"}


def test_unordered_input():
    """Test that detections are correctly ordered regardless of input order."""
    detections = create_test_detections(
        xyxy=np.array(
            [
                [50, 0, 60, 10],  # "O"
                [10, 0, 20, 10],  # "H"
                [70, 0, 80, 10],  # "W"
                [30, 0, 40, 10],  # "L"
            ]
        ),
        class_names=["O", "H", "W", "L"],
    )
    result = stitch_ocr_detections(detections, reading_direction="left_to_right")
    assert result == {"ocr_text": "HLOW"}


@pytest.mark.parametrize(
    "reading_direction",
    [
        "left_to_right",
        "right_to_left",
        "vertical_top_to_bottom",
        "vertical_bottom_to_top",
    ],
)
def test_reading_directions(reading_direction):
    """Test that all reading directions are supported."""
    detections = create_test_detections(
        xyxy=np.array([[0, 0, 10, 10]]), class_names=["A"]  # Single detection
    )
    result = stitch_ocr_detections(detections, reading_direction=reading_direction)
    assert result == {"ocr_text": "A"}  # Should work with any direction
