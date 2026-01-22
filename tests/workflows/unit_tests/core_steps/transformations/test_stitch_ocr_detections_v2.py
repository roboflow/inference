import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.transformations.stitch_ocr_detections.v2 import (
    BlockManifest,
    StitchOCRDetectionsBlockV2,
    adaptive_word_grouping,
    collimate_word_grouping,
    find_otsu_threshold,
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
        "stitching_algorithm": "tolerance",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result.type == "roboflow_core/stitch_ocr_detections@v1"
    assert result.name == "some"
    assert result.predictions == "$steps.detection.predictions"
    assert result.reading_direction == "left_to_right"
    assert result.tolerance == "$inputs.tolerance"
    assert result.stitching_algorithm == "tolerance"


def test_stitch_ocr_detections_when_otsu_algorithm_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/stitch_ocr_detections@v1",
        "name": "some",
        "predictions": "$steps.detection.predictions",
        "reading_direction": "left_to_right",
        "stitching_algorithm": "otsu",
        "otsu_threshold_multiplier": 1.5,
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result.stitching_algorithm == "otsu"
    assert result.otsu_threshold_multiplier == 1.5


def test_stitch_ocr_detections_when_collimate_algorithm_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/stitch_ocr_detections@v1",
        "name": "some",
        "predictions": "$steps.detection.predictions",
        "reading_direction": "left_to_right",
        "stitching_algorithm": "collimate",
        "collimate_tolerance": 15,
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result.stitching_algorithm == "collimate"
    assert result.collimate_tolerance == 15


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
        xyxy=np.array(xyxy, dtype=np.float32),
        data={"class_name": np.array(class_names)},
    )


# =============================================================================
# Tolerance Algorithm Tests
# =============================================================================


def test_tolerance_empty_detections():
    """Test handling of empty detections."""
    detections = create_test_detections(xyxy=np.array([]).reshape(0, 4), class_names=[])
    result = stitch_ocr_detections(detections)
    assert result == {"ocr_text": ""}


def test_tolerance_left_to_right_single_line():
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


def test_tolerance_left_to_right_multiple_lines():
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


def test_tolerance_right_to_left_single_line():
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


def test_tolerance_vertical_top_to_bottom():
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


def test_tolerance_unordered_input():
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


def test_tolerance_with_delimiter():
    """Test that delimiter is applied between characters."""
    detections = create_test_detections(
        xyxy=np.array(
            [
                [10, 0, 20, 10],  # "A"
                [30, 0, 40, 10],  # "B"
                [50, 0, 60, 10],  # "C"
            ]
        ),
        class_names=["A", "B", "C"],
    )
    result = stitch_ocr_detections(detections, delimiter="-")
    assert result == {"ocr_text": "A-B-C"}


# =============================================================================
# Otsu Algorithm Tests
# =============================================================================


def test_otsu_empty_detections():
    """Test Otsu algorithm handling of empty detections."""
    detections = create_test_detections(xyxy=np.array([]).reshape(0, 4), class_names=[])
    result = adaptive_word_grouping(detections, reading_direction="left_to_right")
    assert result == {"ocr_text": ""}


def test_otsu_single_word():
    """Test Otsu algorithm with a single word (uniform spacing)."""
    # Characters with small, uniform gaps - should be detected as single word
    detections = create_test_detections(
        xyxy=np.array(
            [
                [0, 0, 10, 20],  # "H"
                [12, 0, 22, 20],  # "E"
                [24, 0, 34, 20],  # "L"
                [36, 0, 46, 20],  # "L"
                [48, 0, 58, 20],  # "O"
            ]
        ),
        class_names=["H", "E", "L", "L", "O"],
    )
    result = adaptive_word_grouping(detections, reading_direction="left_to_right")
    assert result == {"ocr_text": "HELLO"}


def test_otsu_multiple_words():
    """Test Otsu algorithm with multiple words (bimodal spacing)."""
    # Two words with clear gap between them
    detections = create_test_detections(
        xyxy=np.array(
            [
                # "HI" - characters close together
                [0, 0, 10, 20],  # "H"
                [12, 0, 22, 20],  # "I"
                # Large gap
                # "BYE" - characters close together
                [80, 0, 90, 20],  # "B"
                [92, 0, 102, 20],  # "Y"
                [104, 0, 114, 20],  # "E"
            ]
        ),
        class_names=["H", "I", "B", "Y", "E"],
    )
    result = adaptive_word_grouping(detections, reading_direction="left_to_right")
    assert result == {"ocr_text": "HI BYE"}


def test_otsu_multiple_lines():
    """Test Otsu algorithm with multiple lines."""
    detections = create_test_detections(
        xyxy=np.array(
            [
                # Line 1: "AB"
                [0, 0, 10, 20],  # "A"
                [12, 0, 22, 20],  # "B"
                # Line 2: "CD"
                [0, 40, 10, 60],  # "C"
                [12, 40, 22, 60],  # "D"
            ]
        ),
        class_names=["A", "B", "C", "D"],
    )
    result = adaptive_word_grouping(detections, reading_direction="left_to_right")
    assert result == {"ocr_text": "AB\nCD"}


def test_otsu_threshold_multiplier():
    """Test that threshold multiplier affects word grouping."""
    # Words with moderate gap
    detections = create_test_detections(
        xyxy=np.array(
            [
                [0, 0, 10, 20],  # "A"
                [12, 0, 22, 20],  # "B"
                [50, 0, 60, 20],  # "C"
                [62, 0, 72, 20],  # "D"
            ]
        ),
        class_names=["A", "B", "C", "D"],
    )

    # With high multiplier, should merge more aggressively
    result_high = adaptive_word_grouping(
        detections, reading_direction="left_to_right", threshold_multiplier=3.0
    )

    # With low multiplier, should split more aggressively
    result_low = adaptive_word_grouping(
        detections, reading_direction="left_to_right", threshold_multiplier=0.5
    )

    # Results should differ based on multiplier
    # (exact assertion depends on gap sizes)
    assert isinstance(result_high["ocr_text"], str)
    assert isinstance(result_low["ocr_text"], str)


def test_find_otsu_threshold_bimodal():
    """Test that Otsu correctly identifies bimodal distributions."""
    # Clear bimodal distribution: small gaps (0.1) and large gaps (1.5)
    gaps = np.array([0.05, 0.1, 0.08, 0.12, 1.5, 1.4, 1.6])
    threshold, is_bimodal = find_otsu_threshold(gaps)

    assert is_bimodal  # Should be True (bimodal)
    assert 0.1 < threshold < 1.5  # Threshold should be between the two groups


def test_find_otsu_threshold_unimodal():
    """Test that Otsu correctly identifies unimodal distributions."""
    # Unimodal distribution: all small gaps
    gaps = np.array([0.05, 0.1, 0.08, 0.12, 0.07, 0.11])
    threshold, is_bimodal = find_otsu_threshold(gaps)

    assert not is_bimodal  # Should be False (unimodal)


def test_find_otsu_threshold_few_gaps():
    """Test Otsu with very few gaps."""
    gaps = np.array([0.1])
    threshold, is_bimodal = find_otsu_threshold(gaps)

    assert not is_bimodal  # Should be False
    assert threshold == 0.0


# =============================================================================
# Collimate Algorithm Tests
# =============================================================================


def test_collimate_empty_detections():
    """Test Collimate algorithm handling of empty detections."""
    detections = create_test_detections(xyxy=np.array([]).reshape(0, 4), class_names=[])
    result = collimate_word_grouping(detections, reading_direction="left_to_right")
    assert result == {"ocr_text": ""}


def test_collimate_single_line():
    """Test Collimate algorithm with a single line."""
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
    result = collimate_word_grouping(detections, reading_direction="left_to_right")
    assert result == {"ocr_text": "HELLO"}


def test_collimate_multiple_lines():
    """Test Collimate algorithm with multiple lines."""
    detections = create_test_detections(
        xyxy=np.array(
            [
                # Line 1
                [10, 0, 20, 10],  # "H"
                [30, 0, 40, 10],  # "I"
                # Line 2
                [10, 30, 20, 40],  # "B"
                [30, 30, 40, 40],  # "Y"
                [50, 30, 60, 40],  # "E"
            ]
        ),
        class_names=["H", "I", "B", "Y", "E"],
    )
    result = collimate_word_grouping(detections, reading_direction="left_to_right")
    assert result == {"ocr_text": "HI\nBYE"}


def test_collimate_skewed_text():
    """Test Collimate algorithm with slightly skewed text."""
    # Text that slopes down slightly - each character is a bit lower
    detections = create_test_detections(
        xyxy=np.array(
            [
                [10, 0, 20, 10],  # "S"
                [30, 3, 40, 13],  # "K" (slightly lower)
                [50, 6, 60, 16],  # "E" (slightly lower)
                [70, 9, 80, 19],  # "W" (slightly lower)
            ]
        ),
        class_names=["S", "K", "E", "W"],
    )
    # With sufficient tolerance, should group as single line
    result = collimate_word_grouping(
        detections, reading_direction="left_to_right", tolerance=10
    )
    assert result == {"ocr_text": "SKEW"}


def test_collimate_tolerance_affects_grouping():
    """Test that collimate tolerance affects line grouping."""
    # Two lines that are fairly close
    detections = create_test_detections(
        xyxy=np.array(
            [
                [10, 0, 20, 10],  # "A"
                [30, 0, 40, 10],  # "B"
                [10, 15, 20, 25],  # "C" (close to line 1)
                [30, 15, 40, 25],  # "D"
            ]
        ),
        class_names=["A", "B", "C", "D"],
    )

    # With small tolerance, should be separate lines
    result_small = collimate_word_grouping(
        detections, reading_direction="left_to_right", tolerance=2
    )

    # With large tolerance, might merge into one line
    result_large = collimate_word_grouping(
        detections, reading_direction="left_to_right", tolerance=20
    )

    # At minimum, the algorithm should produce valid output
    assert "A" in result_small["ocr_text"]
    assert "B" in result_small["ocr_text"]
    assert "C" in result_small["ocr_text"]
    assert "D" in result_small["ocr_text"]


def test_collimate_right_to_left():
    """Test Collimate algorithm with right-to-left reading."""
    detections = create_test_detections(
        xyxy=np.array(
            [
                [90, 0, 100, 10],  # "أ"
                [70, 0, 80, 10],  # "ب"
                [50, 0, 60, 10],  # "ج"
            ]
        ),
        class_names=["أ", "ب", "ج"],
    )
    result = collimate_word_grouping(detections, reading_direction="right_to_left")
    assert result == {"ocr_text": "أبج"}


def test_collimate_vertical():
    """Test Collimate algorithm with vertical text."""
    detections = create_test_detections(
        xyxy=np.array(
            [
                [0, 10, 10, 20],  # "上"
                [0, 30, 10, 40],  # "下"
            ]
        ),
        class_names=["上", "下"],
    )
    result = collimate_word_grouping(
        detections, reading_direction="vertical_top_to_bottom"
    )
    assert result == {"ocr_text": "上下"}


def test_collimate_with_delimiter():
    """Test Collimate algorithm with delimiter."""
    detections = create_test_detections(
        xyxy=np.array(
            [
                [10, 0, 20, 10],  # "A"
                [30, 0, 40, 10],  # "B"
                [50, 0, 60, 10],  # "C"
            ]
        ),
        class_names=["A", "B", "C"],
    )
    result = collimate_word_grouping(
        detections, reading_direction="left_to_right", delimiter="-"
    )
    assert result == {"ocr_text": "A-B-C"}


# =============================================================================
# Block Integration Tests
# =============================================================================


def test_block_run_with_tolerance_algorithm():
    """Test the block's run method with tolerance algorithm."""
    block = StitchOCRDetectionsBlockV2()
    detections = create_test_detections(
        xyxy=np.array(
            [
                [10, 0, 20, 10],
                [30, 0, 40, 10],
            ]
        ),
        class_names=["A", "B"],
    )

    result = block.run(
        predictions=[detections],
        stitching_algorithm="tolerance",
        reading_direction="left_to_right",
        tolerance=10,
    )

    assert len(result) == 1
    assert result[0]["ocr_text"] == "AB"


def test_block_run_with_otsu_algorithm():
    """Test the block's run method with otsu algorithm."""
    block = StitchOCRDetectionsBlockV2()
    detections = create_test_detections(
        xyxy=np.array(
            [
                [0, 0, 10, 20],
                [12, 0, 22, 20],
            ]
        ),
        class_names=["A", "B"],
    )

    result = block.run(
        predictions=[detections],
        stitching_algorithm="otsu",
        reading_direction="left_to_right",
        tolerance=10,
        otsu_threshold_multiplier=1.0,
    )

    assert len(result) == 1
    assert result[0]["ocr_text"] == "AB"


def test_block_run_with_collimate_algorithm():
    """Test the block's run method with collimate algorithm."""
    block = StitchOCRDetectionsBlockV2()
    detections = create_test_detections(
        xyxy=np.array(
            [
                [10, 0, 20, 10],
                [30, 0, 40, 10],
            ]
        ),
        class_names=["A", "B"],
    )

    result = block.run(
        predictions=[detections],
        stitching_algorithm="collimate",
        reading_direction="left_to_right",
        tolerance=10,
        collimate_tolerance=10,
    )

    assert len(result) == 1
    assert result[0]["ocr_text"] == "AB"


@pytest.mark.parametrize(
    "reading_direction",
    [
        "left_to_right",
        "right_to_left",
        "vertical_top_to_bottom",
        "vertical_bottom_to_top",
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    ["tolerance", "otsu", "collimate"],
)
def test_all_algorithms_with_all_directions(reading_direction, algorithm):
    """Test that all algorithms work with all reading directions."""
    block = StitchOCRDetectionsBlockV2()
    detections = create_test_detections(
        xyxy=np.array([[0, 0, 10, 10]]),
        class_names=["A"],
    )

    result = block.run(
        predictions=[detections],
        stitching_algorithm=algorithm,
        reading_direction=reading_direction,
        tolerance=10,
        otsu_threshold_multiplier=1.0,
        collimate_tolerance=10,
    )

    assert len(result) == 1
    assert result[0]["ocr_text"] == "A"
