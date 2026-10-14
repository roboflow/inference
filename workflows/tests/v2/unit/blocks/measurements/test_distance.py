"""``roboflow_core/distance_measurement@v1``: gaps, calibration, row choice and errors.

Expected values are worked out from the boxes below, not read from a run.
"""

import pytest
import torch
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.measurements import DistanceMeasurementBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import SegmentationView
from roboflow_workflows.v2.data.masks.grid import MaskGridGeometry

from inference_models.models.base.object_detection import Detections

CLASS_NAMES = {0: "cup", 1: "plate", 2: "card"}
# cup ends at x = 50, plate starts at x = 110: 60 px apart horizontally; their
# y ranges overlap, so there is no vertical gap.
CUP = [10, 10, 50, 40]
PLATE = [110, 15, 150, 45]
# card: 40 x 20 px.
CARD = [200, 100, 240, 120]


def detections(boxes, class_ids, *, rows=None, class_names=CLASS_NAMES):
    image = ImageData.from_tensor(
        torch.zeros((3, 200, 300), dtype=torch.uint8), image_id="scene"
    )
    prediction = Detections(
        xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.tensor(class_ids, dtype=torch.int64),
        confidence=torch.full((len(class_ids),), 0.9),
        image_metadata={**image.prediction_metadata(), "class_names": class_names},
        bboxes_metadata=rows,
    )

    return prediction


def measure(predictions, **overrides):
    params = {
        "object_1_class_name": "cup",
        "object_2_class_name": "plate",
        "reference_axis": "horizontal",
        "calibration_method": "pixel to centimeter",
        "reference_object_class_name": "card",
        "reference_width": 4.0,
        "reference_height": 2.0,
        "pixel_ratio": 20.0,
        **overrides,
    }

    return DistanceMeasurementBlock().run(predictions=predictions, **params)


def test_horizontal_gap_with_pixel_ratio() -> None:
    result = measure(detections([CUP, PLATE], [0, 1]))

    assert result == {"distance_cm": 3.0, "distance_pixel": 60}
    assert isinstance(result["distance_pixel"], int)
    assert isinstance(result["distance_cm"], float)


def test_gap_does_not_depend_on_object_order() -> None:
    result = measure(detections([PLATE, CUP], [1, 0]))

    assert result == {"distance_cm": 3.0, "distance_pixel": 60}


@pytest.mark.parametrize("upper_is_first", [True, False])
def test_vertical_gap_from_either_side(upper_is_first) -> None:
    upper, lower = [10, 10, 50, 40], [20, 70, 60, 90]
    boxes = [upper, lower] if upper_is_first else [lower, upper]

    result = measure(detections(boxes, [0, 1]), reference_axis="vertical")

    assert result == {"distance_cm": 1.5, "distance_pixel": 30}


def test_reference_object_calibration_averages_both_axes() -> None:
    # card 40 x 20 px for 4 x 2 cm: 10 px/cm both ways -> 60 px = 6 cm.
    calibrated = measure(
        detections([CUP, PLATE, CARD], [0, 1, 2]), calibration_method="reference object"
    )
    # 4 x 1 cm: 10 and 20 px/cm, mean 15 px/cm -> 60 px = 4 cm.
    anisotropic = measure(
        detections([CUP, PLATE, CARD], [0, 1, 2]),
        calibration_method="reference object",
        reference_height=1.0,
    )

    assert calibrated == {"distance_cm": 6.0, "distance_pixel": 60}
    assert anisotropic == {"distance_cm": 4.0, "distance_pixel": 60}


def test_unit_follows_calibration_only() -> None:
    predictions = detections([CUP, PLATE], [0, 1])

    coarse = measure(predictions, pixel_ratio=10.0)
    fine = measure(predictions, pixel_ratio=40.0)

    assert (coarse["distance_cm"], fine["distance_cm"]) == (6.0, 1.5)
    assert coarse["distance_pixel"] == fine["distance_pixel"] == 60


@pytest.mark.parametrize(
    "plate, axis",
    [
        ([40, 15, 80, 45], "horizontal"),  # boxes overlap
        ([50, 15, 90, 45], "horizontal"),  # edges touch at x = 50: overlap
        ([110, 15, 150, 45], "vertical"),  # apart in x, no gap in y
    ],
)
def test_overlap_or_no_gap_on_the_axis_measures_zero(plate, axis) -> None:
    result = measure(detections([CUP, plate], [0, 1]), reference_axis=axis)

    assert result == {"distance_cm": 0.0, "distance_pixel": 0}


def test_zero_result_is_returned_before_the_reference_is_looked_up() -> None:
    # V1 order: an overlap measures 0 even though no card is detected.
    result = measure(
        detections([CUP, [40, 15, 80, 45]], [0, 1]),
        calibration_method="reference object",
    )

    assert result == {"distance_cm": 0.0, "distance_pixel": 0}


def test_boxes_are_rounded_half_to_even_before_measuring() -> None:
    # x_max 50.5 -> 50, x_min 109.5 -> 110: 60 px; 51.5 -> 52 gives 58 px.
    even = measure(detections([[10, 10, 50.5, 40], [109.5, 15, 150, 45]], [0, 1]))
    odd = measure(detections([[10, 10, 51.5, 40], [109.5, 15, 150, 45]], [0, 1]))

    assert even["distance_pixel"] == 60
    assert odd["distance_pixel"] == 58


def test_latest_row_of_each_class_is_used_until_both_are_seen() -> None:
    # Rows: cup A, cup B, plate, cup C. The scan stops at the plate, so cup B
    # (ending at x = 90) is measured: 20 px. Cup C is never read.
    boxes = [[0, 10, 30, 40], [60, 10, 90, 40], PLATE, [100, 0, 105, 5]]

    result = measure(detections(boxes, [0, 0, 1, 0]))

    assert result["distance_pixel"] == 20


def test_first_reference_row_calibrates() -> None:
    second_card = [200, 150, 280, 190]  # 80 x 40 px would give 20 px/cm

    result = measure(
        detections([CUP, PLATE, CARD, second_card], [0, 1, 2, 2]),
        calibration_method="reference object",
    )

    assert result["distance_cm"] == 6.0


def test_row_class_name_overrides_the_class_map() -> None:
    rows = [{"class": "plate"}, {}]
    # Row 0 has class id 0 ("cup") but names itself "plate"; row 1 is unmapped.
    predictions = detections([PLATE, CUP], [0, 7], rows=rows)

    result = measure(predictions, object_1_class_name="class_7")

    assert result["distance_pixel"] == 60


@pytest.mark.parametrize(
    "boxes, class_ids",
    [([CUP], [0]), ([], [])],
    ids=["one-object", "empty"],
)
def test_missing_object_raises_the_v1_message(boxes, class_ids) -> None:
    with pytest.raises(ValueError, match="'cup' or 'plate' not found"):
        measure(detections(boxes, class_ids))


def test_missing_reference_object_raises() -> None:
    with pytest.raises(ValueError, match="Reference class 'card' not found"):
        measure(detections([CUP, PLATE], [0, 1]), calibration_method="reference object")


def test_reference_box_without_size_raises() -> None:
    with pytest.raises(ValueError, match="zero width and height"):
        measure(
            detections([CUP, PLATE, [5, 5, 5, 5]], [0, 1, 2]),
            calibration_method="reference object",
        )


def test_same_class_for_both_objects_is_rejected() -> None:
    with pytest.raises(ValueError, match="two different classes"):
        measure(detections([CUP, PLATE], [0, 0]), object_2_class_name="cup")


@pytest.mark.parametrize("field", ["pixel_ratio", "reference_width"])
@pytest.mark.parametrize("value", [0, -1.0])
def test_non_positive_calibration_is_rejected_by_the_params(field, value) -> None:
    params = {
        "predictions": "$steps.model.predictions",
        "object_1_class_name": "cup",
        "object_2_class_name": "plate",
        "reference_axis": "horizontal",
        "calibration_method": "pixel to centimeter",
        field: value,
    }

    with pytest.raises(ValidationError):
        DistanceMeasurementBlock.Params(**params)


def test_view_input_reads_boxes_without_materializing_masks() -> None:
    boxes = detections([CUP, PLATE], [0, 1])
    geometry = MaskGridGeometry.identity((200, 300))
    view = SegmentationView(
        detections=boxes,
        scores=torch.full((2, 200, 300), -1.0),
        score_type="logits",
        threshold=0.0,
        geometry=geometry,
    )

    result = measure(view)

    assert result == {"distance_cm": 3.0, "distance_pixel": 60}
    assert not view.is_full_res_materialized()


def test_input_is_not_modified() -> None:
    rows = [{"detection_id": "a"}, {"detection_id": "b"}]
    predictions = detections([CUP, PLATE], [0, 1], rows=rows)
    boxes_before = predictions.xyxy.clone()

    measure(predictions)

    assert torch.equal(predictions.xyxy, boxes_before)
    assert predictions.bboxes_metadata == [{"detection_id": "a"}, {"detection_id": "b"}]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"pixel_ratio": 0.0}, "Pixel-to-centimeter ratio must be greater than zero"),
        ({"pixel_ratio": float("inf")}, "Pixel-to-centimeter ratio .* finite"),
        (
            {"calibration_method": "reference object", "reference_width": 0.0},
            "Reference object dimensions must be greater than zero",
        ),
        (
            {"calibration_method": "reference object", "reference_width": float("inf")},
            "Reference object dimensions .* finite",
        ),
        (
            {
                "calibration_method": "reference object",
                "reference_height": float("inf"),
            },
            "Reference object dimensions .* finite",
        ),
        ({"reference_axis": "diagonal"}, "reference_axis must be one of"),
    ],
    ids=[
        "pixel-ratio",
        "pixel-ratio-inf",
        "reference-width",
        "reference-width-inf",
        "reference-height-inf",
        "axis",
    ],
)
def test_selector_values_are_checked_when_running(overrides, message) -> None:
    # Selector values reach run() without the Params constraints.
    with pytest.raises(ValueError, match=message):
        measure(detections([CUP, PLATE, CARD], [0, 1, 2]), **overrides)


def test_calibration_is_checked_only_for_objects_apart() -> None:
    # V1 order: overlapping objects measure 0 before the ratio is read.
    result = measure(detections([CUP, [40, 15, 80, 45]], [0, 1]), pixel_ratio=0.0)

    assert result == {"distance_cm": 0.0, "distance_pixel": 0}
