"""``v2/size_measurement@v1``: reference scaling, mask rectangles and errors.

Box sizes are exact. Mask sizes follow OpenCV's pixel-centre contours, so a
filled w x h rectangle measures (w - 1) x (h - 1).
"""

import dataclasses

import cv2
import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils
from roboflow_workflows.v2.blocks.measurements import SizeMeasurementBlock
from roboflow_workflows.v2.blocks.measurements.size import parse_reference_dimensions
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import SegmentationView
from roboflow_workflows.v2.data.masks.grid import MaskGridGeometry

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

SIZE_HW = (60, 80)
# Calibration card: 50 x 30 px box for 5 x 3 cm, i.e. 0.1 cm per pixel.
CARD = [10.0, 10.0, 60.0, 40.0]


def metadata(image=None):
    if image is None:
        image = ImageData.from_tensor(
            torch.zeros((3, *SIZE_HW), dtype=torch.uint8), image_id="scene"
        )

    return {**image.prediction_metadata(), "class_names": {0: "item"}}


def boxes(xyxy, *, confidence=None, image=None):
    count = len(xyxy)
    prediction = Detections(
        xyxy=torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.zeros(count, dtype=torch.int64),
        confidence=torch.tensor(confidence or [0.9] * count, dtype=torch.float32),
        image_metadata=metadata(image),
    )

    return prediction


def with_masks(masks, *, confidence=None):
    masks = torch.as_tensor(np.asarray(masks), dtype=torch.bool).reshape(-1, *SIZE_HW)
    rows = []
    for mask in masks:
        ys, xs = torch.nonzero(mask, as_tuple=True)
        rows.append(
            [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] if len(xs) else [0] * 4
        )
    detections = boxes(rows, confidence=confidence)
    prediction = InstanceDetections(
        xyxy=detections.xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
        mask=masks,
        image_metadata=detections.image_metadata,
    )

    return prediction


def rectangle_mask(x0, y0, x1, y1):
    mask = np.zeros(SIZE_HW, dtype=bool)
    mask[y0:y1, x0:x1] = True

    return mask


def rotated_mask(center, size, angle):
    mask = np.zeros(SIZE_HW, dtype=np.uint8)
    corners = cv2.boxPoints((center, size, angle)).round().astype(np.int32)
    cv2.fillPoly(mask, [corners], 1)

    return mask.astype(bool)


def as_rle(prediction):
    encoded = [
        mask_utils.encode(np.asfortranarray(mask.numpy().astype(np.uint8)))["counts"]
        for mask in prediction.mask
    ]
    converted = dataclasses.replace(
        prediction, mask=InstancesRLEMasks(image_size=SIZE_HW, masks=encoded)
    )

    return converted


def measure(objects, reference, dimensions=(5.0, 3.0)):
    result = SizeMeasurementBlock().run(
        object_predictions=objects,
        reference_predictions=reference,
        reference_dimensions=dimensions,
    )

    return result["dimensions"]


def v1_mask_size(mask):
    # V1's algorithm on the whole mask, written out independently.
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest = max(contours, key=cv2.contourArea)
    box = np.array(cv2.boxPoints(cv2.minAreaRect(largest)), dtype=np.float32)
    edges = (box[1] - box[0], box[2] - box[1])
    lengths = [float(np.linalg.norm(edge)) for edge in edges]
    tilts = []
    for edge in edges:
        angle = abs(np.degrees(np.arctan2(edge[1], edge[0])) % 180)
        tilts.append(min(angle, 180 - angle))
    width_first = tilts[0] < tilts[1]

    return (lengths[0], lengths[1]) if width_first else (lengths[1], lengths[0])


def test_boxes_scale_by_the_reference() -> None:
    dimensions = measure(boxes([[0, 0, 100, 60], [5, 5, 25, 45]]), boxes([CARD]))

    assert dimensions == [
        pytest.approx({"width": 10.0, "height": 6.0, "longer": 10.0, "shorter": 6.0}),
        pytest.approx({"width": 2.0, "height": 4.0, "longer": 4.0, "shorter": 2.0}),
    ]


def test_width_and_height_scale_separately() -> None:
    # 5 x 6 cm card at 50 x 30 px: 0.1 cm/px wide, 0.2 cm/px high.
    dimensions = measure(boxes([[0, 0, 10, 10]]), boxes([CARD]), dimensions=[5, 6])

    assert dimensions[0] == pytest.approx(
        {"width": 1.0, "height": 2.0, "longer": 2.0, "shorter": 1.0}
    )


def test_unit_follows_reference_dimensions() -> None:
    centimetres = measure(boxes([[0, 0, 100, 60]]), boxes([CARD]), "5,3")
    millimetres = measure(boxes([[0, 0, 100, 60]]), boxes([CARD]), "50, 30")

    assert millimetres[0]["width"] == pytest.approx(10 * centimetres[0]["width"])


def test_most_confident_reference_row_is_used_first_on_ties() -> None:
    reference = boxes(
        [[0, 0, 10, 10], CARD, [0, 0, 100, 100]], confidence=[0.5, 0.9, 0.9]
    )

    dimensions = measure(boxes([[0, 0, 100, 60]]), reference)

    assert dimensions[0]["width"] == pytest.approx(10.0)


def test_filled_rectangle_masks_measure_pixel_centre_distances() -> None:
    objects = with_masks([rectangle_mask(5, 5, 26, 16)])  # 21 x 11 pixels
    reference = with_masks([rectangle_mask(30, 20, 41, 31)])  # 11 x 11 pixels

    dimensions = measure(objects, reference, dimensions=(1.0, 1.0))

    # Contours span 20 x 10 and 10 x 10 pixel centres.
    assert dimensions[0] == pytest.approx(
        {"width": 2.0, "height": 1.0, "longer": 2.0, "shorter": 1.0}
    )


def test_rotated_mask_width_is_its_more_horizontal_edge() -> None:
    # A 40 x 10 bar turned 20 degrees: width ~ 40, height ~ 10 pixel-units.
    bar = rotated_mask((40, 30), (40, 10), 20)
    reference = boxes([[0, 0, 10, 10]])

    dimensions = measure(with_masks([bar]), reference, dimensions=(10, 10))

    width, height = dimensions[0]["width"], dimensions[0]["height"]
    assert width == pytest.approx(40, abs=2.5)
    assert height == pytest.approx(10, abs=2.5)
    assert (width, height) == pytest.approx(v1_mask_size(bar))


@pytest.mark.parametrize(
    "mask",
    [
        rotated_mask((40, 30), (40, 10), 20),
        rotated_mask((40, 30), (30, 30), 45),
        rectangle_mask(0, 0, 30, 60),  # touches three image edges
        rectangle_mask(60, 40, 80, 60),  # touches the bottom-right corner
    ],
    ids=["bar", "diamond", "left-edge", "corner"],
)
def test_windowed_contours_equal_v1_on_the_whole_mask(mask) -> None:
    dimensions = measure(with_masks([mask]), boxes([[0, 0, 1, 1]]), (1.0, 1.0))

    expected = v1_mask_size(mask)
    assert (dimensions[0]["width"], dimensions[0]["height"]) == pytest.approx(expected)


def test_largest_blob_of_a_mask_is_measured() -> None:
    mask = rectangle_mask(2, 2, 6, 6) | rectangle_mask(20, 20, 51, 31)

    dimensions = measure(with_masks([mask]), boxes([[0, 0, 1, 1]]), (1.0, 1.0))

    assert (dimensions[0]["width"], dimensions[0]["height"]) == (30.0, 10.0)


def test_rle_and_view_objects_equal_dense_masks() -> None:
    objects = with_masks(
        [rotated_mask((40, 30), (40, 10), 20), rectangle_mask(5, 5, 26, 16)]
    )
    reference = boxes([CARD])

    dense = measure(objects, reference)

    assert measure(as_rle(objects), reference) == dense
    assert measure(SegmentationView.from_dense(objects), reference) == dense


def test_view_reference_materializes_only_the_chosen_row() -> None:
    reference = with_masks(
        [rectangle_mask(30, 20, 41, 31), rectangle_mask(0, 0, 21, 21)],
        confidence=[0.4, 0.8],
    )
    geometry = MaskGridGeometry.identity(SIZE_HW)
    view = SegmentationView(
        detections=Detections(
            xyxy=reference.xyxy,
            class_id=reference.class_id,
            confidence=reference.confidence,
            image_metadata=reference.image_metadata,
        ),
        scores=reference.mask.float() * 2 - 1,
        score_type="logits",
        threshold=0.0,
        geometry=geometry,
    )

    dimensions = measure(boxes([[0, 0, 40, 40]]), view, dimensions=(2.0, 2.0))

    # Row 1 (20 x 20 between pixel centres) is chosen: 0.1 per pixel.
    assert dimensions[0]["width"] == pytest.approx(4.0)
    assert not view.is_full_res_materialized()


def test_rows_without_a_measurable_size_are_none() -> None:
    line = rectangle_mask(5, 5, 30, 6)  # one pixel high: zero contour area
    empty = np.zeros(SIZE_HW, dtype=bool)
    objects = with_masks([line, empty, rectangle_mask(5, 10, 26, 21)])

    dimensions = measure(objects, boxes([CARD]))

    assert dimensions[:2] == [None, None]
    assert dimensions[2] == pytest.approx(
        {"width": 2.0, "height": 1.0, "longer": 2.0, "shorter": 1.0}
    )


def test_zero_size_box_row_is_none() -> None:
    dimensions = measure(boxes([[5, 5, 5, 15], [0, 0, 10, 10]]), boxes([CARD]))

    assert dimensions[0] is None and dimensions[1] is not None


@pytest.mark.parametrize(
    "reference",
    [
        boxes([]),
        boxes([[5, 5, 5, 20]]),
        [],
        [[0, 0]],
        [[0, 0], [10, 0], [20, 0]],  # collinear: no area
    ],
    ids=["no-rows", "zero-size-box", "no-points", "one-point", "collinear"],
)
def test_unusable_reference_gives_no_dimensions(reference) -> None:
    assert measure(boxes([[0, 0, 10, 10]]), reference) is None


@pytest.mark.parametrize(
    "polygon",
    [
        [[10, 10], [60, 10], [60, 40], [10, 40]],
        [[10.0, 10.0], [60.0, 10.0], [60.0, 40.0], [10.0, 40.0]],
    ],
    ids=["integer", "float"],
)
def test_reference_polygon_calibrates(polygon) -> None:
    dimensions = measure(boxes([[0, 0, 100, 60]]), polygon)

    assert dimensions[0]["width"] == pytest.approx(10.0)
    assert dimensions[0]["height"] == pytest.approx(6.0)


@pytest.mark.parametrize(
    "polygon",
    [
        [[0, 0, 1], [10, 0, 1], [10, 10, 1]],
        [0, 0, 10, 0, 10, 10],
        [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 10, "y": 10}],
        [[0, 0], [10, 0], [10]],
        [[0, 0], [10, 0], [10, float("nan")]],
    ],
    ids=["three-values", "flat", "dicts", "ragged", "nan"],
)
def test_malformed_reference_polygon_raises_naming_the_parameter(polygon) -> None:
    # A selector value skips the literal [x, y] bound.
    with pytest.raises(ValueError, match="reference_predictions as a polygon"):
        measure(boxes([[0, 0, 10, 10]]), polygon)


def half_raster_masks():
    # A 20 x 10 object drawn on a mask of half the frame's 60 x 80 size.
    mask = torch.zeros((1, 30, 40), dtype=torch.bool)
    mask[0, 5:15, 5:25] = True
    detections = boxes([[10, 10, 50, 30]])
    prediction = InstanceDetections(
        xyxy=detections.xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
        mask=mask,
        image_metadata=detections.image_metadata,
    )

    return prediction


def test_object_masks_must_match_the_frame() -> None:
    objects = half_raster_masks()
    encoded = mask_utils.encode(
        np.asfortranarray(objects.mask[0].numpy().astype(np.uint8))
    )
    objects_rle = dataclasses.replace(
        objects,
        mask=InstancesRLEMasks(image_size=(30, 40), masks=[encoded["counts"]]),
    )

    for prediction in (objects, objects_rle):
        with pytest.raises(ValueError, match="object_predictions masks are 30x40"):
            measure(prediction, boxes([CARD]))


def test_reference_masks_must_match_the_frame() -> None:
    with pytest.raises(ValueError, match="reference_predictions masks are 30x40"):
        measure(boxes([[0, 0, 10, 10]]), half_raster_masks())


def test_masks_without_frame_metadata_are_measured_as_given() -> None:
    objects = dataclasses.replace(half_raster_masks(), image_metadata=None)

    dimensions = measure(objects, boxes([CARD]))

    assert (dimensions[0]["width"], dimensions[0]["height"]) == pytest.approx(
        (1.9, 0.9)
    )


def test_uint8_255_masks_measure_like_boolean_masks() -> None:
    objects = with_masks([rotated_mask((40, 30), (40, 10), 20)])
    as_255 = dataclasses.replace(objects, mask=objects.mask.to(torch.uint8) * 255)

    assert measure(as_255, boxes([CARD])) == measure(objects, boxes([CARD]))


def test_empty_objects_give_an_empty_list() -> None:
    assert measure(boxes([]), boxes([CARD])) == []


@pytest.mark.parametrize(
    "value, expected",
    [("4.5, 3", (4.5, 3.0)), ((4.5, 3), (4.5, 3.0)), ([2, 1.5], (2.0, 1.5))],
)
def test_reference_dimension_forms(value, expected) -> None:
    assert parse_reference_dimensions(value) == expected


@pytest.mark.parametrize(
    "value, message",
    [
        ("4.5", "format 'width,height'"),
        ("a,b", "Invalid format"),
        ([1, 2, 3], "two values"),
        ([0, 3], "positive and finite"),
        ("-1,2", "positive and finite"),
        ("inf,2", "positive and finite"),
    ],
)
def test_invalid_reference_dimensions_raise(value, message) -> None:
    with pytest.raises(ValueError, match=message):
        measure(boxes([[0, 0, 10, 10]]), boxes([CARD]), value)


def test_crops_of_one_image_share_pixel_size() -> None:
    full = ImageData.from_tensor(
        torch.zeros((3, 120, 160), dtype=torch.uint8), image_id="full"
    )
    objects = boxes([[0, 0, 100, 60]], image=full.crop((40, 30, 140, 90)))

    dimensions = measure(objects, boxes([CARD], image=full))

    assert dimensions[0]["width"] == pytest.approx(10.0)


def test_differently_scaled_images_raise() -> None:
    full = ImageData.from_tensor(
        torch.zeros((3, 120, 160), dtype=torch.uint8), image_id="full"
    )
    half = full.resize((60, 80))

    with pytest.raises(ValueError, match="different pixel sizes"):
        measure(boxes([[0, 0, 50, 30]], image=half), boxes([CARD], image=full))


def test_inputs_are_not_modified() -> None:
    objects = with_masks([rectangle_mask(5, 5, 26, 16)])
    mask_before = objects.mask.clone()

    measure(objects, boxes([CARD]))

    assert torch.equal(objects.mask, mask_before)
    assert objects.bboxes_metadata is None


def test_box_sides_keep_float32_arithmetic() -> None:
    # 0.3 - 0.1 in float32 is not 0.2 in float64; V1 subtracts in float32.
    dimensions = measure(boxes([[0.1, 0.1, 0.3, 0.7]]), boxes([[0, 0, 1, 1]]), (1, 1))

    expected_width = float(np.float32(0.3) - np.float32(0.1))
    assert dimensions[0]["width"] == expected_width
