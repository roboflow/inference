"""``v2/mask_area_measurement@v1``: exact areas per mask representation.

Masks are drawn on a 40 x 60 image with known pixel counts:

    ring    20 x 20 square at (5, 5) minus a 10 x 10 hole: 300 px
    patch   6 x 5 block at (40, 10):                     30 px
"""

import dataclasses

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.measurements import MaskAreaMeasurementBlock
from roboflow_workflows.v2.blocks.measurements import mask_area as mask_area_module
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import SegmentationView
from roboflow_workflows.v2.data.masks.grid import MaskGridGeometry

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

SIZE_HW = (40, 60)
RING_BOX = [5.0, 5.0, 25.0, 25.0]
PATCH_BOX = [40.0, 10.0, 46.0, 15.0]


def image_metadata(size_hw=SIZE_HW):
    image = ImageData.from_tensor(
        torch.zeros((3, *size_hw), dtype=torch.uint8), image_id="scene"
    )

    return {**image.prediction_metadata(), "class_names": {0: "part"}}


def ring_and_patch_masks():
    mask = torch.zeros((2, *SIZE_HW), dtype=torch.bool)
    mask[0, 5:25, 5:25] = True
    mask[0, 10:20, 10:20] = False
    mask[1, 10:15, 40:46] = True

    return mask


def boxes(xyxy, *, rows=None):
    prediction = Detections(
        xyxy=torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.zeros(len(xyxy), dtype=torch.int64),
        confidence=torch.full((len(xyxy),), 0.8),
        image_metadata=image_metadata(),
        bboxes_metadata=rows,
    )

    return prediction


def dense(mask=None, xyxy=(RING_BOX, PATCH_BOX), *, rows=None):
    mask = ring_and_patch_masks() if mask is None else mask
    detections = boxes(list(xyxy), rows=rows)
    prediction = InstanceDetections(
        xyxy=detections.xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
        mask=mask,
        image_metadata=detections.image_metadata,
        bboxes_metadata=rows,
    )

    return prediction


def rle(prediction):
    encoded = [
        mask_utils.encode(np.asfortranarray(mask.numpy().astype(np.uint8)))["counts"]
        for mask in prediction.mask
    ]
    converted = dataclasses.replace(
        prediction, mask=InstancesRLEMasks(image_size=SIZE_HW, masks=encoded)
    )

    return converted


def measure(predictions, pixels_per_unit=1.0):
    result = MaskAreaMeasurementBlock().run(
        predictions=predictions, pixels_per_unit=pixels_per_unit
    )

    return result["predictions"]


def areas(predictions):
    rows = (
        predictions.detections.bboxes_metadata
        if isinstance(predictions, SegmentationView)
        else predictions.bboxes_metadata
    )
    values = [(row["area_px"], row["area_converted"]) for row in rows]

    return values


def test_box_areas_without_masks() -> None:
    measured = measure(boxes([RING_BOX, [0.5, 0.5, 2.0, 4.5]]), pixels_per_unit=2.0)

    assert areas(measured) == [(400.0, 100.0), (6.0, 1.5)]
    assert isinstance(measured, Detections)


def test_dense_masks_count_pixels_and_exclude_holes() -> None:
    measured = measure(dense(), pixels_per_unit=10.0)

    assert areas(measured) == [(300.0, 3.0), (30.0, 0.3)]
    assert isinstance(measured, InstanceDetections)


def uint8_ring_255_and_patch_1():
    # V1 counts nonzero pixels whatever their value; a summing count would
    # report the ring 255 times too large.
    mask = ring_and_patch_masks().to(dtype=torch.uint8)
    mask[0] *= 255

    return mask


def test_uint8_masks_count_nonzero_pixels_not_values() -> None:
    measured = measure(dense(uint8_ring_255_and_patch_1()))

    assert [area for area, _ in areas(measured)] == [300.0, 30.0]


def test_uint8_masks_count_nonzero_pixels_through_a_view() -> None:
    view = SegmentationView.from_dense(dense(uint8_ring_255_and_patch_1()))

    assert [area for area, _ in areas(measure(view))] == [300.0, 30.0]


def test_empty_mask_falls_back_to_its_box_area() -> None:
    mask = ring_and_patch_masks()
    mask[1] = False

    measured = measure(dense(mask))

    assert [area for area, _ in areas(measured)] == [300.0, 30.0]


def test_unit_change_scales_by_the_square_of_pixels_per_unit() -> None:
    for pixels_per_unit, expected in ((1.0, 300.0), (5.0, 12.0), (0.5, 1200.0)):
        measured = measure(dense(), pixels_per_unit=pixels_per_unit)
        assert areas(measured)[0] == (300.0, expected)


def test_rle_masks_count_runs_without_decoding(monkeypatch) -> None:
    def no_decode(*args, **kwargs):
        raise AssertionError("RLE masks were decoded")

    monkeypatch.setattr(mask_utils, "decode", no_decode)
    mask = ring_and_patch_masks()
    mask[1] = False

    measured = measure(rle(dense(mask)))

    assert [area for area, _ in areas(measured)] == [300.0, 30.0]
    assert isinstance(measured.mask, InstancesRLEMasks)


def test_rle_counts_as_text_are_accepted() -> None:
    encoded = rle(dense())
    as_text = dataclasses.replace(
        encoded,
        mask=InstancesRLEMasks(
            image_size=SIZE_HW,
            masks=[counts.decode("ascii") for counts in encoded.mask.masks],
        ),
    )

    assert [area for area, _ in areas(measure(as_text))] == [300.0, 30.0]


def test_malformed_rle_counts_raise() -> None:
    encoded = rle(dense())
    broken = dataclasses.replace(
        encoded,
        mask=InstancesRLEMasks(image_size=SIZE_HW, masks=[b"0", encoded.mask.masks[1]]),
    )

    with pytest.raises(ContractError, match="RLE row 0"):
        measure(broken)


def grid_view(prediction, *, stride=4):
    # Logits on a 4x coarser grid; the reference upsamples them bilinearly
    # and thresholds at full resolution, so it is not cells x 16.
    cells = torch.nn.functional.avg_pool2d(
        prediction.mask.float().unsqueeze(1), stride
    ).squeeze(1)
    height, width = SIZE_HW
    geometry = MaskGridGeometry(
        grid_size_hw=(height // stride, width // stride),
        inference_size_hw=SIZE_HW,
        padding_ltrb=(0, 0, 0, 0),
        scale_xy=(1.0, 1.0),
        crop_size_hw=SIZE_HW,
        original_size_hw=SIZE_HW,
        static_crop_xywh=(0, 0, width, height),
    )
    detections = Detections(
        xyxy=prediction.xyxy,
        class_id=prediction.class_id,
        confidence=prediction.confidence,
        image_metadata=prediction.image_metadata,
        bboxes_metadata=prediction.bboxes_metadata,
    )
    view = SegmentationView(
        detections=detections,
        scores=cells * 16.0 - 8.0,
        score_type="logits",
        threshold=0.0,
        geometry=geometry,
    )

    return view


def test_view_is_measured_on_its_reference_masks_and_stays_a_view() -> None:
    view = grid_view(dense(rows=[{"id": "a"}, {"id": "b"}]))
    reference_counts = view.compute_full_res().mask.sum(dim=(1, 2)).tolist()
    low_res_counts = (view.low_res_binary().sum(dim=(1, 2)) * 16).tolist()

    measured = measure(view)

    assert [area for area, _ in areas(measured)] == [float(c) for c in reference_counts]
    assert reference_counts != low_res_counts
    assert isinstance(measured, SegmentationView)
    assert [row["id"] for row in measured.detections.bboxes_metadata] == ["a", "b"]
    assert view.detections.bboxes_metadata == [{"id": "a"}, {"id": "b"}]
    assert torch.equal(measured.scores, view.scores)


def test_dense_fallback_view_counts_its_dense_masks() -> None:
    measured = measure(SegmentationView.from_dense(dense()))

    assert [area for area, _ in areas(measured)] == [300.0, 30.0]
    assert isinstance(measured, SegmentationView)


def test_mask_raster_must_match_the_frame() -> None:
    small = torch.zeros((2, 20, 30), dtype=torch.bool)

    with pytest.raises(ValueError, match="Masks are 20x30 but the prediction's image"):
        measure(dense(small))


def test_masks_without_frame_metadata_are_counted_as_given() -> None:
    prediction = dataclasses.replace(dense(), image_metadata=None)

    assert [area for area, _ in areas(measure(prediction))] == [300.0, 30.0]


def test_crop_offset_does_not_change_areas() -> None:
    full = ImageData.from_tensor(
        torch.zeros((3, 100, 120), dtype=torch.uint8), image_id="full"
    )
    crop = full.crop((30, 20, 90, 60))
    prediction = dataclasses.replace(
        dense(), image_metadata={**crop.prediction_metadata(), "class_names": {}}
    )

    assert [area for area, _ in areas(measure(prediction))] == [300.0, 30.0]


def empty_dense():
    empty = dense(torch.zeros((0, *SIZE_HW), dtype=torch.bool), xyxy=[])

    return empty


@pytest.mark.parametrize(
    "make_empty",
    [lambda: boxes([]), empty_dense, lambda: rle(empty_dense())],
    ids=["boxes", "dense", "rle"],
)
def test_empty_predictions_stay_empty(make_empty) -> None:
    measured = measure(make_empty())

    assert len(measured.xyxy) == 0
    assert measured.bboxes_metadata == []


def test_input_rows_and_tensors_are_not_modified() -> None:
    rows = [{"id": "a", "kept": 1}, None]
    prediction = dense(rows=rows)
    mask_before = prediction.mask.clone()

    measured = measure(prediction)

    assert prediction.bboxes_metadata == [{"id": "a", "kept": 1}, None]
    assert torch.equal(prediction.mask, mask_before)
    assert measured.mask is prediction.mask
    assert measured.image_metadata is prediction.image_metadata
    assert measured.bboxes_metadata[0] == {
        "id": "a",
        "kept": 1,
        "area_px": 300.0,
        "area_converted": 300.0,
    }


def test_box_areas_keep_float32_arithmetic() -> None:
    # (0.1 - 0) * (0.3 - 0) in float32 is not 0.03 in float64.
    measured = measure(boxes([[0.0, 0.0, 0.1, 0.3]]))
    expected = float(np.float32(0.1) * np.float32(0.3))

    assert areas(measured)[0][0] == expected


@pytest.mark.parametrize("value", [0, -2.0])
def test_non_positive_pixels_per_unit_is_rejected(value) -> None:
    with pytest.raises(ValidationError):
        MaskAreaMeasurementBlock.Params(
            predictions="$steps.model.predictions", pixels_per_unit=value
        )


def test_module_names_its_metadata_keys() -> None:
    assert (mask_area_module.AREA_KEY, mask_area_module.AREA_CONVERTED_KEY) == (
        "area_px",
        "area_converted",
    )


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_selector_pixels_per_unit_is_checked_when_running(value) -> None:
    # Selector values reach run() without the Params constraints.
    with pytest.raises(
        ValueError, match="pixels_per_unit must be greater than zero and finite"
    ):
        measure(dense(), pixels_per_unit=value)
