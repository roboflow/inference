"""Unit tests of ``roboflow_core/dynamic_zone@v1`` on analytic masks."""

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.geometry.dynamic_zone.v1 import DynamicZoneBlock
from roboflow_workflows.v2.blocks.geometry.mask_inputs import boxes_of_view
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import (
    coco_rle_masks_to_numpy_mask,
    torch_mask_to_coco_rle,
)

# Rectangle pixels x 10..30 and y 10..20 inclusive; its outline runs through
# the outer pixel centres.
RECTANGLE = [[10, 10], [10, 20], [30, 20], [30, 10]]


def frame(height=50, width=50):
    return ImageData.from_tensor(
        torch.zeros((3, height, width), dtype=torch.uint8), image_id="frame"
    )


def instances(masks, *, rows=None):
    count = len(masks.masks) if isinstance(masks, InstancesRLEMasks) else len(masks)
    prediction = InstanceDetections(
        xyxy=torch.tensor([[1.0, 2.0, 3.0, 4.0]] * count).reshape(-1, 4),
        class_id=torch.arange(count),
        confidence=torch.full((count,), 0.9),
        mask=masks,
        image_metadata={**frame().prediction_metadata(), "class_names": {0: "a"}},
        bboxes_metadata=rows,
    )

    return with_row_identity(prediction)


def rectangle_mask(size_hw=(50, 50)):
    mask = torch.zeros(size_hw, dtype=torch.bool)
    mask[10:21, 10:31] = True

    return mask


def run(predictions, vertices=4, **options):
    options = {
        "scale_ratio": 1.0,
        "apply_least_squares": False,
        "midpoint_fraction": 1.0,
        **options,
    }

    return DynamicZoneBlock().run(
        predictions=predictions, required_number_of_vertices=vertices, **options
    )


def as_set(zone):
    return sorted(map(tuple, zone))


def test_rectangle_keeps_its_corners_rows_and_mask():
    masks = rectangle_mask().unsqueeze(0)
    source = instances(masks, rows=[{"name": "court", "_host_xyxy": [0, 0, 1, 1]}])
    before = masks.clone()

    result = run(source)

    (zone,) = result["zones"]
    assert as_set(zone) == as_set(RECTANGLE)
    assert result["simplification_converged"] is True
    zoned = result["predictions"]
    assert torch.equal(zoned.mask, masks)
    assert torch.equal(zoned.xyxy, source.xyxy)
    assert zoned.bboxes_metadata == [{"name": "court", "polygon": zone}]
    assert row_record(zoned) == row_record(source)
    assert torch.equal(masks, before)


def test_disc_becomes_a_convex_quadrilateral_inside_the_disc():
    disc = np.zeros((60, 60), dtype=np.uint8)
    cv2.circle(disc, (30, 30), 20, color=1, thickness=-1)

    result = run(instances(torch.from_numpy(disc.astype(bool)).unsqueeze(0)))

    zone = np.array(result["zones"][0], dtype=np.int32)
    assert zone.shape == (4, 2)
    assert cv2.isContourConvex(zone)
    radii = np.linalg.norm(zone - 30, axis=1)
    assert np.all((radii >= 19) & (radii <= 20.5))
    filled = result["predictions"].mask[0].numpy()
    assert filled.sum() < disc.sum() and not (filled & ~disc.astype(bool)).any()


def test_unreachable_vertex_count_pads_with_the_last_vertex():
    result = run(instances(rectangle_mask().unsqueeze(0)), vertices=6)

    (zone,) = result["zones"]
    assert as_set(zone[:4]) == as_set(RECTANGLE)
    assert zone[4:] == [zone[3], zone[3]]
    assert result["simplification_converged"] is False


def test_scale_ratio_grows_the_zone_about_its_centroid_and_its_mask():
    result = run(instances(rectangle_mask().unsqueeze(0)), scale_ratio=2.0)

    # Centroid (20, 15): x 10..30 -> 0..40, y 10..20 -> 5..25.
    assert as_set(result["zones"][0]) == [(0, 5), (0, 25), (40, 5), (40, 25)]
    assert int(result["predictions"].mask[0].sum()) == 41 * 21


def test_zone_mask_is_clipped_to_the_frame():
    result = run(instances(rectangle_mask((25, 35)).unsqueeze(0)), scale_ratio=3.0)

    mask = result["predictions"].mask
    assert mask.shape == (1, 25, 35) and bool(mask[0].all())


def rounded_rectangle_mask(radius):
    # Pixels x 10..60, y 10..40 with each corner rounded by ``radius``.
    ys, xs = np.mgrid[0:60, 0:80]
    mask = (xs >= 10) & (xs <= 60) & (ys >= 10) & (ys <= 40)
    for cx, cy in ((16, 16), (54, 16), (54, 34), (16, 34)):
        in_corner = (np.abs(xs - cx) <= radius) & (np.abs(ys - cy) <= radius)
        beyond = (np.abs(xs - 35) > 19) & (np.abs(ys - 25) > 9)
        outside = (xs - cx) ** 2 + (ys - cy) ** 2 > radius**2
        mask &= ~(in_corner & beyond & outside)

    return torch.from_numpy(mask)


def worst_corner_error(zone, corners):
    vertices = np.array(zone)

    return max(np.abs(vertices - corner).max(axis=1).min() for corner in corners)


def test_least_squares_keeps_exact_corners_and_sharpens_rounded_ones():
    masks = torch.stack([rectangle_mask((60, 80)), rounded_rectangle_mask(6)])
    sharp = [[10, 10], [60, 10], [60, 40], [10, 40]]

    plain = run(instances(masks))
    fitted = run(instances(masks), apply_least_squares=True)

    # The rectangle's vertical edges are fitted too (V1's y = a x + b was not).
    assert as_set(fitted["zones"][0]) == as_set(RECTANGLE)
    # Hull vertices sit on the arcs; the fitted edge lines meet near the corners.
    assert worst_corner_error(plain["zones"][1], sharp) == 3
    assert worst_corner_error(fitted["zones"][1], sharp) == 1
    assert fitted["simplification_converged"] is True


def holed_rectangle_mask():
    # Pixels x 20..99, y 20..69 minus a disc of radius 12 at (60, 45): a
    # top-down court with a player inside its mask.
    ys, xs = np.mgrid[0:90, 0:120]
    rectangle = (xs >= 20) & (xs <= 99) & (ys >= 20) & (ys <= 69)
    hole = (xs - 60) ** 2 + (ys - 45) ** 2 <= 12**2

    return torch.from_numpy(rectangle & ~hole)


@pytest.mark.parametrize("apply_least_squares", [False, True])
def test_hole_does_not_replace_the_outline(apply_least_squares):
    # The straight outline compresses to 4 points; the round hole has more.
    # V1 took the hole's hull. The zone must be the outline's corners.
    masks = holed_rectangle_mask().unsqueeze(0)

    result = run(instances(masks), apply_least_squares=apply_least_squares)

    (zone,) = result["zones"]
    assert as_set(zone) == as_set([[20, 20], [99, 20], [99, 69], [20, 69]])
    assert result["simplification_converged"] is True
    assert int(result["predictions"].mask[0].sum()) == 80 * 50


def test_rows_without_a_contour_keep_their_mask_and_clear_convergence():
    masks = torch.zeros((3, 50, 50), dtype=torch.bool)
    masks[0] = rectangle_mask()
    masks[2, 5, 5] = True  # one pixel: no three-point contour

    result = run(instances(masks, rows=[{"k": 0}, {"k": 1}, {"k": 2}]))

    assert result["zones"][1:] == [None, None]
    assert as_set(result["zones"][0]) == as_set(RECTANGLE)
    assert torch.equal(result["predictions"].mask[1:], masks[1:])
    assert [
        row["polygon"] is None for row in result["predictions"].bboxes_metadata
    ] == [
        False,
        True,
        True,
    ]
    assert result["simplification_converged"] is False


def test_empty_predictions_converge_with_no_zones():
    source = instances(torch.zeros((0, 50, 50), dtype=torch.bool))

    result = run(source)

    assert result["zones"] == []
    assert result["simplification_converged"] is True
    assert result["predictions"].mask.shape == (0, 50, 50)


def test_uint8_noncontiguous_masks_keep_dtype_and_layout_semantics():
    masks = rectangle_mask((50, 50)).T.unsqueeze(0).to(torch.uint8)
    assert not masks.is_contiguous()

    result = run(instances(masks))

    zoned = result["predictions"].mask
    assert zoned.dtype == torch.uint8 and zoned.device == masks.device
    assert as_set(result["zones"][0]) == [(10, 10), (10, 30), (20, 10), (20, 30)]
    assert torch.equal(zoned, masks)


def test_rle_masks_stay_rle():
    dense = rectangle_mask().unsqueeze(0)
    rle = InstancesRLEMasks.from_coco_rle_masks(
        image_size=(50, 50), masks=[torch_mask_to_coco_rle(dense[0])]
    )

    result = run(instances(rle), scale_ratio=2.0)

    assert isinstance(result["predictions"].mask, InstancesRLEMasks)
    assert coco_rle_masks_to_numpy_mask(result["predictions"].mask).sum() == 41 * 21


def grid_view():
    # A 10 x 10 binary grid at stride 5 over a 50 x 50 image.
    grid = torch.zeros((1, 10, 10), dtype=torch.bool)
    grid[0, 2:5, 2:7] = True
    detections = with_row_identity(
        Detections(
            xyxy=torch.tensor([[10.0, 10.0, 35.0, 25.0]]),
            class_id=torch.tensor([0]),
            confidence=torch.tensor([0.9]),
            image_metadata=frame().prediction_metadata(),
            bboxes_metadata=[{"name": "field"}],
        )
    )
    geometry = MaskGridGeometry(
        grid_size_hw=(10, 10),
        inference_size_hw=(50, 50),
        padding_ltrb=(0, 0, 0, 0),
        scale_xy=(1.0, 1.0),
        crop_size_hw=(50, 50),
        original_size_hw=(50, 50),
        static_crop_xywh=(0, 0, 50, 50),
    )

    return SegmentationView(
        detections=detections,
        scores=grid,
        score_type="binary",
        threshold=0.0,
        geometry=geometry,
    )


def test_segmentation_view_uses_reference_masks_only_when_run():
    view = grid_view()
    boxes_of_view(view)
    assert view.materialization_count == 0

    result = run(view)

    reference = view.full_res()
    expected = run(reference)
    assert result["zones"] == expected["zones"]
    assert torch.equal(result["predictions"].mask, expected["predictions"].mask)
    assert result["predictions"].bboxes_metadata[0]["name"] == "field"
    assert row_record(result["predictions"]) == row_record(view.detections)
    assert view.materialization_count == 1


def test_predictions_without_masks_are_refused():
    source = InstanceDetections(
        xyxy=torch.zeros((0, 4)),
        class_id=torch.zeros(0, dtype=torch.long),
        confidence=torch.zeros(0),
        mask=None,
        image_metadata={},
    )

    with pytest.raises(ValueError, match="instance masks"):
        run(source)


@pytest.mark.parametrize(
    "field, value",
    [
        ("required_number_of_vertices", 2),
        ("required_number_of_vertices", 4.0),
        ("scale_ratio", 0),
        ("midpoint_fraction", 1.5),
        ("apply_least_squares", "yes"),
    ],
)
def test_invalid_literals_are_rejected(field, value):
    literals = {"predictions": "$inputs.p", "required_number_of_vertices": 4}

    with pytest.raises(Exception, match=field):
        DynamicZoneBlock.Params.model_validate({**literals, field: value})


def test_block_names_its_v1_origin():
    assert DynamicZoneBlock.metadata == {
        "name": "Dynamic Zone",
        "port_of": "roboflow_core/dynamic_zone@v1",
        "section": "advanced",
    }
