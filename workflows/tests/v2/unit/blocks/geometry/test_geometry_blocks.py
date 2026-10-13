"""Unit tests of the region/parent-mapping blocks and their helpers.

Expected values are hand-computed from the rules in each module docstring.
"""

import sys

import pytest
import torch
from roboflow_workflows.v2.blocks.geometry import (
    BoundingRectBlock,
    DetectionOffsetBlock,
    DetectionsClassesReplacementBlock,
    DetectionsStitchBlock,
    DimensionCollapseBlock,
)
from roboflow_workflows.v2.blocks.geometry.class_replacement import (
    leading_class,
    leading_classes,
)
from roboflow_workflows.v2.blocks.geometry.frames import (
    offset_between,
    translate_prediction,
)
from roboflow_workflows.v2.blocks.geometry.overlap import (
    merge_groups,
    non_max_merge,
    non_max_suppression,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import SegmentationView
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.data.row_identity import (
    RowRef,
    row_record,
    with_classification_parent,
    with_row_identity,
)

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import (
    coco_rle_masks_to_numpy_mask,
    torch_mask_to_coco_rle,
)


def root_image(height=60, width=80, image_id="root"):
    pixels = torch.zeros((3, height, width), dtype=torch.uint8)

    return ImageData.from_tensor(pixels, image_id=image_id)


def detections(source, xyxy, *, confidence=None, class_id=None, mask=None, rows=None):
    count = len(xyxy)
    fields = dict(
        xyxy=torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.tensor(class_id if class_id is not None else [0] * count),
        confidence=torch.tensor(
            confidence if confidence is not None else [0.9] * count,
            dtype=torch.float32,
        ),
        image_metadata={**source.prediction_metadata(), "class_names": {0: "a"}},
        bboxes_metadata=rows,
    )
    if mask is not None:
        return InstanceDetections(mask=mask, **fields)

    return Detections(**fields)


def rle(dense):
    return InstancesRLEMasks.from_coco_rle_masks(
        image_size=tuple(dense.shape[1:]),
        masks=[torch_mask_to_coco_rle(one) for one in dense],
    )


# --- frames ------------------------------------------------------------------


def test_offset_between_a_slice_and_its_image():
    source = root_image()
    crop = source.crop((10, 5, 50, 45))
    slice_ = crop.crop((20, 0, 40, 40))

    assert offset_between(frame_of(detections(slice_, [])), target=crop) == (20, 0)
    assert offset_between(frame_of(detections(slice_, [])), target=source) == (30, 5)


def test_offset_between_refuses_other_roots_and_scales():
    source = root_image()
    other = root_image(image_id="other")
    resized = source.resize((30, 40))

    with pytest.raises(ValueError, match="different workflow inputs"):
        offset_between(frame_of(detections(other, [])), target=source)
    with pytest.raises(ValueError, match="scaled"):
        offset_between(frame_of(detections(resized, [])), target=source)


@pytest.mark.parametrize("encoding", ["dense", "rle"])
def test_translate_pastes_masks_and_moves_row_metadata(encoding):
    source = root_image(10, 12)
    crop = source.crop((4, 2, 10, 8))
    local = torch.zeros((1, 6, 6), dtype=torch.bool)
    local[0, 0:2, 4:6] = True
    mask = rle(local) if encoding == "rle" else local
    prediction = detections(
        crop,
        [[4, 0, 6, 2]],
        mask=mask,
        rows=[{"polygon": [[4, 0], [6, 0], [6, 2]], "_host_xyxy": [4, 0, 6, 2]}],
    )

    moved = translate_prediction(prediction, offset=(4, 2), target=source)

    expected = torch.zeros((1, 10, 12), dtype=torch.bool)
    expected[0, 2:4, 8:10] = True
    dense = moved.mask
    if encoding == "rle":
        dense = torch.from_numpy(coco_rle_masks_to_numpy_mask(dense).astype(bool))
    assert torch.equal(dense, expected)
    assert moved.xyxy.tolist() == [[8.0, 2.0, 10.0, 4.0]]
    assert moved.bboxes_metadata == [{"polygon": [[8, 2], [10, 2], [10, 4]]}]
    assert frame_of(moved).image_id == "root"
    assert moved.image_metadata["class_names"] == {0: "a"}


def test_translate_into_a_crop_cuts_the_mask_and_widens_integer_points():
    source = root_image(10, 12)
    crop = source.crop((4, 2, 10, 8))
    full = torch.zeros((1, 10, 12), dtype=torch.bool)
    full[0, 1:4, 3:6] = True
    key_points = KeyPoints(
        xy=torch.tensor([[[5, 3]]], dtype=torch.uint8),
        class_id=torch.tensor([0]),
        confidence=torch.ones((1, 1)),
        image_metadata=source.prediction_metadata(),
    )

    moved = translate_prediction(
        (key_points, detections(source, [[3, 1, 6, 4]], mask=full)),
        offset=(-4, -2),
        target=crop,
    )

    assert moved[0].xy.dtype == torch.int64 and moved[0].xy.tolist() == [[[1, 1]]]
    assert moved[1].mask.shape == (1, 6, 6)
    assert torch.nonzero(moved[1].mask[0]).tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]


# --- overlap -----------------------------------------------------------------


def boxes(xyxy, confidence, class_id=None, mask=None):
    return detections(
        root_image(), xyxy, confidence=confidence, class_id=class_id, mask=mask
    )


def test_nms_removes_only_above_threshold_and_keeps_input_order():
    # IoU([0,0,10,10],[0,0,10,5]) = 50 / 100 = 0.5 exactly.
    rows = boxes([[0, 0, 10, 5], [0, 0, 10, 10], [50, 50, 60, 60]], [0.8, 0.9, 0.7])

    at_threshold = non_max_suppression(rows, threshold=0.5, use_masks=False)
    below = non_max_suppression(rows, threshold=0.49, use_masks=False)

    assert len(at_threshold.xyxy) == 3
    assert below.confidence.tolist() == pytest.approx([0.9, 0.7])


def test_nms_and_nmm_are_class_aware():
    rows = boxes([[0, 0, 10, 10], [0, 0, 10, 10]], [0.8, 0.9], class_id=[0, 1])

    assert len(non_max_suppression(rows, threshold=0.1, use_masks=False).xyxy) == 2
    assert len(non_max_merge(rows, threshold=0.1, use_masks=False).xyxy) == 2


def test_nmm_merges_at_threshold_with_union_box_and_area_weighted_confidence():
    rows = boxes([[0, 0, 10, 5], [0, 0, 10, 10], [2, 0, 12, 10]], [0.6, 0.9, 0.1])

    merged = non_max_merge(rows, threshold=0.5, use_masks=False)

    # Candidate [0,0,10,10] (0.9) joins [0,0,10,5] (IoU 0.5) and [2,0,12,10]
    # (IoU 80 / 120 = 0.67): union [0,0,12,10].
    assert merged.xyxy.tolist() == [[0.0, 0.0, 12.0, 10.0]]
    expected = (100 * 0.9 + 50 * 0.6 + 100 * 0.1) / 250
    assert merged.confidence.item() == pytest.approx(expected)


def test_nmm_groups_list_the_most_confident_member_first_per_class():
    rows = boxes(
        [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10], [40, 40, 50, 50]],
        [0.5, 0.9, 0.7, 0.6],
        class_id=[1, 1, 1, 0],
    )

    assert merge_groups(rows, threshold=0.5, use_masks=False) == [[3], [1, 2, 0]]


def test_mask_overlap_keeps_l_shapes_whose_boxes_overlap():
    masks = torch.zeros((2, 20, 20), dtype=torch.bool)
    masks[0, 0:10, 0:2] = True  # vertical bar
    masks[0, 8:10, 0:10] = True
    masks[1, 0:2, 0:10] = True  # horizontal bar
    masks[1, 0:10, 8:10] = True
    rows = boxes([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8], mask=masks)

    by_mask = non_max_suppression(rows, threshold=0.3, use_masks=True)
    by_box = non_max_suppression(rows, threshold=0.3, use_masks=False)

    assert len(by_mask.xyxy) == 2 and len(by_box.xyxy) == 1


# --- stitch ------------------------------------------------------------------


def stitch(reference, children, **options):
    defaults = dict(
        overlap_filtering_strategy="none", iou_threshold=0.3, overlap_by="auto"
    )
    defaults.update(options)

    return DetectionsStitchBlock().run(
        reference_image=reference, predictions=Batch.of(children), **defaults
    )["predictions"]


def test_stitch_concatenates_children_with_aligned_rows_and_masks():
    source = root_image(10, 20)
    left, right = source.crop((0, 0, 10, 10)), source.crop((10, 0, 20, 10))
    left_mask = torch.zeros((1, 10, 10), dtype=torch.bool)
    left_mask[0, 0, 0] = True
    right_mask = torch.zeros((1, 10, 10), dtype=torch.bool)
    right_mask[0, 9, 9] = True

    stitched = stitch(
        source,
        [
            detections(left, [[0, 0, 1, 1]], mask=left_mask, rows=[{"id": "L"}]),
            detections(right, [], mask=torch.zeros((0, 10, 10), dtype=torch.bool)),
            detections(right, [[9, 9, 10, 10]], mask=right_mask, rows=[{"id": "R"}]),
        ],
    )

    assert stitched.xyxy.tolist() == [[0.0, 0.0, 1.0, 1.0], [19.0, 9.0, 20.0, 10.0]]
    assert [row["id"] for row in stitched.bboxes_metadata] == ["L", "R"]
    assert torch.nonzero(stitched.mask).tolist() == [[0, 0, 0], [1, 9, 19]]
    assert frame_of(stitched).image_id == "root"


def test_stitch_of_no_children_is_empty_in_the_reference_frame():
    source = root_image()

    stitched = stitch(source, [])

    assert stitched.xyxy.shape == (0, 4) and frame_of(stitched).image_id == "root"


def test_stitch_rejects_mixed_kinds_mask_overlap_without_masks_and_bad_options():
    source = root_image()
    plain = detections(source.crop((0, 0, 10, 10)), [[0, 0, 5, 5]])
    masked = detections(
        source.crop((0, 0, 10, 10)),
        [[0, 0, 5, 5]],
        mask=torch.ones((1, 10, 10), dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="mix"):
        stitch(source, [plain, masked])
    with pytest.raises(ValueError, match="overlap_by='mask'"):
        stitch(source, [plain], overlap_by="mask")
    with pytest.raises(ValueError, match="overlap_filtering_strategy"):
        stitch(source, [plain], overlap_filtering_strategy="soft")
    assert DetectionsStitchBlock().run(
        reference_image=None,
        predictions=Batch.of([]),
        overlap_filtering_strategy="nms",
        iou_threshold=0.3,
        overlap_by="auto",
    ) == {"predictions": None}


# --- offset ------------------------------------------------------------------


def offset(prediction, **options):
    defaults = dict(offset_width=5, offset_height=5, units="Pixels")
    defaults.update(options)

    return DetectionOffsetBlock().run(predictions=prediction, **defaults)["predictions"]


def test_offset_in_pixels_and_percent_with_clipping():
    source = root_image(60, 80)
    rows = detections(source, [[10, 10, 31, 20], [1, 1, 79, 59]])

    pixels = offset(rows)
    percent = offset(rows, offset_width=25, offset_height=25, units="Percent (%)")

    # Pixels: 5 // 2 = 2 per side; second box clips to 0..80 x 0..60.
    assert pixels.xyxy.tolist() == [[8, 8, 33, 22], [0, 0, 80, 60]]
    # Percent: trunc(21 * 25 / 200) = 2, trunc(10 * 25 / 200) = 1.
    assert percent.xyxy[0].tolist() == [8, 9, 33, 21]


def test_offset_renews_ids_keeps_alignment_and_keypoints():
    source = root_image()
    rows = detections(
        source,
        [[10, 10, 20, 20]],
        rows=[{"detection_id": "d1", "_host_xyxy": [1], "tag": "x"}],
    )
    key_points = KeyPoints(
        xy=torch.tensor([[[12.0, 12.0]]]),
        class_id=torch.tensor([0]),
        confidence=torch.ones((1, 1)),
        image_metadata=source.prediction_metadata(),
    )

    moved_points, padded = offset((key_points, rows))

    assert moved_points is key_points
    (row,) = padded.bboxes_metadata
    assert row["parent_id"] == "d1" and row["detection_id"] != "d1"
    assert row["tag"] == "x" and "_host_xyxy" not in row
    assert padded.class_id is rows.class_id


def test_offset_without_a_frame_is_an_error_and_empty_passes_through():
    no_frame = Detections(
        xyxy=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.5]),
        image_metadata={},
    )
    empty = detections(root_image(), [])

    with pytest.raises(Exception):
        offset(no_frame)
    assert offset(empty) is empty


def skeletons(count, *, with_boxes):
    source = root_image()
    key_points = KeyPoints(
        xy=torch.ones((count, 2, 2)),
        class_id=torch.zeros(count, dtype=torch.int64),
        confidence=torch.ones((count, 2)),
        image_metadata=source.prediction_metadata(),
    )
    boxes = detections(source, [[0, 0, 5, 5]] * count) if with_boxes else None

    return with_row_identity((key_points, boxes))


@pytest.mark.parametrize("count", [4, 0])
def test_offset_refuses_keypoints_without_boxes(count):
    # Decision 018: no boxes is a model property, even for a frame without rows.
    with pytest.raises(ContractError, match="v2/detection_offset@v1: the keypoint"):
        offset(skeletons(count, with_boxes=False))


def test_offset_of_keypoints_with_zero_boxes_passes_through():
    empty = skeletons(0, with_boxes=True)

    assert offset(empty) is empty


# --- bounding rect -----------------------------------------------------------


def test_bounding_rect_of_an_axis_aligned_block_and_an_empty_mask():
    masks = torch.zeros((2, 20, 30), dtype=torch.bool)
    masks[0, 5:10, 4:16] = True
    rows = detections(root_image(20, 30), [[4, 5, 16, 10], [0, 0, 1, 1]], mask=masks)

    fitted = BoundingRectBlock().run(predictions=rows)["predictions"]

    first, second = fitted.bboxes_metadata
    # The outline runs through pixel centres 4..15 x 5..9.
    assert sorted(map(tuple, first["rect"])) == [(4, 5), (4, 9), (15, 5), (15, 9)]
    assert {first["width"], first["height"]} == {11.0, 4.0}
    assert fitted.xyxy[0].tolist() == [4, 5, 15, 9]
    assert int(fitted.mask[0].sum()) == 12 * 5
    assert second == {"rect": None, "width": None, "height": None, "angle": None}
    assert fitted.xyxy[1].tolist() == [0, 0, 1, 1]


def test_bounding_rect_keeps_rle_masks_rle_and_requires_masks():
    dense = torch.zeros((1, 10, 10), dtype=torch.bool)
    dense[0, 2:5, 2:8] = True
    source = root_image(10, 10)

    fitted = BoundingRectBlock().run(
        predictions=detections(source, [[2, 2, 8, 5]], mask=rle(dense))
    )["predictions"]

    assert isinstance(fitted.mask, InstancesRLEMasks)
    assert coco_rle_masks_to_numpy_mask(fitted.mask).sum() == 18
    with pytest.raises(ValueError, match="instance masks"):
        BoundingRectBlock().run(
            predictions=InstanceDetections(
                xyxy=torch.zeros((0, 4)),
                class_id=torch.zeros(0, dtype=torch.long),
                confidence=torch.zeros(0),
                mask=None,
                image_metadata={},
            )
        )


@pytest.mark.parametrize("as_rle", [False, True])
@pytest.mark.parametrize("with_hole", [False, True])
def test_bounding_rect_follows_the_outline_and_preserves_rows(as_rle, with_hole):
    masks = torch.zeros((3, 90, 120), dtype=torch.bool)
    masks[0, 20:70, 20:100] = True
    masks[2, 5:10, 5:15] = True
    if with_hole:
        ys, xs = torch.meshgrid(torch.arange(90), torch.arange(120), indexing="ij")
        masks[0] &= (xs - 60) ** 2 + (ys - 45) ** 2 > 12**2
    expected_masks = masks.clone()
    expected_masks[0, 20:70, 20:100] = True
    rows = with_row_identity(
        detections(
            root_image(90, 120),
            [[20, 20, 100, 70], [1, 2, 3, 4], [5, 5, 15, 10]],
            confidence=[0.7, 0.8, 0.9],
            class_id=[2, 1, 0],
            mask=rle(masks) if as_rle else masks,
            rows=[{"tag": "outline"}, {"tag": "empty"}, {"tag": "control"}],
        )
    )
    reordered = select_predictions(rows, indices=[2, 1, 0])

    fitted = BoundingRectBlock().run(predictions=reordered)["predictions"]

    assert fitted.xyxy.tolist() == [[5, 5, 14, 9], [1, 2, 3, 4], [20, 20, 99, 69]]
    assert isinstance(fitted.mask, InstancesRLEMasks) == as_rle
    actual_masks = (
        torch.from_numpy(coco_rle_masks_to_numpy_mask(fitted.mask))
        if as_rle
        else fitted.mask
    )
    assert torch.equal(actual_masks, expected_masks[[2, 1, 0]])
    assert sorted(map(tuple, fitted.bboxes_metadata[2]["rect"])) == [
        (20, 20),
        (20, 69),
        (99, 20),
        (99, 69),
    ]
    assert {
        fitted.bboxes_metadata[2]["width"],
        fitted.bboxes_metadata[2]["height"],
    } == {79.0, 49.0}
    assert fitted.bboxes_metadata[2]["angle"] == 90.0
    assert fitted.bboxes_metadata[1] == {
        "tag": "empty",
        "rect": None,
        "width": None,
        "height": None,
        "angle": None,
    }
    assert [row["tag"] for row in fitted.bboxes_metadata] == [
        "control",
        "empty",
        "outline",
    ]
    assert fitted.image_metadata == reordered.image_metadata
    assert torch.equal(fitted.class_id, reordered.class_id)
    assert torch.equal(fitted.confidence, reordered.confidence)
    assert row_record(fitted).host_ids() == [2, 1, 0]
    assert row_record(fitted).namespace == row_record(rows).namespace
    assert rows.bboxes_metadata == [
        {"tag": "outline"},
        {"tag": "empty"},
        {"tag": "control"},
    ]
    assert torch.equal(
        (
            torch.from_numpy(coco_rle_masks_to_numpy_mask(rows.mask))
            if as_rle
            else rows.mask
        ),
        masks,
    )


def test_bounding_rect_keeps_the_most_points_choice_among_outer_contours():
    masks = torch.zeros((1, 60, 80), dtype=torch.bool)
    masks[0, 20:55, 30:75] = True  # larger rectangle: four contour points
    masks[0, 2:12, 2:4] = True
    masks[0, 10:12, 2:12] = True  # smaller L: six contour points
    rows = detections(root_image(), [[2, 2, 75, 55]], mask=masks)

    fitted = BoundingRectBlock().run(predictions=rows)["predictions"]

    assert fitted.xyxy.tolist() == [[2, 2, 11, 11]]
    expected = torch.zeros_like(masks)
    expected[0, 2:12, 2:12] = True
    assert torch.equal(fitted.mask, expected)


# --- class replacement -------------------------------------------------------


def single_label(confidence, names=None):
    confidence = torch.tensor([confidence])
    return ClassificationPrediction(
        class_id=torch.argmax(confidence, dim=1),
        confidence=confidence,
        images_metadata=[{"class_names": names or {0: "cat", 1: "dog"}}],
    )


def test_leading_class_rules():
    multi = MultiLabelClassificationPrediction(
        class_ids=torch.tensor([0]),
        confidence=torch.tensor([0.6, 0.8]),
        image_metadata={"class_names": {0: "cat", 1: "dog"}},
    )
    nothing = MultiLabelClassificationPrediction(
        class_ids=torch.tensor([], dtype=torch.long),
        confidence=torch.tensor([0.4, 0.3]),
        image_metadata={},
    )

    assert leading_class(single_label([0.2, 0.8])) == ("dog", 1, pytest.approx(0.8))
    # Multi-label: the whole distribution decides once any label is predicted.
    assert leading_class(multi) == ("dog", 1, pytest.approx(0.8))
    assert leading_class(nothing) is None
    assert leading_class("K619") == ("K619", 0, 1.0)
    assert leading_class(["K619", "x"]) == ("K619", 0, 1.0)
    assert leading_class([]) is None
    empty = ClassificationPrediction(
        class_id=torch.tensor([0]),
        confidence=torch.zeros((1, 0)),
        images_metadata=[{}],
    )
    assert leading_class(empty) is None


def replace_classes(rows, children, *, name=None, class_id=None, crops=None):
    # A Group delivers full indices under the parent (0,), position by position.
    batch = Batch(
        list(children),
        indices=[(0, k) for k in range(len(children))],
        parent_index=(0,),
    )

    return DetectionsClassesReplacementBlock().run(
        object_detection_predictions=rows,
        classification_predictions=batch,
        crops=crops,
        fallback_class_name=name,
        fallback_class_id=class_id,
    )["predictions"]


def classified(prediction, targets, row_id):
    # What a classifier does with a crop of ``targets`` row ``row_id``.
    parent = RowRef(namespace=row_record(targets).namespace, id=row_id)

    return with_classification_parent(prediction, parent)


def crops_for(targets, row_ids):
    record = row_record(targets)
    crops = [
        root_image(10, 10, image_id=f"crop-{k}").with_parent_row(
            RowRef(namespace=record.namespace, id=row_id)
        )
        for k, row_id in enumerate(row_ids)
    ]
    batch = Batch(crops, indices=[(0, k) for k in range(len(crops))], parent_index=(0,))

    return batch


def three_rows():
    source = root_image()
    rows = detections(
        source,
        [[0, 0, 5, 5], [5, 5, 9, 9], [10, 10, 20, 20]],
        rows=[{"n": 0}, {"n": 1}, {"n": 2, "_host_class_id": 0}],
    )

    return with_row_identity(rows)


def test_replacement_joins_by_row_key_and_applies_the_fallback():
    rows = three_rows()
    # Results arrive in another order than the rows; text binds through crops.
    children = ["plate", classified(single_label([0.9, 0.1]), rows, 0)]
    crops = crops_for(rows, [2, 0])

    with_fallback = replace_classes(
        rows, children, name="unknown", class_id=-3, crops=crops
    )
    without = replace_classes(rows, children, crops=crops)

    assert [r["class"] for r in with_fallback.bboxes_metadata] == [
        "cat",
        "unknown",
        "plate",
    ]
    assert with_fallback.class_id.tolist() == [0, sys.maxsize, 0]
    assert with_fallback.confidence.tolist() == pytest.approx([0.9, 0.0, 1.0])
    # Ids 0 belong to "cat" first; the per-row name says "plate".
    assert with_fallback.image_metadata["class_names"] == {
        0: "cat",
        sys.maxsize: "unknown",
    }
    assert "_host_class_id" not in with_fallback.bboxes_metadata[2]
    assert [r["n"] for r in without.bboxes_metadata] == [0, 2]
    assert torch.equal(without.xyxy, rows.xyxy[[0, 2]])
    # Relabelling is one-to-one: kept rows keep their row keys.
    assert row_record(without).namespace == row_record(rows).namespace
    assert row_record(without).host_ids() == [0, 2]


def test_replacement_follows_rows_through_filter_and_sort():
    rows = three_rows()
    # [A, B, C] -> keep C and A, C first -> crop 0 is C, crop 1 is A.
    cropped = select_predictions(rows, indices=[2, 0])
    children = [
        classified(
            single_label([0.1, 0.9]), cropped, row_record(cropped).host_ids()[0]
        ),
        classified(
            single_label([0.8, 0.2]), cropped, row_record(cropped).host_ids()[1]
        ),
    ]

    on_original = replace_classes(rows, children, name="unknown")
    on_selection = replace_classes(cropped, children)

    assert [r["class"] for r in on_original.bboxes_metadata] == [
        "cat",
        "unknown",
        "dog",
    ]
    assert [r["class"] for r in on_selection.bboxes_metadata] == ["dog", "cat"]


def test_repeated_rows_share_one_result_and_stale_results_are_ignored():
    rows = three_rows()
    repeated = select_predictions(rows, indices=[1, 1])
    children = [
        classified(single_label([0.1, 0.9]), rows, 1),
        classified(single_label([0.9, 0.1]), rows, 2),
    ]

    replaced = replace_classes(repeated, children)

    assert [r["class"] for r in replaced.bboxes_metadata] == ["dog", "dog"]


def test_replacement_refuses_results_it_cannot_join():
    rows = three_rows()
    other = three_rows()
    unbound = single_label([0.9, 0.1])

    with pytest.raises(ContractError, match="carries no detection row; classify"):
        replace_classes(rows, [unbound])
    with pytest.raises(ContractError, match="with_classification_parent"):
        replace_classes(rows, [unbound])
    with pytest.raises(ContractError, match="bind 'crops'"):
        replace_classes(rows, ["plate"])
    with pytest.raises(ContractError, match="other predictions"):
        replace_classes(rows, [classified(unbound, other, 0)])
    with pytest.raises(ContractError, match="both for row 1"):
        replace_classes(rows, ["a", "b"], crops=crops_for(rows, [1, 1]))
    with pytest.raises(ContractError, match="was made for row"):
        replace_classes(
            rows, [classified(unbound, rows, 0)], crops=crops_for(rows, [1])
        )
    without_identity = detections(root_image(), [[0, 0, 5, 5]])
    with pytest.raises(ContractError, match="detections_classes_replacement"):
        replace_classes(without_identity, [classified(unbound, rows, 0)])
    # The targets lack identity: the error names them, not the classifier.
    with pytest.raises(ContractError, match="with_row_identity"):
        replace_classes(without_identity, [unbound])
    unlabelled_crop = Batch([root_image(10, 10)], indices=[(0, 0)], parent_index=(0,))
    with pytest.raises(ContractError, match="has no parent_row"):
        replace_classes(rows, ["plate"], crops=unlabelled_crop)


def test_replacement_of_empty_and_absent_inputs():
    rows = three_rows()
    block = DetectionsClassesReplacementBlock()

    def run(targets, group, name=None):
        return block.run(
            object_detection_predictions=targets,
            classification_predictions=group,
            crops=None,
            fallback_class_name=name,
            fallback_class_id=None,
        )["predictions"]

    assert run(None, Batch.of([])) is None
    absent_group = run(rows, None, name="unknown")
    assert [r["class"] for r in absent_group.bboxes_metadata] == ["unknown"] * 3
    assert len(run(rows, Batch.of([])).xyxy) == 0
    # Without results nothing is joined, so row identity is not needed.
    plain = detections(root_image(), [[0, 0, 5, 5]])
    assert len(run(plain, Batch.of([]), name="x").xyxy) == 1


def test_replacement_keeps_keypoints_aligned_with_kept_rows():
    source = root_image()
    key_points = KeyPoints(
        xy=torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]]]),
        class_id=torch.tensor([0, 0]),
        confidence=torch.ones((2, 1)),
        image_metadata=source.prediction_metadata(),
    )
    pair = with_row_identity(
        (key_points, detections(source, [[0, 0, 5, 5], [5, 5, 9, 9]]))
    )

    kept_points, kept = replace_classes(
        pair, [classified(single_label([0.9, 0.1]), pair, 1)]
    )

    assert kept_points.xy.tolist() == [[[2.0, 2.0]]]
    assert kept.xyxy.tolist() == [[5.0, 5.0, 9.0, 9.0]]


@pytest.mark.parametrize("children", [[], ["red"]])
def test_replacement_refuses_keypoints_without_boxes(children):
    with pytest.raises(
        ContractError, match="v2/detections_classes_replacement@v1: the keypoint"
    ):
        replace_classes(skeletons(4, with_boxes=False), children, name="unknown")


def test_leading_classes_equal_leading_class_one_by_one():
    values = [
        single_label([0.2, 0.8]),
        "K619",
        [],
        MultiLabelClassificationPrediction(
            class_ids=torch.tensor([0]),
            confidence=torch.tensor([0.0, 0.0]),
            image_metadata={},
        ),
        single_label([0.7, 0.3]),
    ]

    assert leading_classes(values) == [leading_class(value) for value in values]
    assert leading_classes(values)[3] is None


# --- collapse ----------------------------------------------------------------


def collapse(survivors, *, known, parent=(0,)):
    group = Batch(
        [value for _, value in survivors],
        indices=[(*parent, k) for k, _ in survivors],
        known_indices=[(*parent, k) for k in known],
        parent_index=parent,
    )

    return DimensionCollapseBlock().run(data=group)["output"]


def test_collapse_keeps_skipped_children_as_none():
    assert collapse([(0, "a"), (2, "c")], known=range(4)) == ["a", None, "c", None]
    assert collapse([], known=range(2)) == [None, None]
    assert collapse([(0, "a"), (1, "b")], known=range(2)) == ["a", "b"]
    # Positions are local to the exact parent of a nested group.
    assert collapse([(1, "b")], known=range(3), parent=(1, 4)) == [None, "b", None]


def test_collapse_of_true_empty_and_absent_groups():
    assert DimensionCollapseBlock().run(data=Batch.of([])) == {"output": []}
    assert DimensionCollapseBlock().run(data=None) == {"output": None}


# --- row identity and views across the geometry blocks ------------------------


def slice_children(source):
    left, right = source.crop((0, 0, 10, 10)), source.crop((10, 0, 20, 10))
    masks = torch.zeros((1, 10, 10), dtype=torch.bool)
    masks[0, 2:5, 2:5] = True
    children = [
        with_row_identity(detections(left, [[2, 2, 5, 5]], mask=masks)),
        with_row_identity(detections(right, [[2, 2, 5, 5]], mask=masks.clone())),
    ]

    return children


def test_stitched_rows_are_a_new_result_with_a_new_identity():
    source = root_image(10, 20)
    children = slice_children(source)

    stitched = stitch(source, children)

    record = row_record(stitched)
    assert record.rows == 2 and record.host_ids() == [0, 1]
    assert record.namespace not in {row_record(c).namespace for c in children}
    assert row_record(stitch(source, children, overlap_filtering_strategy="nmm")).rows


def test_stitch_of_views_equals_stitch_of_their_predictions():
    source = root_image(10, 20)
    children = slice_children(source)

    from_views = stitch(source, [SegmentationView.from_dense(c) for c in children])
    from_dense = stitch(source, children)

    assert torch.equal(from_views.xyxy, from_dense.xyxy)
    assert torch.equal(from_views.mask, from_dense.mask)


def test_stitch_of_an_absent_group_is_empty_and_without_reference_absent():
    source = root_image()
    options = dict(
        overlap_filtering_strategy="nms", iou_threshold=0.3, overlap_by="auto"
    )

    empty = DetectionsStitchBlock().run(
        reference_image=source, predictions=None, **options
    )["predictions"]
    absent = DetectionsStitchBlock().run(
        reference_image=None, predictions=None, **options
    )["predictions"]

    assert empty.xyxy.shape == (0, 4) and frame_of(empty).image_id == "root"
    assert row_record(empty).rows == 0
    assert absent is None


def test_offset_and_bounding_rect_keep_row_keys():
    masks = torch.zeros((2, 20, 30), dtype=torch.bool)
    masks[0, 5:10, 4:16] = True
    masks[1, 1:3, 1:3] = True
    rows = with_row_identity(
        detections(root_image(20, 30), [[4, 5, 16, 10], [1, 1, 3, 3]], mask=masks)
    )
    reordered = select_predictions(rows, indices=[1, 0])

    assert row_record(offset(reordered)).host_ids() == [1, 0]
    fitted = BoundingRectBlock().run(predictions=reordered)["predictions"]
    assert row_record(fitted).host_ids() == [1, 0]
    assert row_record(fitted).namespace == row_record(rows).namespace


def test_bounding_rect_of_a_view_equals_its_predictions():
    masks = torch.zeros((1, 20, 30), dtype=torch.bool)
    masks[0, 5:10, 4:16] = True
    rows = detections(root_image(20, 30), [[4, 5, 16, 10]], mask=masks)

    from_view = BoundingRectBlock().run(predictions=SegmentationView.from_dense(rows))
    from_dense = BoundingRectBlock().run(predictions=rows)

    assert torch.equal(from_view["predictions"].mask, from_dense["predictions"].mask)
    assert (
        from_view["predictions"].bboxes_metadata
        == from_dense["predictions"].bboxes_metadata
    )
