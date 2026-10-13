"""Analytic and latest V1 reference checks for crop rollup."""

import importlib

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils
from roboflow_workflows.v2.blocks.fusion.rollup import DetectionsListRollUpBlock
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks


def image(size=(80, 120)):
    return ImageData.from_tensor(
        torch.zeros((3, *size), dtype=torch.uint8), image_id="root"
    )


def det(img, boxes, *, scores=None, classes=None, metadata=None, rows=None):
    n = len(boxes)
    return Detections(
        xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.tensor(
            classes if classes is not None else [0] * n, dtype=torch.int64
        ),
        confidence=torch.tensor(scores if scores is not None else [0.8] * n),
        image_metadata={
            **(img.prediction_metadata() if img is not None else {}),
            **(metadata or {}),
        },
        bboxes_metadata=rows if rows is not None else [{} for _ in boxes],
    )


def seg(img, boxes, *, masks=None, rle=False, **kwargs):
    d = det(img, boxes, **kwargs)
    if masks is None:
        masks = torch.zeros((len(boxes), *img.size_hw), dtype=torch.bool)
        for mask, (x0, y0, x1, y1) in zip(masks, boxes):
            mask[int(y0) : int(y1), int(x0) : int(x1)] = True
    if rle:
        encoded = [
            mask_utils.encode(np.asfortranarray(m.numpy().astype(np.uint8)))
            for m in masks
        ]
        masks = InstancesRLEMasks.from_coco_rle_masks(img.size_hw, encoded)
    return InstanceDetections(**d.__dict__, mask=masks)


def kp(img, boxes, points, **kwargs):
    d = det(img, boxes, **kwargs)
    xy = torch.tensor(points, dtype=torch.float32)
    k = KeyPoints(
        xy=xy,
        class_id=d.class_id,
        confidence=torch.ones(xy.shape[:2]),
        image_metadata=dict(d.image_metadata),
    )
    return k, d


def roll(parent, children, **kwargs):
    options = dict(
        confidence_strategy="max",
        overlap_threshold=0.0,
        keypoint_merge_threshold=10.0,
        raise_on_class_name_conflict=False,
    )
    options.update(kwargs)
    return DetectionsListRollUpBlock().run(
        parent_detection=parent, child_detections=children, **options
    )["rolled_up_detections"]


def dense(prediction):
    if isinstance(prediction.mask, InstancesRLEMasks):
        return torch.from_numpy(
            mask_utils.decode(prediction.mask.to_coco_rle_masks()).transpose(2, 0, 1)
        ).bool()
    return prediction.mask


@pytest.mark.parametrize("family", ["boxes", "dense", "rle", "view"])
@pytest.mark.parametrize(
    "strategy,expected", [("max", 0.8), ("mean", 0.6), ("min", 0.4)]
)
def test_translation_and_confidence_match_latest_v1(family, strategy, expected):
    img = image()
    parent = det(img, [[10, 20, 40, 50], [20, 20, 50, 50]])
    crops = [img.crop([int(v) for v in box]) for box in parent.xyxy.tolist()]
    boxes = [[[5, 5, 25, 20]], [[0, 5, 20, 20]]]
    make = det if family == "boxes" else seg
    children = [
        make(crop, box, scores=[score])
        for crop, box, score in zip(crops, boxes, [0.4, 0.8])
    ]
    reference_children = children
    if family == "rle":
        children = [
            seg(crop, box, scores=[score], rle=True)
            for crop, box, score in zip(crops, boxes, [0.4, 0.8])
        ]
    elif family == "view":
        children = [SegmentationView.from_dense(child) for child in children]
    result = roll(parent, children, confidence_strategy=strategy)
    # Legacy import is a test-only oracle, never a runtime dependency.
    v1 = importlib.import_module(
        "roboflow_workflows.core_steps.fusion.detections_list_rollup.v1_tensor"
    )
    reference, _ = v1.merge_crop_predictions(
        parent, reference_children, strategy, 0.0, 10.0
    )
    torch.testing.assert_close(result.xyxy, reference.xyxy)
    torch.testing.assert_close(result.confidence, reference.confidence)
    assert result.confidence.tolist() == pytest.approx([expected])
    assert result.xyxy.tolist() == [[15, 25, 40, 40]]
    if family != "boxes":
        assert torch.equal(dense(result), reference.mask)
        assert dense(result).sum() == 375
    if family == "rle":
        assert isinstance(result.mask, InstancesRLEMasks)


@pytest.mark.parametrize(
    "boxes,threshold,expected",
    [
        ([[0, 0, 10, 10], [10, 0, 20, 10]], 0, 2),
        ([[0, 0, 10, 10], [0, 0, 10, 20]], 0.5, 1),
        ([[0, 0, 10, 10], [0, 0, 10, 20]], 0.5001, 2),
        ([[0, 0, 10, 10], [9, 0, 19, 10], [18, 0, 28, 10]], 0, 1),
    ],
)
def test_box_touching_inclusive_iou_and_transitive_groups(boxes, threshold, expected):
    img = image()
    result = roll(
        det(img, [[0, 0, 120, 80]]), [det(img, boxes)], overlap_threshold=threshold
    )
    assert len(result.xyxy) == expected
    if expected == 1:
        assert result.xyxy[0].tolist() == [
            0,
            0,
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ]


def test_mask_box_prefilter_and_empty_unions():
    img = image((20, 20))
    masks = torch.zeros((3, 20, 20), dtype=torch.bool)
    masks[:2, 4:8, 4:8] = True
    # Identical masks with disjoint declared boxes remain separate, as V1.
    child = seg(img, [[0, 0, 5, 5], [5, 0, 10, 5], [0, 0, 20, 20]], masks=masks)
    result = roll(det(img, [[0, 0, 20, 20]]), [child])
    assert result.xyxy.tolist() == [[4, 4, 8, 8], [4, 4, 8, 8]]
    assert dense(result).sum() == 32


@pytest.mark.parametrize("family", ["boxes", "dense", "rle", "keypoints"])
def test_empty_children_keep_family_and_parent_canvas(family):
    img = image()
    parent = det(img, [[0, 0, 20, 20], [0, 0, 20, 20]])
    if family == "boxes":
        child = det(img, [])
    elif family == "keypoints":
        child = (
            KeyPoints(
                torch.zeros(0, 2, 2),
                torch.zeros(0, dtype=torch.int64),
                torch.zeros(0, 2),
                img.prediction_metadata(),
            ),
            det(img, []),
        )
    else:
        child = seg(img, [], rle=family == "rle")
    result = roll(parent, [None, child])
    carrier = result[1] if family == "keypoints" else result
    assert carrier.xyxy.shape == (0, 4)
    assert frame_of(carrier).size_hw == img.size_hw
    if family in ("dense", "rle"):
        assert isinstance(result, InstanceDetections)
        assert (
            result.mask.image_size if family == "rle" else result.mask.shape[1:]
        ) == img.size_hw
    if family == "keypoints":
        assert result[0].xy.shape == (0, 2, 2)


def test_keypoints_seed_greedy_strict_threshold_mean_box_and_extras():
    img = image()
    points = [[[0, 1]], [[6, 1]], [[12, 1]]]
    child = kp(
        img,
        [[0, 0, 4, 4], [6, 0, 10, 4], [12, 0, 16, 4]],
        points,
        scores=[0.2, 0.6, 0.9],
    )
    child[0].covariance = torch.eye(2).reshape(1, 1, 2, 2).repeat(3, 1, 1, 1)
    child[0].detection_confidence = torch.tensor([0.2, 0.6, 0.9])
    k, boxes = roll(
        det(img, [[0, 0, 120, 80]]),
        [child],
        keypoint_merge_threshold=10,
        confidence_strategy="mean",
    )
    assert k.xy.tolist() == [[[3, 1]], [[12, 1]]]
    assert boxes.xyxy.tolist() == [[3, 0, 7, 4], [12, 0, 16, 4]]
    assert k.detection_confidence.tolist() == pytest.approx([0.4, 0.9])
    torch.testing.assert_close(
        k.covariance, torch.eye(2).reshape(1, 1, 2, 2).repeat(2, 1, 1, 1)
    )
    strict = roll(det(img, [[0, 0, 120, 80]]), [child], keypoint_merge_threshold=6)
    assert len(strict[1].xyxy) == 3


def test_unequal_keypoint_counts_are_padded_and_never_merged():
    img = image()
    children = [
        kp(img, [[0, 0, 4, 4]], [[[1, 1]]]),
        kp(img, [[0, 0, 4, 4]], [[[1, 1], [2, 2]]]),
    ]
    result, boxes = roll(det(img, [[0, 0, 10, 10]] * 2), children)
    assert len(boxes.xyxy) == 2
    assert result.xy.shape == (2, 2, 2)


@pytest.mark.parametrize(
    "box,expected_origin",
    [([10.6, 20.4, 30, 40], [11, 20]), ([-4, -6, 30, 40], [0, 0])],
)
def test_rounded_clipped_lineage_beats_raw_parent_box(box, expected_origin):
    img = image()
    crop = img.crop([round(v) for v in box])
    result = roll(det(img, [box]), [det(crop, [[1, 2, 5, 6]])])
    x, y = expected_origin
    assert result.xyxy.tolist() == [[x + 1, y + 2, x + 5, y + 6]]
    fallback = roll(det(img, [box]), [det(None, [[1, 2, 5, 6]])])
    assert fallback.xyxy.tolist() == [
        [int(box[0]) + 1, int(box[1]) + 2, int(box[0]) + 5, int(box[1]) + 6]
    ]


@pytest.mark.parametrize("family", ["boxes", "keypoints", "dense", "rle", "view"])
def test_nested_resized_children_map_into_parent_analytically(family):
    root = image()
    # Parent itself is resized: root (10,8) maps to parent (0,0), 2 root px/local px.
    parent_image = root.crop([10, 8, 90, 68]).resize((30, 40))
    child_image = parent_image.crop([4, 3, 24, 23]).resize((10, 10))
    parent = det(parent_image, [[4, 3, 24, 23]])
    if family == "boxes":
        child = det(child_image, [[1, 2, 4, 5]])
    elif family == "keypoints":
        child = kp(child_image, [[1, 2, 4, 5]], [[[2, 3]]])
        child[0].covariance = torch.eye(2).reshape(1, 1, 2, 2)
    else:
        child = seg(child_image, [[1, 2, 4, 5]], rle=family == "rle")
        if family == "view":
            child = SegmentationView.from_dense(child)
    result = roll(parent, [child])
    carrier = result[1] if family == "keypoints" else result
    assert frame_of(carrier) == frame_of(parent)
    assert carrier.xyxy.tolist() == [[6, 7, 12, 13]]
    if family == "keypoints":
        assert result[0].xy.tolist() == [[[8, 9]]]
        torch.testing.assert_close(
            result[0].covariance, 4 * torch.eye(2).reshape(1, 1, 2, 2)
        )
    elif family != "boxes":
        expected = torch.zeros((1, 30, 40), dtype=torch.bool)
        expected[:, 7:13, 6:12] = True
        assert torch.equal(dense(result), expected)


@pytest.mark.parametrize(
    "key",
    [
        "parent_id",
        "parent_frame_id",
        "root_parent_id",
        "scaling_relative_to_parent",
        "scaling_relative_to_root_parent",
    ],
)
def test_partial_known_frame_never_uses_fallback(key):
    img = image()
    with pytest.raises(ValueError):
        roll(
            det(img, [[10, 10, 20, 20]]),
            [det(None, [[0, 0, 1, 1]], metadata={key: "known"})],
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"is_composite": True},
        {"composite_sources": []},
        {"is_composite": False, "composite_sources": []},
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_partial_known_composite_metadata_never_uses_fallback(metadata, empty):
    img = image((20, 20))
    child = det(None, [] if empty else [[0, 0, 2, 2]], metadata=metadata)
    with pytest.raises(ValueError):
        roll(det(img, [[7, 8, 12, 13]]), [child])


def test_conflicting_and_unrelated_frames_rejected():
    img = image()
    child = det(img, [[0, 0, 1, 1]])
    child.image_metadata["root_parent_id"] = "other"
    with pytest.raises(ValueError):
        roll(det(img, [[0, 0, 10, 10]]), [child])
    other = ImageData.from_tensor(
        torch.zeros((3, 10, 10), dtype=torch.uint8), image_id="other"
    )
    with pytest.raises(ValueError, match="root"):
        roll(det(img, [[0, 0, 10, 10]]), [det(other, [[0, 0, 1, 1]])])


def test_metadata_class_order_fallback_conflicts_fresh_identity_and_input_immutability(
    caplog,
):
    img = image()
    first = with_row_identity(
        det(
            img,
            [[0, 0, 5, 5], [20, 0, 25, 5]],
            classes=[2, 0],
            metadata={"class_names": {2: "two", 0: "old"}, "inference_id": "first"},
            rows=[
                {"detection_id": "a", "label": "first", "_host_xyxy": [99]},
                {"label": "second"},
            ],
        )
    )
    second = det(
        img,
        [[40, 0, 45, 5]],
        classes=[0],
        metadata={"class_names": {0: "new"}},
        rows=[{}],
    )
    result = roll(det(img, [[0, 0, 100, 80]] * 2), [first, second])
    assert result.class_id.tolist() == [2, 0, 0]
    assert result.image_metadata["class_names"] == {2: "two", 0: "new"}
    assert [r["class"] for r in result.bboxes_metadata] == ["two", "old", "new"]
    assert [r["label"] for r in result.bboxes_metadata] == ["first", "second", "second"]
    assert result.image_metadata["inference_id"] == "first"
    assert result.image_metadata["prediction_type"] == "object-detection"
    assert all(
        r["detection_id"] != "a" and "_host_xyxy" not in r
        for r in result.bboxes_metadata
    )
    assert "Conflicting class names" in caplog.text
    assert first.bboxes_metadata[0]["detection_id"] == "a"
    assert row_record(result).namespace != row_record(first).namespace
    with pytest.raises(ValueError, match="Conflicting class names"):
        roll(
            det(img, [[0, 0, 100, 80]] * 2),
            [first, second],
            raise_on_class_name_conflict=True,
        )


@pytest.mark.parametrize(
    "options",
    [
        {"confidence_strategy": "bad"},
        {"overlap_threshold": -0.1},
        {"overlap_threshold": 1.1},
        {"overlap_threshold": float("nan")},
        {"keypoint_merge_threshold": -1},
        {"keypoint_merge_threshold": float("nan")},
    ],
)
def test_runtime_parameter_validation(options):
    img = image()
    with pytest.raises(ValueError):
        roll(det(img, []), [], **options)


def test_count_mixed_family_and_missing_keypoint_boxes_errors():
    img = image()
    parent = det(img, [[0, 0, 10, 10]] * 2)
    with pytest.raises(ValueError, match="must match"):
        roll(parent, [None])
    with pytest.raises(ValueError, match="different families"):
        roll(parent, [det(img, [[0, 0, 2, 2]]), seg(img, [[0, 0, 2, 2]])])
    with pytest.raises(ValueError, match="without boxes"):
        roll((kp(img, [[0, 0, 2, 2]], [[[1, 1]]])[0], None), [])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA qualification requires a CUDA device"
)
def test_cuda_dense_masks_stay_on_device():
    img = image()
    parent = det(img, [[0, 0, 120, 80]])
    child = seg(img, [[1, 2, 5, 6], [2, 2, 6, 6]])
    for carrier in (parent, child):
        for name in ("xyxy", "class_id", "confidence", "mask"):
            if hasattr(carrier, name):
                setattr(carrier, name, getattr(carrier, name).cuda())
    result = roll(parent, [child])
    assert result.mask.is_cuda and result.xyxy.is_cuda
    assert result.xyxy.cpu().tolist() == [[1, 2, 6, 6]]


def test_empty_children_still_validate_known_lineage_and_mixed_empty_family():
    img = image()
    with pytest.raises(ValueError):
        roll(
            det(img, [[0, 0, 10, 10]]),
            [det(None, [], metadata={"parent_id": "partial"})],
        )
    child = kp(img, [[0, 0, 4, 4]], [[[1, 1]]])
    points, boxes = roll(det(img, [[0, 0, 10, 10]] * 2), [det(img, []), child])
    assert points.xy.tolist() == [[[1, 1]]] and len(boxes.xyxy) == 1


@pytest.mark.parametrize("strategy,score", [("max", 0.8), ("mean", 0.6), ("min", 0.4)])
def test_keypoints_latest_v1_mean_geometry_and_confidence(strategy, score):
    img = image()
    parent = det(img, [[0, 0, 120, 80]])
    child = kp(
        img,
        [[0, 0, 4, 4], [2, 2, 6, 6]],
        [[[1, 1], [2, 2]], [[3, 3], [4, 4]]],
        scores=[0.4, 0.8],
    )
    child[1].image_metadata["prediction_type"] = "keypoint-detection"
    for row, points in zip(child[1].bboxes_metadata, child[0].xy.tolist()):
        row["keypoints_xy"] = points
        row["keypoints_confidence"] = [1.0, 1.0]
    v1 = importlib.import_module(
        "roboflow_workflows.core_steps.fusion.detections_list_rollup.v1_tensor"
    )
    reference_pair, _ = v1.merge_crop_predictions(
        parent, [child[1]], strategy, 0.0, 10.0
    )
    reference_points, reference = reference_pair
    points, boxes = roll(parent, [child], confidence_strategy=strategy)
    torch.testing.assert_close(boxes.xyxy, reference.xyxy)
    torch.testing.assert_close(boxes.confidence, reference.confidence)
    assert boxes.confidence.tolist() == pytest.approx([score])
    torch.testing.assert_close(points.xy, reference_points.xy)
    assert points.xy.tolist() == [[[2, 2], [3, 3]]]


@pytest.mark.parametrize("threshold,count", [(0.5, 1), (0.5001, 2)])
@pytest.mark.parametrize("rle", [False, True])
def test_exact_mask_pixel_iou_threshold(threshold, count, rle):
    img = image((20, 20))
    child = seg(img, [[0, 0, 10, 10], [0, 0, 10, 20]], rle=rle)
    result = roll(det(img, [[0, 0, 20, 20]]), [child], overlap_threshold=threshold)
    assert len(result.xyxy) == count


@pytest.mark.parametrize("family", ["dense", "rle", "view", "keypoints"])
def test_parent_native_families_only_supply_their_boxes(family):
    img = image((20, 20))
    if family == "keypoints":
        parent = kp(img, [[3, 4, 13, 14]], [[[5, 5]]])
    else:
        parent = seg(img, [[3, 4, 13, 14]], rle=family == "rle")
        if family == "view":
            parent = SegmentationView.from_dense(parent)
    child = det(img.crop([3, 4, 13, 14]), [[1, 2, 5, 6]])
    result = roll(parent, [child])
    assert result.xyxy.tolist() == [[4, 6, 8, 10]]


def test_mixed_dense_rle_and_fully_empty_mask_rows():
    img = image((20, 20))
    parent = det(img, [[0, 0, 20, 20]] * 2)
    result = roll(
        parent, [seg(img, [[1, 2, 5, 6]]), seg(img, [[2, 2, 6, 6]], rle=True)]
    )
    assert isinstance(result.mask, InstancesRLEMasks)
    assert result.xyxy.tolist() == [[1, 2, 6, 6]]
    empty = roll(
        det(img, [[0, 0, 20, 20]]),
        [seg(img, [[1, 2, 5, 6]], masks=torch.zeros(1, 20, 20, dtype=torch.bool))],
    )
    assert isinstance(empty, InstanceDetections) and empty.mask.shape == (0, 20, 20)


@pytest.mark.parametrize("empty_first", [False, True])
def test_empty_rle_child_controls_mask_storage(empty_first):
    img = image((20, 20))
    children = [seg(img, [[1, 2, 5, 6]]), seg(img, [], rle=True)]
    if empty_first:
        children.reverse()

    result = roll(det(img, [[0, 0, 20, 20]] * 2), children)
    assert isinstance(result.mask, InstancesRLEMasks)
    assert result.xyxy.tolist() == [[1, 2, 5, 6]]
    expected = torch.zeros((1, 20, 20), dtype=torch.bool)
    expected[:, 2:6, 1:5] = True
    assert torch.equal(dense(result), expected)


@pytest.mark.parametrize("empty_first", [False, True])
def test_empty_keypoint_child_controls_padded_width(empty_first):
    img = image((20, 20))
    empty_boxes = det(img, [])
    empty_points = KeyPoints(
        xy=torch.zeros((0, 3, 2)),
        confidence=torch.zeros((0, 3)),
        class_id=empty_boxes.class_id,
        image_metadata=dict(empty_boxes.image_metadata),
    )
    children = [kp(img, [[0, 0, 4, 4]], [[[1, 2]]]), (empty_points, empty_boxes)]
    if empty_first:
        children.reverse()

    points, boxes = roll(det(img, [[0, 0, 20, 20]] * 2), children)
    assert points.xy.tolist() == [[[1, 2], [0, 0], [0, 0]]]
    assert points.confidence.tolist() == [[1, 0, 0]]
    assert boxes.xyxy.tolist() == [[0, 0, 4, 4]]


@pytest.mark.parametrize("with_frame", [False, True])
@pytest.mark.parametrize("rle", [False, True])
def test_empty_child_rejects_stray_mask_rows_with_or_without_lineage(with_frame, rle):
    img = image((20, 20))
    child = seg(
        img,
        [],
        masks=torch.ones((1, 20, 20), dtype=torch.bool),
        rle=rle,
    )
    if not with_frame:
        child.image_metadata = {}

    with pytest.raises(ValueError, match="(mask|shape)"):
        roll(det(img, [[0, 0, 20, 20]]), [child])


@pytest.mark.parametrize("with_frame", [False, True])
@pytest.mark.parametrize("family", ["boxes", "dense", "rle", "keypoints", "view"])
def test_valid_empty_children_keep_family_without_materializing_masks(
    family, with_frame, monkeypatch
):
    img = image((20, 20))
    boxes = det(img if with_frame else None, [])
    if family == "boxes":
        child = boxes
    elif family == "keypoints":
        child = (
            KeyPoints(
                xy=torch.zeros((0, 3, 2)),
                class_id=boxes.class_id,
                confidence=torch.zeros((0, 3)),
                image_metadata=dict(boxes.image_metadata),
            ),
            boxes,
        )
    elif family == "view":
        child = SegmentationView(
            detections=boxes,
            scores=torch.zeros((0, 2, 2)),
            score_type="logits",
            threshold=0.0,
            geometry=MaskGridGeometry.identity((2, 2)),
        )

        def unexpected_materialization(self):
            raise AssertionError("Empty view must not materialize its masks")

        monkeypatch.setattr(SegmentationView, "full_res", unexpected_materialization)
    else:
        child = seg(img, [], rle=family == "rle")
        if not with_frame:
            child.image_metadata = {}

    result = roll(det(img, [[0, 0, 20, 20]]), [child])
    carrier = result[1] if family == "keypoints" else result
    assert len(carrier.xyxy) == 0
    assert frame_of(carrier) == frame_of(det(img, []))
    if family in ("dense", "rle", "view"):
        assert isinstance(result, InstanceDetections)
        assert isinstance(result.mask, InstancesRLEMasks) == (family == "rle")
    elif family == "keypoints":
        assert result[0].xy.shape == (0, 3, 2)
    if family == "view":
        assert child.materialization_count == 0
        assert not child.is_full_res_materialized()


@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8])
def test_mask_payload_normalization_preserves_pixels_and_ownership(dtype):
    from roboflow_workflows.v2.blocks.fusion.rollup import _mask_payloads
    from torch.utils._python_dispatch import TorchDispatchMode

    masks = torch.tensor([[[0, 1], [1, 0]]], dtype=dtype).transpose(1, 2)
    before = masks.clone()
    normalizations = []

    class RecordNormalization(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.ne.Scalar:
                normalizations.append(args[0].dtype)
            result = func(*args, **(kwargs or {}))

            return result

    with RecordNormalization():
        dense_masks, rle_masks = _mask_payloads(
            [InstanceDetections(**det(None, [[0, 0, 2, 2]]).__dict__, mask=masks)],
            rle_storage=False,
        )

    assert normalizations == ([] if dtype == torch.bool else [dtype])
    assert rle_masks is None
    assert dense_masks.dtype == torch.bool
    assert torch.equal(dense_masks, before != 0)
    assert dense_masks.data_ptr() != masks.data_ptr()
    dense_masks.zero_()
    assert torch.equal(masks, before)
