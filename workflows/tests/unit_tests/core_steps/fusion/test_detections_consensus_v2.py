from copy import deepcopy

import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError
from roboflow_workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
)
from roboflow_workflows.core_steps.fusion.detections_consensus.v2 import (
    AggregationMode,
    BlockManifest,
    DetectionsConsensusBlockV2,
    MaskAggregationMode,
    calculate_iou,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from supervision.config import ORIENTED_BOX_COORDINATES


def test_detections_consensus_v2_validation_when_valid_specification_given() -> None:
    # given
    specification = {
        "type": "roboflow_core/detections_consensus@v2",
        "name": "some",
        "predictions": [
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        "image_metadata": "$steps.detection.image",
        "required_votes": 3,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type="roboflow_core/detections_consensus@v2",
        name="some",
        predictions_batches=[
            "$steps.detection.predictions",
            "$steps.detection_2.predictions",
        ],
        image_metadata="$steps.detection.image",
        required_votes=3,
        class_aware=True,
        iou_threshold=0.3,
        confidence=0.0,
        classes_to_consider=None,
        required_objects=None,
        presence_confidence_aggregation=AggregationMode.MAX,
        detections_merge_confidence_aggregation=AggregationMode.AVERAGE,
        detections_merge_coordinates_aggregation=AggregationMode.AVERAGE,
    )


def test_detections_consensus_v2_validation_rejects_v1_type_alias() -> None:
    # given
    specification = {
        "type": "roboflow_core/detections_consensus@v1",
        "name": "some",
        "predictions": ["$steps.detection.predictions"],
        "image_metadata": "$steps.detection.image",
        "required_votes": 1,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def _rotated_rect(
    cx: float, cy: float, w: float, h: float, angle_deg: float
) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    rot = np.array([[cos, -sin], [sin, cos]])
    corners = np.array(
        [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]
    )
    return (corners @ rot.T + [cx, cy]).astype(np.float32)


def test_calculate_iou_uses_box_iou_when_no_richer_geometry_present() -> None:
    # given - partially overlapping plain boxes, same expectation as the v1
    # behaviour: box A size = box B size = 800, intersection = 200,
    # expected result = 200 / 1400 = 1 / 7
    detections = sv.Detections(
        xyxy=np.array([[80, 190, 120, 210], [100, 200, 140, 220]], dtype=np.float64),
        confidence=np.array([0.5, 0.6], dtype=np.float64),
        class_id=np.array([1, 2]),
        data={"class_name": np.array(["a", "b"])},
    )
    detection_a = detections[0]
    detection_b = detections[1]

    # when
    result = calculate_iou(
        detection_a=detection_a,
        detection_b=detection_b,
    )

    # then
    assert abs(result - 1 / 7) < 1e-5


def test_calculate_iou_uses_mask_iou_when_both_have_masks() -> None:
    # given - identical AABB, disjoint masks. AABB IoU would be 1.0;
    # mask IoU is 0.
    mask_a = np.zeros((1, 10, 10), dtype=bool)
    mask_a[0, 0:5, 0:5] = True
    mask_b = np.zeros((1, 10, 10), dtype=bool)
    mask_b[0, 5:10, 5:10] = True
    xyxy = np.array([[0, 0, 10, 10]], dtype=np.float32)

    detection_a = sv.Detections(xyxy=xyxy, mask=mask_a)
    detection_b = sv.Detections(xyxy=xyxy, mask=mask_b)

    # when
    result = calculate_iou(
        detection_a=detection_a,
        detection_b=detection_b,
    )

    # then
    assert result == 0.0


def test_calculate_iou_uses_obb_iou_when_both_have_oriented_boxes() -> None:
    # given - crossed thin rectangles at +/-45 deg: near-identical AABBs but
    # barely-overlapping oriented bodies.
    quad_a = _rotated_rect(50, 50, 100, 10, +45)
    quad_b = _rotated_rect(50, 50, 100, 10, -45)
    aabb = np.array(
        [
            [
                quad_a[:, 0].min(),
                quad_a[:, 1].min(),
                quad_a[:, 0].max(),
                quad_a[:, 1].max(),
            ]
        ],
        dtype=np.float32,
    )

    detection_a = sv.Detections(
        xyxy=aabb, data={ORIENTED_BOX_COORDINATES: quad_a[None, ...]}
    )
    detection_b = sv.Detections(
        xyxy=aabb, data={ORIENTED_BOX_COORDINATES: quad_b[None, ...]}
    )

    # when
    result = calculate_iou(
        detection_a=detection_a,
        detection_b=detection_b,
    )

    # then
    assert result < 0.2


def _mask_sources(masks, *, boxes=None, detection_ids=None):
    height, width = masks[0].shape[1:]
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="image"),
        numpy_image=np.zeros((height, width, 3), dtype=np.uint8),
    )
    sources = []
    for index, mask in enumerate(masks):
        count = len(mask)
        source = sv.Detections(
            xyxy=np.array(
                boxes[index] if boxes is not None else [[0, 0, width, height]] * count,
                dtype=float,
            ),
            mask=mask,
            class_id=np.zeros(count, dtype=int),
            confidence=np.full(count, 0.9),
            data={
                "class_name": np.full(count, "object"),
                "detection_id": np.array(
                    detection_ids[index]
                    if detection_ids is not None
                    else [f"{index}:{row}" for row in range(count)]
                ),
                "image_dimensions": np.tile([height, width], (count, 1)),
            },
        )
        attach_parents_coordinates_to_sv_detections(source, image)
        sources.append(source)
    return sources


def _run_mask_consensus(sources, *, block=None, **updates):
    kwargs = dict(
        predictions_batches=[[source] for source in sources],
        required_votes=2,
        class_aware=True,
        iou_threshold=0.3,
        confidence=0.0,
        classes_to_consider=None,
        required_objects=None,
        presence_confidence_aggregation=AggregationMode.MAX,
        detections_merge_confidence_aggregation=AggregationMode.AVERAGE,
        detections_merge_coordinates_aggregation=AggregationMode.AVERAGE,
        detections_merge_mask_aggregation=MaskAggregationMode.UNION,
    )
    kwargs.update(updates)
    return (block or DetectionsConsensusBlockV2()).run(**kwargs)[0]


@pytest.mark.parametrize(
    "dtype, on_a, on_b", [(bool, True, True), (np.uint8, 255, 255), (np.uint8, 1, 2)]
)
@pytest.mark.parametrize(
    "boxes", [([[0, 0, 2, 2]], [[0, 0, 2, 2]]), ([[0, 0, 2, 2]], [[8, 8, 10, 10]])]
)
def test_mask_pixels_outside_detector_boxes_determine_consensus(
    dtype, on_a, on_b, boxes
):
    mask_a = np.zeros((1, 10, 10), dtype=dtype)
    mask_b = np.zeros((1, 10, 10), dtype=dtype)
    mask_a[0, 2:6, 2:6] = on_a
    mask_b[0, 2:6, 4:8] = on_b
    sources = _mask_sources([mask_a, mask_b], boxes=boxes)
    before = deepcopy(sources)
    # Each mask has area 16, with eight shared pixels: IoU is 8 / 24.
    assert calculate_iou(*sources) == pytest.approx(1 / 3)
    for threshold, count in [(0.3, 1), (1 / 3, 0)]:
        result = _run_mask_consensus(sources, iou_threshold=threshold)
        assert len(result["predictions"]) == count
        assert result["object_present"] is bool(count)
        if count:
            np.testing.assert_array_equal(result["predictions"].xyxy, [[2, 2, 7, 5]])
            expected_mask = np.zeros((1, 10, 10), dtype=bool)
            expected_mask[0, 2:6, 2:8] = True
            np.testing.assert_array_equal(result["predictions"].mask, expected_mask)
    for original, source in zip(before, sources):
        np.testing.assert_array_equal(source.xyxy, original.xyxy)
        np.testing.assert_array_equal(source.mask, original.mask)
        assert source.metadata == original.metadata
        assert source.data.keys() == original.data.keys()
        for key in original.data:
            np.testing.assert_array_equal(source.data[key], original.data[key])


@pytest.mark.parametrize("foreground", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_mask_consensus_retains_last_canvas_pixel_and_empty_masks(foreground, compact):
    mask = np.zeros((1, 10, 10), dtype=bool)
    mask[0, 9, 9] = foreground
    sources = _mask_sources([mask.copy(), mask.copy()])
    if compact:
        for source in sources:
            source.mask = sv.CompactMask.from_dense(
                source.mask, sv.mask_to_xyxy(source.mask), (10, 10)
            )
    result = _run_mask_consensus(sources)
    assert len(result["predictions"]) == int(foreground)
    if foreground:
        np.testing.assert_array_equal(result["predictions"].xyxy, [[9, 9, 9, 9]])
        np.testing.assert_array_equal(result["predictions"].mask, mask)


def test_mask_consensus_recomputes_geometry_on_later_calls_with_the_same_ids():
    mask = np.zeros((1, 10, 10), dtype=bool)
    mask[0, 2:6, 2:6] = True
    sources = _mask_sources([mask.copy(), mask.copy()])
    block = DetectionsConsensusBlockV2()
    first = _run_mask_consensus(sources, block=block)["predictions"]
    np.testing.assert_array_equal(first.xyxy, [[2, 2, 5, 5]])
    for source in sources:
        source.mask[:] = False
        source.mask[0, 8:10, 8:10] = True
    second = _run_mask_consensus(sources, block=block)["predictions"]
    np.testing.assert_array_equal(second.xyxy, [[8, 8, 9, 9]])
    np.testing.assert_array_equal(second.mask, sources[0].mask)


@pytest.mark.parametrize("identity", ["duplicate", 7, np.nan])
def test_aliased_or_legacy_detection_ids_keep_exact_disjoint_mask_votes(identity):
    mask_a = np.zeros((2, 10, 10), dtype=bool)
    mask_b = np.zeros((2, 10, 10), dtype=bool)
    mask_a[0, 0:2, 0:2] = True
    mask_a[1, 2:4, 2:4] = True
    mask_b[0, 5:7, 5:7] = True
    mask_b[1, 7:9, 7:9] = True
    sources = _mask_sources([mask_a, mask_b], detection_ids=[[identity] * 2] * 2)
    result = _run_mask_consensus(sources)
    assert len(result["predictions"]) == 0
    assert result["object_present"] is False


class _UnreadMasks(np.ndarray):
    """Fail if a no-candidate call starts reducing or comparing mask pixels."""

    def any(self, *args, **kwargs):
        raise AssertionError("Mask pixels were scanned without a candidate")

    def __array_function__(self, *args, **kwargs):
        raise AssertionError("Mask pixels were scanned without a candidate")

    def __array_ufunc__(self, *args, **kwargs):
        raise AssertionError("Mask pixels were scanned without a candidate")


@pytest.mark.parametrize("with_ids", [False, True])
def test_mask_consensus_does_not_scan_masks_when_class_filter_removes_all_candidates(
    with_ids,
):
    sources = _mask_sources([np.ones((2, 10, 10), dtype=bool)] * 2)
    for source in sources:
        source.mask = source.mask.view(_UnreadMasks)
        if not with_ids:
            del source.data["detection_id"]
    result = _run_mask_consensus(sources, classes_to_consider=["absent"])
    assert len(result["predictions"]) == 0
    assert result["object_present"] is False


def test_different_mask_canvases_keep_the_existing_numpy_merge_error():
    # NumPy overlap broadcasts the one-row canvas; merging incompatible canvases
    # still raises rather than silently replacing that overlap with a zero vote.
    mask_a = np.ones((1, 1, 10), dtype=bool)
    mask_b = np.zeros((1, 10, 10), dtype=bool)
    mask_b[0, 2:6, :] = True
    sources = _mask_sources([mask_a, mask_b])
    with pytest.raises(ValueError):
        _run_mask_consensus(sources)
