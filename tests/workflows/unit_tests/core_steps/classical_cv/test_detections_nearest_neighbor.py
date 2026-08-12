import math

import numpy as np
import pytest
import supervision as sv
import torch

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.workflows.core_steps.classical_cv.detections_nearest_neighbor.v1 import (
    NEAREST_TARGET_DISTANCE_KEY,
    OUTPUT_KEY_MATCHED_QUERY_DETECTIONS,
    OUTPUT_KEY_MATCHED_TARGET_DETECTIONS,
    OUTPUT_KEY_QUERY_PREDICTIONS,
    DetectionsNearestNeighborBlockV1,
)
from inference.core.workflows.core_steps.classical_cv.detections_nearest_neighbor.v1_tensor import (
    DetectionsNearestNeighborBlockV1 as TensorNativeDetectionsNearestNeighborBlockV1,
)
from inference.core.workflows.execution_engine.constants import CLASS_NAMES_KEY
from inference_models.models.base.instance_segmentation import (
    InstanceDetections as NativeInstanceDetections,
)
from inference_models.models.base.keypoints_detection import (
    KeyPoints as NativeKeyPoints,
)
from inference_models.models.base.object_detection import Detections as NativeDetections

# The sv-based tests below exercise the numpy v1 module directly, which is
# flag-agnostic - they pass in both flag directions, so no _NUMPY_ONLY marker is
# needed. Each has a `*_tensor_native` parity sibling (skipped when the flag is
# off) feeding the native inference_models representation into the v1_tensor
# block and asserting the same scenario, with per-box results read from
# `bboxes_metadata` (the native equivalent of the sv `.data` columns).
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)


def make_detections(xyxy, detection_ids=None, class_name="object") -> sv.Detections:
    xyxy = np.array(xyxy, dtype=np.float64)
    n = len(xyxy)
    data = {"class_name": np.array([class_name] * n, dtype="<U32")}
    if detection_ids is not None:
        data["detection_id"] = np.array(detection_ids, dtype="<U36")
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.array([0.9] * n),
        class_id=np.array([0] * n),
        data=data,
    )


def make_keypoint_detections(
    xyxy, keypoints_xy, keypoints_class_name, detection_ids=None
) -> sv.Detections:
    detections = make_detections(xyxy=xyxy, detection_ids=detection_ids)
    detections.data["keypoints_xy"] = np.array(keypoints_xy, dtype=np.float64)
    detections.data["keypoints_class_name"] = np.array(
        keypoints_class_name, dtype=object
    )
    return detections


def run_block(
    query,
    target,
    query_point="CENTER",
    target_point="CENTER",
    query_keypoint_name=None,
    target_keypoint_name=None,
    max_distance=None,
):
    block = DetectionsNearestNeighborBlockV1()
    return block.run(
        query_predictions=query,
        target_predictions=target,
        query_point=query_point,
        target_point=target_point,
        query_keypoint_name=query_keypoint_name,
        target_keypoint_name=target_keypoint_name,
        max_distance=max_distance,
    )


def test_single_nearest_match() -> None:
    # given
    query = make_detections(xyxy=[[0, 0, 10, 10]], detection_ids=["q1"])
    target = make_detections(
        xyxy=[
            [100, 100, 110, 110],  # center (105, 105) -> far
            [10, 10, 20, 20],  # center (15, 15) -> near
        ],
        detection_ids=["t1", "t2"],
    )

    # when
    result = run_block(query, target)

    # then
    query_out = result[OUTPUT_KEY_QUERY_PREDICTIONS]
    matched_query = result[OUTPUT_KEY_MATCHED_QUERY_DETECTIONS]
    matched_target = result[OUTPUT_KEY_MATCHED_TARGET_DETECTIONS]
    expected_distance = math.hypot(15 - 5, 15 - 5)
    assert math.isclose(
        query_out.data[NEAREST_TARGET_DISTANCE_KEY][0], expected_distance, rel_tol=1e-6
    )
    assert len(matched_query) == 1
    assert len(matched_target) == 1
    assert matched_query.data["detection_id"][0] == "q1"
    assert matched_target.data["detection_id"][0] == "t2"


def test_exact_tie_within_epsilon() -> None:
    # given: two targets equidistant (well within the 1px epsilon) from the query
    query = make_detections(xyxy=[[0, 0, 10, 10]], detection_ids=["q1"])
    target = make_detections(
        xyxy=[
            [10, 10, 20, 20],  # center (15, 15)
            [10, 10.2, 20, 20.2],  # center (15, 15.2) -> distance differs by < 1px
            [1000, 1000, 1010, 1010],  # clearly far away
        ],
        detection_ids=["t1", "t2", "t3"],
    )

    # when
    result = run_block(query, target)

    # then: the query row is duplicated once per tied match
    matched_query = result[OUTPUT_KEY_MATCHED_QUERY_DETECTIONS]
    matched_target = result[OUTPUT_KEY_MATCHED_TARGET_DETECTIONS]
    assert len(matched_query) == 2
    assert len(matched_target) == 2
    assert list(matched_query.data["detection_id"]) == ["q1", "q1"]
    assert set(matched_target.data["detection_id"]) == {"t1", "t2"}


def test_no_tie_clear_separation() -> None:
    # given
    query = make_detections(xyxy=[[0, 0, 10, 10]], detection_ids=["q1"])
    target = make_detections(
        xyxy=[
            [10, 10, 20, 20],  # near
            [1000, 1000, 1010, 1010],  # far
        ],
        detection_ids=["t1", "t2"],
    )

    # when
    result = run_block(query, target)

    # then
    matched_query = result[OUTPUT_KEY_MATCHED_QUERY_DETECTIONS]
    matched_target = result[OUTPUT_KEY_MATCHED_TARGET_DETECTIONS]
    assert len(matched_query) == 1
    assert len(matched_target) == 1
    assert matched_target.data["detection_id"][0] == "t1"


def test_max_distance_excludes_matches_beyond_the_limit() -> None:
    # given: the only target is farther than max_distance
    query = make_detections(
        xyxy=[[0, 0, 10, 10]], detection_ids=["q1"]
    )  # center (5, 5)
    target = make_detections(
        xyxy=[[100, 100, 110, 110]], detection_ids=["t1"]
    )  # center (105, 105) -> distance ~141.4

    # when
    result = run_block(query, target, max_distance=50)

    # then
    query_out = result[OUTPUT_KEY_QUERY_PREDICTIONS]
    matched_query = result[OUTPUT_KEY_MATCHED_QUERY_DETECTIONS]
    matched_target = result[OUTPUT_KEY_MATCHED_TARGET_DETECTIONS]
    assert query_out.data[NEAREST_TARGET_DISTANCE_KEY][0] is None
    assert len(matched_query) == 0
    assert len(matched_target) == 0


def test_max_distance_retains_in_range_target_while_excluding_out_of_range_target() -> (
    None
):
    # given: t1 is beyond max_distance and t2 is within it. Note t2 is
    # genuinely the nearer of the two here - a target closer than an excluded
    # one is, by construction, also within range, so there's no geometry where
    # the true nearest target is excluded yet a farther target is retained.
    # This test instead confirms that an out-of-range candidate doesn't
    # prevent a valid in-range candidate from being matched.
    query = make_detections(
        xyxy=[[0, 0, 10, 10]], detection_ids=["q1"]
    )  # center (5, 5)
    target = make_detections(
        xyxy=[
            [100, 100, 110, 110],  # center (105, 105) -> distance ~141.4, excluded
            [40, 5, 50, 15],  # center (45, 10) -> distance ~40.3, within limit
        ],
        detection_ids=["t1", "t2"],
    )

    # when
    result = run_block(query, target, max_distance=45)

    # then
    matched_target = result[OUTPUT_KEY_MATCHED_TARGET_DETECTIONS]
    assert len(matched_target) == 1
    assert matched_target.data["detection_id"][0] == "t2"


def test_max_distance_boundary_is_inclusive() -> None:
    # given: the target sits exactly on the max_distance boundary
    query = make_detections(xyxy=[[0, 0, 0, 0]], detection_ids=["q1"])  # center (0, 0)
    target = make_detections(
        xyxy=[[50, 0, 50, 0]], detection_ids=["t1"]
    )  # center (50, 0) -> distance exactly 50

    # when
    result = run_block(query, target, max_distance=50)

    # then: equal-to-limit is included, not excluded
    matched_target = result[OUTPUT_KEY_MATCHED_TARGET_DETECTIONS]
    assert len(matched_target) == 1
    assert matched_target.data["detection_id"][0] == "t1"


def test_empty_target_set() -> None:
    # given
    query = make_detections(
        xyxy=[[0, 0, 10, 10], [20, 20, 30, 30]], detection_ids=["q1", "q2"]
    )
    target = sv.Detections.empty()

    # when
    result = run_block(query, target)

    # then
    query_out = result[OUTPUT_KEY_QUERY_PREDICTIONS]
    matched_query = result[OUTPUT_KEY_MATCHED_QUERY_DETECTIONS]
    matched_target = result[OUTPUT_KEY_MATCHED_TARGET_DETECTIONS]
    assert list(query_out.data[NEAREST_TARGET_DISTANCE_KEY]) == [None, None]
    assert len(matched_query) == 0
    assert len(matched_target) == 0


def test_self_match_exclusion_same_set_as_query_and_target() -> None:
    # given: the same detections passed as both query and target
    detections = make_detections(
        xyxy=[
            [0, 0, 10, 10],  # center (5, 5)
            [10, 10, 20, 20],  # center (15, 15) -> nearest to detection 0
            [1000, 1000, 1010, 1010],  # center (1005, 1005) -> far
        ],
        detection_ids=["a", "b", "c"],
    )

    # when
    result = run_block(detections, detections)

    # then
    query_out = result[OUTPUT_KEY_QUERY_PREDICTIONS]
    matched_query = result[OUTPUT_KEY_MATCHED_QUERY_DETECTIONS]
    matched_target = result[OUTPUT_KEY_MATCHED_TARGET_DETECTIONS]
    pairs = dict(
        zip(matched_query.data["detection_id"], matched_target.data["detection_id"])
    )
    # detection "a" must not match itself, nearest is "b"; "b" must not match
    # itself either, nearest is "a"
    assert pairs["a"] == "b"
    assert pairs["b"] == "a"
    assert query_out.data[NEAREST_TARGET_DISTANCE_KEY][0] > 0


def test_self_match_not_excluded_when_detection_id_missing() -> None:
    # given: detections lacking `detection_id` on both sides - self-exclusion
    # must be skipped gracefully (not raise), so a detection matches itself
    xyxy = np.array([[0, 0, 10, 10], [1000, 1000, 1010, 1010]], dtype=np.float64)
    detections = sv.Detections(
        xyxy=xyxy,
        confidence=np.array([0.9, 0.9]),
        class_id=np.array([0, 0]),
        data={"class_name": np.array(["object", "object"], dtype="<U32")},
    )

    # when
    result = run_block(detections, detections)

    # then
    query_out = result[OUTPUT_KEY_QUERY_PREDICTIONS]
    assert query_out.data[NEAREST_TARGET_DISTANCE_KEY][0] == 0.0


@pytest.mark.parametrize(
    "point_option,expected_point",
    [
        ("CENTER", (5, 10)),
        ("CENTER_LEFT", (0, 10)),
        ("CENTER_RIGHT", (10, 10)),
        ("TOP_CENTER", (5, 0)),
        ("TOP_LEFT", (0, 0)),
        ("TOP_RIGHT", (10, 0)),
        ("BOTTOM_LEFT", (0, 20)),
        ("BOTTOM_CENTER", (5, 20)),
        ("BOTTOM_RIGHT", (10, 20)),
    ],
)
def test_bbox_anchor_point_options(point_option, expected_point) -> None:
    # given
    query = make_detections(xyxy=[[0, 0, 10, 20]], detection_ids=["q1"])
    target = make_detections(xyxy=[[100, 100, 100, 100]], detection_ids=["t1"])

    # when
    result = run_block(query, target, query_point=point_option, target_point="CENTER")

    # then
    ex, ey = expected_point
    expected_distance = math.hypot(100 - ex, 100 - ey)
    query_out = result[OUTPUT_KEY_QUERY_PREDICTIONS]
    assert math.isclose(
        query_out.data[NEAREST_TARGET_DISTANCE_KEY][0], expected_distance, rel_tol=1e-6
    )


def test_keypoint_anchor_point_option() -> None:
    # given
    query = make_keypoint_detections(
        xyxy=[[0, 0, 10, 10]],
        keypoints_xy=[[[3, 4], [7, 8]]],
        keypoints_class_name=[["left_shoulder", "right_shoulder"]],
        detection_ids=["q1"],
    )
    target = make_detections(xyxy=[[103, 104, 103, 104]], detection_ids=["t1"])

    # when
    result = run_block(
        query,
        target,
        query_point="KEYPOINT",
        target_point="CENTER",
        query_keypoint_name="left_shoulder",
    )

    # then
    query_out = result[OUTPUT_KEY_QUERY_PREDICTIONS]
    assert math.isclose(
        query_out.data[NEAREST_TARGET_DISTANCE_KEY][0],
        100.0 * math.sqrt(2),
        rel_tol=1e-6,
    )


def test_keypoint_missing_on_detection_excludes_it_from_matching() -> None:
    # given: target detection's keypoints do not include the requested name
    query = make_detections(xyxy=[[0, 0, 10, 10]], detection_ids=["q1"])
    target = make_keypoint_detections(
        xyxy=[[10, 10, 20, 20], [200, 200, 210, 210]],
        keypoints_xy=[[[15, 15]], [[0, 0]]],
        keypoints_class_name=[["nose"], ["left_eye"]],
        detection_ids=["t1", "t2"],
    )

    # when
    result = run_block(
        query,
        target,
        query_point="CENTER",
        target_point="KEYPOINT",
        target_keypoint_name="nose",
    )

    # then: only t1 has a "nose" keypoint, t2 must be excluded from matching
    matched_query = result[OUTPUT_KEY_MATCHED_QUERY_DETECTIONS]
    matched_target = result[OUTPUT_KEY_MATCHED_TARGET_DETECTIONS]
    assert len(matched_query) == 1
    assert len(matched_target) == 1
    assert matched_target.data["detection_id"][0] == "t1"


def test_query_point_keypoint_requires_keypoint_name() -> None:
    # given
    query = make_detections(xyxy=[[0, 0, 10, 10]], detection_ids=["q1"])
    target = make_detections(xyxy=[[10, 10, 20, 20]], detection_ids=["t1"])

    # when / then
    with pytest.raises(
        expected_exception=ValueError,
        match="query_keypoint_name",
    ):
        run_block(query, target, query_point="KEYPOINT")


def test_keypoint_option_requires_keypoint_predictions() -> None:
    # given: query_point is KEYPOINT but query predictions have no keypoint data
    query = make_detections(xyxy=[[0, 0, 10, 10]], detection_ids=["q1"])
    target = make_detections(xyxy=[[10, 10, 20, 20]], detection_ids=["t1"])

    # when / then
    with pytest.raises(expected_exception=ValueError, match="keypoint data"):
        run_block(
            query,
            target,
            query_point="KEYPOINT",
            query_keypoint_name="left_shoulder",
        )


def make_native_detections(
    xyxy, detection_ids=None, class_name="object"
) -> NativeDetections:
    n = len(xyxy)
    bboxes_metadata = None
    if detection_ids is not None:
        bboxes_metadata = [
            {"detection_id": detection_id} for detection_id in detection_ids
        ]
    return NativeDetections(
        xyxy=torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.zeros(n, dtype=torch.long),
        confidence=torch.full((n,), 0.9, dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {0: class_name}},
        bboxes_metadata=bboxes_metadata,
    )


def make_native_empty_detections() -> NativeDetections:
    return NativeDetections(
        xyxy=torch.zeros((0, 4), dtype=torch.float32),
        class_id=torch.zeros((0,), dtype=torch.long),
        confidence=torch.zeros((0,), dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {}},
        bboxes_metadata=None,
    )


def make_native_keypoint_prediction(
    xyxy, keypoints_xy, keypoints_class_name, detection_ids=None
):
    """Build the native keypoint-detection shape: a ``(KeyPoints, Detections)``
    tuple whose bbox component carries the flattened per-box ``keypoints_xy`` /
    ``keypoints_class_name`` payloads in ``bboxes_metadata`` (the native mirror
    of the sv data columns)."""
    detections = make_native_detections(xyxy=xyxy, detection_ids=detection_ids)
    n = len(xyxy)
    if detections.bboxes_metadata is None:
        detections.bboxes_metadata = [{} for _ in range(n)]
    max_keypoints = max(len(instance_xy) for instance_xy in keypoints_xy)
    padded_xy = torch.zeros((n, max_keypoints, 2), dtype=torch.float32)
    padded_confidence = torch.zeros((n, max_keypoints), dtype=torch.float32)
    for index, (instance_xy, instance_names) in enumerate(
        zip(keypoints_xy, keypoints_class_name)
    ):
        detections.bboxes_metadata[index]["keypoints_xy"] = [
            [float(x), float(y)] for x, y in instance_xy
        ]
        detections.bboxes_metadata[index]["keypoints_class_name"] = list(instance_names)
        if len(instance_xy) > 0:
            padded_xy[index, : len(instance_xy)] = torch.tensor(
                instance_xy, dtype=torch.float32
            )
            padded_confidence[index, : len(instance_xy)] = 1.0
    key_points = NativeKeyPoints(
        xy=padded_xy,
        class_id=torch.zeros(n, dtype=torch.long),
        confidence=padded_confidence,
        image_metadata=detections.image_metadata,
    )
    return key_points, detections


def bbox_component(prediction) -> NativeDetections:
    if isinstance(prediction, tuple):
        return prediction[1]
    return prediction


def nearest_distances(prediction) -> list:
    detections = bbox_component(prediction)
    return [
        box_metadata[NEAREST_TARGET_DISTANCE_KEY]
        for box_metadata in detections.bboxes_metadata
    ]


def run_tensor_block(
    query,
    target,
    query_point="CENTER",
    target_point="CENTER",
    query_keypoint_name=None,
    target_keypoint_name=None,
    max_distance=None,
):
    block = TensorNativeDetectionsNearestNeighborBlockV1()
    return block.run(
        query_predictions=query,
        target_predictions=target,
        query_point=query_point,
        target_point=target_point,
        query_keypoint_name=query_keypoint_name,
        target_keypoint_name=target_keypoint_name,
        max_distance=max_distance,
    )


@_TENSOR_ONLY
def test_single_nearest_match_tensor_native() -> None:
    # given
    query = make_native_detections(xyxy=[[0, 0, 10, 10]], detection_ids=["q1"])
    target = make_native_detections(
        xyxy=[
            [100, 100, 110, 110],  # center (105, 105) -> far
            [10, 10, 20, 20],  # center (15, 15) -> near
        ],
        detection_ids=["t1", "t2"],
    )

    # when
    result = run_tensor_block(query, target)

    # then
    query_out = result[OUTPUT_KEY_QUERY_PREDICTIONS]
    matched_query = result[OUTPUT_KEY_MATCHED_QUERY_DETECTIONS]
    matched_target = result[OUTPUT_KEY_MATCHED_TARGET_DETECTIONS]
    expected_distance = math.hypot(15 - 5, 15 - 5)
    assert nearest_distances(query_out)[0] == pytest.approx(
        expected_distance, rel=1e-6
    )
    assert len(matched_query) == 1
    assert len(matched_target) == 1
    assert matched_query.bboxes_metadata[0]["detection_id"] == "q1"
    assert matched_target.bboxes_metadata[0]["detection_id"] == "t2"
    # the sliced query rows carry the enriched per-box scalar
    assert matched_query.bboxes_metadata[0][NEAREST_TARGET_DISTANCE_KEY] == (
        pytest.approx(expected_distance, rel=1e-6)
    )


@_TENSOR_ONLY
def test_exact_tie_within_epsilon_tensor_native() -> None:
    # given: two targets equidistant (well within the 1px epsilon) from the query
    query = make_native_detections(xyxy=[[0, 0, 10, 10]], detection_ids=["q1"])
    target = make_native_detections(
        xyxy=[
            [10, 10, 20, 20],  # center (15, 15)
            [10, 10.2, 20, 20.2],  # center (15, 15.2) -> distance differs by < 1px
            [1000, 1000, 1010, 1010],  # clearly far away
        ],
        detection_ids=["t1", "t2", "t3"],
    )

    # when
    result = run_tensor_block(query, target)

    # then: the query row is duplicated once per tied match
    matched_query = result[OUTPUT_KEY_MATCHED_QUERY_DETECTIONS]
    matched_target = result[OUTPUT_KEY_MATCHED_TARGET_DETECTIONS]
    assert len(matched_query) == 2
    assert len(matched_target) == 2
    assert [
        box_metadata["detection_id"] for box_metadata in matched_query.bboxes_metadata
    ] == ["q1", "q1"]
    assert {
        box_metadata["detection_id"] for box_metadata in matched_target.bboxes_metadata
    } == {"t1", "t2"}


@_TENSOR_ONLY
def test_no_tie_clear_separation_tensor_native() -> None:
    # given
    query = make_native_detections(xyxy=[[0, 0, 10, 10]], detection_ids=["q1"])
    target = make_native_detections(
        xyxy=[
            [10, 10, 20, 20],  # near
            [1000, 1000, 1010, 1010],  # far
        ],
        detection_ids=["t1", "t2"],
    )

    # when
    result = run_tensor_block(query, target)

    # then
    matched_query = result[OUTPUT_KEY_MATCHED_QUERY_DETECTIONS]
    matched_target = result[OUTPUT_KEY_MATCHED_TARGET_DETECTIONS]
    assert len(matched_query) == 1
    assert len(matched_target) == 1
    assert matched_target.bboxes_metadata[0]["detection_id"] == "t1"


@_TENSOR_ONLY
def test_max_distance_excludes_matches_beyond_the_limit_tensor_native() -> None:
    # given: the only target is farther than max_distance
    query = make_native_detections(
        xyxy=[[0, 0, 10, 10]], detection_ids=["q1"]
    )  # center (5, 5)
    target = make_native_detections(
        xyxy=[[100, 100, 110, 110]], detection_ids=["t1"]
    )  # center (105, 105) -> distance ~141.4

    # when
    result = run_tensor_block(query, target, max_distance=50)

    # then
    query_out = result[OUTPUT_KEY_QUERY_PREDICTIONS]
    matched_query = result[OUTPUT_KEY_MATCHED_QUERY_DETECTIONS]
    matched_target = result[OUTPUT_KEY_MATCHED_TARGET_DETECTIONS]
    assert nearest_distances(query_out)[0] is None
    assert len(matched_query) == 0
    assert len(matched_target) == 0


@_TENSOR_ONLY
def test_max_distance_retains_in_range_target_while_excluding_out_of_range_target_tensor_native() -> (
    None
):
    # given: t1 is beyond max_distance and t2 is within it
    query = make_native_detections(
        xyxy=[[0, 0, 10, 10]], detection_ids=["q1"]
    )  # center (5, 5)
    target = make_native_detections(
        xyxy=[
            [100, 100, 110, 110],  # center (105, 105) -> distance ~141.4, excluded
            [40, 5, 50, 15],  # center (45, 10) -> distance ~40.3, within limit
        ],
        detection_ids=["t1", "t2"],
    )

    # when
    result = run_tensor_block(query, target, max_distance=45)

    # then
    matched_target = result[OUTPUT_KEY_MATCHED_TARGET_DETECTIONS]
    assert len(matched_target) == 1
    assert matched_target.bboxes_metadata[0]["detection_id"] == "t2"


@_TENSOR_ONLY
def test_max_distance_boundary_is_inclusive_tensor_native() -> None:
    # given: the target sits exactly on the max_distance boundary
    query = make_native_detections(
        xyxy=[[0, 0, 0, 0]], detection_ids=["q1"]
    )  # center (0, 0)
    target = make_native_detections(
        xyxy=[[50, 0, 50, 0]], detection_ids=["t1"]
    )  # center (50, 0) -> distance exactly 50

    # when
    result = run_tensor_block(query, target, max_distance=50)

    # then: equal-to-limit is included, not excluded
    matched_target = result[OUTPUT_KEY_MATCHED_TARGET_DETECTIONS]
    assert len(matched_target) == 1
    assert matched_target.bboxes_metadata[0]["detection_id"] == "t1"


@_TENSOR_ONLY
def test_empty_target_set_tensor_native() -> None:
    # given
    query = make_native_detections(
        xyxy=[[0, 0, 10, 10], [20, 20, 30, 30]], detection_ids=["q1", "q2"]
    )
    target = make_native_empty_detections()

    # when
    result = run_tensor_block(query, target)

    # then
    query_out = result[OUTPUT_KEY_QUERY_PREDICTIONS]
    matched_query = result[OUTPUT_KEY_MATCHED_QUERY_DETECTIONS]
    matched_target = result[OUTPUT_KEY_MATCHED_TARGET_DETECTIONS]
    assert nearest_distances(query_out) == [None, None]
    assert len(matched_query) == 0
    assert len(matched_target) == 0


@_TENSOR_ONLY
def test_self_match_exclusion_same_set_as_query_and_target_tensor_native() -> None:
    # given: the same detections passed as both query and target
    detections = make_native_detections(
        xyxy=[
            [0, 0, 10, 10],  # center (5, 5)
            [10, 10, 20, 20],  # center (15, 15) -> nearest to detection 0
            [1000, 1000, 1010, 1010],  # center (1005, 1005) -> far
        ],
        detection_ids=["a", "b", "c"],
    )

    # when
    result = run_tensor_block(detections, detections)

    # then
    query_out = result[OUTPUT_KEY_QUERY_PREDICTIONS]
    matched_query = result[OUTPUT_KEY_MATCHED_QUERY_DETECTIONS]
    matched_target = result[OUTPUT_KEY_MATCHED_TARGET_DETECTIONS]
    pairs = dict(
        zip(
            [entry["detection_id"] for entry in matched_query.bboxes_metadata],
            [entry["detection_id"] for entry in matched_target.bboxes_metadata],
        )
    )
    # detection "a" must not match itself, nearest is "b"; "b" must not match
    # itself either, nearest is "a"
    assert pairs["a"] == "b"
    assert pairs["b"] == "a"
    assert nearest_distances(query_out)[0] > 0


@_TENSOR_ONLY
def test_self_match_not_excluded_when_detection_id_missing_tensor_native() -> None:
    # given: detections lacking `detection_id` on both sides - self-exclusion
    # must be skipped gracefully (not raise), so a detection matches itself
    detections = NativeDetections(
        xyxy=torch.tensor(
            [[0, 0, 10, 10], [1000, 1000, 1010, 1010]], dtype=torch.float32
        ),
        class_id=torch.zeros(2, dtype=torch.long),
        confidence=torch.full((2,), 0.9, dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {0: "object"}},
        bboxes_metadata=[{}, {}],
    )

    # when
    result = run_tensor_block(detections, detections)

    # then
    query_out = result[OUTPUT_KEY_QUERY_PREDICTIONS]
    assert nearest_distances(query_out)[0] == 0.0


@_TENSOR_ONLY
@pytest.mark.parametrize(
    "point_option,expected_point",
    [
        ("CENTER", (5, 10)),
        ("CENTER_LEFT", (0, 10)),
        ("CENTER_RIGHT", (10, 10)),
        ("TOP_CENTER", (5, 0)),
        ("TOP_LEFT", (0, 0)),
        ("TOP_RIGHT", (10, 0)),
        ("BOTTOM_LEFT", (0, 20)),
        ("BOTTOM_CENTER", (5, 20)),
        ("BOTTOM_RIGHT", (10, 20)),
    ],
)
def test_bbox_anchor_point_options_tensor_native(point_option, expected_point) -> None:
    # given
    query = make_native_detections(xyxy=[[0, 0, 10, 20]], detection_ids=["q1"])
    target = make_native_detections(xyxy=[[100, 100, 100, 100]], detection_ids=["t1"])

    # when
    result = run_tensor_block(
        query, target, query_point=point_option, target_point="CENTER"
    )

    # then
    ex, ey = expected_point
    expected_distance = math.hypot(100 - ex, 100 - ey)
    query_out = result[OUTPUT_KEY_QUERY_PREDICTIONS]
    assert nearest_distances(query_out)[0] == pytest.approx(
        expected_distance, rel=1e-6
    )


@_TENSOR_ONLY
def test_keypoint_anchor_point_option_tensor_native() -> None:
    # given
    query = make_native_keypoint_prediction(
        xyxy=[[0, 0, 10, 10]],
        keypoints_xy=[[[3, 4], [7, 8]]],
        keypoints_class_name=[["left_shoulder", "right_shoulder"]],
        detection_ids=["q1"],
    )
    target = make_native_detections(xyxy=[[103, 104, 103, 104]], detection_ids=["t1"])

    # when
    result = run_tensor_block(
        query,
        target,
        query_point="KEYPOINT",
        target_point="CENTER",
        query_keypoint_name="left_shoulder",
    )

    # then: the keypoint-detection tuple shape survives, with the per-box
    # scalar written into the bbox component's metadata
    query_out = result[OUTPUT_KEY_QUERY_PREDICTIONS]
    assert isinstance(query_out, tuple)
    assert nearest_distances(query_out)[0] == pytest.approx(
        100.0 * math.sqrt(2), rel=1e-6
    )


@_TENSOR_ONLY
def test_keypoint_missing_on_detection_excludes_it_from_matching_tensor_native() -> (
    None
):
    # given: target detection's keypoints do not include the requested name
    query = make_native_detections(xyxy=[[0, 0, 10, 10]], detection_ids=["q1"])
    target = make_native_keypoint_prediction(
        xyxy=[[10, 10, 20, 20], [200, 200, 210, 210]],
        keypoints_xy=[[[15, 15]], [[0, 0]]],
        keypoints_class_name=[["nose"], ["left_eye"]],
        detection_ids=["t1", "t2"],
    )

    # when
    result = run_tensor_block(
        query,
        target,
        query_point="CENTER",
        target_point="KEYPOINT",
        target_keypoint_name="nose",
    )

    # then: only t1 has a "nose" keypoint, t2 must be excluded from matching;
    # the matched target keeps the tuple shape with the KeyPoints component
    # sliced in lockstep
    matched_query = result[OUTPUT_KEY_MATCHED_QUERY_DETECTIONS]
    matched_target = result[OUTPUT_KEY_MATCHED_TARGET_DETECTIONS]
    assert len(matched_query) == 1
    assert isinstance(matched_target, tuple)
    matched_target_key_points, matched_target_detections = matched_target
    assert len(matched_target_detections) == 1
    assert int(matched_target_key_points.xy.shape[0]) == 1
    assert matched_target_detections.bboxes_metadata[0]["detection_id"] == "t1"


@_TENSOR_ONLY
def test_query_point_keypoint_requires_keypoint_name_tensor_native() -> None:
    # given
    query = make_native_detections(xyxy=[[0, 0, 10, 10]], detection_ids=["q1"])
    target = make_native_detections(xyxy=[[10, 10, 20, 20]], detection_ids=["t1"])

    # when / then
    with pytest.raises(
        expected_exception=ValueError,
        match="query_keypoint_name",
    ):
        run_tensor_block(query, target, query_point="KEYPOINT")


@_TENSOR_ONLY
def test_keypoint_option_requires_keypoint_predictions_tensor_native() -> None:
    # given: query_point is KEYPOINT but query predictions have no keypoint data
    query = make_native_detections(xyxy=[[0, 0, 10, 10]], detection_ids=["q1"])
    target = make_native_detections(xyxy=[[10, 10, 20, 20]], detection_ids=["t1"])

    # when / then
    with pytest.raises(expected_exception=ValueError, match="keypoint data"):
        run_tensor_block(
            query,
            target,
            query_point="KEYPOINT",
            query_keypoint_name="left_shoulder",
        )


@_TENSOR_ONLY
def test_instance_segmentation_input_preserves_type_and_masks_tensor_native() -> None:
    # given: instance-segmentation query - the matched output must stay an
    # InstanceDetections with the surviving row's mask carried over
    masks = torch.zeros((2, 40, 40), dtype=torch.bool)
    masks[0, :10, :10] = True
    masks[1, 20:30, 20:30] = True
    query = NativeInstanceDetections(
        xyxy=torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=torch.float32),
        class_id=torch.zeros(2, dtype=torch.long),
        confidence=torch.full((2,), 0.9, dtype=torch.float32),
        mask=masks,
        image_metadata={CLASS_NAMES_KEY: {0: "object"}},
        bboxes_metadata=[{"detection_id": "q1"}, {"detection_id": "q2"}],
    )
    target = make_native_detections(xyxy=[[10, 10, 20, 20]], detection_ids=["t1"])

    # when
    result = run_tensor_block(query, target)

    # then
    query_out = result[OUTPUT_KEY_QUERY_PREDICTIONS]
    matched_query = result[OUTPUT_KEY_MATCHED_QUERY_DETECTIONS]
    assert isinstance(query_out, NativeInstanceDetections)
    assert isinstance(matched_query, NativeInstanceDetections)
    assert len(matched_query) == 2
    assert matched_query.mask.shape == (2, 40, 40)
    assert all(
        box_metadata[NEAREST_TARGET_DISTANCE_KEY] is not None
        for box_metadata in matched_query.bboxes_metadata
    )
