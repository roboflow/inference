"""Tensor-native sibling of detections_nearest_neighbor/v1.py.

The numpy block performs a nearest-neighbor spatial join on ``sv.Detections``:
anchor points are resolved per detection (bbox anchors or a named keypoint),
a brute-force pairwise Euclidean pixel-distance matrix is built, self-matches
(shared ``detection_id``) and targets beyond ``max_distance`` are excluded, and
each query's minimum distance (with a 1 px tie epsilon) selects the matched
target rows. The nearest distance lands in
``query_predictions.data["nearest_target_distance"]``.

This sibling keeps exactly those semantics on the native inference_models
dataclasses: anchor points and the distance matrix are computed as torch ops on
the prediction's device (invalid candidates are masked instead of NaN-assigned,
with the same nan-propagation behaviour for missing keypoints), and the single
device->host hops are the batched ``.cpu()`` of the per-query minima and the
tie-mask indices. The computed per-box scalar is written to
``bboxes_metadata[i][NEAREST_TARGET_DISTANCE_KEY]`` (``None`` when a query has
no eligible target) - the native equivalent of the ``sv.Detections.data``
column, mutating the input prediction in place like the numpy block. Keypoint
input arrives as a ``(KeyPoints, Detections)`` tuple; the bbox component
carries the flattened per-box ``keypoints_xy`` / ``keypoints_class_name``
payloads in ``bboxes_metadata`` (the native mirror of the sv data columns) and
the matched outputs are sliced with ``take_prediction_by_indices`` so keypoint
tensors and instance masks ride along, duplicated rows included on ties.
"""

import math
from typing import List, Optional, Tuple, Type, Union

import torch

from inference.core.workflows.core_steps.classical_cv.detections_nearest_neighbor.v1 import (
    KEYPOINT_POINT_OPTION,
    OUTPUT_KEY_MATCHED_QUERY_DETECTIONS,
    OUTPUT_KEY_MATCHED_TARGET_DETECTIONS,
    OUTPUT_KEY_QUERY_PREDICTIONS,
    TIE_EPSILON_PX,
    BlockManifest,
)
from inference.core.workflows.core_steps.common.tensor_native import (
    split_key_point_prediction,
    take_prediction_by_indices,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    NEAREST_TARGET_DISTANCE_KEY,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

TensorNativeDetections = Union[Detections, InstanceDetections]
KeyPointPrediction = Tuple[KeyPoints, Optional[Detections]]
NearestNeighborInput = Union[Detections, InstanceDetections, KeyPointPrediction]


class DetectionsNearestNeighborBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        query_predictions: NearestNeighborInput,
        target_predictions: NearestNeighborInput,
        query_point: str,
        target_point: str,
        query_keypoint_name: Optional[str],
        target_keypoint_name: Optional[str],
        max_distance: Optional[int],
    ) -> BlockResult:
        if query_point == KEYPOINT_POINT_OPTION and not query_keypoint_name:
            raise ValueError(
                "`query_keypoint_name` must be provided when `query_point` is set to 'KEYPOINT'."
            )
        if target_point == KEYPOINT_POINT_OPTION and not target_keypoint_name:
            raise ValueError(
                "`target_keypoint_name` must be provided when `target_point` is set to 'KEYPOINT'."
            )

        # The keypoint-detection kind is a (KeyPoints, Detections) tuple; the
        # bbox component carries xyxy, detection ids and the flattened per-box
        # keypoint payloads used below.
        _, query_detections = split_key_point_prediction(query_predictions)
        _, target_detections = split_key_point_prediction(target_predictions)

        query_points = resolve_anchor_points(
            detections=query_detections,
            point=query_point,
            keypoint_name=query_keypoint_name,
        )
        target_points = resolve_anchor_points(
            detections=target_detections,
            point=target_point,
            keypoint_name=target_keypoint_name,
        )

        distances, matched_query_indices, matched_target_indices = (
            match_query_to_targets(
                query_detections=query_detections,
                target_detections=target_detections,
                query_points=query_points,
                target_points=target_points,
                max_distance=max_distance,
            )
        )
        # Mutates the input prediction in place (same convention as the numpy
        # block and Velocity/Time in Zone): the per-box scalar lands in
        # `bboxes_metadata`, so `matched_query_detections` below picks it up
        # for free via the index-slice. Entry dicts are copied so the write
        # cannot leak into other references of the same metadata dicts.
        number_of_queries = int(query_detections.xyxy.shape[0])
        bboxes_metadata = query_detections.bboxes_metadata
        if bboxes_metadata is None:
            bboxes_metadata = [{} for _ in range(number_of_queries)]
        else:
            bboxes_metadata = [
                dict(box_metadata) if box_metadata is not None else {}
                for box_metadata in bboxes_metadata
            ]
        for box_metadata, distance in zip(bboxes_metadata, distances):
            box_metadata[NEAREST_TARGET_DISTANCE_KEY] = distance
        query_detections.bboxes_metadata = bboxes_metadata

        return {
            OUTPUT_KEY_QUERY_PREDICTIONS: query_predictions,
            OUTPUT_KEY_MATCHED_QUERY_DETECTIONS: take_prediction_by_indices(
                query_predictions, matched_query_indices
            ),
            OUTPUT_KEY_MATCHED_TARGET_DETECTIONS: take_prediction_by_indices(
                target_predictions, matched_target_indices
            ),
        }


# Bbox anchor resolution: (x, y) per option as (column selectors) over
# [x_min, y_min, x_max, y_max, center_x, center_y] - the torch mirror of
# ``sv.Detections.get_anchors_coordinates(anchor=sv.Position[point])``.
_X_MIN, _Y_MIN, _X_MAX, _Y_MAX, _CENTER_X, _CENTER_Y = range(6)
_BBOX_ANCHOR_COLUMNS = {
    "CENTER": (_CENTER_X, _CENTER_Y),
    "CENTER_LEFT": (_X_MIN, _CENTER_Y),
    "CENTER_RIGHT": (_X_MAX, _CENTER_Y),
    "TOP_CENTER": (_CENTER_X, _Y_MIN),
    "TOP_LEFT": (_X_MIN, _Y_MIN),
    "TOP_RIGHT": (_X_MAX, _Y_MIN),
    "BOTTOM_LEFT": (_X_MIN, _Y_MAX),
    "BOTTOM_CENTER": (_CENTER_X, _Y_MAX),
    "BOTTOM_RIGHT": (_X_MAX, _Y_MAX),
}


def resolve_anchor_points(
    detections: TensorNativeDetections,
    point: str,
    keypoint_name: Optional[str],
) -> torch.Tensor:
    """Resolve one (x, y) anchor per detection as an ``(N, 2)`` float32 tensor
    on the prediction's device."""
    if point == KEYPOINT_POINT_OPTION:
        return resolve_keypoint_anchor_points(
            detections=detections, keypoint_name=keypoint_name
        )
    if point not in _BBOX_ANCHOR_COLUMNS:
        raise ValueError(
            f"Invalid anchor point option '{point}'. Supported options: "
            f"{sorted(_BBOX_ANCHOR_COLUMNS)} or '{KEYPOINT_POINT_OPTION}'."
        )
    xyxy = detections.xyxy.detach().to(dtype=torch.float32)
    centers = (xyxy[:, :2] + xyxy[:, 2:]) * 0.5
    columns = torch.cat([xyxy, centers], dim=1)  # (N, 6)
    x_column, y_column = _BBOX_ANCHOR_COLUMNS[point]
    return torch.stack([columns[:, x_column], columns[:, y_column]], dim=1)


def resolve_keypoint_anchor_points(
    detections: TensorNativeDetections,
    keypoint_name: str,
) -> torch.Tensor:
    """Read the named keypoint per detection from the flattened
    ``bboxes_metadata`` payloads (``keypoints_xy`` / ``keypoints_class_name`` -
    the native mirror of the sv data columns the numpy block reads).

    Detections whose keypoint set does not include ``keypoint_name`` (e.g. an
    occluded joint) get a NaN anchor point, which propagates through the
    distance matrix and naturally excludes them from matching - the same
    graceful degradation as the numpy block.
    """
    number_of_detections = int(detections.xyxy.shape[0])
    bboxes_metadata = detections.bboxes_metadata or []
    has_keypoint_payload = any(
        box_metadata is not None
        and KEYPOINTS_XY_KEY_IN_SV_DETECTIONS in box_metadata
        and KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS in box_metadata
        for box_metadata in bboxes_metadata
    )
    if number_of_detections > 0 and not has_keypoint_payload:
        raise ValueError(
            "`query_point`/`target_point` set to 'KEYPOINT' but the corresponding "
            "predictions do not contain keypoint data. Provide keypoint detection "
            "predictions to use this option."
        )
    points = torch.full((number_of_detections, 2), float("nan"), dtype=torch.float32)
    for index in range(number_of_detections):
        box_metadata = (
            bboxes_metadata[index] if index < len(bboxes_metadata) else None
        ) or {}
        keypoint_names = box_metadata.get(KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS)
        keypoints_xy = box_metadata.get(KEYPOINTS_XY_KEY_IN_SV_DETECTIONS)
        if keypoint_names is None or keypoints_xy is None:
            continue
        for name, coordinates in zip(keypoint_names, keypoints_xy):
            if str(name) == keypoint_name:
                points[index, 0] = float(coordinates[0])
                points[index, 1] = float(coordinates[1])
                break
    return points.to(device=detections.xyxy.device)


def _detection_ids(detections: TensorNativeDetections) -> List[Optional[str]]:
    """Per-box ``detection_id`` (``None`` when absent) - the native mirror of
    the ``sv.Detections.data["detection_id"]`` column."""
    number_of_detections = int(detections.xyxy.shape[0])
    bboxes_metadata = detections.bboxes_metadata or []
    return [
        (
            (bboxes_metadata[index] or {}).get(DETECTION_ID_KEY)
            if index < len(bboxes_metadata)
            else None
        )
        for index in range(number_of_detections)
    ]


def match_query_to_targets(
    query_detections: TensorNativeDetections,
    target_detections: TensorNativeDetections,
    query_points: torch.Tensor,
    target_points: torch.Tensor,
    max_distance: Optional[int],
) -> Tuple[List[Optional[float]], List[int], List[int]]:
    num_query = int(query_points.shape[0])
    if int(target_points.shape[0]) == 0:
        return [None] * num_query, [], []

    device = query_points.device
    target_points = target_points.to(device)

    # Plain brute-force pairwise distance matrix (not a KD-tree), same as the
    # numpy block: typical detection counts here are tens per set.
    diff = query_points[:, None, :] - target_points[None, :, :]
    distance_matrix = torch.sqrt((diff * diff).sum(dim=-1))  # (Q, T)

    # Invalid candidates are tracked in a boolean mask instead of NaN-assigning
    # the matrix (torch mirror of the numpy block's NaN bookkeeping). NaN
    # distances from missing-keypoint anchors start out invalid.
    valid = ~torch.isnan(distance_matrix)

    # A target counts as a self-match (and is excluded) whenever it shares the
    # query detection's `detection_id`. Boxes without a `detection_id` are
    # never treated as self-matches - when neither side carries ids the
    # exclusion is skipped entirely, mirroring the numpy block's behaviour for
    # a missing `detection_id` column.
    query_ids = _detection_ids(query_detections)
    target_ids = _detection_ids(target_detections)
    if any(query_id is not None for query_id in query_ids) and any(
        target_id is not None for target_id in target_ids
    ):
        self_match = torch.tensor(
            [
                [
                    query_id is not None and query_id == target_id
                    for target_id in target_ids
                ]
                for query_id in query_ids
            ],
            dtype=torch.bool,
            device=device,
        )
        valid &= ~self_match

    if max_distance is not None:
        # Candidates beyond the limit are dropped before ranking (boundary
        # inclusive), so a tie can only form among in-range targets.
        valid &= distance_matrix <= max_distance

    infinity = float("inf")
    filled = torch.where(
        valid, distance_matrix, torch.full_like(distance_matrix, infinity)
    )
    # An all-invalid query row keeps +inf as its minimum and is omitted from
    # the paired outputs (no placeholder row); `query_predictions` still
    # carries it, with `nearest_target_distance` set to `None`.
    min_per_row = filled.min(dim=1).values  # (Q,)

    # A tie duplicates the query row once per tied target; `torch.nonzero` is
    # row-major like `np.where`, keeping the two paired outputs index-aligned.
    tie_mask = valid & (filled <= (min_per_row[:, None] + TIE_EPSILON_PX))

    # The only device->host hops: the batched minima and the tie indices.
    matched_pairs = torch.nonzero(tie_mask, as_tuple=False).cpu().tolist()
    matched_query_indices = [pair[0] for pair in matched_pairs]
    matched_target_indices = [pair[1] for pair in matched_pairs]
    distances = [
        None if math.isinf(distance) else float(distance)
        for distance in min_per_row.cpu().tolist()
    ]
    return distances, matched_query_indices, matched_target_indices
