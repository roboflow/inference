"""Pair query detections with nearest targets: ``v2/detections_nearest_neighbor@v1``.

Port of V1 ``roboflow_core/detections_nearest_neighbor@v1``. As V1::

    q, t      = one anchor point per query / target row
    d[i, j]   = |q_i - t_j|                     (pixels, brute force)
    eligible  = not the same row, and d <= max_distance when it is set
    nearest_i = min over eligible j of d[i, j]  (None when nothing is eligible)
    pairs     = every eligible (i, j) with d[i, j] <= nearest_i + 1 px

A tie within ``TIE_EPSILON_PX`` repeats the query row once per tied target, so
``matched_query_detections[k]`` and ``matched_target_detections[k]`` are pair
``k``. Pairs are in query order, then target order. Matched rows are selected
with their masks, keypoints, row metadata and row identity.

Anchors are box points (``CENTER``, corners, edge centres) or, with
``KEYPOINT``, the named keypoint of each skeleton. Names come from the
prediction's ``key_points_classes`` for the row's class; a skeleton without
that name, or with the keypoint's confidence at most 0, has no anchor and
takes part in no pair.

Distances, ``max_distance`` and the 1 px tie rule use one shared local pixel
frame. Known frame conflicts are rejected; missing metadata stays unknown.
Align crop, resize or different-camera predictions before matching.

Limits as V1: 1000 rows per side, 10 000 pairs, 100 pairs when masks are
present.

Differences from V1: ``query_predictions`` is a copy with
``nearest_target_distance`` in each row's metadata (V1 writes into the input);
a row is "the same row" when both predictions carry the same V2 row identity
(V1: equal ``detection_id``); a keypoint prediction without boxes is a
``ContractError``; a missing keypoint is found from confidence (V1: from the
flattened per-box keypoint lists).
"""

from dataclasses import replace
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

import torch
from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.execution_engine.constants import NEAREST_TARGET_DISTANCE_KEY
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import (
    FRAME_RECORD_KEY,
    FrameRef,
    _frame_fields,
    _legacy_frame_fields,
    _same_value,
)
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import row_record
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

PREDICTION_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
)
TIE_EPSILON_PX = 1.0
MAX_DETECTIONS_PER_SET = 1_000
MAX_MATCHED_PAIRS = 10_000
MAX_MATCHED_PAIRS_WITH_MASKS = 100
KEYPOINT = "KEYPOINT"
# Anchor (x, y) as columns of [x_min, y_min, x_max, y_max, center_x, center_y].
BOX_ANCHOR_COLUMNS = {
    "CENTER": (4, 5),
    "CENTER_LEFT": (0, 5),
    "CENTER_RIGHT": (2, 5),
    "TOP_CENTER": (4, 1),
    "TOP_LEFT": (0, 1),
    "TOP_RIGHT": (2, 1),
    "BOTTOM_LEFT": (0, 3),
    "BOTTOM_CENTER": (4, 3),
    "BOTTOM_RIGHT": (2, 3),
}
AnchorPoint = Literal[
    "CENTER",
    "CENTER_LEFT",
    "CENTER_RIGHT",
    "TOP_CENTER",
    "TOP_LEFT",
    "TOP_RIGHT",
    "BOTTOM_LEFT",
    "BOTTOM_CENTER",
    "BOTTOM_RIGHT",
    "KEYPOINT",
]


class DetectionsNearestNeighborBlock(Block):
    """Find, for every query detection, the closest target detection.

    Example: query = people, target = forklifts, ``max_distance`` 200; each
    person within 200 px of a forklift is paired with the closest one, and
    ``query_predictions`` tells every person's distance to it.
    """

    type = "v2/detections_nearest_neighbor@v1"
    outputs = {
        "query_predictions": Output(
            *PREDICTION_KINDS,
            source="query_predictions",
            description="All query rows with nearest_target_distance (None if none).",
        ),
        "matched_query_detections": Output(
            *PREDICTION_KINDS,
            source="query_predictions",
            description="Query row of every pair; repeated on ties.",
        ),
        "matched_target_detections": Output(
            *PREDICTION_KINDS,
            source="target_predictions",
            description="Target row of every pair, aligned with the query rows.",
        ),
    }

    class Params(BlockParams):
        query_predictions: Ref(*PREDICTION_KINDS) = Field(
            description="Detections to find neighbours for."
        )
        target_predictions: Ref(*PREDICTION_KINDS) = Field(
            description="Detections to search; may be the query itself."
        )
        query_point: AnchorPoint | Ref(STRING_KIND) = Field(
            default="CENTER", description="Anchor of each query row."
        )
        target_point: AnchorPoint | Ref(STRING_KIND) = Field(
            default="CENTER", description="Anchor of each target row."
        )
        query_keypoint_name: Optional[str] | Ref(STRING_KIND) = Field(
            default=None, description="Keypoint name when query_point is KEYPOINT."
        )
        target_keypoint_name: Optional[str] | Ref(STRING_KIND) = Field(
            default=None, description="Keypoint name when target_point is KEYPOINT."
        )
        max_distance: Optional[StrictInt] | Ref(INTEGER_KIND) = Field(
            default=None,
            ge=0,
            description="Farthest pair in pixels, inclusive; None for no limit.",
        )

        @model_validator(mode="after")
        def _keypoint_anchor_has_a_name(
            self,
        ) -> "DetectionsNearestNeighborBlock.Params":
            # The engine reruns this check after resolving selected values.
            for side in ("query", "target"):
                if getattr(self, f"{side}_point") == KEYPOINT and not getattr(
                    self, f"{side}_keypoint_name"
                ):
                    raise ValueError(
                        f"{side}_keypoint_name is required when {side}_point is "
                        f"{KEYPOINT}"
                    )

            return self

    def run(
        self,
        *,
        query_predictions: Any,
        target_predictions: Any,
        query_point: str,
        target_point: str,
        query_keypoint_name: Optional[str],
        target_keypoint_name: Optional[str],
        max_distance: Optional[int],
    ) -> Dict[str, Any]:
        """Pair query rows with their nearest target rows.

        Args:
            query_predictions: Detections, instance or keypoint predictions.
            target_predictions: Same families as ``query_predictions``.
            query_point: Anchor option of the query rows.
            target_point: Anchor option of the target rows.
            query_keypoint_name: Keypoint for a ``KEYPOINT`` query anchor.
            target_keypoint_name: Keypoint for a ``KEYPOINT`` target anchor.
            max_distance: Largest pair distance in pixels, or ``None``.

        Returns:
            ``query_predictions``, ``matched_query_detections`` and
            ``matched_target_detections``.

        Raises:
            ContractError: On keypoints without boxes, an unknown anchor, more
                than 1000 rows on a side, incompatible known frames or more pairs
                than the limit.
        """
        query_boxes = self._boxes(query_predictions, side="query")
        target_boxes = self._boxes(target_predictions, side="target")
        query_anchors = anchor_points(
            query_predictions, point=query_point, keypoint_name=query_keypoint_name
        )
        target_anchors = anchor_points(
            target_predictions, point=target_point, keypoint_name=target_keypoint_name
        )
        if len(query_boxes.xyxy) and len(target_boxes.xyxy):
            _require_shared_local_frame(query_predictions, target_predictions)

        distances = (
            query_anchors[:, None, :]
            .sub(target_anchors.to(query_anchors.device)[None, :, :])
            .norm(dim=-1)
        )
        eligible = ~distances.isnan() & ~same_rows(query_boxes, target_boxes)
        if max_distance is not None:
            eligible &= distances <= max_distance

        nearest, ties = nearest_targets(distances, eligible)
        pair_limit = MAX_MATCHED_PAIRS
        if _has_masks(query_boxes) or _has_masks(target_boxes):
            pair_limit = MAX_MATCHED_PAIRS_WITH_MASKS
        # Counted on the mask: widespread ties could make far more pairs than
        # rows, and every pair copies its rows (masks included).
        pair_count = int(ties.sum())
        if pair_count > pair_limit:
            raise ContractError(
                f"{self.type}: {pair_count} query-target pairs exceed the limit "
                f"of {pair_limit} (100 when masks are present); many rows share "
                "nearly the same anchor point. Reduce or separate the detections."
            )

        pairs = torch.nonzero(ties)  # row-major: query order, then target order
        result = {
            "query_predictions": _with_distances(query_predictions, nearest),
            "matched_query_detections": select_predictions(
                query_predictions, indices=pairs[:, 0]
            ),
            "matched_target_detections": select_predictions(
                target_predictions, indices=pairs[:, 1]
            ),
        }

        return result

    def _boxes(self, prediction: Any, *, side: str) -> Any:
        boxes = prediction[1] if isinstance(prediction, tuple) else prediction
        if boxes is None:
            raise ContractError(
                f"{self.type}: the {side} keypoint prediction has no bounding "
                "boxes; use a keypoint model that also returns boxes"
            )
        if len(boxes.xyxy) > MAX_DETECTIONS_PER_SET:
            raise ContractError(
                f"{self.type}: {side}_predictions has {len(boxes.xyxy)} rows, "
                f"more than the {MAX_DETECTIONS_PER_SET} a full pairwise "
                "comparison allows; filter them upstream"
            )

        return boxes


def _require_shared_local_frame(query: Any, target: Any) -> None:
    # Use the shared frame readers, preserving each missing field as unknown.
    # Both tuple components must agree, including the one supplying anchors.
    known = {}
    for side, prediction in (("query", query), ("target", target)):
        carriers = prediction if isinstance(prediction, tuple) else (prediction,)
        for carrier in carriers:
            metadata = carrier.image_metadata
            if metadata is None:
                continue

            try:
                if not isinstance(metadata, Mapping):
                    raise ContractError("image_metadata must be a dict")

                fields = _legacy_frame_fields(metadata)
                record = metadata.get(FRAME_RECORD_KEY)
                if record is not None:
                    recorded = _frame_fields(FrameRef.from_record(record))
                    for name, value in fields.items():
                        if value is not None and not _same_value(
                            value, recorded[name], exact_integers=True
                        ):
                            raise ContractError(
                                f"legacy {name} disagrees with the frame record"
                            )
                    fields = recorded

                for name, value in fields.items():
                    if value is None:
                        continue
                    if name in known and not _same_value(
                        value, known[name], exact_integers=True
                    ):
                        raise ContractError(
                            f"conflicting {name}: {value!r} vs {known[name]!r}"
                        )
                    known.setdefault(name, value)
            except ContractError as error:
                raise ContractError(
                    f"{DetectionsNearestNeighborBlock.type}: {side} prediction "
                    f"frame metadata is incompatible: {error}. Align both "
                    "predictions into the same local pixel frame before matching "
                    "(including crops, resizes and camera sources); distances, "
                    "max_distance and ties use that frame's pixels."
                ) from error


def anchor_points(
    prediction: Any, *, point: str, keypoint_name: Optional[str]
) -> torch.Tensor:
    """Return one ``(x, y)`` anchor per row; NaN where a keypoint is missing.

    Args:
        prediction: Detections, instance detections or a
            ``(KeyPoints, Detections)`` tuple.
        point: A ``BOX_ANCHOR_COLUMNS`` option or ``"KEYPOINT"``.
        keypoint_name: Keypoint name for ``"KEYPOINT"``.

    Returns:
        ``(N, 2)`` ``float32`` tensor on the rows' device.

    Raises:
        ContractError: On an unknown option, or ``"KEYPOINT"`` without keypoints.
    """
    if point == KEYPOINT:
        if not isinstance(prediction, tuple):
            raise ContractError(
                "KEYPOINT anchors need a keypoint prediction, got "
                f"{type(prediction).__name__}"
            )
        return keypoint_anchor_points(prediction[0], keypoint_name=keypoint_name)
    if point not in BOX_ANCHOR_COLUMNS:
        raise ContractError(
            f"unknown anchor point {point!r}; use one of "
            f"{sorted(BOX_ANCHOR_COLUMNS) + [KEYPOINT]}"
        )

    boxes = prediction[1] if isinstance(prediction, tuple) else prediction
    xyxy = boxes.xyxy.to(torch.float32)
    columns = torch.cat([xyxy, (xyxy[:, :2] + xyxy[:, 2:]) * 0.5], dim=1)
    x_column, y_column = BOX_ANCHOR_COLUMNS[point]

    return columns[:, [x_column, y_column]]


def keypoint_anchor_points(key_points: Any, *, keypoint_name: str) -> torch.Tensor:
    """Return the named keypoint of every skeleton, NaN where it is missing.

    Args:
        key_points: ``KeyPoints`` whose ``image_metadata`` has
            ``key_points_classes``, one list of names per class id.
        keypoint_name: Name to look up.

    Returns:
        ``(N, 2)`` ``float32`` tensor on the keypoints' device.

    Raises:
        ContractError: When the names are missing.
    """
    names_per_class = (key_points.image_metadata or {}).get("key_points_classes")
    if names_per_class is None:
        raise ContractError(
            "KEYPOINT anchors need key_points_classes in the keypoint "
            "prediction's image_metadata"
        )

    device = key_points.xy.device
    rows = len(key_points.class_id)
    # Position of the name in each class's keypoint list; -1 when the class
    # lacks it. One extra -1 entry serves class ids beyond the list.
    positions = [
        names.index(keypoint_name) if keypoint_name in names else -1
        for names in names_per_class
    ]
    position_of_class = torch.tensor(positions + [-1], dtype=torch.int64, device=device)
    class_id = key_points.class_id.clamp(0, len(positions))
    position = position_of_class[class_id]
    if rows == 0 or key_points.xy.shape[1] == 0:
        return torch.full((rows, 2), float("nan"), device=device)

    row_index = torch.arange(rows, device=device)
    safe_position = position.clamp(min=0)
    points = key_points.xy[row_index, safe_position].to(torch.float32)
    present = (position >= 0) & (key_points.confidence[row_index, safe_position] > 0)

    return torch.where(present[:, None], points, float("nan"))


def same_rows(query_boxes: Any, target_boxes: Any) -> torch.Tensor:
    """Mark query/target pairs that are the same row of one result.

    Args:
        query_boxes: Query row carrier.
        target_boxes: Target row carrier.

    Returns:
        ``(Q, T)`` bool tensor on the query rows' device; all false unless both
        carry a row record of the same namespace.
    """
    query_record, target_record = row_record(query_boxes), row_record(target_boxes)
    device = query_boxes.xyxy.device
    shape = (len(query_boxes.xyxy), len(target_boxes.xyxy))
    if (
        query_record is None
        or target_record is None
        or query_record.namespace != target_record.namespace
    ):
        return torch.zeros(shape, dtype=torch.bool, device=device)

    query_ids = _row_ids(query_record, device=device)
    target_ids = _row_ids(target_record, device=device)

    return query_ids[:, None] == target_ids[None, :]


def nearest_targets(
    distances: torch.Tensor, eligible: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find each query's nearest eligible targets, ties within 1 px included.

    Args:
        distances: ``(Q, T)`` pixel distances.
        eligible: ``(Q, T)`` bool, pairs that may match.

    Returns:
        ``(Q,)`` nearest distances (``inf`` without an eligible target) and
        the ``(Q, T)`` bool mask of matched pairs.
    """
    filled = torch.where(eligible, distances, float("inf"))
    if filled.shape[1] == 0:
        nearest = filled.new_full((filled.shape[0],), float("inf"))
    else:
        nearest = filled.amin(dim=1)
    ties = eligible & (filled <= nearest[:, None] + TIE_EPSILON_PX)

    return nearest, ties


def _row_ids(record: Any, *, device: torch.device) -> torch.Tensor:
    if record.ids is None:
        return torch.arange(record.rows, device=device)

    return record.ids.to(device)


def _has_masks(boxes: Any) -> bool:
    return getattr(boxes, "mask", None) is not None


def _with_distances(prediction: Any, nearest: torch.Tensor) -> Any:
    # A copy of the prediction whose rows carry their nearest distance.
    key_points, boxes = (
        prediction if isinstance(prediction, tuple) else (None, prediction)
    )
    rows = boxes.bboxes_metadata or [None] * len(boxes.xyxy)
    distances = [
        None if distance == float("inf") else distance
        for distance in nearest.cpu().tolist()
    ]
    annotated = replace(
        boxes,
        bboxes_metadata=[
            {**(row or {}), NEAREST_TARGET_DISTANCE_KEY: distance}
            for row, distance in zip(rows, distances)
        ],
    )

    return annotated if key_points is None else (key_points, annotated)
