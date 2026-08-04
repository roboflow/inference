from typing import List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    NEAREST_TARGET_DISTANCE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = (
    "Find the closest target detection for each query detection using pixel distance."
)

LONG_DESCRIPTION = """
Perform a nearest-neighbor spatial join between two detection sets by finding, for each detection in a query set, the closest detection (or detections, on a tie) in a target set, using plain 2D Euclidean pixel distance between configurable anchor points.

## How This Block Works

This block matches each detection in a query set to its nearest neighbor(s) in a target set. The block:

1. Receives query predictions and target predictions (object detection, instance segmentation, or keypoint detection) - the same set can be passed as both query and target
2. Resolves an anchor point for every detection in each set, based on the selected `query_point` and `target_point` options: the center, one of the four corners, one of the four edge midpoints of the bounding box, or a named keypoint (requires keypoint detection predictions and `query_keypoint_name`/`target_keypoint_name`)
3. Computes a full pairwise pixel-distance matrix between every query anchor point and every target anchor point (brute-force, not a spatial tree - appropriate for the tens-of-detections-per-frame scale this block targets)
4. For each query detection, excludes any target detection that shares its `detection_id` (self-match exclusion), so passing the same detections as both query and target does not match a detection against itself
5. If `max_distance` is set, discards any remaining candidate target farther than `max_distance` pixels from the query anchor point before ranking - a target beyond the limit is treated the same as an excluded self-match
6. Finds the minimum distance for each query detection, then includes every remaining target detection whose distance is within a fixed 1-pixel epsilon of that minimum, so exact or near-exact ties produce multiple matches instead of an arbitrary single winner
7. Attaches the nearest distance (in pixels) to each query detection's metadata; detections with no eligible target (empty target set, every target excluded as a self-match, or every remaining target farther than `max_distance`) get `None`
8. Returns three flat, aligned outputs: the enriched query predictions, and two same-length/same-order detection sets - `matched_query_detections` and `matched_target_detections` - with one row per query-target pairing. A query detection with a tie appears once per tied match (its row is duplicated); a query detection with no valid match is omitted from both, and only shows up in `query_predictions` (with `nearest_target_distance` set to `None`)

Distance is always plain, uncalibrated 2D Euclidean pixel distance - there is no real-world unit conversion, unlike the Distance Measurement block. This keeps the block simple for use cases that only need relative/pixel-space proximity.

## Common Use Cases

- **Zone/Anchor Assignment**: Assign each detected object to its nearest reference point or anchor detection (e.g. match people to the nearest exit sign, match vehicles to the nearest parking spot marker), enabling proximity-based assignment workflows
- **Cross-Class Association**: Associate detections of one class with the nearest detection of another class (e.g. match detected hands to the nearest detected tool, match players to the nearest ball detection), enabling relationship analysis between different object types
- **Duplicate/Overlap Analysis**: Compare a detection set against itself to find each detection's closest neighbor (e.g. find the closest other person in a crowd, identify tightly clustered detections), enabling density and clustering analysis
- **Tracking Handoff**: Match detections between two model outputs on the same frame (e.g. match a fast general detector's output to a slower specialist model's output) to combine results, enabling multi-model fusion workflows

## Connecting to Other Blocks

This block receives two detection sets and produces the enriched query predictions plus two flat, aligned detection sets representing every query-target pairing - no dimensionality tricks or intermediate merge step required, since `matched_query_detections` and `matched_target_detections` are ordinary detection sets of the same shape any detection-producing block returns:

- **After object detection, instance segmentation, or keypoint detection model blocks** to build the query and target sets to match, enabling detection-to-matching workflows
- **Before logic blocks** like Continue If or Detections Filter to make decisions based on `nearest_target_distance` (e.g. continue if the nearest match is within a threshold distance) or to filter `matched_query_detections`/`matched_target_detections` directly, enabling proximity-based decision workflows
- **Before visualization blocks** like Label Visualization or Bounding Box Visualization to display the enriched `query_predictions`, or `matched_query_detections`/`matched_target_detections` side by side to draw matched pairs, enabling distance-annotated visualizations
- **Before data storage blocks** like CSV Formatter to export every query-target pairing as a row, enabling matched-pair logging workflows

## Requirements

This block requires two sets of detection predictions (object detection, instance segmentation, or keypoint detection); the same set can be used for both `query_predictions` and `target_predictions`. To use the `KEYPOINT` anchor option for either set, that set must be keypoint detection predictions and the corresponding `query_keypoint_name`/`target_keypoint_name` must be provided. Self-match exclusion relies on `detection_id` being present on both sets - this is populated automatically for all Roboflow object detection, instance segmentation, and keypoint detection model blocks. `max_distance` is optional; leave it unset to match every query detection to its nearest target regardless of distance.

Note that `query_predictions` is enriched in place - the same `sv.Detections` object passed in is mutated (a new `nearest_target_distance` field is added to its `.data`) and returned, the same convention used by blocks like Velocity and Time in Zone. Avoid feeding the same selector into two independent branches of a workflow if each branch needs to see its own, unmodified `nearest_target_distance`.
"""

TIE_EPSILON_PX = 1.0

KEYPOINT_POINT_OPTION = "KEYPOINT"
ANCHOR_POINT_OPTIONS = [
    "CENTER",
    "CENTER_LEFT",
    "CENTER_RIGHT",
    "TOP_CENTER",
    "TOP_LEFT",
    "TOP_RIGHT",
    "BOTTOM_LEFT",
    "BOTTOM_CENTER",
    "BOTTOM_RIGHT",
    KEYPOINT_POINT_OPTION,
]

OUTPUT_KEY_QUERY_PREDICTIONS = "query_predictions"
OUTPUT_KEY_MATCHED_QUERY_DETECTIONS = "matched_query_detections"
OUTPUT_KEY_MATCHED_TARGET_DETECTIONS = "matched_target_detections"

DETECTIONS_KIND = [
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
]


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Nearest Neighbor Detection Match",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-crosshairs",
                "blockPriority": 12,
            },
        }
    )

    type: Literal["roboflow_core/detections_nearest_neighbor@v1"]

    query_predictions: Selector(kind=DETECTIONS_KIND) = Field(
        title="Query Detections",
        description="Detections to find nearest-neighbor matches for. Each detection in this set is matched against `target_predictions`. The same selector can be used for both `query_predictions` and `target_predictions` (e.g. to find each detection's closest neighbor within one set) - a detection is never matched against itself.",
        examples=["$steps.model.predictions"],
    )

    target_predictions: Selector(kind=DETECTIONS_KIND) = Field(
        title="Target Detections",
        description="Detections to search for the nearest match within, for every detection in `query_predictions`. Can be the same selector as `query_predictions`.",
        examples=["$steps.model.predictions"],
    )

    query_point: Union[
        Literal[tuple(ANCHOR_POINT_OPTIONS)],
        Selector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        title="Query Anchor Point",
        default="CENTER",
        description="Anchor point used to represent each query detection's location: CENTER, a corner (TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT), an edge midpoint (TOP_CENTER, BOTTOM_CENTER, CENTER_LEFT, CENTER_RIGHT), or KEYPOINT (requires `query_predictions` to be keypoint detection predictions and `query_keypoint_name` to be set).",
        examples=["CENTER", "$inputs.query_point"],
    )

    target_point: Union[
        Literal[tuple(ANCHOR_POINT_OPTIONS)],
        Selector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        title="Target Anchor Point",
        default="CENTER",
        description="Anchor point used to represent each target detection's location. Same options as `query_point`, set independently. Use KEYPOINT to match against a named keypoint on `target_predictions` (requires `target_predictions` to be keypoint detection predictions and `target_keypoint_name` to be set).",
        examples=["CENTER", "$inputs.target_point"],
    )

    query_keypoint_name: Optional[Union[str, Selector(kind=[STRING_KIND])]] = Field(
        title="Query Keypoint Name",
        default=None,
        description="Name of the keypoint class to use as the query anchor point. Required only when `query_point` is set to KEYPOINT; `query_predictions` must then be keypoint detection predictions.",
        examples=["left_shoulder", "$inputs.query_keypoint_name"],
        json_schema_extra={
            "relevant_for": {
                "query_point": {
                    "values": [KEYPOINT_POINT_OPTION],
                    "required": True,
                },
            },
        },
    )

    target_keypoint_name: Optional[Union[str, Selector(kind=[STRING_KIND])]] = Field(
        title="Target Keypoint Name",
        default=None,
        description="Name of the keypoint class to use as the target anchor point. Required only when `target_point` is set to KEYPOINT; `target_predictions` must then be keypoint detection predictions.",
        examples=["left_shoulder", "$inputs.target_keypoint_name"],
        json_schema_extra={
            "relevant_for": {
                "target_point": {
                    "values": [KEYPOINT_POINT_OPTION],
                    "required": True,
                },
            },
        },
    )

    max_distance: Optional[Union[int, Selector(kind=[INTEGER_KIND])]] = Field(
        title="Maximum Match Distance",
        default=None,
        description="Maximum allowed pixel distance between a query detection and a candidate target. Targets farther than this are excluded before ranking, the same way self-matches are excluded. A query detection whose nearest remaining target is still farther than this (or has no remaining target at all) gets `nearest_target_distance = None` and is omitted from `matched_query_detections`/`matched_target_detections`. Leave unset for no limit.",
        examples=[50, "$inputs.max_distance"],
        ge=0,
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_KEY_QUERY_PREDICTIONS,
                kind=DETECTIONS_KIND,
            ),
            OutputDefinition(
                name=OUTPUT_KEY_MATCHED_QUERY_DETECTIONS,
                kind=DETECTIONS_KIND,
            ),
            OutputDefinition(
                name=OUTPUT_KEY_MATCHED_TARGET_DETECTIONS,
                kind=DETECTIONS_KIND,
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class DetectionsNearestNeighborBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        query_predictions: sv.Detections,
        target_predictions: sv.Detections,
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

        query_points = resolve_anchor_points(
            detections=query_predictions,
            point=query_point,
            keypoint_name=query_keypoint_name,
        )
        target_points = resolve_anchor_points(
            detections=target_predictions,
            point=target_point,
            keypoint_name=target_keypoint_name,
        )

        distances, matched_query_indices, matched_target_indices = (
            match_query_to_targets(
                query_detections=query_predictions,
                target_detections=target_predictions,
                query_points=query_points,
                target_points=target_points,
                max_distance=max_distance,
            )
        )
        # Mutates the input `sv.Detections` object in place (same convention
        # as Velocity/Time in Zone), so `matched_query_detections` below picks
        # up this field for free via the index-slice.
        query_predictions.data[NEAREST_TARGET_DISTANCE_KEY] = np.array(
            distances, dtype=object
        )

        return {
            OUTPUT_KEY_QUERY_PREDICTIONS: query_predictions,
            OUTPUT_KEY_MATCHED_QUERY_DETECTIONS: query_predictions[
                matched_query_indices
            ],
            OUTPUT_KEY_MATCHED_TARGET_DETECTIONS: target_predictions[
                matched_target_indices
            ],
        }


def resolve_anchor_points(
    detections: sv.Detections,
    point: str,
    keypoint_name: Optional[str],
) -> np.ndarray:
    if point == KEYPOINT_POINT_OPTION:
        return resolve_keypoint_anchor_points(
            detections=detections, keypoint_name=keypoint_name
        )
    return detections.get_anchors_coordinates(anchor=getattr(sv.Position, point))


def resolve_keypoint_anchor_points(
    detections: sv.Detections,
    keypoint_name: str,
) -> np.ndarray:
    keypoints_xy = detections.data.get(KEYPOINTS_XY_KEY_IN_SV_DETECTIONS)
    keypoints_class_name = detections.data.get(
        KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS
    )
    if keypoints_xy is None or keypoints_class_name is None:
        raise ValueError(
            "`query_point`/`target_point` set to 'KEYPOINT' but the corresponding "
            "predictions do not contain keypoint data. Provide keypoint detection "
            "predictions to use this option."
        )
    # Detections whose keypoint set does not include `keypoint_name` (e.g. an
    # occluded joint) get a NaN anchor point, which propagates through the
    # distance matrix and naturally excludes them from matching, rather than
    # raising - consistent with the block's other graceful-degradation cases.
    points = np.full((len(detections), 2), np.nan, dtype=np.float64)
    for i, class_names in enumerate(keypoints_class_name):
        matching_indices = np.where(np.asarray(class_names) == keypoint_name)[0]
        if len(matching_indices) > 0:
            points[i] = keypoints_xy[i][matching_indices[0]]
    return points


def match_query_to_targets(
    query_detections: sv.Detections,
    target_detections: sv.Detections,
    query_points: np.ndarray,
    target_points: np.ndarray,
    max_distance: Optional[int],
) -> Tuple[List[Optional[float]], List[int], List[int]]:
    num_query = len(query_detections)
    if len(target_detections) == 0:
        return [None] * num_query, [], []

    # Plain brute-force pairwise distance matrix (not a KD-tree): typical
    # detection counts here are tens per set (single-frame use case), so
    # tree-construction overhead isn't worth the added complexity a KD-tree
    # would introduce for tie handling.
    diff = query_points[:, None, :] - target_points[None, :, :]
    distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    query_ids = query_detections.data.get(DETECTION_ID_KEY)
    target_ids = target_detections.data.get(DETECTION_ID_KEY)
    if query_ids is not None and target_ids is not None:
        # A target counts as a self-match (and is excluded) whenever it shares
        # the query detection's `detection_id` - this covers the same-set case
        # automatically without a separate config flag. If `detection_id` is
        # missing on either set, self-exclusion is skipped entirely rather
        # than raising.
        distance_matrix[
            np.asarray(query_ids)[:, None] == np.asarray(target_ids)[None, :]
        ] = np.nan

    if max_distance is not None:
        # Candidates beyond the limit are dropped before ranking, so a tie
        # can only ever form among targets that are themselves within
        # `max_distance` - not merely within epsilon of an out-of-range
        # minimum.
        distance_matrix[distance_matrix > max_distance] = np.nan

    # An all-NaN query row (no valid target) keeps NaN as its minimum - such
    # rows are sliced out before `np.nanmin` only to avoid its all-NaN-slice
    # warning. NaN compares False in the tie mask below, so those rows are
    # omitted from the paired outputs entirely (no placeholder row);
    # `query_predictions` still carries them, with `nearest_target_distance`
    # left as `None`.
    valid_rows = ~np.all(np.isnan(distance_matrix), axis=1)
    min_per_row = np.full(num_query, np.nan)
    min_per_row[valid_rows] = np.nanmin(distance_matrix[valid_rows], axis=1)

    # A tie duplicates the query row once per tied target; row-major `np.where`
    # keeps the two paired outputs the same length and index-aligned.
    tie_mask = distance_matrix <= (min_per_row[:, None] + TIE_EPSILON_PX)
    matched_query_indices, matched_target_indices = (
        x.tolist() for x in np.where(tie_mask)
    )
    distances = [None if np.isnan(d) else float(d) for d in min_per_row]
    return distances, matched_query_indices, matched_target_indices
