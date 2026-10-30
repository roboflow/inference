"""Opt-in mask/OBB consensus, ported from contributor PR #2420.

Reuse v1 voting and aggregation. Native geometry, front and identity come from
one highest-confidence contributor of the class chosen by that aggregation.
"""

import math
from functools import lru_cache
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field
from roboflow_workflows.core_steps.common.oriented import (
    NATIVE_OBB,
    _native_obb_to_sv,
    merge_native_semantics,
    native_obb_to_tensor,
    serialise_native_obb,
)
from roboflow_workflows.core_steps.fusion.detections_consensus import v1
from roboflow_workflows.core_steps.fusion.detections_consensus.v1 import (
    AggregationMode,
    MaskAggregationMode,
)
from roboflow_workflows.environment import ENABLE_TENSOR_DATA_REPRESENTATION
from roboflow_workflows.execution_engine.constants import DETECTION_ID_KEY
from roboflow_workflows.execution_engine.entities.base import OutputDefinition
from roboflow_workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_ORIENTED_OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    ORIENTED_OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RuntimeRestriction,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import DependentResource
from shapely.geometry import Polygon
from supervision.config import ORIENTED_BOX_COORDINATES

_MaskGeometry = Tuple[np.ndarray, Tuple[int, int, int, int], int]

_SOURCE_ORDER = "_consensus_source_order"
_NATIVE_KIND = (
    TENSOR_NATIVE_ORIENTED_OBJECT_DETECTION_PREDICTION_KIND
    if ENABLE_TENSOR_DATA_REPRESENTATION
    else ORIENTED_OBJECT_DETECTION_PREDICTION_KIND
)
_INPUT_KINDS = [_NATIVE_KIND]
if not ENABLE_TENSOR_DATA_REPRESENTATION:
    _INPUT_KINDS += [
        OBJECT_DETECTION_PREDICTION_KIND,
        INSTANCE_SEGMENTATION_PREDICTION_KIND,
        KEYPOINT_DETECTION_PREDICTION_KIND,
    ]


class BlockManifest(v1.BlockManifest):
    """Select geometry-aware voting explicitly with the v2 block type."""

    model_config = ConfigDict(
        json_schema_extra={
            **v1.BlockManifest.model_config["json_schema_extra"],
            "version": "v2",
            "long_description": (
                "Combine detections using v1 voting and class/confidence aggregation. "
                "Compare masks when both ordinary inputs carry masks, rotated corners "
                "when both carry them, and AABBs otherwise. Native OBB inputs require "
                "frozen class heading semantics and ordered corners: the output keeps "
                "the highest-confidence contributor of the chosen output class, "
                "including its footprint, front and identity (source-order ties). "
                "Native and ordinary inputs cannot be mixed. Native inputs support "
                "NumPy and tensor carriers through CPU geometry conversion. Ordinary "
                "tensor inputs are unsupported; use v1 for those inputs."
            ),
        }
    )
    type: Literal["roboflow_core/detections_consensus@v2"] = Field(
        description="Explicit opt-in to mask and oriented-geometry voting."
    )
    predictions_batches: List[Selector(kind=_INPUT_KINDS)] = Field(
        min_length=1,
        validation_alias=AliasChoices("predictions_batches", "predictions"),
        description="Predictions for the same image. Native OBB sources must share frozen class heading semantics. Ordinary inputs are supported in NumPy mode; tensor mode accepts native OBB only.",
        examples=[["$inputs.first", "$inputs.second"]],
    )
    detections_merge_coordinates_aggregation: AggregationMode = Field(
        default=AggregationMode.AVERAGE,
        description="Ordinary box coordinate aggregation: average, largest (max), or smallest (min). Native OBB geometry and front are retained atomically from the highest-confidence contributor of the aggregated output class, with source-order ties; this setting does not refit native OBBs.",
        examples=["average"],
    )

    @classmethod
    @lru_cache(maxsize=None)
    def describe_outputs(cls) -> List[OutputDefinition]:
        """Describe consensus outputs, including the native oriented carrier.

        Returns:
            Output definitions with native geometry compatibility.
        """
        outputs = [
            output.model_copy(deep=True) for output in super().describe_outputs()
        ]
        outputs[0].kind = [_NATIVE_KIND] + (
            [OBJECT_DETECTION_PREDICTION_KIND, INSTANCE_SEGMENTATION_PREDICTION_KIND]
            if not ENABLE_TENSOR_DATA_REPRESENTATION
            else []
        )
        return outputs

    def discover_work_operations(self) -> List[WorkOperation]:
        """Declare geometry-aware detection processing.

        Returns:
            The work performed by this block.
        """
        return [WorkOperation.DETECTION_PROCESSING]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        """Declare unrestricted execution.

        Args:
            ignore_environment_restrictions: Whether to omit host restrictions.

        Returns:
            A complete empty restriction declaration.
        """
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )

    def discover_dependent_resources(self) -> List[DependentResource]:
        """Declare that no external resources are required.

        Returns:
            A complete empty resource declaration.
        """
        return []


class DetectionsConsensusBlockV2(v1.DetectionsConsensusBlockV1):
    """Run geometry-aware voting with v1's established consensus traversal."""

    @classmethod
    def get_manifest(cls):
        """Return the explicit v2 manifest.

        Returns:
            Manifest supporting native oriented predictions.
        """
        return BlockManifest

    def _agree_on_consensus(self, **kwargs):
        sources = kwargs.pop("detections_from_sources")
        tensor = [not isinstance(source, sv.Detections) for source in sources]
        if any(tensor) and not all(tensor):
            raise ValueError("Consensus sources must use the same carrier")

        if any(tensor):
            from inference_models.models.base.object_detection import Detections

            if any(not isinstance(source, Detections) for source in sources):
                raise ValueError("Unsupported consensus tensor carrier")
            sources = [_native_obb_to_sv(source) for source in sources]
            if not all(_is_native(source) for source in sources):
                raise ValueError("V2 ordinary tensor consensus is unsupported; use v1")

        native = [_is_native(source) for source in sources]
        policy = None
        if any(native):
            if not all(native):
                raise ValueError("Native OBB consensus cannot mix ordinary detections")
            policy_carrier = sources[0][:0]
            for source in sources:
                # Validate every source before filtering/voting, including empty maps.
                serialise_native_obb(source)
                if any(
                    type(value) is not bool
                    for value in source.metadata["class_heading"].values()
                ):
                    raise ValueError(
                        "Native OBB requires frozen boolean class semantics"
                    )
                if any(
                    source.metadata["class_heading"].get(str(name)) is not bool(heading)
                    for name, heading in zip(
                        source.data.get("class_name", []),
                        source.data.get("heading", []),
                    )
                ):
                    raise ValueError(
                        "Native OBB heading must match frozen class semantics"
                    )
                if NATIVE_OBB in source.data and not np.all(source.data[NATIVE_OBB]):
                    raise ValueError(
                        "Native OBB consensus cannot mix ordinary detections"
                    )
                policy = merge_native_semantics(policy_carrier, source)
                policy_carrier.metadata = {
                    **policy_carrier.metadata,
                    "class_heading": policy,
                }
            sources = [source[:] for source in sources]
            for index, source in enumerate(sources):
                source.metadata = {**source.metadata, "class_heading": policy}
                # Class names are already per-row; tensor image metadata only
                # describes classes observed by that source and may differ.
                source.metadata.pop("class_names", None)
                source.data[_SOURCE_ORDER] = np.full(len(source), index)
                corners = source.data.get(ORIENTED_BOX_COORDINATES)
                if len(source):
                    source.xyxy = np.concatenate(
                        (corners.min(1), corners.max(1)), axis=1
                    )

        result = v1.agree_on_consensus_for_all_detections_sources(
            detections_from_sources=sources,
            _iou_calculator=(
                calculate_iou if any(native) else _make_mask_iou_calculator(sources)
            ),
            _detection_merger=merge_detections,
            **kwargs,
        )
        parent_id, present, confidence, detections = result
        if any(native):
            if not len(detections):
                detections = sources[0][:0]
            detections.data.pop(_SOURCE_ORDER, None)
            detections.metadata = {**sources[0].metadata, "class_heading": policy}
            if all(tensor):
                if not len(detections):
                    detections.data.setdefault("class_name", np.asarray([], dtype=str))
                detections = native_obb_to_tensor(detections)
        return parent_id, present, confidence, detections


def _is_native(detection):
    return bool(detection.metadata.get(NATIVE_OBB)) or bool(
        np.any(detection.data.get(NATIVE_OBB, False))
    )


def _make_mask_iou_calculator(
    sources: List[sv.Detections],
) -> Callable[[sv.Detections, sv.Detections], float]:
    """Keep lazily derived foreground geometry within one consensus call."""
    masked_sources = [
        source for source in sources if len(source) and source.mask is not None
    ]
    if len(masked_sources) < 2 or any(
        not isinstance(source.mask, np.ndarray) for source in masked_sources
    ):
        return calculate_iou
    if len({source.mask.shape[1:] for source in masked_sources}) != 1:
        return calculate_iou

    identities = []
    for source in masked_sources:
        source_ids = source.data.get(DETECTION_ID_KEY)
        if source_ids is None or any(
            not isinstance(value, str) for value in source_ids
        ):
            return calculate_iou
        identities.extend(source_ids)
    # v1 does not require unique input IDs; aliases must not share cached geometry.
    if len(set(identities)) != len(identities):
        return calculate_iou

    geometry: Optional[dict[str, _MaskGeometry]] = None

    def _calculate_pair_iou(
        detection_a: sv.Detections, detection_b: sv.Detections
    ) -> float:
        if detection_a.mask is None or detection_b.mask is None:
            return calculate_iou(detection_a, detection_b)

        nonlocal geometry
        if geometry is None:
            geometry = {}
            for source in masked_sources:
                boxes = sv.mask_to_xyxy(source.mask)
                for identity, mask, box in zip(
                    source.data[DETECTION_ID_KEY], source.mask, boxes
                ):
                    x_min, y_min, x_max, y_max = map(int, box)
                    # Supervision bounds are inclusive; extraction windows are half-open.
                    x_max += 1
                    y_max += 1
                    area = int(np.count_nonzero(mask[y_min:y_max, x_min:x_max]))
                    geometry[identity] = (mask, (x_min, y_min, x_max, y_max), area)

        mask_a, bounds_a, area_a = geometry[detection_a.data[DETECTION_ID_KEY][0]]
        mask_b, bounds_b, area_b = geometry[detection_b.data[DETECTION_ID_KEY][0]]
        if not area_a or not area_b:
            return 0.0

        x_min, y_min = max(bounds_a[0], bounds_b[0]), max(bounds_a[1], bounds_b[1])
        x_max, y_max = min(bounds_a[2], bounds_b[2]), min(bounds_a[3], bounds_b[3])
        intersection = int(
            np.count_nonzero(
                np.logical_and(
                    mask_a[y_min:y_max, x_min:x_max], mask_b[y_min:y_max, x_min:x_max]
                )
            )
        )
        union = area_a + area_b - intersection
        return float(intersection) / float(union)

    return _calculate_pair_iou


def _mask_pair_iou(detection_a: sv.Detections, detection_b: sv.Detections) -> float:
    """Compare full foreground masks; detector boxes may not contain them."""
    mask_a, mask_b = detection_a.mask[0], detection_b.mask[0]
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def _oriented_box_pair_iou(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    """Exact oriented-box IoU for a pair of (4, 2) corner arrays.

    Computed via polygon intersection rather than coordinate-dependent rasterization.
    """
    polygon_a = Polygon(corners_a)
    polygon_b = Polygon(corners_b)
    if not polygon_a.is_valid or not polygon_b.is_valid:
        return 0.0
    intersection = polygon_a.intersection(polygon_b).area
    union = polygon_a.area + polygon_b.area - intersection
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


def calculate_iou(detection_a: sv.Detections, detection_b: sv.Detections) -> float:
    """Compute IoU on the geometry the detections actually carry.

    Dispatch order: segmentation masks when both sides have them, oriented
    box corners (``data["xyxyxyxy"]``) when both sides have them, and the
    axis-aligned ``xyxy`` boxes otherwise. The v1 block always compared
    ``xyxy``, which over-matches detections whose boxes overlap even though
    their masks or rotated bodies do not.

    Mask pixels can extend beyond detector boxes, so masks are compared before
    the box gate. Oriented corners are contained in their axis-aligned bounds.

    Args:
        detection_a: One detection in the shared image frame.
        detection_b: One candidate from another prediction source.

    Returns:
        Intersection over union of the available geometry.

    Raises:
        ValueError: For mixed native/ordinary or incomplete native geometry.
    """
    native_a = _is_native(detection_a)
    native_b = _is_native(detection_b)
    if native_a or native_b:
        if not (native_a and native_b):
            raise ValueError("Native OBB consensus cannot mix ordinary detections")
        for detection in (detection_a, detection_b):
            if (
                ORIENTED_BOX_COORDINATES not in detection.data
                or "heading" not in detection.data
            ):
                raise ValueError("Native OBB overlap requires corners and heading")
    if not native_a and detection_a.mask is not None and detection_b.mask is not None:
        return _mask_pair_iou(detection_a, detection_b)
    box_iou = float(sv.box_iou_batch(detection_a.xyxy, detection_b.xyxy)[0][0])
    if math.isnan(box_iou) or box_iou == 0.0:
        return 0.0
    if (
        ORIENTED_BOX_COORDINATES in detection_a.data
        and ORIENTED_BOX_COORDINATES in detection_b.data
    ):
        corners_a = np.asarray(
            detection_a.data[ORIENTED_BOX_COORDINATES], dtype=np.float64
        ).reshape(4, 2)
        corners_b = np.asarray(
            detection_b.data[ORIENTED_BOX_COORDINATES], dtype=np.float64
        ).reshape(4, 2)
        return _oriented_box_pair_iou(corners_a, corners_b)
    return box_iou


def merge_detections(
    detections,
    confidence_aggregation_mode,
    boxes_aggregation_mode,
    mask_aggregation_mode,
):
    """Merge ordinary predictions or retain one native class contributor.

    Args:
        detections: Matched single-image predictions.
        confidence_aggregation_mode: Existing class and confidence selection policy.
        boxes_aggregation_mode: Ordinary AABB aggregation policy.
        mask_aggregation_mode: Ordinary mask aggregation policy.

    Returns:
        One consensus detection with native contributor geometry kept atomically.
    """
    if not _is_native(detections):
        result = v1.merge_detections(
            detections,
            confidence_aggregation_mode,
            boxes_aggregation_mode,
            mask_aggregation_mode,
        )
        return result

    serialise_native_obb(detections)
    class_name, class_id = v1.AGGREGATION_MODE2CLASS_SELECTOR[
        confidence_aggregation_mode
    ](detections)
    indices = np.flatnonzero(
        (detections.class_id == class_id) & (detections["class_name"] == class_name)
    )
    order = detections.data.get(_SOURCE_ORDER, np.arange(len(detections)))
    winner_index = indices[
        np.lexsort((order[indices], -detections.confidence[indices]))[0]
    ]
    winner = detections[int(winner_index)]
    winner.confidence = np.array(
        [
            v1.aggregate_field_values(
                detections=detections,
                field="confidence",
                aggregation_mode=confidence_aggregation_mode,
            )
        ],
        dtype=np.float64,
    )
    return winner
