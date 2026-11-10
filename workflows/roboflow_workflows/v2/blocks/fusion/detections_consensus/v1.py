"""Agree on detections seen by several models: ``roboflow_core/detections_consensus@v1``.

Port of V1 ``roboflow_core/detections_consensus@v1``. Every source is one
model's prediction of the same image. A row becomes a consensus detection when
enough other sources saw an overlapping box::

    for source s, row r in input order (the anchor), skipping used rows:
        best[o] = row of every other source o with the highest box IoU to r,
                  unused, same class name (class_aware), IoU > iou_threshold;
                  ties keep the earlier row
        if len(best) < required_votes - 1:  skip r
        group  = [r] + best[o] in source order
        merged = merge(group);  if merged.confidence < confidence: skip r
        emit merged; mark r and every best[o] used

Matching compares boxes only, never masks. One ``(N, N)`` IoU over all rows is
computed on the predictions' device and copied to the host once, with the boxes,
confidences and class ids; the greedy loop and the box and confidence
aggregation then run on the host in float64, as in V1, and the result is
uploaded once. The IoU is float32 (``fusion.pairwise``), like V1's overlap
matrix. So a pair whose exact IoU equals a threshold that float32 cannot
represent (e.g. 1/3) may still match. Last-bit equality with V1 across
backends and devices is not claimed.

Merging a group::

    class       average: most common (name, id), ties to the earlier row
                max / min: the row with the highest / lowest confidence
    confidence  mean / max / min of the group, as float32
    box         no mask in the group: mean box / largest / smallest area
                any mask in the group: the box of the aggregated mask
                (half-open pixel extents; an empty mask gives (0, 0, 0, 0))
    mask        union / intersection / largest / smallest pixel count;
                rows without a mask count as empty masks

Masks are read only for groups that contain an instance row, and only for
those rows: each source's mask provider answers at the frame raster, so a
low-resolution grid, an RLE mask and a dense mask of one image combine. The
sources must describe one image in one coordinate frame; predictions of a
crop are rejected, not resized into the parent. All known image ids must be
equal; an unknown id agrees with any of them.

Differences from V1 (decision 044):

* Rows are told apart by their row identity, else their ``detection_id``,
  else their source and position; V1 required a ``detection_id``.
* Class names follow the V2 row rule (``bboxes_metadata[i]["class"]``, else
  ``class_names``); unnamed rows match as ``class_<id>`` like V1, but
  ``classes_to_consider`` drops them. Every merged row keeps its own name, so
  a class-id collision between models no longer relabels rows; the
  ``class_names`` map still keeps the later name and warns, or raises with
  ``raise_on_class_name_conflict``.
* Merged rows are a new result with a fresh row identity. Each keeps the
  metadata of the row its class came from, without that row's
  ``detection_id``.
* ``prediction_type`` names the actual output kind.
* ``object_present`` is declared boolean and ``presence_confidence``
  dictionary, the values V1 always returned; V1 declared wider unions.
* The output describes the sources' common frame, also when no row survives.
"""

import logging
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Literal, Optional, Sequence, Tuple
import numpy as np
import torch
from pydantic import AliasChoices, Field, StrictBool, StrictFloat, StrictInt
from roboflow_workflows.v2.blocks.fusion.pairwise import box_iou_between
from roboflow_workflows.v2.blocks.geometry._rows import without_host_mirrors
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import FrameRef, frame_of
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
)
from roboflow_workflows.v2.data.masks import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    SegmentationView,
)
from roboflow_workflows.v2.data.masks.providers import mask_provider_of
from roboflow_workflows.v2.data.prediction_ops import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    class_names_by_id,
    row_class_names,
)
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections


logger = logging.getLogger(__name__)

BLOCK_TYPE = "roboflow_core/detections_consensus@v1"

INPUT_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_VIEW_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
)

AGGREGATION_MODES = ("average", "max", "min")

MASK_AGGREGATION_MODES = ("intersection", "union", "max", "min")

ANY_OBJECT = "any_object"

DETECTION_ID_KEY = "detection_id"

PREDICTION_TYPE_KEY = "prediction_type"

AggregationMode = Literal["average", "max", "min"]

MaskAggregationMode = Literal["intersection", "union", "max", "min"]

FIELD_AGGREGATORS = {"average": statistics.mean, "max": max, "min": min}

class DetectionsConsensusBlock(Block):
    """Keep the objects that enough models agree on, merged into one row each.

    Example: model A sees a person at ``(10, 10, 50, 90)`` with 0.9, model B
    at ``(12, 8, 52, 88)`` with 0.7. With ``required_votes=2`` both boxes
    overlap above ``iou_threshold``, so one person at ``(11, 9, 51, 89)``
    with 0.8 is emitted, ``object_present`` is ``True`` and
    ``presence_confidence`` is ``{"person": 0.8}``. A row only one model saw
    is dropped.
    """

    type = BLOCK_TYPE
    metadata = {
        "name": "Detections Consensus",
        "port_of": "roboflow_core/detections_consensus@v1",
        "section": "fusion",
    }
    outputs = {
        "predictions": Output(
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            source="predictions_batches",
            description=(
                "Merged rows with a fresh row identity, in the sources' common "
                "frame. Instance segmentation with dense masks when any merged "
                "row had a mask (box-only rows then carry an empty mask), "
                "otherwise object detection."
            ),
        ),
        "object_present": Output(
            BOOLEAN_KIND,
            description="Whether merged rows exist and satisfy required_objects.",
        ),
        "presence_confidence": Output(
            DICTIONARY_KIND,
            description=(
                "{} when no object is present, {'any_object': c} when not class "
                "aware, else {class_name: c}, aggregated over merged rows."
            ),
        ),
    }

    class Params(BlockParams):
        predictions_batches: List[Ref(*INPUT_KINDS)] = Field(
            min_length=1,
            validation_alias=AliasChoices("predictions_batches", "predictions"),
            description=(
                "Predictions of the same image by several models: object "
                "detection, instance segmentation (dense, RLE or view) or "
                "keypoints with boxes. Order matters: earlier sources anchor "
                "first and win ties."
            ),
            examples=[["$steps.a.predictions", "$steps.b.predictions"]],
        )
        required_votes: StrictInt | Ref(INTEGER_KIND) = Field(
            ge=1,
            description=(
                "Sources that must see an object, the anchor included, for it "
                "to be kept."
            ),
            examples=[2, "$inputs.required_votes"],
        )
        class_aware: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description=(
                "Only rows with the same class name vote for each other; "
                "presence is reported per class."
            ),
            examples=[True, "$inputs.class_aware"],
        )
        iou_threshold: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            default=0.3,
            ge=0,
            le=1,
            description="Box IoU that two rows must exceed to match, in [0, 1].",
            examples=[0.3, "$inputs.iou_threshold"],
        )
        confidence: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            default=0.0,
            ge=0,
            le=1,
            description=(
                "Merged rows below this confidence are dropped, in [0, 1]; their "
                "rows stay available to later anchors."
            ),
            examples=[0.1, "$inputs.confidence"],
        )
        classes_to_consider: Optional[List[str]] | Ref(LIST_OF_VALUES_KIND) = Field(
            default=None,
            description=(
                "Class names that take part; other and unnamed rows are dropped "
                "first. None or empty keeps every row."
            ),
            examples=[["a", "b"], "$inputs.classes_to_consider"],
        )
        required_objects: Optional[StrictInt | Dict[str, StrictInt]] | Ref(
            INTEGER_KIND, DICTIONARY_KIND
        ) = Field(
            default=None,
            description=(
                "Minimum merged rows for object_present: a positive total, or a "
                "positive count per class name (summed when not class aware)."
            ),
            examples=[3, {"a": 7, "b": 2}, "$inputs.required_objects"],
        )
        presence_confidence_aggregation: AggregationMode = Field(
            default="max",
            description="How presence_confidence combines merged confidences.",
            examples=["max", "min"],
        )
        detections_merge_confidence_aggregation: AggregationMode = Field(
            default="average",
            description=(
                "Merged confidence: mean, max or min. Also picks the class: "
                "average takes the most common, max / min the class of the most "
                "/ least confident row."
            ),
            examples=["min", "max"],
        )
        detections_merge_coordinates_aggregation: AggregationMode = Field(
            default="average",
            description=(
                "Merged box without masks: mean box, largest or smallest area."
            ),
            examples=["min", "max"],
        )
        detections_merge_mask_aggregation: MaskAggregationMode = Field(
            default="union",
            description=(
                "Merged mask when the group has one: union, intersection, or "
                "the mask with the most (max) or fewest (min) pixels."
            ),
            examples=["union", "intersection"],
        )
        raise_on_class_name_conflict: StrictBool = Field(
            default=False,
            description=(
                "Raise when merged sources map one class id to different names, "
                "instead of warning and keeping the later name in class_names."
            ),
            examples=[False, True],
        )

    def run(
        self,
        *,
        predictions_batches: List[Any],
        required_votes: int,
        class_aware: bool,
        iou_threshold: float,
        confidence: float,
        classes_to_consider: Optional[List[str]],
        required_objects: Optional[Any],
        presence_confidence_aggregation: str,
        detections_merge_confidence_aggregation: str,
        detections_merge_coordinates_aggregation: str,
        detections_merge_mask_aggregation: str,
        raise_on_class_name_conflict: bool,
    ) -> Dict[str, Any]:
        """Agree on the detections of one image.

        Args:
            predictions_batches: One prediction per source.
            required_votes: Sources that must agree, at least 1.
            class_aware: Match only rows of the same class name.
            iou_threshold: Box IoU a match must exceed.
            confidence: Lowest merged confidence kept.
            classes_to_consider: Class names that take part, or ``None``.
            required_objects: ``None``, a positive int or class -> count.
            presence_confidence_aggregation: ``average``, ``max`` or ``min``.
            detections_merge_confidence_aggregation: ``average``, ``max`` or
                ``min``.
            detections_merge_coordinates_aggregation: ``average``, ``max`` or
                ``min``.
            detections_merge_mask_aggregation: ``intersection``, ``union``,
                ``max`` or ``min``.
            raise_on_class_name_conflict: Raise on a class-name map collision.

        Returns:
            ``predictions``, ``object_present`` and ``presence_confidence``.

        Raises:
            ValueError: On invalid options, sources of different images or
                devices, keypoints without boxes, or a class-name conflict
                when ``raise_on_class_name_conflict`` is set.
        """
        result = detections_consensus(
            predictions_batches,
            required_votes=required_votes,
            class_aware=class_aware,
            iou_threshold=iou_threshold,
            confidence=confidence,
            classes_to_consider=classes_to_consider,
            required_objects=required_objects,
            presence_confidence_aggregation=presence_confidence_aggregation,
            confidence_aggregation=detections_merge_confidence_aggregation,
            coordinates_aggregation=detections_merge_coordinates_aggregation,
            mask_aggregation=detections_merge_mask_aggregation,
            raise_on_class_name_conflict=raise_on_class_name_conflict,
        )

        return result

@dataclass(frozen=True)
class _Source:
    """One model's rows: box carrier plus the payload holding its masks."""

    boxes: Any
    masks: Any

@dataclass(frozen=True)
class _HostRows:
    """Host copies of every row, indexed by the row's global position."""

    iou: np.ndarray
    xyxy: np.ndarray
    confidence: np.ndarray
    class_ids: List[int]
    names: List[Optional[Any]]
    keys: List[Hashable]
    source_of: List[int]
    local_of: List[int]

@dataclass(frozen=True)
class _Merged:
    """One emitted consensus row, before its tensors are built."""

    rows: Tuple[int, ...]
    class_id: int
    class_name: Any
    selected_row: int
    confidence: float
    xyxy: Tuple[float, float, float, float]

def detections_consensus(
    predictions: Sequence[Any],
    *,
    required_votes: int,
    class_aware: bool = True,
    iou_threshold: float = 0.3,
    confidence: float = 0.0,
    classes_to_consider: Optional[List[str]] = None,
    required_objects: Optional[Any] = None,
    presence_confidence_aggregation: str = "max",
    confidence_aggregation: str = "average",
    coordinates_aggregation: str = "average",
    mask_aggregation: str = "union",
    raise_on_class_name_conflict: bool = False,
) -> Dict[str, Any]:
    """Run consensus over the predictions several models made of one image.

    Args:
        predictions: One payload per source; see ``INPUT_KINDS``.
        required_votes: Sources that must agree, at least 1.
        class_aware: Match only rows of the same class name.
        iou_threshold: Box IoU a match must exceed.
        confidence: Lowest merged confidence kept.
        classes_to_consider: Class names that take part, or ``None``.
        required_objects: ``None``, a positive int or class -> positive count.
        presence_confidence_aggregation: ``average``, ``max`` or ``min``.
        confidence_aggregation: ``average``, ``max`` or ``min``.
        coordinates_aggregation: ``average``, ``max`` or ``min``.
        mask_aggregation: ``intersection``, ``union``, ``max`` or ``min``.
        raise_on_class_name_conflict: Raise on a class-name map collision.

    Returns:
        ``predictions`` (``Detections`` or ``InstanceDetections`` with a fresh
        row identity), ``object_present`` (bool) and ``presence_confidence``
        (dict).

    Raises:
        ValueError: See ``DetectionsConsensusBlock.run``.
    """
    _check_options(
        predictions,
        required_votes=required_votes,
        classes_to_consider=classes_to_consider,
        required_objects=required_objects,
        aggregations=(
            presence_confidence_aggregation,
            confidence_aggregation,
            coordinates_aggregation,
        ),
        mask_aggregation=mask_aggregation,
    )
    sources = [_source_of(payload, position=i) for i, payload in enumerate(predictions)]

    if all(len(source.boxes.xyxy) == 0 for source in sources):
        empty = _empty_result(sources)
        return empty

    frame, device = _common_frame_and_device(sources)
    host = _host_rows(sources)
    kept = _candidate_rows(sources, host=host, classes=classes_to_consider)
    merged = _agree(
        kept,
        host=host,
        sources=sources,
        required_votes=required_votes,
        class_aware=class_aware,
        iou_threshold=iou_threshold,
        confidence=confidence,
        confidence_aggregation=confidence_aggregation,
        coordinates_aggregation=coordinates_aggregation,
        raise_on_class_name_conflict=raise_on_class_name_conflict,
    )
    consensus = _build_predictions(
        merged,
        host=host,
        sources=sources,
        frame=frame,
        device=device,
        mask_aggregation=mask_aggregation,
        raise_on_class_name_conflict=raise_on_class_name_conflict,
    )
    object_present, presence_confidence = _presence(
        [item.class_name for item in merged],
        [item.confidence for item in merged],
        class_aware=class_aware,
        required_objects=required_objects,
        aggregation=presence_confidence_aggregation,
    )

    result = {
        "predictions": consensus,
        "object_present": object_present,
        "presence_confidence": presence_confidence,
    }

    return result

def _check_options(
    predictions: Sequence[Any],
    *,
    required_votes: Any,
    classes_to_consider: Any,
    required_objects: Any,
    aggregations: Tuple[str, ...],
    mask_aggregation: str,
) -> None:
    if len(predictions) < 1:
        raise ValueError(f"{BLOCK_TYPE} needs at least one source of predictions")
    if not _is_positive_int(required_votes):
        raise ValueError(
            f"{BLOCK_TYPE}: required_votes must be a positive int, got "
            f"{required_votes!r}"
        )
    if classes_to_consider is not None and not isinstance(
        classes_to_consider, (list, tuple)
    ):
        raise ValueError(
            f"{BLOCK_TYPE}: classes_to_consider must be a list of class names or "
            f"None, got {type(classes_to_consider).__name__}"
        )

    required_is_valid = (
        required_objects is None
        or _is_positive_int(required_objects)
        or (
            isinstance(required_objects, dict)
            and all(
                isinstance(name, str) and _is_positive_int(count)
                for name, count in required_objects.items()
            )
        )
    )
    if not required_is_valid:
        raise ValueError(
            f"{BLOCK_TYPE}: required_objects must be None, a positive int or a "
            f"dict of class name -> positive int, got {required_objects!r}"
        )
    for mode in aggregations:
        if mode not in AGGREGATION_MODES:
            raise ValueError(
                f"{BLOCK_TYPE}: aggregation mode must be one of "
                f"{list(AGGREGATION_MODES)}, got {mode!r}"
            )
    if mask_aggregation not in MASK_AGGREGATION_MODES:
        raise ValueError(
            f"{BLOCK_TYPE}: mask aggregation must be one of "
            f"{list(MASK_AGGREGATION_MODES)}, got {mask_aggregation!r}"
        )

def _is_positive_int(value: Any) -> bool:
    is_positive = isinstance(value, int) and not isinstance(value, bool) and value > 0

    return is_positive

def _source_of(payload: Any, *, position: int) -> _Source:
    if isinstance(payload, tuple) and len(payload) == 2:
        boxes = payload[1]
        if boxes is None:
            raise ValueError(
                f"{BLOCK_TYPE}: keypoint prediction {position} has no "
                "bounding-box Detections, which consensus requires"
            )
        source = _Source(boxes=boxes, masks=None)
        return source

    if isinstance(payload, SegmentationView):
        source = _Source(boxes=payload.detections, masks=payload)
        return source

    if isinstance(payload, InstanceDetections):
        source = _Source(boxes=payload, masks=payload)
        return source

    if isinstance(payload, Detections):
        source = _Source(boxes=payload, masks=None)
        return source

    raise ValueError(
        f"{BLOCK_TYPE}: prediction {position} must be Detections, "
        "InstanceDetections, a SegmentationView or a keypoint tuple, got "
        f"{type(payload).__name__}"
    )

def _frame_of_source(source: _Source, *, position: int) -> FrameRef:
    try:
        frame = frame_of(source.boxes)
    except ContractError as error:
        raise ValueError(
            f"{BLOCK_TYPE}: prediction {position} has no readable frame, so it "
            f"cannot be matched with the other sources: {error}"
        ) from error

    return frame

def _common_frame_and_device(
    sources: List[_Source],
) -> Tuple[FrameRef, torch.device]:
    """Check that non-empty sources share one image, frame and device.

    Two frames align when they have the same size, parent and root mappings
    and composite placements, and the same image id where both know it.
    Box coordinates of such predictions mean the same pixels. Every frame is
    compared with the first frame that knows its image id, so an unknown id
    never lets two different known ids pass.
    """
    present = [
        (position, source)
        for position, source in enumerate(sources)
        if len(source.boxes.xyxy) > 0
    ]
    frames = [
        (position, _frame_of_source(source, position=position))
        for position, source in present
    ]
    reference_position, reference = next(
        ((position, frame) for position, frame in frames if frame.image_id is not None),
        frames[0],
    )
    for position, frame in frames:
        if not _same_coordinates(reference, frame):
            raise ValueError(
                f"{BLOCK_TYPE}: predictions {reference_position} and {position} "
                "describe different images or coordinate frames "
                f"({_describe(reference)} vs {_describe(frame)}). Consensus "
                "compares boxes of one image; map crop predictions into their "
                "parent image before consensus."
            )

    device = present[0][1].boxes.xyxy.device
    for position, source in present[1:]:
        if source.boxes.xyxy.device != device:
            raise ValueError(
                f"{BLOCK_TYPE}: prediction {position} is on "
                f"{source.boxes.xyxy.device}, but prediction {present[0][0]} is on "
                f"{device}"
            )

    return reference, device

def _same_coordinates(first: FrameRef, other: FrameRef) -> bool:
    same_geometry = (
        first.size_hw == other.size_hw
        and first.parent == other.parent
        and first.root == other.root
        and first.composite_sources == other.composite_sources
    )
    ids_agree = (
        first.image_id is None
        or other.image_id is None
        or first.image_id == other.image_id
    )

    return same_geometry and ids_agree

def _describe(frame: FrameRef) -> str:
    root = frame.root
    description = (
        f"image {frame.image_id!r} {tuple(frame.size_hw)} at offset "
        f"{root.offset_xy}, scale {root.scale_xy} in root {root.frame_id!r}"
    )

    return description

def _host_rows(sources: List[_Source]) -> _HostRows:
    """Copy what the greedy loop reads to the host, once per tensor field."""
    present = [source for source in sources if len(source.boxes.xyxy) > 0]
    xyxy = torch.cat([source.boxes.xyxy for source in present])
    iou = box_iou_between(xyxy, xyxy).cpu().numpy()
    host_xyxy = xyxy.cpu().numpy().astype(np.float64)
    host_confidence = (
        torch.cat([source.boxes.confidence for source in present])
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    host_class_ids = torch.cat([source.boxes.class_id for source in present]).tolist()

    names: List[Optional[Any]] = []
    keys: List[Hashable] = []
    source_of: List[int] = []
    local_of: List[int] = []
    offset = 0
    for position, source in enumerate(sources):
        rows = len(source.boxes.xyxy)
        if rows == 0:
            continue
        boxes = source.boxes
        end = offset + rows
        class_ids = host_class_ids[offset:end]
        row_metadata = boxes.bboxes_metadata or [{} for _ in range(rows)]
        names.extend(
            row_class_names(
                class_ids,
                row_metadata=row_metadata,
                names_by_id=class_names_by_id(boxes.image_metadata),
            )
        )
        keys.extend(_row_keys(boxes, position=position, row_metadata=row_metadata))
        source_of.extend([position] * rows)
        local_of.extend(range(rows))
        offset += rows

    host = _HostRows(
        iou=iou,
        xyxy=host_xyxy,
        confidence=host_confidence,
        class_ids=host_class_ids,
        names=names,
        keys=keys,
        source_of=source_of,
        local_of=local_of,
    )

    return host

def _row_keys(
    boxes: Any, *, position: int, row_metadata: Sequence[Dict[str, Any]]
) -> List[Hashable]:
    """Identity of each row: row record, else ``detection_id``, else position."""
    record = row_record(boxes)
    if record is not None:
        keys = [("row", record.namespace, row_id) for row_id in record.host_ids()]
        return keys

    keys = [
        (
            ("detection_id", row[DETECTION_ID_KEY])
            if DETECTION_ID_KEY in row
            else ("position", position, local)
        )
        for local, row in enumerate(row_metadata)
    ]

    return keys

def _candidate_rows(
    sources: List[_Source], *, host: _HostRows, classes: Optional[List[str]]
) -> List[List[int]]:
    """Global row positions of each source that take part, in order."""
    kept: List[List[int]] = [[] for _ in sources]
    for row, source in enumerate(host.source_of):
        name = host.names[row]
        if classes and (name is None or name not in classes):
            continue
        kept[source].append(row)

    return kept

def _match_name(host: _HostRows, row: int) -> Any:
    """Name used to compare and report a row; unnamed rows as in V1."""
    name = host.names[row]
    if name is None:
        name = f"class_{host.class_ids[row]}"

    return name

def _agree(
    kept: List[List[int]],
    *,
    host: _HostRows,
    sources: List[_Source],
    required_votes: int,
    class_aware: bool,
    iou_threshold: float,
    confidence: float,
    confidence_aggregation: str,
    coordinates_aggregation: str,
    raise_on_class_name_conflict: bool,
) -> List[_Merged]:
    """The V1 greedy loop over host data; see the module docstring."""
    # Keys, not positions: rows sharing an identity (e.g. one detection_id) are
    # used together, as in V1.
    used: set = set()
    merged_rows: List[_Merged] = []
    for source, anchors in enumerate(kept):
        for anchor in anchors:
            if host.keys[anchor] in used:
                continue

            best = _best_matches(
                anchor,
                source=source,
                kept=kept,
                host=host,
                used=used,
                class_aware=class_aware,
                iou_threshold=iou_threshold,
            )
            if len(best) < required_votes - 1:
                continue

            group = (anchor, *best.values())
            # V1 unions the class maps of a group's sources while merging it,
            # before the confidence filter, so a conflict warns or raises even
            # for a group that is dropped afterwards.
            _check_group_class_names(
                group,
                host=host,
                sources=sources,
                raise_on_class_name_conflict=raise_on_class_name_conflict,
            )
            merged = _merge_group(
                group,
                host=host,
                confidence_aggregation=confidence_aggregation,
                coordinates_aggregation=coordinates_aggregation,
            )
            if merged.confidence < confidence:
                continue

            merged_rows.append(merged)
            used.update(host.keys[row] for row in group)

    return merged_rows

def _best_matches(
    anchor: int,
    *,
    source: int,
    kept: List[List[int]],
    host: _HostRows,
    used: set,
    class_aware: bool,
    iou_threshold: float,
) -> Dict[int, int]:
    """Best unused row of every other source; insertion is source order."""
    anchor_name = _match_name(host, anchor)
    best: Dict[int, Tuple[int, float]] = {}
    for other_source, candidates in enumerate(kept):
        if other_source == source:
            continue
        for candidate in candidates:
            if host.keys[candidate] in used:
                continue
            if class_aware and _match_name(host, candidate) != anchor_name:
                continue
            iou = float(host.iou[anchor, candidate])
            if iou <= iou_threshold:
                continue
            if other_source not in best or best[other_source][1] < iou:
                best[other_source] = (candidate, iou)

    matches = {other: row for other, (row, _) in best.items()}

    return matches

def _merge_group(
    group: Tuple[int, ...],
    *,
    host: _HostRows,
    confidence_aggregation: str,
    coordinates_aggregation: str,
) -> _Merged:
    confidences = [float(host.confidence[row]) for row in group]
    if confidence_aggregation == "average":
        pairs = [(_match_name(host, row), host.class_ids[row]) for row in group]
        winner = Counter(pairs).most_common(1)[0][0]
        selected = group[pairs.index(winner)]
    elif confidence_aggregation == "max":
        selected = group[confidences.index(max(confidences))]
    else:
        selected = group[confidences.index(min(confidences))]

    merged_confidence = FIELD_AGGREGATORS[confidence_aggregation](confidences)
    # V1 stores the merged confidence as float32 before comparing it.
    stored_confidence = float(np.float32(merged_confidence))

    merged = _Merged(
        rows=group,
        class_id=host.class_ids[selected],
        class_name=_match_name(host, selected),
        selected_row=selected,
        confidence=stored_confidence,
        xyxy=_merged_box(host.xyxy[list(group)], mode=coordinates_aggregation),
    )

    return merged

def _merged_box(xyxy: np.ndarray, *, mode: str) -> Tuple[float, ...]:
    """V1 box aggregation in float64; replaced later when the group has masks."""
    if mode == "average":
        box = tuple(float(value) for value in np.mean(xyxy, axis=0))
        return box

    areas = ((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])).tolist()
    target = max(areas) if mode == "max" else min(areas)
    box = tuple(float(value) for value in xyxy[areas.index(target)])

    return box

def _check_group_class_names(
    group: Tuple[int, ...],
    *,
    host: _HostRows,
    sources: List[_Source],
    raise_on_class_name_conflict: bool,
) -> None:
    """Warn or raise when the group's sources map one class id to two names.

    The class maps are unioned in group order, as V1 did; the union itself is
    not used.
    """
    maps = [
        class_names_by_id(sources[host.source_of[row]].boxes.image_metadata) or {}
        for row in group
    ]
    _union_class_names(maps, raise_on_class_name_conflict=raise_on_class_name_conflict)

def _union_class_names(
    maps: Sequence[Dict[int, Any]], *, raise_on_class_name_conflict: bool
) -> Dict[int, Any]:
    union: Dict[int, Any] = {}
    for names in maps:
        for class_id, name in names.items():
            class_id = int(class_id)
            existing = union.get(class_id)
            if existing is not None and existing != name:
                _class_name_conflict(
                    class_id,
                    existing=existing,
                    new=name,
                    raise_on_class_name_conflict=raise_on_class_name_conflict,
                )
            union[class_id] = name

    return union

def _class_name_conflict(
    class_id: int, *, existing: Any, new: Any, raise_on_class_name_conflict: bool
) -> None:
    message = (
        f"{BLOCK_TYPE}: conflicting class names for class_id={class_id}: "
        f"existing {existing!r} vs incoming {new!r}."
    )
    if raise_on_class_name_conflict:
        raise ValueError(
            f"{message} Disable 'raise_on_class_name_conflict' to keep the later "
            "name in class_names and continue."
        )
    logger.warning(
        "%s Keeping the later name %r in class_names; each row keeps its own "
        "name. Enable 'raise_on_class_name_conflict' to raise instead.",
        message,
        new,
    )

def _build_predictions(
    merged: List[_Merged],
    *,
    host: _HostRows,
    sources: List[_Source],
    frame: FrameRef,
    device: torch.device,
    mask_aggregation: str,
    raise_on_class_name_conflict: bool,
) -> Any:
    """Upload the merged rows and attach their masks and metadata."""
    class_names = _union_class_names(
        [{item.class_id: item.class_name} for item in merged],
        raise_on_class_name_conflict=raise_on_class_name_conflict,
    )
    xyxy = torch.tensor(
        [item.xyxy for item in merged], dtype=torch.float32, device=device
    ).reshape(-1, 4)
    class_id = torch.tensor(
        [item.class_id for item in merged], dtype=torch.int64, device=device
    )
    confidence = torch.tensor(
        [item.confidence for item in merged], dtype=torch.float32, device=device
    )
    bboxes_metadata = [
        _merged_row_metadata(item, host=host, sources=sources) for item in merged
    ]

    masked = [
        index
        for index, item in enumerate(merged)
        if any(sources[host.source_of[row]].masks is not None for row in item.rows)
    ]
    masks = None
    if masked:
        aggregated = _aggregated_masks(
            [merged[index] for index in masked],
            host=host,
            sources=sources,
            frame=frame,
            mode=mask_aggregation,
        )
        positions = torch.tensor(masked, dtype=torch.int64, device=device)
        xyxy[positions] = _mask_boxes(aggregated)
        masks = torch.zeros(
            (len(merged), *frame.size_hw), dtype=torch.bool, device=device
        )
        masks[positions] = aggregated

    image_metadata = {
        **frame.prediction_metadata(),
        CLASS_NAMES_KEY: class_names,
        PREDICTION_TYPE_KEY: (
            "instance-segmentation" if masks is not None else "object-detection"
        ),
    }
    fields = {
        "xyxy": xyxy,
        "class_id": class_id,
        "confidence": confidence,
        "image_metadata": image_metadata,
        "bboxes_metadata": bboxes_metadata,
    }
    if masks is None:
        predictions = Detections(**fields)
    else:
        predictions = InstanceDetections(mask=masks, **fields)
    identified = with_row_identity(predictions)

    return identified

def _merged_row_metadata(
    item: _Merged, *, host: _HostRows, sources: List[_Source]
) -> Dict[str, Any]:
    """The class-selected row's metadata, without its identity, plus the name."""
    boxes = sources[host.source_of[item.selected_row]].boxes
    rows = boxes.bboxes_metadata
    selected = rows[host.local_of[item.selected_row]] if rows is not None else {}
    metadata = {
        key: value
        for key, value in without_host_mirrors(selected).items()
        if key != DETECTION_ID_KEY
    }
    metadata[CLASS_NAME_KEY] = item.class_name

    return metadata

def _aggregated_masks(
    merged: List[_Merged],
    *,
    host: _HostRows,
    sources: List[_Source],
    frame: FrameRef,
    mode: str,
) -> torch.Tensor:
    """One aggregated ``(H, W)`` mask per merged row, stacked.

    Each source's provider is asked once, for exactly the rows these groups
    use, at the frame raster. Rows without a mask count as empty.
    """
    needed: Dict[int, List[int]] = {}
    for item in merged:
        for row in item.rows:
            source = host.source_of[row]
            if sources[source].masks is not None:
                needed.setdefault(source, []).append(host.local_of[row])

    dense: Dict[Tuple[int, int], torch.Tensor] = {}
    for source, local_rows in needed.items():
        local_rows = sorted(set(local_rows))
        binary = (
            mask_provider_of(sources[source].masks)
            .select(indices=local_rows)
            .binary(fidelity="reference")
        )
        mask_frame = binary.frame
        if not _same_coordinates(frame, mask_frame):
            raise ValueError(
                f"{BLOCK_TYPE}: masks of prediction {source} describe "
                f"{_describe(mask_frame)}, not {_describe(frame)}"
            )
        for index, local in enumerate(local_rows):
            dense[(source, local)] = binary.predictions.mask[index]

    empty = None
    aggregated = []
    for item in merged:
        stack = []
        for row in item.rows:
            mask = dense.get((host.source_of[row], host.local_of[row]))
            if mask is None:
                if empty is None:
                    empty = torch.zeros(
                        frame.size_hw,
                        dtype=torch.bool,
                        device=next(iter(dense.values())).device,
                    )
                mask = empty
            stack.append(mask.to(torch.bool))
        aggregated.append(_aggregate_masks(torch.stack(stack), mode=mode))
    stacked = torch.stack(aggregated)

    return stacked

def _aggregate_masks(stack: torch.Tensor, *, mode: str) -> torch.Tensor:
    if mode == "union":
        union = stack.any(dim=0)
        return union

    if mode == "intersection":
        intersection = stack.all(dim=0)
        return intersection

    areas = stack.flatten(1).sum(dim=1)
    # argmax / argmin return the first extreme, as V1's list.index did.
    index = areas.argmax() if mode == "max" else areas.argmin()
    chosen = stack[index]

    return chosen

def _mask_boxes(masks: torch.Tensor) -> torch.Tensor:
    """``(M, 4)`` float32 half-open pixel extents of ``(M, H, W)`` masks.

    Upper bounds include the full last pixel, so boxes compose with crops.
    An empty mask gives ``(0, 0, 0, 0)``.
    """
    height, width = masks.shape[1:]
    rows_any = masks.any(dim=2).to(torch.uint8)
    cols_any = masks.any(dim=1).to(torch.uint8)
    x_min = cols_any.argmax(dim=1)
    x_max = width - cols_any.flip(1).argmax(dim=1)
    y_min = rows_any.argmax(dim=1)
    y_max = height - rows_any.flip(1).argmax(dim=1)
    boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1).to(torch.float32)
    present = rows_any.amax(dim=1).bool().unsqueeze(1)
    boxes = torch.where(present, boxes, torch.zeros_like(boxes))

    return boxes

def _empty_result(sources: List[_Source]) -> Dict[str, Any]:
    """V1's result when no source has a row: no frame check, no rows."""
    first = sources[0].boxes
    device = first.xyxy.device
    try:
        image_metadata = frame_of(first).prediction_metadata()
    except ContractError:
        image_metadata = {}
    image_metadata[CLASS_NAMES_KEY] = {}
    image_metadata[PREDICTION_TYPE_KEY] = "object-detection"
    predictions = Detections(
        xyxy=torch.zeros((0, 4), dtype=torch.float32, device=device),
        class_id=torch.zeros((0,), dtype=torch.int64, device=device),
        confidence=torch.zeros((0,), dtype=torch.float32, device=device),
        image_metadata=image_metadata,
        bboxes_metadata=[],
    )

    result = {
        "predictions": with_row_identity(predictions),
        "object_present": False,
        "presence_confidence": {},
    }

    return result

def _presence(
    class_names: List[Any],
    confidences: List[float],
    *,
    class_aware: bool,
    required_objects: Optional[Any],
    aggregation: str,
) -> Tuple[bool, Dict[str, float]]:
    """V1 presence rule over the merged rows."""
    if not class_names:
        return False, {}

    required = 0 if required_objects is None else required_objects
    if isinstance(required, dict) and not class_aware:
        required = sum(required.values())
    if isinstance(required, int) and len(class_names) < required:
        return False, {}

    aggregate = FIELD_AGGREGATORS[aggregation]
    if not class_aware:
        presence = {ANY_OBJECT: aggregate(confidences)}
        return True, presence

    by_class: Dict[Any, List[float]] = {}
    for name, value in zip(class_names, confidences):
        by_class.setdefault(name, []).append(value)
    if isinstance(required, dict):
        for name, count in required.items():
            if len(by_class.get(name, ())) < count:
                return False, {}

    presence = {name: aggregate(values) for name, values in by_class.items()}

    return True, presence

__all__ = [
    "BLOCK_TYPE",
    "DetectionsConsensusBlock",
    "detections_consensus",
]
