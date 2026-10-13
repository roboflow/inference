"""Roll crop detections back onto their parent: ``v2/detections_list_rollup@v1``.

Port of the latest V1 ``roboflow_core/detections_list_rollup@v1`` (tensor
implementation). Child ``k`` holds the predictions made on the crop of parent
row ``k``::

    detect -> v2/dynamic_crop@v1 -> detect on each crop -> v2/dimension_collapse@v1
           -> this block (parent_detection, child_detections)

    for each child (None or no rows: nothing to add):
        move its rows into the parent's frame              (geometry.frames)
    group all rows by class id, in order of first appearance
    merge each class by the family of the children:
        boxes      connected groups of overlapping boxes -> envelope box
        masks      connected groups of overlapping masks -> union mask and its
                   box (last pixel inclusive); an empty union drops the row
        keypoints  each unused row seeds a group of later rows with as many
                   keypoints and mean keypoint distance < keypoint_merge_threshold
                   -> mean keypoints, keypoint confidences and box
    confidence = max / mean / min of the group (confidence_strategy)

Two rows overlap when they intersect (``overlap_threshold`` 0; touching is not
enough) or when their IoU is at least ``overlap_threshold``. Masks are only
compared when their boxes intersect, and then by exact pixel counts. A merged
row has the class id and metadata of its lowest row; groups are ordered by
their lowest row.

Coordinates. A child with frame metadata moves by its exact crop origin
relative to the parent's frame, including nested resized crops. Masks are
resampled by the shared coordinate API; resizing cannot recover lost detail.
A child with no frame metadata at all moves by ``int(parent_box[:2])``, V1's
rule. Masks are pasted on the parent frame's canvas. ``crop_zones`` lists the
corners of every parent box.

Differences from V1:

* V1 moved every child by ``int(parent_box[:2])``. ``v2/dynamic_crop@v1`` cuts
  at the rounded, clamped box, so that rule is off by a pixel or more.
* V1's mask canvas was the root image, also for a parent that is a crop.
* Children that mix boxes with masks or keypoints with anything else, and
  keypoints with masks, raise; V1 crashed or misaligned keypoints. ``None``
  children (empty crops) add nothing.
* Keypoints come from the native ``KeyPoints``; V1 read row metadata.
  Covariance and ``detection_confidence`` are kept when every keypoint child
  has them (mean and ``confidence_strategy``); V1 dropped them.
* Every row keeps its own class name in ``bboxes_metadata[i]["class"]``. The
  ``class_names`` map keeps V1's rule: a later child wins with a warning, or
  ``raise_on_class_name_conflict`` raises.
* ``prediction_type`` names the actual output kind. An empty result keeps the
  children's family; V1 returned empty object detections.
* An unknown ``confidence_strategy`` raises instead of meaning max or min.
* The result is a new prediction with a fresh row identity.

Box IoU uses the float32 ``fusion.pairwise`` helper; V1 used float64, so a
pair exactly at ``overlap_threshold`` can decide differently.

CPU boundary: parent boxes (for ``crop_zones``), class ids, the pairwise IoU,
pixel-count and keypoint-distance matrices and the merge groups reach the host
once each; merged tensors are computed on the rows' device. Dense masks stay on
their device. RLE masks are merged with pycocotools on the host, without a
dense canvas. A ``SegmentationView`` child gives its reference masks.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from pycocotools import mask as mask_utils
from pydantic import Field
from roboflow_workflows.v2.blocks.fusion.pairwise import box_iou_between
from roboflow_workflows.v2.blocks.geometry._rows import without_host_mirrors
from roboflow_workflows.v2.blocks.geometry.mask_inputs import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    boxes_of_view,
    instance_predictions_of,
)
from roboflow_workflows.v2.data.coordinates import prediction_to_frame
from roboflow_workflows.v2.data.frames import FRAME_RECORD_KEY, FrameRef, frame_of
from roboflow_workflows.v2.data.image import FrameMapping
from roboflow_workflows.v2.data.kinds import (
    FLOAT_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.masks import SegmentationView
from roboflow_workflows.v2.data.prediction_ops import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    class_names_by_id,
    row_class_name,
    validated_row_count,
)
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import with_row_identity
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

logger = logging.getLogger(__name__)

PARENT_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_VIEW_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
)
OUTPUT_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
)
STRATEGIES = ("max", "mean", "min")
BOXES, MASKS, KEYPOINTS = "boxes", "masks", "keypoints"
PREDICTION_TYPES = {
    BOXES: "object-detection",
    MASKS: "instance-segmentation",
    KEYPOINTS: "keypoint-detection",
}
DETECTION_ID_KEY = "detection_id"
INFERENCE_ID_KEY = "inference_id"
PREDICTION_TYPE_KEY = "prediction_type"
# Keys V1 never copied from a child row: identities, keypoint lists (the
# KeyPoints carrier holds them) and per-image values (the parent frame holds
# them).
DROPPED_ROW_KEYS = frozenset(
    {
        DETECTION_ID_KEY,
        INFERENCE_ID_KEY,
        CLASS_NAME_KEY,
        "keypoints_xy",
        "keypoints_class_name",
        "keypoints_class_id",
        "keypoints_confidence",
        PREDICTION_TYPE_KEY,
        "parent_id",
        "root_parent_id",
        "image_dimensions",
        "parent_dimensions",
        "parent_coordinates",
        "root_parent_dimensions",
        "root_parent_coordinates",
    }
)
# Any of these means a prediction states its frame; frame_of reads them.
FRAME_KEYS = (
    FRAME_RECORD_KEY,
    "parent_id",
    "parent_frame_id",
    "root_parent_id",
    "image_dimensions",
    "parent_dimensions",
    "parent_coordinates",
    "root_parent_dimensions",
    "root_parent_coordinates",
    "scaling_relative_to_parent",
    "scaling_relative_to_root_parent",
    "is_composite",
    "composite_sources",
)

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Zone = List[Tuple[float, float]]
Pairs = Set[Tuple[int, int]]

__all__ = ["DetectionsListRollUpBlock"]


class DetectionsListRollUpBlock(Block):
    """Merge the detections made on each parent row's crop into the parent.

    Example: parent rows ``[100, 50, 200, 150]`` and ``[180, 50, 300, 150]``
    are cropped, and a detector finds a car at ``[70, 10, 100, 60]`` in crop 0
    and at ``[0, 10, 30, 60]`` in crop 1. Both become ``[170, 60, 200, 110]``
    and ``[180, 60, 210, 110]`` in the parent's image; they intersect, so the
    result is one car at ``[170, 60, 210, 110]`` with the higher confidence.

    The block is stateless, does not change its inputs and is prunable.
    """

    type = "v2/detections_list_rollup@v1"
    aliases = ("DetectionsListRollUp",)
    prunable = True
    metadata = {
        "name": "Detections List Roll-Up",
        "port_of": "roboflow_core/detections_list_rollup@v1",
        "section": "advanced",
    }
    outputs = {
        "rolled_up_detections": Output(
            *OUTPUT_KINDS,
            source="parent_detection",
            description=(
                "Children's rows merged in the parent's image, in the children's "
                "family; ordered by class first appearance, then lowest row."
            ),
        ),
        "crop_zones": Output(
            LIST_OF_VALUES_KIND,
            source="parent_detection",
            description=(
                "Per parent row, its box corners: top-left, top-right, "
                "bottom-right, bottom-left (x, y)."
            ),
        ),
    }

    class Params(BlockParams):
        parent_detection: Ref(*PARENT_KINDS) = Field(
            description="Detections whose rows were cropped; only boxes are read.",
            examples=["$steps.detector.predictions"],
        )
        child_detections: Ref(LIST_OF_VALUES_KIND) = Field(
            description=(
                "One prediction (or None) per parent row, made on that row's crop; "
                "usually the output of v2/dimension_collapse@v1."
            ),
            examples=["$steps.collapse.output"],
        )
        confidence_strategy: Literal["max", "mean", "min"] | Ref(STRING_KIND) = Field(
            default="max",
            description="Confidence of a merged row: 'max', 'mean' or 'min'.",
            examples=["max", "$inputs.confidence_strategy"],
        )
        overlap_threshold: UnitFloat | Ref(FLOAT_KIND) = Field(
            default=0.0,
            description=(
                "IoU in [0, 1] at which rows merge; 0 merges any intersection."
            ),
            examples=[0.0, 0.5],
        )
        keypoint_merge_threshold: NonNegativeFloat | Ref(FLOAT_KIND) = Field(
            default=10.0,
            description=(
                "Keypoint rows merge below this mean keypoint distance in pixels."
            ),
            examples=[10.0, 20.0],
        )
        raise_on_class_name_conflict: bool = Field(
            default=False,
            description=(
                "Raise when children name one class id differently; otherwise "
                "the later name wins in class_names and a warning is logged."
            ),
            examples=[False, True],
        )

    def run(
        self,
        *,
        parent_detection: Any,
        child_detections: Sequence[Any],
        confidence_strategy: str,
        overlap_threshold: float,
        keypoint_merge_threshold: float,
        raise_on_class_name_conflict: bool,
    ) -> Dict[str, Any]:
        """Move every child into the parent's frame and merge overlapping rows.

        Args:
            parent_detection: Detections, instance segmentation (predictions or
                view) or keypoints with boxes; one row per crop.
            child_detections: One prediction or ``None`` per parent row.
            confidence_strategy: ``"max"``, ``"mean"`` or ``"min"``.
            overlap_threshold: Merge IoU in ``[0, 1]``; 0 means any intersection.
            keypoint_merge_threshold: Nonnegative keypoint merge distance.
            raise_on_class_name_conflict: Raise on a class id named differently.

        Returns:
            ``rolled_up_detections`` in the parent's frame with a fresh row
            identity, and ``crop_zones``, one corner list per parent row.

        Raises:
            ValueError: On an invalid parameter value, a child count other
                than the parent row count, children of mixed families, a
                child with unrelated, incomplete or conflicting frame metadata,
                or a class-name conflict with ``raise_on_class_name_conflict``.
        """
        _check_parameters(
            confidence_strategy=confidence_strategy,
            overlap_threshold=overlap_threshold,
            keypoint_merge_threshold=keypoint_merge_threshold,
        )
        parent_boxes = _parent_boxes(parent_detection)
        children = list(child_detections)
        if len(parent_boxes.xyxy) != len(children):
            raise ValueError(
                f"Number of detections in parent_detection ({len(parent_boxes.xyxy)}) "
                f"must match number of child predictions ({len(children)})"
            )

        frame = frame_of(parent_boxes)
        host_boxes = parent_boxes.xyxy.detach().cpu().tolist()
        crop_zones = [_zone(box) for box in host_boxes]
        family = _family(children)
        class_names = _merged_class_names(
            children, raise_on_conflict=raise_on_class_name_conflict
        )

        rows = _gather_rows(
            children,
            frame=frame,
            parent_boxes=host_boxes,
            family=family,
            class_names=class_names,
        )
        groups = _merge_groups(
            rows,
            overlap_threshold=overlap_threshold,
            keypoint_merge_threshold=keypoint_merge_threshold,
        )
        rolled_up = _merged_prediction(
            rows,
            groups=groups,
            frame=frame,
            class_names=class_names,
            confidence_strategy=confidence_strategy,
            inference_id=_inference_id(children),
            device=parent_boxes.xyxy.device,
        )
        result = {"rolled_up_detections": rolled_up, "crop_zones": crop_zones}

        return result


@dataclass
class _Rows:
    """Every child row in the parent's frame, concatenated in child order."""

    family: str
    xyxy: Optional[torch.Tensor]
    class_id: Optional[torch.Tensor]
    confidence: Optional[torch.Tensor]
    metadata: List[Dict[str, Any]]
    names: List[str]
    host_class_ids: List[int]
    dense_masks: Optional[torch.Tensor]
    rle_masks: Optional[List[dict]]
    rle_storage: bool
    key_points: Optional[List[KeyPoints]]
    key_point_counts: Optional[List[int]]
    key_point_width: int

    def __len__(self) -> int:
        return len(self.host_class_ids)


def _check_parameters(
    *, confidence_strategy, overlap_threshold, keypoint_merge_threshold
) -> None:
    if confidence_strategy not in STRATEGIES:
        raise ValueError(
            f"confidence_strategy must be one of {list(STRATEGIES)}, got "
            f"{confidence_strategy!r}"
        )

    if not 0 <= overlap_threshold <= 1:
        raise ValueError(
            f"overlap_threshold must be in [0, 1], got {overlap_threshold}"
        )

    if not keypoint_merge_threshold >= 0:
        raise ValueError(
            "keypoint_merge_threshold must be nonnegative, got "
            f"{keypoint_merge_threshold}"
        )


def _parent_boxes(parent: Any) -> Any:
    if isinstance(parent, tuple):
        boxes = parent[1]
        if boxes is None:
            raise ValueError(
                "parent_detection is a keypoint prediction without boxes; the "
                "crops are described by boxes"
            )
        return boxes

    boxes = boxes_of_view(parent)

    return boxes


def _zone(box: Sequence[float]) -> Zone:
    x0, y0, x1, y1 = (float(value) for value in box)
    zone = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    return zone


def _family_of(child: Any) -> str:
    if isinstance(child, tuple):
        if len(child) != 2 or child[1] is None:
            raise ValueError(
                "A keypoint child needs its boxes: (KeyPoints, Detections), got "
                "a prediction without boxes"
            )
        if isinstance(child[1], InstanceDetections):
            raise ValueError(
                "Keypoint children with instance masks cannot be rolled up"
            )
        return KEYPOINTS
    if isinstance(child, (InstanceDetections, SegmentationView)):
        return MASKS
    if isinstance(child, Detections):
        return BOXES

    raise TypeError(
        "child_detections must hold native detections, instance segmentation, "
        f"(KeyPoints, Detections) tuples or None, got {type(child).__name__}"
    )


def _family(children: List[Any]) -> str:
    present = [child for child in children if child is not None]
    families = [_family_of(child) for child in present]
    with_rows = {
        family
        for family, child in zip(families, present)
        if len(_boxes_of(child).xyxy) > 0
    }
    if len(with_rows) > 1:
        raise ValueError(
            "child_detections mix predictions of different families "
            f"({', '.join(sorted(with_rows))}); roll up one family at a time"
        )

    if with_rows:
        return with_rows.pop()

    # Only empty children: keep their family when they agree, else boxes (V1).
    family = families[0] if len(set(families)) == 1 else BOXES

    return family


def _boxes_of(child: Any) -> Any:
    if isinstance(child, tuple):
        return child[1]

    boxes = boxes_of_view(child)

    return boxes


def _merged_class_names(
    children: List[Any], *, raise_on_conflict: bool
) -> Dict[int, Any]:
    """Union of the children's ``class_names`` maps; a later child wins."""
    merged: Dict[int, Any] = {}
    for child in children:
        if child is None:
            continue
        names = class_names_by_id(_boxes_of(child).image_metadata) or {}
        for class_id, name in names.items():
            class_id = int(class_id)
            existing = merged.get(class_id)
            if existing is not None and existing != name:
                _class_name_conflict(
                    class_id, existing=existing, new=name, raise_error=raise_on_conflict
                )
            merged[class_id] = name

    return merged


def _class_name_conflict(class_id: int, *, existing, new, raise_error: bool) -> None:
    message = (
        f"Conflicting class names for class_id={class_id} while rolling up "
        f"detections: existing '{existing}' vs incoming '{new}'."
    )
    if raise_error:
        raise ValueError(
            f"{message} Disable 'raise_on_class_name_conflict' to keep the later "
            "name in class_names and continue."
        )

    logger.warning(
        f"{message} Keeping the later value ('{new}') in class_names; each row "
        "keeps its own name. Enable 'raise_on_class_name_conflict' to raise."
    )


def _gather_rows(
    children: List[Any],
    *,
    frame: FrameRef,
    parent_boxes: List[List[float]],
    family: str,
    class_names: Dict[int, Any],
) -> _Rows:
    # All children decide output layout, including empty RLE/keypoint children.
    # Only nonempty children contribute payload rows below.
    rle_storage = family == MASKS and any(
        isinstance(getattr(child, "mask", None), InstancesRLEMasks)
        for child in children
        if child is not None
    )
    key_point_width = (
        max(
            (
                int(child[0].xy.shape[1])
                for child in children
                if isinstance(child, tuple)
            ),
            default=0,
        )
        if family == KEYPOINTS
        else 0
    )
    moved, metadata, names, host_class_ids = [], [], [], []
    for child, parent_box in zip(children, parent_boxes):
        if child is None:
            continue

        carrier = _boxes_of(child) if isinstance(child, SegmentationView) else child
        # Views validate their score rows at construction; inspect their boxes
        # without materializing masks. Native carriers also validate mask rows.
        child_rows_count = validated_row_count(carrier)
        mapping = _child_mapping(carrier, frame=frame, parent_box=parent_box)
        if child_rows_count == 0:
            # Mapping validation also applies to children without payload rows.
            continue

        if family == MASKS:
            child = instance_predictions_of(child)
        child = prediction_to_frame(child, mapping=mapping, frame=frame)
        boxes = _boxes_of(child)
        own_names = class_names_by_id(boxes.image_metadata) or {}
        host_ids = [int(class_id) for class_id in boxes.class_id.cpu().tolist()]
        child_rows = boxes.bboxes_metadata or [{} for _ in host_ids]
        for row, class_id in zip(child_rows, host_ids):
            name = row_class_name(row or {})
            if name is None:
                name = own_names.get(class_id, class_names.get(class_id))
            names.append(f"class_{class_id}" if name is None else name)
            metadata.append(without_host_mirrors(row))
        host_class_ids.extend(host_ids)
        moved.append(child)

    boxes = [_boxes_of(child) for child in moved]
    dense_masks, rle_masks = (
        _mask_payloads(boxes, rle_storage=rle_storage)
        if family == MASKS
        else (None, None)
    )
    key_points = [child[0] for child in moved] if family == KEYPOINTS else None
    key_point_counts = (
        [int(points.xy.shape[1]) for points in key_points for _ in points.xy]
        if key_points is not None
        else None
    )
    rows = _Rows(
        family=family,
        xyxy=torch.cat([b.xyxy for b in boxes]) if boxes else None,
        class_id=torch.cat([b.class_id for b in boxes]) if boxes else None,
        confidence=torch.cat([b.confidence for b in boxes]) if boxes else None,
        metadata=metadata,
        names=names,
        host_class_ids=host_class_ids,
        dense_masks=dense_masks,
        rle_masks=rle_masks,
        rle_storage=rle_storage,
        key_points=key_points,
        key_point_counts=key_point_counts,
        key_point_width=key_point_width,
    )

    return rows


def _child_mapping(
    child: Any, *, frame: FrameRef, parent_box: List[float]
) -> FrameMapping:
    """Validate stated lineage and derive the child-to-parent placement."""
    if _states_frame(child):
        child_frame = frame_of(child)
        local, target = child_frame.root, frame.root
        if local.frame_id != target.frame_id:
            raise ValueError(
                f"Child root {local.frame_id!r} differs from parent root "
                f"{target.frame_id!r}; cannot roll up unrelated workflow inputs"
            )
        if local.frame_id is None and (
            child_frame.image_id is None or child_frame.image_id != frame.image_id
        ):
            raise ValueError("Cannot establish a shared root for child and parent")
        # Both mappings describe root pixels per local pixel. Divide out the
        # parent's placement to express child pixels in the parent's frame.
        scale = tuple(local.scale_xy[i] / target.scale_xy[i] for i in range(2))
        offset = tuple(
            (local.offset_xy[i] - target.offset_xy[i]) / target.scale_xy[i]
            for i in range(2)
        )
    else:
        # V1's rule for a child that does not say where it was cut.
        scale = (1.0, 1.0)
        offset = (int(parent_box[0]), int(parent_box[1]))
    mapping = FrameMapping(
        frame_id=frame.image_id,
        frame_size_hw=frame.size_hw,
        scale_xy=scale,
        offset_xy=offset,
    )

    return mapping


def _states_frame(child: Any) -> bool:
    carriers = child if isinstance(child, tuple) else (child,)
    states = any(
        key in (carrier.image_metadata or {})
        for carrier in carriers
        if carrier is not None
        for key in FRAME_KEYS
    )

    return states


def _mask_payloads(
    boxes: List[Any], *, rle_storage: bool
) -> Tuple[Optional[torch.Tensor], Optional[List[dict]]]:
    """Collect nonempty mask rows in the layout chosen from all children."""
    masks = [b.mask for b in boxes]
    if not rle_storage:
        dense_masks = (
            torch.cat(
                [mask if mask.dtype == torch.bool else mask != 0 for mask in masks]
            )
            if masks
            else None
        )

        return dense_masks, None

    # Some child stores RLE: dense rows are encoded on the host, as stitch does.
    encoded = []
    for mask in masks:
        if isinstance(mask, InstancesRLEMasks):
            size = [int(mask.image_size[0]), int(mask.image_size[1])]
            encoded.extend({"size": size, "counts": c} for c in mask.masks)
            continue

        for one in (mask != 0).cpu().numpy().astype(np.uint8):
            encoded.append(mask_utils.encode(np.asfortranarray(one)))

    return None, encoded


def _merge_groups(
    rows: _Rows, *, overlap_threshold: float, keypoint_merge_threshold: float
) -> List[List[int]]:
    """Row positions of every merged row, in output order."""
    if not len(rows):
        return []

    by_class: Dict[int, List[int]] = {}
    for position, class_id in enumerate(rows.host_class_ids):
        by_class.setdefault(class_id, []).append(position)

    if rows.family == KEYPOINTS:
        distances = _key_point_distances(rows)
        groups = [
            group
            for members in by_class.values()
            for group in _seeded_groups(
                members, distances=distances, threshold=keypoint_merge_threshold
            )
        ]
        return groups

    pairs = [
        (first, second)
        for members in by_class.values()
        for index, first in enumerate(members)
        for second in members[index + 1 :]
    ]
    if rows.family == MASKS:
        overlapping = _overlapping_masks(rows, pairs=pairs, threshold=overlap_threshold)
    else:
        overlapping = _overlapping_boxes(rows, pairs=pairs, threshold=overlap_threshold)
    groups = [
        group
        for members in by_class.values()
        for group in _connected_groups(members, overlapping=overlapping)
    ]

    return groups


def _connected_groups(members: List[int], *, overlapping: Pairs) -> List[List[int]]:
    """Connected components of ``members``, ordered by their lowest member."""
    owner = {member: member for member in members}

    def find(member: int) -> int:
        while owner[member] != member:
            owner[member] = owner[owner[member]]
            member = owner[member]
        return member

    for index, first in enumerate(members):
        for second in members[index + 1 :]:
            if (first, second) in overlapping:
                owner[find(first)] = find(second)

    components: Dict[int, List[int]] = {}
    for member in members:
        components.setdefault(find(member), []).append(member)
    groups = list(components.values())

    return groups


def _overlapping_boxes(
    rows: _Rows, *, pairs: List[Tuple[int, int]], threshold: float
) -> Pairs:
    if not pairs:
        return set()

    iou = box_iou_between(rows.xyxy, rows.xyxy).cpu().tolist()
    overlapping = {
        (a, b)
        for a, b in pairs
        if (iou[a][b] > 0 if threshold <= 0 else iou[a][b] >= threshold)
    }

    return overlapping


def _overlapping_masks(
    rows: _Rows, *, pairs: List[Tuple[int, int]], threshold: float
) -> Pairs:
    """Pixel overlap of pairs whose boxes intersect; V1 skips the others."""
    boxes = rows.xyxy.cpu().tolist()
    candidates = [(a, b) for a, b in pairs if _boxes_intersect(boxes[a], boxes[b])]
    if not candidates:
        return set()

    if rows.rle_storage:
        ious = [
            float(mask_utils.iou([rows.rle_masks[a]], [rows.rle_masks[b]], [0])[0][0])
            for a, b in candidates
        ]
        decisions = [iou > 0 if threshold <= 0 else iou >= threshold for iou in ious]
    else:
        decisions = _dense_mask_decisions(
            rows.dense_masks, candidates, threshold=threshold
        )
    overlapping = {pair for pair, overlap in zip(candidates, decisions) if overlap}

    return overlapping


def _boxes_intersect(a: Sequence[float], b: Sequence[float]) -> bool:
    disjoint = a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]

    return not disjoint


def _dense_mask_decisions(
    masks: torch.Tensor, pairs: List[Tuple[int, int]], *, threshold: float
) -> List[bool]:
    # Exact integer pixel counts, computed on the masks' device, read once.
    intersections = torch.stack([(masks[a] & masks[b]).sum() for a, b in pairs])
    if threshold <= 0:
        decisions = (intersections > 0).cpu().tolist()
        return decisions

    areas = masks.flatten(1).sum(dim=1)
    host = torch.cat([intersections, areas]).cpu().tolist()
    counts, areas = host[: len(pairs)], host[len(pairs) :]
    decisions = []
    for (a, b), intersection in zip(pairs, counts):
        union = areas[a] + areas[b] - intersection
        iou = intersection / union if union > 0 else 0.0
        decisions.append(iou >= threshold)

    return decisions


def _key_point_distances(rows: _Rows) -> List[List[float]]:
    """Mean keypoint distance of every row pair; NaN for different counts."""
    width = rows.key_point_width
    xy = torch.cat([_pad_slots(points.xy, width=width) for points in rows.key_points])
    xy = xy.to(torch.float64)
    counts = torch.tensor(rows.key_point_counts, device=xy.device)
    valid = torch.arange(width, device=xy.device)[None, :] < counts[:, None]
    gaps = torch.linalg.vector_norm(xy[:, None] - xy[None, :], dim=-1)
    mean = (gaps * valid[:, None, :]).sum(dim=-1) / counts[:, None]
    same_count = counts[:, None] == counts[None, :]
    distances = torch.where(same_count, mean, torch.nan).cpu().tolist()

    return distances


def _seeded_groups(
    members: List[int], *, distances: List[List[float]], threshold: float
) -> List[List[int]]:
    """V1's keypoint grouping: later rows join the first unused row as seed."""
    used = set()
    groups = []
    for index, seed in enumerate(members):
        if seed in used:
            continue
        group = [seed]
        used.add(seed)
        for other in members[index + 1 :]:
            if other not in used and distances[seed][other] < threshold:
                group.append(other)
                used.add(other)
        groups.append(group)

    return groups


def _pad_slots(values: torch.Tensor, *, width: int) -> torch.Tensor:
    missing = width - values.shape[1]
    if missing == 0:
        return values

    shape = (values.shape[0], missing, *values.shape[2:])
    padding = torch.zeros(shape, dtype=values.dtype, device=values.device)
    padded = torch.cat([values, padding], dim=1)

    return padded


def _merged_prediction(
    rows: _Rows,
    *,
    groups: List[List[int]],
    frame: FrameRef,
    class_names: Dict[int, Any],
    confidence_strategy: str,
    inference_id: Optional[str],
    device: torch.device,
) -> Any:
    image_metadata = {
        **frame.prediction_metadata(),
        PREDICTION_TYPE_KEY: PREDICTION_TYPES[rows.family],
    }
    masks = None
    if rows.family == MASKS and groups:
        masks, mask_boxes, kept = _merged_masks(rows, groups=groups, frame=frame)
        groups = [groups[index] for index in kept]
    if not groups:
        if class_names:
            image_metadata[CLASS_NAMES_KEY] = dict(class_names)
        empty = _empty_prediction(
            rows, frame=frame, image_metadata=image_metadata, device=device
        )
        return with_row_identity(empty)

    output_ids = [rows.host_class_ids[group[0]] for group in groups]
    image_metadata[CLASS_NAMES_KEY] = {
        class_id: class_names.get(class_id, f"class_{class_id}")
        for class_id in output_ids
    }
    if inference_id is not None:
        image_metadata[INFERENCE_ID_KEY] = inference_id

    merge = _GroupMerge.of(groups, rows=len(rows), device=rows.xyxy.device)
    fields = {
        "xyxy": None,
        "class_id": rows.class_id.index_select(0, merge.first),
        "confidence": merge.reduce(rows.confidence, how=confidence_strategy).to(
            torch.float32
        ),
        "image_metadata": image_metadata,
        "bboxes_metadata": _merged_row_metadata(rows, groups=groups),
    }
    if rows.family == MASKS:
        merged = InstanceDetections(**{**fields, "xyxy": mask_boxes}, mask=masks)
    elif rows.family == KEYPOINTS:
        xyxy = merge.reduce(rows.xyxy, how="mean").to(torch.float32)
        boxes = Detections(**{**fields, "xyxy": xyxy})
        key_points = _merged_key_points(
            rows,
            merge=merge,
            confidence_strategy=confidence_strategy,
            image_metadata=dict(image_metadata),
        )
        merged = (key_points, boxes)
    else:
        corners = (
            merge.reduce(rows.xyxy[:, :2], how="min"),
            merge.reduce(rows.xyxy[:, 2:], how="max"),
        )
        merged = Detections(**{**fields, "xyxy": torch.cat(corners, dim=1)})
    identified = with_row_identity(merged)

    return identified


@dataclass
class _GroupMerge:
    """Row -> group assignment on the rows' device, for per-group reductions."""

    owner: torch.Tensor
    first: torch.Tensor
    count: int

    @classmethod
    def of(cls, groups: List[List[int]], *, rows: int, device) -> "_GroupMerge":
        # Rows of dropped groups go to one extra bucket that is cut off.
        owner = torch.full((rows,), len(groups), dtype=torch.int64)
        for index, group in enumerate(groups):
            owner[group] = index
        first = torch.tensor([group[0] for group in groups], dtype=torch.int64)
        merge = cls(owner=owner.to(device), first=first.to(device), count=len(groups))

        return merge

    def reduce(self, values: torch.Tensor, *, how: str) -> torch.Tensor:
        """Per-group ``max`` / ``min`` / ``mean`` (float64) of ``values`` rows."""
        reduced = _reduce(values, owner=self.owner, groups=self.count + 1, how=how)

        return reduced[: self.count]


def _reduce(
    values: torch.Tensor, *, owner: torch.Tensor, groups: int, how: str
) -> torch.Tensor:
    """Per-group max / min / mean of ``values`` rows; means in float64."""
    shape = (groups, *values.shape[1:])
    index = owner.reshape(-1, *([1] * (values.dim() - 1))).expand_as(values)
    if how == "mean":
        sums = torch.zeros(shape, dtype=torch.float64, device=values.device)
        sums.scatter_add_(0, index, values.to(torch.float64))
        counts = torch.bincount(owner, minlength=groups).to(torch.float64)
        reduced = sums / counts.reshape(-1, *([1] * (values.dim() - 1)))
        return reduced

    reduced = torch.zeros(shape, dtype=values.dtype, device=values.device)
    reduced.scatter_reduce_(
        0, index, values, reduce="amax" if how == "max" else "amin", include_self=False
    )

    return reduced


def _merged_masks(
    rows: _Rows, *, groups: List[List[int]], frame: FrameRef
) -> Tuple[Any, torch.Tensor, List[int]]:
    """Union masks, their boxes and the groups whose union is not empty."""
    device = rows.xyxy.device
    if rows.rle_storage:
        unions = [
            (
                rows.rle_masks[group[0]]
                if len(group) == 1
                else mask_utils.merge([rows.rle_masks[i] for i in group], intersect=0)
            )
            for group in groups
        ]
        kept = [index for index, rle in enumerate(unions) if mask_utils.area(rle) > 0]
        boxes = [mask_utils.toBbox(unions[index]).tolist() for index in kept]
        xyxy = torch.tensor(
            [[x, y, x + w, y + h] for x, y, w, h in boxes], dtype=torch.float32
        ).reshape(-1, 4)
        masks = InstancesRLEMasks(
            image_size=frame.size_hw, masks=[unions[index]["counts"] for index in kept]
        )
        return masks, xyxy.to(device), kept

    dense = rows.dense_masks
    unions = torch.stack(
        [
            dense[group[0]] if len(group) == 1 else dense[group].any(dim=0)
            for group in groups
        ]
    )
    xyxy, nonempty = _mask_boxes(unions)
    kept = [index for index, present in enumerate(nonempty.cpu().tolist()) if present]
    if len(kept) < len(groups):
        selected = torch.tensor(kept, dtype=torch.int64, device=device)
        unions, xyxy = unions.index_select(0, selected), xyxy.index_select(0, selected)

    return unions, xyxy, kept


def _mask_boxes(masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Box of every ``(G, H, W)`` mask, last pixel inclusive, and nonempty flags."""
    height, width = masks.shape[1:]
    rows = masks.any(dim=2).to(torch.int32)
    columns = masks.any(dim=1).to(torch.int32)
    y0 = rows.argmax(dim=1)
    y1 = height - rows.flip(1).argmax(dim=1)
    x0 = columns.argmax(dim=1)
    x1 = width - columns.flip(1).argmax(dim=1)
    xyxy = torch.stack([x0, y0, x1, y1], dim=1).to(torch.float32)
    nonempty = rows.any(dim=1)

    return xyxy, nonempty


def _merged_key_points(
    rows: _Rows,
    *,
    merge: _GroupMerge,
    confidence_strategy: str,
    image_metadata: Dict[str, Any],
) -> KeyPoints:
    """Mean keypoints and keypoint confidences of every group, padded to one width."""
    width, parts = rows.key_point_width, rows.key_points
    xy = torch.cat([_pad_slots(points.xy, width=width) for points in parts])
    point_confidence = torch.cat(
        [_pad_slots(points.confidence, width=width) for points in parts]
    )

    covariance = None
    if all(points.covariance is not None for points in parts):
        stacked = torch.cat([_pad_slots(p.covariance, width=width) for p in parts])
        covariance = merge.reduce(stacked, how="mean").to(stacked.dtype)
    detection_confidence = None
    if all(points.detection_confidence is not None for points in parts):
        stacked = torch.cat([points.detection_confidence for points in parts])
        detection_confidence = merge.reduce(stacked, how=confidence_strategy)
        detection_confidence = detection_confidence.to(stacked.dtype)
    key_points_metadata = None
    if any(points.key_points_metadata is not None for points in parts):
        entries = [
            dict(entry)
            for points in parts
            for entry in (points.key_points_metadata or [{} for _ in points.xy])
        ]
        key_points_metadata = [entries[row] for row in merge.first.cpu().tolist()]

    class_id = torch.cat([points.class_id for points in parts])
    merged = KeyPoints(
        xy=merge.reduce(xy, how="mean").to(xy.dtype),
        class_id=class_id.index_select(0, merge.first.to(class_id.device)),
        confidence=merge.reduce(point_confidence, how="mean").to(
            point_confidence.dtype
        ),
        image_metadata=image_metadata,
        key_points_metadata=key_points_metadata,
        covariance=covariance,
        detection_confidence=detection_confidence,
    )

    return merged


def _merged_row_metadata(
    rows: _Rows, *, groups: List[List[int]]
) -> List[Dict[str, Any]]:
    """V1's row values: the lowest row's, else the class's first row's, else None."""
    keys = list(
        dict.fromkeys(
            key for row in rows.metadata for key in row if key not in DROPPED_ROW_KEYS
        )
    )
    with_ids = any(DETECTION_ID_KEY in row for row in rows.metadata)
    first_of_class: Dict[int, Dict[str, Any]] = {}
    for row, class_id in zip(rows.metadata, rows.host_class_ids):
        first_of_class.setdefault(class_id, row)

    merged = []
    for group in groups:
        lowest = rows.metadata[group[0]]
        fallback = first_of_class[rows.host_class_ids[group[0]]]
        row = {key: lowest[key] if key in lowest else fallback.get(key) for key in keys}
        if with_ids:
            row[DETECTION_ID_KEY] = str(uuid.uuid4())
        row[CLASS_NAME_KEY] = rows.names[group[0]]
        merged.append(row)

    return merged


def _empty_prediction(
    rows: _Rows, *, frame: FrameRef, image_metadata: Dict[str, Any], device
) -> Any:
    fields = {
        "xyxy": torch.zeros((0, 4), dtype=torch.float32, device=device),
        "class_id": torch.zeros((0,), dtype=torch.int64, device=device),
        "confidence": torch.zeros((0,), dtype=torch.float32, device=device),
        "image_metadata": image_metadata,
        "bboxes_metadata": None,
    }
    if rows.family == MASKS:
        if rows.rle_storage:
            mask = InstancesRLEMasks(image_size=frame.size_hw, masks=[])
        else:
            mask = torch.zeros((0, *frame.size_hw), dtype=torch.bool, device=device)
        empty = InstanceDetections(mask=mask, **fields)
        return empty

    empty = Detections(**fields)
    if rows.family == KEYPOINTS:
        width = rows.key_point_width
        key_points = KeyPoints(
            xy=torch.zeros((0, width, 2), dtype=torch.float32, device=device),
            class_id=torch.zeros((0,), dtype=torch.int64, device=device),
            confidence=torch.zeros((0, width), dtype=torch.float32, device=device),
            image_metadata=dict(image_metadata),
        )
        empty = (key_points, empty)

    return empty


def _inference_id(children: List[Any]) -> Optional[str]:
    """V1: the first child's id when it has rows; a new id when only others have one."""
    present = [child for child in children if child is not None]
    if not any(
        INFERENCE_ID_KEY in (_boxes_of(c).image_metadata or {}) for c in present
    ):
        return None

    first = children[0] if children else None
    if first is not None and len(_boxes_of(first).xyxy) > 0:
        sample = (_boxes_of(first).image_metadata or {}).get(INFERENCE_ID_KEY)
        if sample:
            return str(sample)

    inference_id = str(uuid.uuid4())

    return inference_id
