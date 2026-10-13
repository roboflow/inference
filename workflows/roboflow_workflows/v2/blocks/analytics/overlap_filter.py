"""``v2/overlap_filter@v1``: keep the rows that overlap a reference class.

Port of V1 ``roboflow_core/overlap@v1`` ("Overlap Filter"); its tensor
variant is the reference. It is a filter, unlike ``v2/overlap_analysis@v1``,
which measures overlap. Rows of ``overlap_class_name`` are references; the
output keeps every other row that overlaps at least one reference, in input
order, and drops the references themselves::

    Center Overlap   x1 < cx < x2 and y1 < cy < y2   strict: a centre on an
                     (cx, cy) = row box centre       edge is outside
    Any Overlap      the boxes intersect or touch    inclusive: shared edges
                                                     and corners count

    (x1, y1, x2, y2) = a reference box

So ``bicycle`` as the overlap class keeps the people riding bicycles and
drops the bicycles. No reference row, or no other row, gives an empty
prediction. Only boxes are compared, also for instance segmentations.

The output has the input's family. Kept rows keep their masks (a
segmentation view stays low resolution), metadata and row identity; no row
is changed. A row's class is its own ``class`` name, else
``image_metadata["class_names"]`` of its class id, else ``class_<id>``.

Device and CPU boundary: the comparison runs on the boxes' device, in
float64 as V1 does (float32 on MPS, which has no float64). Class ids are
copied to the host once to name rows; the selection follows the native row
selection's host work.

Differences from V1:

* The output keeps instance segmentations; V1 declared only object
  detection output.
* ``overlap_class_name`` and ``overlap_type`` can be selected; a selected
  ``overlap_type`` is checked in the call.
* A segmentation view is accepted and stays lazy.
"""

from typing import Any, Dict, List, Literal

import torch
from pydantic import Field, StrictStr
from roboflow_workflows.v2.blocks.query.detections import boxes_of, select_rows
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.data.masks import INSTANCE_SEGMENTATION_VIEW_KIND
from roboflow_workflows.v2.data.prediction_ops import (
    RowSelection,
    class_names_by_id,
    row_class_names,
)
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

__all__ = [
    "OVERLAP_TYPES",
    "OverlapFilterBlock",
]

OVERLAP_TYPES = ("Center Overlap", "Any Overlap")
OverlapType = Literal["Center Overlap", "Any Overlap"]

FILTERED_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_VIEW_KIND,
)


class OverlapFilterBlock(Block):
    """Keep rows overlapping any row of a reference class; drop the references.

    Example, ``overlap_class_name="bicycle"``, ``Center Overlap``::

        row  class    box                  kept
        0    person   (12, 0, 28, 30)      yes: centre (20, 15) in bicycle
        1    bicycle  (0, 10, 40, 40)      no: reference
        2    person   (60, 0, 80, 30)      no: centre (70, 15) outside
    """

    type = "v2/overlap_filter@v1"
    prunable = True
    metadata = {
        "name": "Overlap Filter",
        "port_of": "roboflow_core/overlap@v1",
        "section": "flow_control",
    }
    outputs = {
        "overlaps": Output(
            *FILTERED_KINDS,
            source="predictions",
            description=(
                "Rows overlapping a reference row, in input order, without "
                "the reference rows."
            ),
        ),
    }

    class Params(BlockParams):
        predictions: Ref(*FILTERED_KINDS) = Field(
            description="Detections or instance segmentations to filter.",
            examples=["$steps.object_detection_model.predictions"],
        )
        overlap_class_name: StrictStr | Ref(STRING_KIND) = Field(
            description=(
                "Class of the reference rows, e.g. 'bicycle' or 'pallet'. "
                "They are removed from the output."
            ),
            examples=["bicycle", "$inputs.reference_class"],
        )
        overlap_type: OverlapType | Ref(STRING_KIND) = Field(
            default="Center Overlap",
            description=(
                "'Center Overlap': the row's box centre is strictly inside a "
                "reference box. 'Any Overlap': the boxes intersect or touch."
            ),
            examples=["Center Overlap", "Any Overlap"],
        )

    def run(
        self,
        *,
        predictions: Any,
        overlap_class_name: str,
        overlap_type: str = "Center Overlap",
    ) -> Dict[str, Any]:
        """Select the rows overlapping a reference row.

        Args:
            predictions: Rows to filter; not modified.
            overlap_class_name: Class of the reference rows.
            overlap_type: ``"Center Overlap"`` or ``"Any Overlap"``.

        Returns:
            ``overlaps``: the selected rows, in the input's family.

        Raises:
            ValueError: When a selected ``overlap_type`` or a direct call's
                parameter is invalid.
        """
        if overlap_type not in OVERLAP_TYPES:
            raise ValueError(
                f"overlap_type must be one of {', '.join(OVERLAP_TYPES)}; got "
                f"{overlap_type!r}"
            )
        if not isinstance(overlap_class_name, str):
            raise ValueError(
                f"overlap_class_name must be a string, got {overlap_class_name!r}"
            )

        boxes = boxes_of(predictions)
        xyxy = boxes.xyxy.detach()
        dtype = torch.float32 if xyxy.device.type == "mps" else torch.float64
        xyxy = xyxy.to(dtype)
        is_reference = [name == overlap_class_name for name in _class_names(boxes)]
        reference_mask = torch.tensor(
            is_reference, dtype=torch.bool, device=xyxy.device
        )
        references = xyxy[reference_mask]
        if overlap_type == "Center Overlap":
            overlapping = _centres_inside(xyxy, references)
        else:
            overlapping = _boxes_touch(xyxy, references)
        kept = overlapping & ~reference_mask
        selection = RowSelection.of(len(boxes), mask=kept)
        result = {"overlaps": select_rows(predictions, selection)}

        return result


def _class_names(boxes: Any) -> List[str]:
    rows_metadata = list(boxes.bboxes_metadata or [])
    rows_metadata += [{} for _ in range(len(boxes) - len(rows_metadata))]
    class_ids = [int(class_id) for class_id in boxes.class_id.tolist()]
    names = row_class_names(
        class_ids,
        row_metadata=[row or {} for row in rows_metadata],
        names_by_id=class_names_by_id(boxes.image_metadata),
    )
    named = [
        str(name) if name is not None else f"class_{class_id}"
        for name, class_id in zip(names, class_ids)
    ]

    return named


def _centres_inside(xyxy: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    """``(N,)``: row centre strictly inside at least one reference box."""
    # V1's centre: x1 + (x2 - x1) / 2.
    centres = xyxy[:, :2] + (xyxy[:, 2:] - xyxy[:, :2]) / 2
    x, y = centres[:, None, 0], centres[:, None, 1]
    inside = (
        (x > references[None, :, 0])
        & (x < references[None, :, 2])
        & (y > references[None, :, 1])
        & (y < references[None, :, 3])
    )

    return inside.any(dim=1)


def _boxes_touch(xyxy: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    """``(N,)``: row box intersects or touches at least one reference box."""
    rows, others = xyxy[:, None, :], references[None, :, :]
    apart = (
        (rows[..., 2] < others[..., 0])
        | (rows[..., 0] > others[..., 2])
        | (rows[..., 3] < others[..., 1])
        | (rows[..., 1] > others[..., 3])
    )

    return (~apart).any(dim=1)
