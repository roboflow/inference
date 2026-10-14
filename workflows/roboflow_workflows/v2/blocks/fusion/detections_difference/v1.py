"""Compare before/after detections with V1's greedy one-to-one assignment.

``reference_predictions`` is the before set; ``candidate_predictions`` is the
after set. Matching compares raw xyxy coordinates, so both views must use a
comparable coordinate basis and viewpoint. Different image/frame identities
are expected; no reprojection, resize or same-frame check is performed.

    spatial = 1 - IoU                           overlapping boxes
              1 + centre_distance/enclosing_diagonal otherwise
    cost = spatial_weight * spatial + class_mismatch_penalty * mismatch

Strict class matching forbids mismatches. Repeatedly take the cheapest pair,
breaking ties by reference row then candidate row, and remove its row/column.
A cost equal to ``reject_cost`` is accepted; greedy stops at the first
non-finite cheapest cost. Removed rows come from the reference; persisted/new
rows come from the candidate, in their input order.

An infinite soft ``class_mismatch_penalty`` matches nothing, as in V1: a
same-class cell becomes ``0 * inf = NaN`` and a cross-class cell becomes
``inf``, so no cell is finite. This is deliberate V1 parity, not a NaN bug.

Box IoU uses the cleared native float32 helper. Only its small matrix and the
box/class values needed by matching reach CPU tensors. V1's float32 spatial
arithmetic, promotion for disjoint distances, and float64 greedy loop are kept.
Predictions, masks and aligned row metadata are never transported through
NumPy. The helper's float32 box areas can change exact-boundary decisions from
V1's float64 areas.
"""

import math
from typing import Annotated, Any, Dict, List, Tuple, Union
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.fusion.pairwise import box_iou_between
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.v2.data.prediction_ops import (
    CLASS_NAMES_KEY,
    RowSelection,
    take_rows,
)
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections


DetectionsInput = Union[Detections, InstanceDetections]

DETECTION_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]

NonNegativeFloat = Annotated[float, Field(ge=0.0)]

PositiveInt = Annotated[int, Field(ge=1)]

__all__ = ["DetectionsDifferenceBlock"]

class DetectionsDifferenceBlock(Block):
    """Report removed, persisted and new objects between two views of a scene.

    Class names come from each prediction's ``image_metadata['class_names']``
    map when both maps are nonempty, falling back to ``class_<id>`` for an
    unmapped id. Otherwise class ids are compared. This is the latest tensor
    V1 rule: per-row class-name overrides do not change matching. Confidence
    and masks do not participate; they follow their selected rows unchanged.

    Empty subsets preserve the source prediction's frame and row identity.
    The block is stateless, does not mutate inputs, and is prunable.
    """

    type = "roboflow_core/detections_difference@v1"
    aliases = ("DetectionsDifference",)
    prunable = True
    metadata = {
        "name": "Detections Difference",
        "port_of": "roboflow_core/detections_difference@v1",
        "section": "advanced",
    }
    outputs = {
        "removed_detections": Output(
            *DETECTION_KINDS,
            source="reference_predictions",
            description="Unmatched before rows in reference coordinates and order.",
        ),
        "persisted_detections": Output(
            *DETECTION_KINDS,
            source="candidate_predictions",
            description="Matched after rows in candidate coordinates and order.",
        ),
        "new_detections": Output(
            *DETECTION_KINDS,
            source="candidate_predictions",
            description="Unmatched after rows in candidate coordinates and order.",
        ),
        "removed_count": Output(
            INTEGER_KIND,
            source="reference_predictions",
            description="Number of unmatched reference rows.",
        ),
        "new_count": Output(
            INTEGER_KIND,
            source="candidate_predictions",
            description="Number of unmatched candidate rows.",
        ),
        "verified": Output(
            BOOLEAN_KIND,
            source="reference_predictions",
            description="Nonempty reference and enough removed rows to verify.",
        ),
    }

    class Params(BlockParams):
        reference_predictions: Ref(*DETECTION_KINDS) = Field(
            description='Native detections from the reference ("before") view.',
            examples=["$steps.before_model.predictions"],
        )
        candidate_predictions: Ref(*DETECTION_KINDS) = Field(
            description='Native detections from the candidate ("after") view.',
            examples=["$steps.after_model.predictions"],
        )
        spatial_weight: UnitFloat | Ref(FLOAT_KIND) = Field(
            default=0.4,
            description="Weight in [0,1] of IoU or normalized centre distance.",
            examples=[0.4, "$inputs.spatial_weight"],
        )
        class_mismatch_penalty: NonNegativeFloat | Ref(FLOAT_KIND) = Field(
            default=0.15,
            description="Nonnegative mismatch cost; ignored with strict classes.",
            examples=[0.15, "$inputs.class_mismatch_penalty"],
        )
        reject_cost: NonNegativeFloat | Ref(FLOAT_KIND) = Field(
            default=0.75,
            description="Nonnegative maximum accepted pair cost, inclusive.",
            examples=[0.75, "$inputs.reject_cost"],
        )
        class_strict: bool | Ref(BOOLEAN_KIND) = Field(
            default=False,
            description="Forbid pairs with different class names or ids.",
            examples=[False, "$inputs.class_strict"],
        )
        min_removed_to_verify: PositiveInt | Ref(INTEGER_KIND) = Field(
            default=1,
            description="At least this many removals verify a nonempty reference.",
            examples=[1, "$inputs.min_removed_to_verify"],
        )

    def run(
        self,
        *,
        reference_predictions: DetectionsInput,
        candidate_predictions: DetectionsInput,
        spatial_weight: float,
        class_mismatch_penalty: float,
        reject_cost: float,
        class_strict: bool,
        min_removed_to_verify: int,
    ) -> Dict[str, Any]:
        """Match two predictions and select their removed/persisted/new rows.

        Args:
            reference_predictions: Before rows in comparable xyxy coordinates.
            candidate_predictions: After rows in comparable xyxy coordinates.
            spatial_weight: Spatial cost weight, from zero to one.
            class_mismatch_penalty: Nonnegative soft class cost.
            reject_cost: Nonnegative inclusive cost limit.
            class_strict: Whether to forbid cross-class matches.
            min_removed_to_verify: Positive required removal count.

        Returns:
            All six declared outputs. Native subsets keep their source device,
            frame, masks, row ids and aligned metadata.

        Raises:
            ValueError: When a selected numeric parameter violates its bounds.
        """
        _check_bounds(
            spatial_weight=spatial_weight,
            class_mismatch_penalty=class_mismatch_penalty,
            reject_cost=reject_cost,
            min_removed_to_verify=min_removed_to_verify,
        )
        n_reference, n_candidate = len(reference_predictions), len(
            candidate_predictions
        )
        matches = []
        if n_reference and n_candidate:
            costs = _pairwise_costs(
                reference=reference_predictions,
                candidate=candidate_predictions,
                spatial_weight=spatial_weight,
                class_mismatch_penalty=class_mismatch_penalty,
                class_strict=class_strict,
            )
            matches = _greedy_match(costs, reject_cost=reject_cost)

        matched_reference = {i for i, _ in matches}
        matched_candidate = {j for _, j in matches}
        removed = [i for i in range(n_reference) if i not in matched_reference]
        persisted = sorted(matched_candidate)
        new = [j for j in range(n_candidate) if j not in matched_candidate]
        result = {
            "removed_detections": _subset(reference_predictions, removed),
            "persisted_detections": _subset(candidate_predictions, persisted),
            "new_detections": _subset(candidate_predictions, new),
            "removed_count": len(removed),
            "new_count": len(new),
            "verified": n_reference > 0 and len(removed) >= min_removed_to_verify,
        }

        return result

def _check_bounds(
    *, spatial_weight, class_mismatch_penalty, reject_cost, min_removed_to_verify
) -> None:
    if not 0 <= spatial_weight <= 1:
        raise ValueError("spatial_weight must be in [0,1]")

    for name, value in (
        ("class_mismatch_penalty", class_mismatch_penalty),
        ("reject_cost", reject_cost),
    ):
        if not value >= 0:
            raise ValueError(f"{name} must be nonnegative")

    if min_removed_to_verify < 1:
        raise ValueError("min_removed_to_verify must be at least 1")

def _pairwise_costs(
    *, reference, candidate, spatial_weight, class_mismatch_penalty, class_strict
) -> torch.Tensor:
    # Only box geometry may move between devices; output payloads stay put.
    # One CPU cost matrix avoids a device synchronization per greedy argmin.
    iou = (
        box_iou_between(reference.xyxy, candidate.xyxy.to(device=reference.xyxy.device))
        .detach()
        .to(device="cpu")
    )
    spatial = 1.0 - iou
    disjoint = iou <= 0.0
    if torch.any(disjoint):
        a = reference.xyxy.detach().to(device="cpu", dtype=torch.float64)
        b = candidate.xyxy.detach().to(device="cpu", dtype=torch.float64)
        centres_a, centres_b = (a[:, :2] + a[:, 2:]) / 2, (b[:, :2] + b[:, 2:]) / 2
        distance = torch.linalg.vector_norm(centres_a[:, None] - centres_b[None], dim=2)
        enclosing = torch.maximum(a[:, None, 2:], b[None, :, 2:]) - torch.minimum(
            a[:, None, :2], b[None, :, :2]
        )
        diagonal = torch.hypot(enclosing[..., 0], enclosing[..., 1])
        normalized = torch.zeros_like(distance)
        nonzero = diagonal > 0
        normalized[nonzero] = distance[nonzero] / diagonal[nonzero]
        spatial = torch.where(disjoint, 1.0 + normalized, spatial)

    cost = spatial_weight * spatial
    mismatch = _class_mismatch(reference, candidate)
    if class_strict:
        cost = torch.where(mismatch, torch.inf, cost)
    else:
        # NumPy's bool * Python float is float64, even when V1's
        # all-overlapping spatial cost is still float32.
        cost = cost + class_mismatch_penalty * mismatch.to(dtype=torch.float64)

    return cost

def _class_mismatch(reference, candidate) -> torch.Tensor:
    maps = [
        (prediction.image_metadata or {}).get(CLASS_NAMES_KEY) or {}
        for prediction in (reference, candidate)
    ]
    if all(maps):
        labels = [
            [
                mapping.get(int(cid), f"class_{int(cid)}")
                for cid in prediction.class_id.detach().cpu().tolist()
            ]
            for prediction, mapping in zip((reference, candidate), maps)
        ]
        mismatch = torch.tensor(
            [[a != b for b in labels[1]] for a in labels[0]], dtype=torch.bool
        )
    else:
        a, b = reference.class_id.detach().cpu(), candidate.class_id.detach().cpu()
        mismatch = a[:, None] != b[None, :]

    return mismatch

def _greedy_match(costs: torch.Tensor, *, reject_cost: float) -> List[Tuple[int, int]]:
    cost = costs.to(dtype=torch.float64).clone()
    matches = []
    for _ in range(min(cost.shape)):
        flat = int(torch.argmin(cost))
        row, column = divmod(flat, cost.shape[1])
        cheapest = float(cost[row, column])
        if not math.isfinite(cheapest) or cheapest > reject_cost:
            break

        matches.append((row, column))
        cost[row, :] = torch.inf
        cost[:, column] = torch.inf

    return matches

def _subset(prediction: DetectionsInput, indices: List[int]) -> DetectionsInput:
    selection = RowSelection.of(len(prediction), indices=indices)
    selected = take_rows(prediction, selection)

    return selected
