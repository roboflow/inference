from typing import List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    INTEGER_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    FloatZeroToOne,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Compare two sets of detections of the same scene taken at different times and
report what was removed, what persisted, and what is new.

The canonical use case is **cleanup verification**: run a detector on a
"before" photo of a littered area and again on an "after" photo taken from
roughly the same viewpoint once cleanup is claimed. Detections from the
"before" image that cannot be matched in the "after" image are `removed` —
evidence the cleanup happened. The same pattern generalises to any
before/after audit: shelf restocking checks (which products disappeared /
appeared), construction-site or warehouse inspection rounds, parking-lot
occupancy deltas, or comparing the output of two different models on the same
image to see where they disagree at the object level.

`reference_predictions` is the "before" set, `candidate_predictions` the
"after" set. Each reference detection is matched against at most one candidate
detection (and vice versa) by minimising a pairwise cost:

    cost = spatial_weight * spatial_term (+ class_mismatch_penalty)

where `spatial_term` is `1 - IoU` for overlapping boxes; for disjoint boxes it
falls back to `1 + d`, with `d` the centre distance normalised by the diagonal
of the pair's smallest enclosing box, so any overlapping pair is always
cheaper than any disjoint pair while disjoint pairs still rank by proximity.
The class penalty applies when the two detections carry different classes;
with `class_strict` enabled cross-class matches are forbidden outright.
Matched pairs whose cost exceeds `reject_cost` are discarded, turning both
sides of the pair into an unmatched removal/appearance.

Outputs: `removed_detections` (unmatched reference detections, in the
*reference* image's coordinates), `persisted_detections` and `new_detections`
(matched / unmatched candidate detections, in the *candidate* image's
coordinates), the corresponding counts, and a `verified` flag that is true
when the reference set was non-empty and at least `min_removed_to_verify`
detections were removed.

This block is the cross-time counterpart of `roboflow_core/overlap_analysis@v1`,
which relates two detection sets from the *same* image geometrically; this
block instead assigns detections one-to-one across two images of the same
scene and reports the set difference.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detections Difference",
            "version": "v1",
            "short_description": (
                "Match detections across before/after images and report "
                "removed, persisted and new objects."
            ),
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "fusion",
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-not-equal",
                "blockPriority": 11,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/detections_difference@v1", "DetectionsDifference"]
    reference_predictions: Selector(kind=[OBJECT_DETECTION_PREDICTION_KIND]) = Field(
        description=(
            'Detections from the reference ("before") image. Reference '
            "detections that cannot be matched in `candidate_predictions` are "
            "reported as `removed_detections`."
        ),
        examples=["$steps.before_model.predictions"],
    )
    candidate_predictions: Selector(kind=[OBJECT_DETECTION_PREDICTION_KIND]) = Field(
        description=(
            'Detections from the candidate ("after") image. Candidate '
            "detections matched to a reference detection are reported as "
            "`persisted_detections`; unmatched ones as `new_detections`."
        ),
        examples=["$steps.after_model.predictions"],
    )
    spatial_weight: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = (
        Field(
            default=0.4,
            description=(
                "Weight of the spatial term in the matching cost. The spatial "
                "term is `1 - IoU` for overlapping boxes and `1 + normalised "
                "centre distance` for disjoint boxes, so it ranges over [0, 2)."
            ),
            examples=[0.4, "$inputs.spatial_weight"],
        )
    )
    class_mismatch_penalty: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=0.15,
        ge=0.0,
        description=(
            "Cost added to a pair whose detections carry different classes. "
            "Ignored when `class_strict` is enabled (cross-class pairs are "
            "then forbidden entirely)."
        ),
        examples=[0.15, "$inputs.class_mismatch_penalty"],
    )
    reject_cost: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=0.75,
        ge=0.0,
        description=(
            "Pairs whose matching cost exceeds this value are not matched; "
            "both detections are then reported as removed / new respectively."
        ),
        examples=[0.75, "$inputs.reject_cost"],
    )
    class_strict: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description=(
            "When enabled, detections of different classes can never match, "
            "regardless of spatial proximity."
        ),
        examples=[False, "$inputs.class_strict"],
    )
    min_removed_to_verify: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=1,
        ge=1,
        description=(
            "Minimum number of removed detections required (together with a "
            "non-empty reference set) for the `verified` output to be true."
        ),
        examples=[1, "$inputs.min_removed_to_verify"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="removed_detections", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
            OutputDefinition(
                name="persisted_detections", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
            OutputDefinition(
                name="new_detections", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
            OutputDefinition(name="removed_count", kind=[INTEGER_KIND]),
            OutputDefinition(name="new_count", kind=[INTEGER_KIND]),
            OutputDefinition(name="verified", kind=[BOOLEAN_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class DetectionsDifferenceBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        reference_predictions: sv.Detections,
        candidate_predictions: sv.Detections,
        spatial_weight: float,
        class_mismatch_penalty: float,
        reject_cost: float,
        class_strict: bool,
        min_removed_to_verify: int,
    ) -> BlockResult:
        n_reference = len(reference_predictions)
        n_candidate = len(candidate_predictions)
        if n_reference == 0 or n_candidate == 0:
            matches: List[Tuple[int, int]] = []
        else:
            cost_matrix = compute_pairwise_costs(
                reference=reference_predictions,
                candidate=candidate_predictions,
                spatial_weight=spatial_weight,
                class_mismatch_penalty=class_mismatch_penalty,
                class_strict=class_strict,
            )
            matches = greedy_match(cost_matrix=cost_matrix, reject_cost=reject_cost)
        matched_reference = {i for i, _ in matches}
        matched_candidate = {j for _, j in matches}
        removed_indices = [i for i in range(n_reference) if i not in matched_reference]
        persisted_indices = sorted(matched_candidate)
        new_indices = [j for j in range(n_candidate) if j not in matched_candidate]
        removed_count = len(removed_indices)
        verified = n_reference > 0 and removed_count >= min_removed_to_verify
        return {
            "removed_detections": _subset(reference_predictions, removed_indices),
            "persisted_detections": _subset(candidate_predictions, persisted_indices),
            "new_detections": _subset(candidate_predictions, new_indices),
            "removed_count": removed_count,
            "new_count": len(new_indices),
            "verified": verified,
        }


def compute_pairwise_costs(
    reference: sv.Detections,
    candidate: sv.Detections,
    spatial_weight: float,
    class_mismatch_penalty: float,
    class_strict: bool,
) -> np.ndarray:
    """Return the (n_reference, n_candidate) matching-cost matrix.

    The spatial term is `1 - IoU` for overlapping pairs. Disjoint pairs get
    `1 + centre_distance / enclosing_box_diagonal` instead — strictly greater
    than any overlapping pair's term, so overlap always wins, while disjoint
    pairs still rank by proximity (a DIoU-style distance gradient).
    """
    iou = sv.box_iou_batch(reference.xyxy, candidate.xyxy)
    spatial = 1.0 - iou
    disjoint = iou <= 0.0
    if np.any(disjoint):
        reference_centers = (reference.xyxy[:, :2] + reference.xyxy[:, 2:]) / 2
        candidate_centers = (candidate.xyxy[:, :2] + candidate.xyxy[:, 2:]) / 2
        deltas = reference_centers[:, None, :] - candidate_centers[None, :, :]
        center_distance = np.linalg.norm(deltas, axis=2)
        enclosing_x1 = np.minimum(
            reference.xyxy[:, None, 0], candidate.xyxy[None, :, 0]
        )
        enclosing_y1 = np.minimum(
            reference.xyxy[:, None, 1], candidate.xyxy[None, :, 1]
        )
        enclosing_x2 = np.maximum(
            reference.xyxy[:, None, 2], candidate.xyxy[None, :, 2]
        )
        enclosing_y2 = np.maximum(
            reference.xyxy[:, None, 3], candidate.xyxy[None, :, 3]
        )
        enclosing_diagonal = np.hypot(
            enclosing_x2 - enclosing_x1, enclosing_y2 - enclosing_y1
        )
        normalized_distance = np.divide(
            center_distance,
            enclosing_diagonal,
            out=np.zeros_like(center_distance),
            where=enclosing_diagonal > 0,
        )
        spatial = np.where(disjoint, 1.0 + normalized_distance, spatial)
    cost = spatial_weight * spatial
    mismatch = _class_mismatch_matrix(reference=reference, candidate=candidate)
    if mismatch is not None:
        if class_strict:
            cost = np.where(mismatch, np.inf, cost)
        else:
            cost = cost + class_mismatch_penalty * mismatch
    return cost


def greedy_match(cost_matrix: np.ndarray, reject_cost: float) -> List[Tuple[int, int]]:
    """Greedy lowest-cost one-to-one assignment.

    Repeatedly picks the globally cheapest remaining (reference, candidate)
    pair, stopping once the cheapest pair exceeds `reject_cost` (pairs at
    exactly `reject_cost` are accepted). `scipy.optimize.linear_sum_assignment`
    would give the optimal (Hungarian) solution, but scipy is not part of this
    package's base requirements, so a dependency-free greedy pass is used
    instead; for the low detection counts typical of before/after comparisons
    the assignments rarely differ.
    """
    matches: List[Tuple[int, int]] = []
    if cost_matrix.size == 0:
        return matches
    cost = cost_matrix.astype(float, copy=True)
    n_rows, n_cols = cost.shape
    for _ in range(min(n_rows, n_cols)):
        flat_index = int(np.argmin(cost))
        i, j = divmod(flat_index, n_cols)
        cheapest = cost[i, j]
        if not np.isfinite(cheapest) or cheapest > reject_cost:
            break
        matches.append((i, j))
        cost[i, :] = np.inf
        cost[:, j] = np.inf
    return matches


def _class_mismatch_matrix(
    reference: sv.Detections, candidate: sv.Detections
) -> Optional[np.ndarray]:
    """Boolean (n_reference, n_candidate) matrix of class disagreements.

    Compares class names when both sides carry them, falling back to class
    ids. Returns None when class identity is unavailable on either side —
    the two label spaces cannot be compared, so no penalty is applied.
    """
    extractors = (
        lambda detections: detections.data.get(CLASS_NAME_DATA_FIELD),
        lambda detections: detections.class_id,
    )
    for extract in extractors:
        reference_labels = extract(reference)
        candidate_labels = extract(candidate)
        if (
            reference_labels is not None
            and candidate_labels is not None
            and len(reference_labels) == len(reference)
            and len(candidate_labels) == len(candidate)
        ):
            return (
                np.asarray(reference_labels)[:, None]
                != np.asarray(candidate_labels)[None, :]
            )
    return None


def _subset(detections: sv.Detections, indices: List[int]) -> sv.Detections:
    if len(indices) == 0:
        return sv.Detections.empty()
    # Fancy indexing propagates every `data` key (detection ids, parent
    # coordinates, ...) so downstream re-projection keeps working.
    return detections[np.asarray(indices, dtype=int)]
