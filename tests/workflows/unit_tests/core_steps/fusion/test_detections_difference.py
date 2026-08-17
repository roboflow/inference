"""
Unit tests for `DetectionsDifferenceBlockV1` (`roboflow_core/detections_difference@v1`).

The test set exercises the block's logical branches:

- manifest validation (defaults, alias, parameter bounds),
- perfect persistence / full removal / partial removal outcomes,
- appearance of new detections,
- the reject-cost edge (a pair whose cost lands just above / just below /
  exactly at `reject_cost`),
- class handling (`class_mismatch_penalty`, `class_strict`, class-less inputs),
- `min_removed_to_verify` gating of the `verified` output,
- empty inputs on either / both sides,
- greedy one-to-one assignment (incl. the documented greedy-vs-optimal
  trade-off) via the pure `greedy_match` helper.

Cost model under test (see the block docstring):

    cost = spatial_weight * spatial_term (+ class_mismatch_penalty)

with `spatial_term = 1 - IoU` for overlapping boxes and
`1 + centre_distance / enclosing_diagonal` for disjoint boxes.
"""

from typing import List

import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.fusion.detections_difference.v1 import (
    BlockManifest,
    DetectionsDifferenceBlockV1,
    compute_pairwise_costs,
    greedy_match,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _detections_from_xyxy(
    xyxy: np.ndarray,
    *,
    class_names: List[str] = None,
    confidences: List[float] = None,
    detection_ids: List[str] = None,
) -> sv.Detections:
    n = len(xyxy)
    data = {}
    if class_names is not None:
        data["class_name"] = np.array(class_names)
    if detection_ids is not None:
        data["detection_id"] = np.array(detection_ids)
    return sv.Detections(
        xyxy=np.asarray(xyxy, dtype=float),
        confidence=(
            np.asarray(confidences, dtype=float)
            if confidences is not None
            else np.ones(n, dtype=float)
        ),
        class_id=np.zeros(n, dtype=int),
        data=data,
    )


def _run_block(reference: sv.Detections, candidate: sv.Detections, **overrides):
    params = {
        "spatial_weight": 0.4,
        "class_mismatch_penalty": 0.15,
        "reject_cost": 0.75,
        "class_strict": False,
        "min_removed_to_verify": 1,
    }
    params.update(overrides)
    return DetectionsDifferenceBlockV1().run(
        reference_predictions=reference,
        candidate_predictions=candidate,
        **params,
    )


def _detection_ids(detections: sv.Detections) -> set:
    return set(detections.data.get("detection_id", []))


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_validation_when_valid_input_provided() -> None:
    # given
    specification = {
        "type": "roboflow_core/detections_difference@v1",
        "name": "difference",
        "reference_predictions": "$steps.before_model.predictions",
        "candidate_predictions": "$steps.after_model.predictions",
        "spatial_weight": 0.5,
        "class_mismatch_penalty": 0.2,
        "reject_cost": 0.6,
        "class_strict": True,
        "min_removed_to_verify": 2,
    }

    # when
    manifest = BlockManifest.model_validate(specification)

    # then
    assert manifest.type == "roboflow_core/detections_difference@v1"
    assert manifest.reference_predictions == "$steps.before_model.predictions"
    assert manifest.candidate_predictions == "$steps.after_model.predictions"
    assert manifest.spatial_weight == 0.5
    assert manifest.class_mismatch_penalty == 0.2
    assert manifest.reject_cost == 0.6
    assert manifest.class_strict is True
    assert manifest.min_removed_to_verify == 2


def test_manifest_accepts_short_type_alias_and_applies_defaults() -> None:
    # given
    specification = {
        "type": "DetectionsDifference",
        "name": "difference",
        "reference_predictions": "$steps.before_model.predictions",
        "candidate_predictions": "$steps.after_model.predictions",
    }

    # when
    manifest = BlockManifest.model_validate(specification)

    # then
    assert manifest.type == "DetectionsDifference"
    assert manifest.spatial_weight == 0.4
    assert manifest.class_mismatch_penalty == 0.15
    assert manifest.reject_cost == 0.75
    assert manifest.class_strict is False
    assert manifest.min_removed_to_verify == 1


@pytest.mark.parametrize(
    "field, value",
    [
        ("spatial_weight", 1.5),
        ("spatial_weight", -0.1),
        ("class_mismatch_penalty", -0.15),
        ("reject_cost", -1.0),
        ("min_removed_to_verify", 0),
    ],
)
def test_manifest_rejects_out_of_bounds_parameters(field: str, value) -> None:
    # given
    specification = {
        "type": "roboflow_core/detections_difference@v1",
        "name": "difference",
        "reference_predictions": "$steps.before_model.predictions",
        "candidate_predictions": "$steps.after_model.predictions",
        field: value,
    }

    # when
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(specification)


def test_describe_outputs_matches_run_keys() -> None:
    # given
    reference = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]), detection_ids=["r0"]
    )
    candidate = sv.Detections.empty()

    # when
    declared = {output.name for output in BlockManifest.describe_outputs()}
    result = _run_block(reference, candidate)

    # then
    assert declared == set(result.keys())


# ---------------------------------------------------------------------------
# Core outcomes — removed / persisted / new
# ---------------------------------------------------------------------------


def test_identical_sets_all_persist() -> None:
    # given: before == after (same three boxes, same classes)
    xyxy = np.array([[0, 0, 100, 100], [200, 0, 300, 100], [0, 200, 100, 300]])
    reference = _detections_from_xyxy(
        xyxy, class_names=["bottle", "can", "bag"], detection_ids=["r0", "r1", "r2"]
    )
    candidate = _detections_from_xyxy(
        xyxy, class_names=["bottle", "can", "bag"], detection_ids=["c0", "c1", "c2"]
    )

    # when
    result = _run_block(reference, candidate)

    # then
    assert result["removed_count"] == 0
    assert result["new_count"] == 0
    assert len(result["persisted_detections"]) == 3
    assert _detection_ids(result["persisted_detections"]) == {"c0", "c1", "c2"}
    assert result["verified"] is False


def test_full_cleanup_all_reference_detections_removed() -> None:
    # given: the after image shows nothing
    reference = _detections_from_xyxy(
        np.array([[0, 0, 100, 100], [200, 0, 300, 100]]),
        class_names=["bottle", "can"],
        detection_ids=["r0", "r1"],
    )
    candidate = sv.Detections.empty()

    # when
    result = _run_block(reference, candidate)

    # then
    assert result["removed_count"] == 2
    assert _detection_ids(result["removed_detections"]) == {"r0", "r1"}
    assert len(result["persisted_detections"]) == 0
    assert len(result["new_detections"]) == 0
    assert result["new_count"] == 0
    assert result["verified"] is True


def test_partial_cleanup_reports_removed_and_persisted() -> None:
    # given: 3 before-objects; after image only shows the middle one (shifted
    # slightly, as a fresh detection with its own id)
    reference = _detections_from_xyxy(
        np.array([[0, 0, 100, 100], [200, 0, 300, 100], [0, 200, 100, 300]]),
        class_names=["bottle", "can", "bag"],
        detection_ids=["r0", "r1", "r2"],
    )
    candidate = _detections_from_xyxy(
        np.array([[205, 0, 305, 100]]),
        class_names=["can"],
        detection_ids=["c0"],
    )

    # when
    result = _run_block(reference, candidate)

    # then
    assert result["removed_count"] == 2
    assert _detection_ids(result["removed_detections"]) == {"r0", "r2"}
    assert _detection_ids(result["persisted_detections"]) == {"c0"}
    assert result["new_count"] == 0
    assert result["verified"] is True


def test_new_detections_reported_for_unmatched_candidates() -> None:
    # given: after image shows the original object plus a fresh, disjoint one
    reference = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]),
        class_names=["bottle"],
        detection_ids=["r0"],
    )
    candidate = _detections_from_xyxy(
        np.array([[0, 0, 100, 100], [500, 500, 600, 600]]),
        class_names=["bottle", "bag"],
        detection_ids=["c0", "c1"],
    )

    # when
    result = _run_block(reference, candidate)

    # then
    assert result["removed_count"] == 0
    assert _detection_ids(result["persisted_detections"]) == {"c0"}
    assert result["new_count"] == 1
    assert _detection_ids(result["new_detections"]) == {"c1"}
    assert result["verified"] is False


def test_outputs_are_sv_detections() -> None:
    # given
    reference = _detections_from_xyxy(np.array([[0, 0, 100, 100]]))
    candidate = _detections_from_xyxy(np.array([[500, 0, 600, 100]]))

    # when
    result = _run_block(reference, candidate)

    # then
    assert isinstance(result["removed_detections"], sv.Detections)
    assert isinstance(result["persisted_detections"], sv.Detections)
    assert isinstance(result["new_detections"], sv.Detections)
    assert isinstance(result["removed_count"], int)
    assert isinstance(result["new_count"], int)
    assert isinstance(result["verified"], bool)


# ---------------------------------------------------------------------------
# Reject-cost edge
# ---------------------------------------------------------------------------

# Disjoint same-class pair: ref (0,0,10,10) vs cand (90,0,100,10).
# centres (5,5) / (95,5) -> distance 90; enclosing box (0,0,100,10) ->
# diagonal sqrt(10100) ~= 100.4988; normalised distance ~= 0.895534;
# spatial term = 1.895534; cost = 0.4 * 1.895534 ~= 0.758214.
_DISJOINT_PAIR_COST = 0.4 * (1.0 + 90.0 / np.sqrt(10100.0))


def test_reject_cost_edge_pair_just_above_threshold_is_rejected() -> None:
    # given: pair cost ~0.7582 > default reject_cost 0.75
    assert _DISJOINT_PAIR_COST > 0.75
    reference = _detections_from_xyxy(
        np.array([[0, 0, 10, 10]]), class_names=["bottle"], detection_ids=["r0"]
    )
    candidate = _detections_from_xyxy(
        np.array([[90, 0, 100, 10]]), class_names=["bottle"], detection_ids=["c0"]
    )

    # when
    result = _run_block(reference, candidate)

    # then: no match — the reference is removed AND the candidate is new
    assert result["removed_count"] == 1
    assert _detection_ids(result["removed_detections"]) == {"r0"}
    assert result["new_count"] == 1
    assert _detection_ids(result["new_detections"]) == {"c0"}
    assert len(result["persisted_detections"]) == 0
    assert result["verified"] is True


def test_reject_cost_edge_same_pair_matches_when_threshold_raised() -> None:
    # given: same geometry, reject_cost lifted just above the pair cost
    reference = _detections_from_xyxy(
        np.array([[0, 0, 10, 10]]), class_names=["bottle"], detection_ids=["r0"]
    )
    candidate = _detections_from_xyxy(
        np.array([[90, 0, 100, 10]]), class_names=["bottle"], detection_ids=["c0"]
    )

    # when
    result = _run_block(reference, candidate, reject_cost=_DISJOINT_PAIR_COST + 1e-6)

    # then
    assert result["removed_count"] == 0
    assert _detection_ids(result["persisted_detections"]) == {"c0"}
    assert result["new_count"] == 0
    assert result["verified"] is False


def test_reject_cost_edge_on_overlapping_pair_via_iou() -> None:
    # given: spatial_weight=1.0 so cost == 1 - IoU.
    # ref (0,0,100,100) vs cand (80,0,180,100): IoU = 2000/18000 = 1/9,
    # cost = 8/9 ~= 0.889 > 0.75 -> rejected.
    reference = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]), class_names=["bottle"], detection_ids=["r0"]
    )
    candidate_far = _detections_from_xyxy(
        np.array([[80, 0, 180, 100]]), class_names=["bottle"], detection_ids=["c0"]
    )
    # cand (50,0,150,100): IoU = 5000/15000 = 1/3, cost = 2/3 <= 0.75 -> matched.
    candidate_near = _detections_from_xyxy(
        np.array([[50, 0, 150, 100]]), class_names=["bottle"], detection_ids=["c0"]
    )

    # when
    rejected = _run_block(reference, candidate_far, spatial_weight=1.0)
    matched = _run_block(reference, candidate_near, spatial_weight=1.0)

    # then
    assert rejected["removed_count"] == 1
    assert rejected["new_count"] == 1
    assert matched["removed_count"] == 0
    assert _detection_ids(matched["persisted_detections"]) == {"c0"}


def test_greedy_match_accepts_pair_at_exactly_reject_cost() -> None:
    # given: `> reject_cost` rejects, so equality must be accepted
    at_threshold = np.array([[0.75]])
    above_threshold = np.array([[0.75 + 1e-9]])

    # when / then
    assert greedy_match(cost_matrix=at_threshold, reject_cost=0.75) == [(0, 0)]
    assert greedy_match(cost_matrix=above_threshold, reject_cost=0.75) == []


# ---------------------------------------------------------------------------
# Class handling
# ---------------------------------------------------------------------------


def test_class_mismatch_penalty_can_push_pair_over_reject_cost() -> None:
    # given: identical boxes -> spatial cost 0; cross-class pair costs exactly
    # class_mismatch_penalty (0.15). reject_cost=0.1 sits between 0 and 0.15.
    reference = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]), class_names=["bottle"], detection_ids=["r0"]
    )
    candidate_same = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]), class_names=["bottle"], detection_ids=["c0"]
    )
    candidate_other = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]), class_names=["can"], detection_ids=["c0"]
    )

    # when
    same_class = _run_block(reference, candidate_same, reject_cost=0.1)
    cross_class = _run_block(reference, candidate_other, reject_cost=0.1)

    # then
    assert _detection_ids(same_class["persisted_detections"]) == {"c0"}
    assert cross_class["removed_count"] == 1
    assert cross_class["new_count"] == 1


def test_class_strict_forbids_cross_class_matches_regardless_of_cost() -> None:
    # given: identical boxes, different classes, absurdly permissive reject_cost
    reference = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]), class_names=["bottle"], detection_ids=["r0"]
    )
    candidate = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]), class_names=["can"], detection_ids=["c0"]
    )

    # when
    result = _run_block(reference, candidate, class_strict=True, reject_cost=10.0)

    # then
    assert result["removed_count"] == 1
    assert result["new_count"] == 1
    assert len(result["persisted_detections"]) == 0


def test_detections_without_class_information_match_spatially() -> None:
    # given: no class_name and no class_id on either side
    reference = sv.Detections(
        xyxy=np.array([[0, 0, 100, 100]], dtype=float),
        data={"detection_id": np.array(["r0"])},
    )
    candidate = sv.Detections(
        xyxy=np.array([[0, 0, 100, 100]], dtype=float),
        data={"detection_id": np.array(["c0"])},
    )

    # when
    result = _run_block(reference, candidate)

    # then: no class penalty applicable — pure spatial match
    assert result["removed_count"] == 0
    assert _detection_ids(result["persisted_detections"]) == {"c0"}


# ---------------------------------------------------------------------------
# verified gating
# ---------------------------------------------------------------------------


def test_verified_requires_min_removed_to_verify() -> None:
    # given: exactly one removal
    reference = _detections_from_xyxy(
        np.array([[0, 0, 100, 100], [200, 0, 300, 100]]),
        class_names=["bottle", "can"],
        detection_ids=["r0", "r1"],
    )
    candidate = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]),
        class_names=["bottle"],
        detection_ids=["c0"],
    )

    # when
    default_gate = _run_block(reference, candidate)
    raised_gate = _run_block(reference, candidate, min_removed_to_verify=2)

    # then
    assert default_gate["removed_count"] == 1
    assert default_gate["verified"] is True
    assert raised_gate["removed_count"] == 1
    assert raised_gate["verified"] is False


def test_verified_false_when_reference_is_empty() -> None:
    # given: nothing on the before image — nothing to verify against
    reference = sv.Detections.empty()
    candidate = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]), class_names=["bottle"], detection_ids=["c0"]
    )

    # when
    result = _run_block(reference, candidate)

    # then
    assert result["removed_count"] == 0
    assert result["new_count"] == 1
    assert _detection_ids(result["new_detections"]) == {"c0"}
    assert result["verified"] is False


def test_both_sides_empty_yield_empty_outputs() -> None:
    # when
    result = _run_block(sv.Detections.empty(), sv.Detections.empty())

    # then
    assert len(result["removed_detections"]) == 0
    assert len(result["persisted_detections"]) == 0
    assert len(result["new_detections"]) == 0
    assert result["removed_count"] == 0
    assert result["new_count"] == 0
    assert result["verified"] is False


# ---------------------------------------------------------------------------
# Assignment behaviour
# ---------------------------------------------------------------------------


def test_two_by_two_assignment_pairs_each_reference_with_nearest_candidate() -> None:
    # given: A~X (IoU .818), B~Y (IoU .667), weaker cross pairings
    reference = _detections_from_xyxy(
        np.array([[0, 0, 100, 100], [100, 0, 200, 100]]),
        class_names=["bottle", "bottle"],
        detection_ids=["a", "b"],
    )
    candidate = _detections_from_xyxy(
        np.array([[10, 0, 110, 100], [120, 0, 220, 100]]),
        class_names=["bottle", "bottle"],
        detection_ids=["x", "y"],
    )

    # when
    result = _run_block(reference, candidate)

    # then: both matched, nothing removed or new
    assert result["removed_count"] == 0
    assert result["new_count"] == 0
    assert _detection_ids(result["persisted_detections"]) == {"x", "y"}


def test_candidate_matched_at_most_once() -> None:
    # given: two overlapping reference boxes but a single candidate — only one
    # reference can claim it, the other must be reported removed
    reference = _detections_from_xyxy(
        np.array([[0, 0, 100, 100], [10, 10, 110, 110]]),
        class_names=["bottle", "bottle"],
        detection_ids=["r0", "r1"],
    )
    candidate = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]),
        class_names=["bottle"],
        detection_ids=["c0"],
    )

    # when
    result = _run_block(reference, candidate)

    # then
    assert result["removed_count"] == 1
    assert _detection_ids(result["removed_detections"]) == {"r1"}
    assert _detection_ids(result["persisted_detections"]) == {"c0"}
    assert result["new_count"] == 0


def test_greedy_match_picks_globally_cheapest_pairs_first() -> None:
    # given
    cost_matrix = np.array(
        [
            [0.10, 0.20],
            [0.15, 0.90],
        ]
    )

    # when
    matches = greedy_match(cost_matrix=cost_matrix, reject_cost=1.0)

    # then
    assert matches == [(0, 0), (1, 1)]


def test_greedy_match_documented_suboptimality() -> None:
    # given: Hungarian assignment would pair (0,1)+(1,0) for total 0.31 and
    # two matches; greedy takes (0,0) first and leaves row 1 unmatchable.
    # This pins the documented greedy trade-off so a future switch to an
    # optimal solver shows up as a deliberate test change.
    cost_matrix = np.array(
        [
            [0.10, 0.20],
            [0.11, np.inf],
        ]
    )

    # when
    matches = greedy_match(cost_matrix=cost_matrix, reject_cost=1.0)

    # then
    assert matches == [(0, 0)]


# ---------------------------------------------------------------------------
# Cost matrix
# ---------------------------------------------------------------------------


def test_compute_pairwise_costs_overlapping_pair_uses_one_minus_iou() -> None:
    # given: IoU = 5000/15000 = 1/3
    reference = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]), class_names=["bottle"]
    )
    candidate = _detections_from_xyxy(
        np.array([[50, 0, 150, 100]]), class_names=["bottle"]
    )

    # when
    costs = compute_pairwise_costs(
        reference=reference,
        candidate=candidate,
        spatial_weight=1.0,
        class_mismatch_penalty=0.15,
        class_strict=False,
    )

    # then
    assert costs.shape == (1, 1)
    assert costs[0, 0] == pytest.approx(2.0 / 3.0)


def test_compute_pairwise_costs_disjoint_pair_exceeds_any_overlapping_pair() -> None:
    # given: one overlapping and one disjoint candidate for the same reference
    reference = _detections_from_xyxy(
        np.array([[0, 0, 100, 100]]), class_names=["bottle"]
    )
    candidate = _detections_from_xyxy(
        np.array([[99, 0, 199, 100], [101, 0, 201, 100]]),
        class_names=["bottle", "bottle"],
    )

    # when
    costs = compute_pairwise_costs(
        reference=reference,
        candidate=candidate,
        spatial_weight=1.0,
        class_mismatch_penalty=0.15,
        class_strict=False,
    )

    # then: barely-overlapping (IoU ~ 0.005) still beats barely-disjoint
    assert costs[0, 0] < 1.0 <= costs[0, 1]


def test_compute_pairwise_costs_disjoint_pairs_rank_by_distance() -> None:
    # given: two disjoint candidates at different distances
    reference = _detections_from_xyxy(
        np.array([[0, 0, 10, 10]]), class_names=["bottle"]
    )
    candidate = _detections_from_xyxy(
        np.array([[20, 0, 30, 10], [200, 0, 210, 10]]),
        class_names=["bottle", "bottle"],
    )

    # when
    costs = compute_pairwise_costs(
        reference=reference,
        candidate=candidate,
        spatial_weight=1.0,
        class_mismatch_penalty=0.15,
        class_strict=False,
    )

    # then
    assert costs[0, 0] < costs[0, 1]


# ---------------------------------------------------------------------------
# Loader registration
# ---------------------------------------------------------------------------


def test_block_is_registered_in_loader() -> None:
    # a block is invisible to the execution engine until it is listed in
    # `load_blocks()` — this pins the registration. The loader swaps in the
    # tensor-native sibling under ENABLE_TENSOR_DATA_REPRESENTATION, so the
    # expected class follows the flag.
    from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
    from inference.core.workflows.core_steps import loader

    if ENABLE_TENSOR_DATA_REPRESENTATION:
        from inference.core.workflows.core_steps.fusion.detections_difference.v1_tensor import (
            DetectionsDifferenceBlockV1 as expected_block,
        )
    else:
        expected_block = DetectionsDifferenceBlockV1

    assert expected_block in loader.load_blocks()
