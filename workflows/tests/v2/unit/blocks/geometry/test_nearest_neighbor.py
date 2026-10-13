"""Unit tests of ``v2/detections_nearest_neighbor@v1`` with hand-computed distances."""

import pytest
import torch
from roboflow_workflows.v2.blocks.geometry.nearest_neighbor import (
    DetectionsNearestNeighborBlock,
)
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections


def boxes_centred_at(points, masks=False):
    # 2 x 2 boxes whose CENTER anchors are ``points``.
    centres = torch.tensor(points, dtype=torch.float32).reshape(-1, 2)
    fields = {
        "xyxy": torch.cat([centres - 1, centres + 1], dim=1),
        "class_id": torch.arange(len(centres)),
        "confidence": torch.full((len(centres),), 0.9),
        "image_metadata": {},
        "bboxes_metadata": [{"name": f"row-{i}"} for i in range(len(centres))],
    }
    if masks:
        mask = torch.zeros((len(centres), 30, 30), dtype=torch.bool)
        for row in range(len(centres)):
            mask[row, row, row] = True
        return with_row_identity(InstanceDetections(mask=mask, **fields))

    return with_row_identity(Detections(**fields))


def run(query, target, **options):
    arguments = {
        "query_point": "CENTER",
        "target_point": "CENTER",
        "query_keypoint_name": None,
        "target_keypoint_name": None,
        "max_distance": None,
    }
    arguments.update(options)

    return DetectionsNearestNeighborBlock().run(
        query_predictions=query, target_predictions=target, **arguments
    )


def names(prediction):
    return [row["name"] for row in prediction.bboxes_metadata]


def distances(prediction):
    return [row["nearest_target_distance"] for row in prediction.bboxes_metadata]


def test_three_four_five_triangle_picks_the_closer_target():
    query = boxes_centred_at([(10, 10)])
    target = boxes_centred_at([(16, 18), (13, 14)])  # distances 10 and 5

    result = run(query, target)

    assert distances(result["query_predictions"]) == [5.0]
    assert names(result["matched_query_detections"]) == ["row-0"]
    assert names(result["matched_target_detections"]) == ["row-1"]


def test_targets_within_one_pixel_of_the_nearest_tie_and_repeat_the_query():
    query = boxes_centred_at([(0, 0), (100, 0)])  # row 1: only (5.8, 0) at 94.2
    # Distances from (0, 0): 5.0, 5.8 (tie, within 1 px) and 6.1 (no tie).
    target = boxes_centred_at([(3, 4), (5.8, 0), (0, 6.1)])

    result = run(query, target)

    assert names(result["matched_query_detections"]) == ["row-0", "row-0", "row-1"]
    assert names(result["matched_target_detections"]) == ["row-0", "row-1", "row-1"]
    assert distances(result["query_predictions"])[0] == 5.0


def test_same_prediction_on_both_sides_excludes_each_row_itself():
    people = boxes_centred_at([(0, 0), (3, 4), (20, 0)])

    result = run(people, people)

    assert names(result["matched_target_detections"]) == ["row-1", "row-0", "row-1"]
    assert distances(result["query_predictions"]) == pytest.approx(
        [5.0, 5.0, 17.4642], abs=1e-4
    )


def test_rows_without_a_shared_row_identity_are_never_the_same_row():
    query = boxes_centred_at([(0, 0)])
    target = boxes_centred_at([(0, 0)])  # same position, separate result

    result = run(query, target)

    assert distances(result["query_predictions"]) == [0.0]
    assert row_record(query).namespace != row_record(target).namespace


@pytest.mark.parametrize("limit, expected", [(5, [5.0]), (4, [None])])
def test_max_distance_is_inclusive(limit, expected):
    result = run(
        boxes_centred_at([(0, 0)]), boxes_centred_at([(3, 4)]), max_distance=limit
    )

    assert distances(result["query_predictions"]) == expected
    assert len(result["matched_query_detections"].xyxy) == len(
        [d for d in expected if d]
    )


def test_empty_targets_give_no_pairs_and_no_distances():
    query = boxes_centred_at([(0, 0), (1, 1)])

    result = run(query, boxes_centred_at([]))

    assert distances(result["query_predictions"]) == [None, None]
    assert len(result["matched_query_detections"].xyxy) == 0
    assert len(result["matched_target_detections"].xyxy) == 0


def test_input_rows_are_not_modified():
    query = boxes_centred_at([(0, 0)])

    run(query, boxes_centred_at([(3, 4)]))

    assert query.bboxes_metadata == [{"name": "row-0"}]


def test_corner_anchors_follow_the_box_corners():
    query = boxes_centred_at([(10, 10)])  # TOP_LEFT (9, 9)
    target = boxes_centred_at([(15, 20)])  # BOTTOM_RIGHT (16, 21)

    result = run(query, target, query_point="TOP_LEFT", target_point="BOTTOM_RIGHT")

    assert distances(result["query_predictions"]) == [pytest.approx(13.8924, abs=1e-4)]


def test_keypoint_anchor_uses_the_named_point_and_skips_invisible_ones():
    boxes = boxes_centred_at([(50, 50), (60, 60)])
    skeletons = KeyPoints(
        xy=torch.tensor([[[0.0, 0.0], [3.0, 4.0]], [[90.0, 90.0], [30.0, 40.0]]]),
        class_id=torch.tensor([0, 0]),
        confidence=torch.tensor([[0.9, 0.9], [0.9, 0.0]]),  # row 1's nose unseen
        image_metadata={"key_points_classes": [["eye", "nose"]]},
    )
    target = boxes_centred_at([(3, 8)])

    result = run(
        (skeletons, boxes), target, query_point="KEYPOINT", query_keypoint_name="nose"
    )

    assert distances(result["query_predictions"][1]) == [4.0, None]
    matched_skeletons, matched_boxes = result["matched_query_detections"]
    assert names(matched_boxes) == ["row-0"]
    assert matched_skeletons.xy.tolist() == [[[0.0, 0.0], [3.0, 4.0]]]


def test_masks_follow_the_matched_rows():
    query = boxes_centred_at([(0, 0)])
    target = boxes_centred_at([(20, 20), (3, 4)], masks=True)

    matched = run(query, target)["matched_target_detections"]

    assert torch.equal(matched.mask, target.mask[[1]])
    assert names(matched) == ["row-1"]


def test_too_many_pairs_with_masks_is_a_contract_error():
    # 11 x 10 rows on one spot: 110 tied pairs, above the masked limit of 100.
    query = boxes_centred_at([(5, 5)] * 11)
    target = boxes_centred_at([(5, 5)] * 10, masks=True)

    with pytest.raises(
        ContractError, match="110 query-target pairs exceed the limit of 100"
    ):
        run(query, target)


def test_keypoints_without_boxes_are_a_contract_error():
    skeletons = KeyPoints(
        xy=torch.zeros((1, 1, 2)),
        class_id=torch.tensor([0]),
        confidence=torch.ones((1, 1)),
        image_metadata={"key_points_classes": [["nose"]]},
    )

    with pytest.raises(ContractError, match="no bounding boxes"):
        run((skeletons, None), boxes_centred_at([(0, 0)]))


def test_keypoint_anchor_without_a_name_is_rejected():
    with pytest.raises(Exception, match="query_keypoint_name is required"):
        DetectionsNearestNeighborBlock.Params.model_validate(
            {
                "query_predictions": "$steps.a.predictions",
                "target_predictions": "$steps.b.predictions",
                "query_point": "KEYPOINT",
            }
        )


def test_negative_max_distance_is_rejected():
    with pytest.raises(Exception, match="max_distance"):
        DetectionsNearestNeighborBlock.Params.model_validate(
            {
                "query_predictions": "$steps.a.predictions",
                "target_predictions": "$steps.b.predictions",
                "max_distance": -1,
            }
        )
