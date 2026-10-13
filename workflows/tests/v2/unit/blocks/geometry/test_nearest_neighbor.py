"""Unit tests of ``v2/detections_nearest_neighbor@v1`` with hand-computed distances."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone

import pytest
import torch
from roboflow_workflows.v2.blocks.geometry.nearest_neighbor import (
    DetectionsNearestNeighborBlock,
)
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import ImageData
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


def frame_image():
    return ImageData.from_tensor(
        torch.zeros((3, 100, 200), dtype=torch.uint8), image_id="root"
    )


@pytest.mark.parametrize("other_frame", ["crop", "resize", "camera"])
def test_known_different_local_frames_are_rejected(other_frame):
    root = frame_image()
    target_image = {
        "crop": root.crop((100, 0, 200, 100)),
        "resize": root.resize((50, 100)),
        "camera": replace(root, image_id="other-camera"),
    }[other_frame]
    query = boxes_centred_at([(110, 10)])
    target = boxes_centred_at([(10, 10) if other_frame == "crop" else (110, 10)])
    query.image_metadata = root.prediction_metadata()
    target.image_metadata = target_image.prediction_metadata()
    query_before = deepcopy(query.image_metadata)
    target_before = deepcopy(target.image_metadata)

    with pytest.raises(ContractError, match="same local pixel frame"):
        run(query, target, max_distance=5)

    assert query.image_metadata == query_before
    assert target.image_metadata == target_before
    assert query.bboxes_metadata == [{"name": "row-0"}]


@pytest.mark.parametrize("mapping_name", ["parent", "root"])
@pytest.mark.parametrize(
    "field, value",
    [
        ("scale_xy", (2.0, 1.0)),
        ("offset_xy", (1.0, 0.0)),
        ("frame_id", "other"),
        ("frame_size_hw", (100, 201)),
    ],
)
def test_mapping_conflicts_are_detected_even_with_the_same_image_id(
    mapping_name, field, value
):
    frame = FrameRef.from_image(frame_image())
    mapping = replace(getattr(frame, mapping_name), **{field: value})
    changed = replace(frame, **{mapping_name: mapping})
    query, target = boxes_centred_at([(0, 0)]), boxes_centred_at([(0, 0)])
    query.image_metadata = frame.prediction_metadata()
    target.image_metadata = changed.prediction_metadata()

    with pytest.raises(ContractError, match=rf"conflicting {mapping_name}\."):
        run(query, target)


@pytest.mark.parametrize(
    "field, value", [("image_id", "other"), ("size_hw", (100, 201))]
)
def test_local_identity_and_size_conflicts_are_detected(field, value):
    frame = FrameRef.from_image(frame_image())
    query, target = boxes_centred_at([(0, 0)]), boxes_centred_at([(0, 0)])
    query.image_metadata = frame.prediction_metadata()
    target.image_metadata = replace(frame, **{field: value}).prediction_metadata()

    with pytest.raises(ContractError, match=f"conflicting {field}"):
        run(query, target)


def test_known_source_conflict_is_detected_with_the_same_geometry_and_image_id():
    video = VideoMetadata(
        video_identifier="camera-a",
        frame_number=3,
        frame_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    frame = replace(FrameRef.from_image(frame_image()), video_metadata=video)
    other = replace(
        frame, video_metadata=video.model_copy(update={"video_identifier": "camera-b"})
    )
    query, target = boxes_centred_at([(0, 0)]), boxes_centred_at([(0, 0)])
    query.image_metadata = frame.prediction_metadata()
    target.image_metadata = other.prediction_metadata()

    with pytest.raises(ContractError, match="conflicting video_identifier"):
        run(query, target)


@pytest.mark.parametrize(
    "limit, expected_distance, expected_rows",
    [(4, None, []), (5, 5.0, [0]), (6, 5.0, [0, 1]), (None, 5.0, [0, 1])],
)
def test_equivalent_separate_frame_metadata_keeps_distances_ties_masks_and_identity(
    limit, expected_distance, expected_rows
):
    query = boxes_centred_at([(0, 0)], masks=True)
    target = boxes_centred_at([(3, 4), (6, 0), (6.1, 0)], masks=True)
    image = ImageData.from_tensor(
        torch.zeros((3, 30, 30), dtype=torch.uint8), image_id="same-frame"
    )
    frame = FrameRef.from_image(image)
    query.image_metadata = frame.prediction_metadata()
    target.image_metadata = frame.prediction_metadata()
    assert query.image_metadata == target.image_metadata
    assert query.image_metadata is not target.image_metadata
    query, target = with_row_identity(query), with_row_identity(target)
    metadata_before = deepcopy(query.image_metadata)

    result = run(query, target, max_distance=limit)

    assert distances(result["query_predictions"]) == [expected_distance]
    assert result["query_predictions"] is not query
    assert torch.equal(result["query_predictions"].xyxy, query.xyxy)
    assert query.image_metadata == metadata_before
    assert query.bboxes_metadata == [{"name": "row-0"}]
    for key, original, indices in [
        ("matched_query_detections", query, [0] * len(expected_rows)),
        ("matched_target_detections", target, expected_rows),
    ]:
        matched = result[key]
        assert type(matched) is type(original)
        assert matched.xyxy.tolist() == original.xyxy[indices].tolist()
        assert torch.equal(matched.mask, original.mask[indices])
        assert row_record(matched).namespace == row_record(original).namespace
        assert row_record(matched).host_ids() == indices


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"parent_id": None},
        {"image_dimensions": [50, 100]},
        {"root_parent_id": "root"},
        {"root_parent_coordinates": [0, 0]},
        {"scaling_relative_to_root_parent": None},
        {"scaling_relative_to_root_parent": [0.5, 0.5]},
    ],
)
def test_missing_and_partial_metadata_remains_unknown(metadata):
    query, target = boxes_centred_at([(0, 0)]), boxes_centred_at([(3, 4)])
    # Local pixels count twice in the root. An absent legacy scale is unknown,
    # not the identity scale assumed by the complete-frame conversion reader.
    query.image_metadata = frame_image().resize((50, 100)).prediction_metadata()
    target.image_metadata = metadata

    result = run(query, target, max_distance=5)

    assert distances(result["query_predictions"]) == [5.0]


def test_conflicting_partial_metadata_is_rejected():
    query, target = boxes_centred_at([(0, 0)]), boxes_centred_at([(0, 0)])
    query.image_metadata = {"root_parent_coordinates": [0, 0]}
    target.image_metadata = {"root_parent_coordinates": [1, 0]}

    with pytest.raises(ContractError, match="conflicting root.offset_xy"):
        run(query, target)


def test_typed_record_and_legacy_mirror_must_agree():
    query, target = boxes_centred_at([(0, 0)]), boxes_centred_at([(0, 0)])
    query.image_metadata = frame_image().prediction_metadata()
    query.image_metadata["root_parent_coordinates"] = [1, 0]

    with pytest.raises(ContractError, match="legacy root.offset_xy disagrees"):
        run(query, target)


@pytest.mark.parametrize("side", ["query", "target"])
@pytest.mark.parametrize("point", ["CENTER", "KEYPOINT"])
def test_tuple_keypoint_and_box_frames_must_agree_for_either_anchor(side, point):
    boxes = boxes_centred_at([(0, 0)])
    boxes.image_metadata = frame_image().prediction_metadata()
    skeletons = KeyPoints(
        xy=torch.zeros((1, 1, 2)),
        class_id=torch.tensor([0]),
        confidence=torch.ones((1, 1)),
        image_metadata={
            **frame_image().crop((100, 0, 200, 100)).prediction_metadata(),
            "key_points_classes": [["nose"]],
        },
    )
    other = boxes_centred_at([(0, 0)])
    query, target = (
        ((skeletons, boxes), other) if side == "query" else (other, (skeletons, boxes))
    )

    with pytest.raises(ContractError, match="same local pixel frame"):
        run(query, target, **{f"{side}_point": point, f"{side}_keypoint_name": "nose"})


def test_keypoint_anchor_frame_is_checked_when_boxes_have_no_frame():
    boxes = boxes_centred_at([(0, 0)])
    skeletons = KeyPoints(
        xy=torch.zeros((1, 1, 2)),
        class_id=torch.tensor([0]),
        confidence=torch.ones((1, 1)),
        image_metadata={
            **frame_image().crop((100, 0, 200, 100)).prediction_metadata(),
            "key_points_classes": [["nose"]],
        },
    )
    target = boxes_centred_at([(0, 0)])
    target.image_metadata = frame_image().prediction_metadata()

    with pytest.raises(ContractError, match="same local pixel frame"):
        run(
            (skeletons, boxes),
            target,
            query_point="KEYPOINT",
            query_keypoint_name="nose",
        )


def test_equivalent_keypoint_and_box_frames_keep_keypoints_and_rows():
    query = boxes_centred_at([(50, 50)])
    target = boxes_centred_at([(3, 8)])
    metadata = frame_image().prediction_metadata()
    query.image_metadata = deepcopy(metadata)
    target.image_metadata = deepcopy(metadata)
    query, target = with_row_identity(query), with_row_identity(target)
    skeletons = KeyPoints(
        xy=torch.tensor([[[3.0, 4.0]]]),
        class_id=torch.tensor([0]),
        confidence=torch.ones((1, 1)),
        image_metadata={**deepcopy(metadata), "key_points_classes": [["nose"]]},
    )

    result = run(
        (skeletons, query), target, query_point="KEYPOINT", query_keypoint_name="nose"
    )

    assert result["query_predictions"][0] is skeletons
    assert distances(result["query_predictions"][1]) == [4.0]
    matched_keypoints, matched_boxes = result["matched_query_detections"]
    assert torch.equal(matched_keypoints.xy, skeletons.xy)
    assert row_record(matched_boxes).namespace == row_record(query).namespace
    assert row_record(matched_boxes).host_ids() == [0]


@pytest.mark.parametrize("empty_side", ["query", "target"])
def test_empty_inputs_keep_no_pairs_even_with_different_frame_metadata(empty_side):
    query = boxes_centred_at([] if empty_side == "query" else [(110, 10)])
    target = boxes_centred_at([] if empty_side == "target" else [(10, 10)])
    query.image_metadata = frame_image().prediction_metadata()
    target.image_metadata = frame_image().crop((100, 0, 200, 100)).prediction_metadata()

    result = run(query, target)

    assert len(result["matched_query_detections"].xyxy) == 0
    assert len(result["matched_target_detections"].xyxy) == 0
    assert distances(result["query_predictions"]) == (
        [] if empty_side == "query" else [None]
    )


@pytest.mark.parametrize("unknown_field", ["image_id", "video_metadata"])
def test_unknown_typed_identity_or_source_does_not_prove_a_conflict(unknown_field):
    video = VideoMetadata(
        video_identifier="camera-a",
        frame_number=3,
        frame_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    frame = replace(FrameRef.from_image(frame_image()), video_metadata=video)
    query, target = boxes_centred_at([(0, 0)]), boxes_centred_at([(3, 4)])
    query.image_metadata = frame.prediction_metadata()
    target.image_metadata = replace(
        frame, **{unknown_field: None}
    ).prediction_metadata()

    assert distances(run(query, target)["query_predictions"]) == [5.0]


def test_disjoint_partial_frame_fields_do_not_prove_a_conflict():
    query, target = boxes_centred_at([(0, 0)]), boxes_centred_at([(3, 4)])
    query.image_metadata = {"parent_id": "known-image"}
    target.image_metadata = {"root_parent_coordinates": [7, 9]}

    assert distances(run(query, target)["query_predictions"]) == [5.0]


def test_same_known_frame_and_row_identity_still_excludes_self_pairs():
    people = boxes_centred_at([(0, 0), (3, 4)])
    people.image_metadata.update(frame_image().prediction_metadata())

    result = run(people, people)

    assert distances(result["query_predictions"]) == [5.0, 5.0]
    assert row_record(result["matched_query_detections"]).host_ids() == [0, 1]
    assert row_record(result["matched_target_detections"]).host_ids() == [1, 0]
