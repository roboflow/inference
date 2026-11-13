import numpy as np
import pytest
import supervision as sv
import torch
from roboflow_workflows.core_steps.transformations.detections_merge.v1 import (
    DetectionsMergeBlockV1,
    DetectionsMergeManifest,
    calculate_union_bbox,
)
from roboflow_workflows.core_steps.transformations.detections_merge.v1_tensor import (
    DetectionsMergeBlockV1 as NativeDetectionsMergeBlockV1,
)

from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections as NativeDetections


def test_calculate_union_bbox():
    # given
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 20, 20], [15, 15, 25, 25]]),
    )

    # when
    union_bbox = calculate_union_bbox(detections)

    # then
    expected_bbox = np.array([[10, 10, 25, 25]])
    assert np.allclose(
        union_bbox, expected_bbox
    ), f"Expected bounding box to be {expected_bbox}, but got {union_bbox}"


@pytest.mark.parametrize("type_alias", ["roboflow_core/detections_merge@v1"])
def test_detections_merge_validation_when_valid_manifest_is_given(
    type_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "detections_merge",
        "predictions": "$steps.od_model.predictions",
        "class_name": "custom_merged",
    }

    # when
    result = DetectionsMergeManifest.model_validate(data)

    # then
    assert result == DetectionsMergeManifest(
        type=type_alias,
        name="detections_merge",
        predictions="$steps.od_model.predictions",
        class_name="custom_merged",
    )


def test_detections_merge_block() -> None:
    # given
    block = DetectionsMergeBlockV1()
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 20, 20], [15, 15, 25, 25]]),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([1, 1]),
        data={
            "class_name": np.array(["person", "person"]),
        },
    )

    # when
    output = block.run(predictions=detections)

    # then
    assert isinstance(output, dict)
    assert "predictions" in output
    assert len(output["predictions"]) == 1
    assert np.allclose(output["predictions"].xyxy, np.array([[10, 10, 25, 25]]))
    assert np.allclose(output["predictions"].confidence, np.array([0.8]))
    assert np.allclose(output["predictions"].class_id, np.array([0]))
    assert output["predictions"].data["class_name"][0] == "merged_detection"
    assert isinstance(output["predictions"].data["detection_id"][0], str)


def test_detections_merge_block_with_custom_class() -> None:
    # given
    block = DetectionsMergeBlockV1()
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 20, 20], [15, 15, 25, 25]]),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([1, 1]),
        data={
            "class_name": np.array(["person", "person"]),
        },
    )

    # when
    output = block.run(predictions=detections, class_name="custom_merged")

    # then
    assert isinstance(output, dict)
    assert "predictions" in output
    assert len(output["predictions"]) == 1
    assert np.allclose(output["predictions"].xyxy, np.array([[10, 10, 25, 25]]))
    assert np.allclose(output["predictions"].confidence, np.array([0.8]))
    assert np.allclose(output["predictions"].class_id, np.array([0]))
    assert output["predictions"].data["class_name"][0] == "custom_merged"
    assert isinstance(output["predictions"].data["detection_id"][0], str)


def test_detections_merge_block_empty_input() -> None:
    # given
    block = DetectionsMergeBlockV1()
    empty_detections = sv.Detections(xyxy=np.array([], dtype=np.float32).reshape(0, 4))

    # when
    output = block.run(predictions=empty_detections)

    # then
    assert isinstance(output, dict)
    assert "predictions" in output
    assert len(output["predictions"]) == 0
    assert isinstance(output["predictions"].xyxy, np.ndarray)
    assert output["predictions"].xyxy.shape == (0, 4)


@pytest.mark.parametrize("shared_frame", [True, False])
def test_detections_merge_carries_only_shared_frame_data(shared_frame: bool) -> None:
    first_scale = 0.5
    second_scale = first_scale if shared_frame else 0.25
    predictions = sv.Detections(
        xyxy=np.array([[10, 10, 20, 20], [15, 15, 25, 25]], dtype=np.float32),
        confidence=np.array([0.8, 0.9]),
        data={
            "parent_id": np.array(["crop", "crop"]),
            "scaling_relative_to_parent": np.array([0.75, 0.75]),
            "scaling_relative_to_root_parent": np.array([first_scale, second_scale]),
            "detection_id": np.array(["first", "second"]),
            "custom": np.array([1, 2]),
        },
    )

    merged = DetectionsMergeBlockV1().run(predictions=predictions)["predictions"]

    assert merged.xyxy.tolist() == [[10, 10, 25, 25]]
    assert merged.data["detection_id"][0] not in {"first", "second"}
    assert "custom" not in merged.data
    if shared_frame:
        assert merged.data["parent_id"].tolist() == ["crop"]
        assert merged.data["scaling_relative_to_parent"].tolist() == [0.75]
        assert merged.data["scaling_relative_to_root_parent"].tolist() == [0.5]
        assert not np.shares_memory(
            merged.data["scaling_relative_to_root_parent"],
            predictions.data["scaling_relative_to_root_parent"],
        )
    else:
        assert "parent_id" not in merged.data
        assert "scaling_relative_to_parent" not in merged.data
        assert "scaling_relative_to_root_parent" not in merged.data


@pytest.mark.parametrize("empty", [False, True])
def test_detections_merge_leaves_unknown_image_dimensions_absent(empty: bool) -> None:
    predictions = sv.Detections(
        xyxy=np.empty((0, 4)) if empty else np.array([[1, 2, 3, 4]]),
        data={} if empty else {"image_dimensions": np.array([[None, None]])},
        metadata={"image_dimensions": [None, None]} if empty else {},
    )

    merged = DetectionsMergeBlockV1().run(predictions=predictions)["predictions"]

    assert "image_dimensions" not in merged.data
    assert "image_dimensions" not in merged.metadata


def test_native_detections_merge_keeps_keypoint_frame_without_boxes() -> None:
    keypoints = KeyPoints(
        xy=torch.empty((0, 0, 2)),
        class_id=torch.empty((0,), dtype=torch.long),
        confidence=torch.empty((0, 0)),
        image_metadata={
            "image_dimensions": [60, 80],
            "root_parent_dimensions": [120, 160],
            "root_parent_coordinates": [40, 30],
        },
    )

    merged = NativeDetectionsMergeBlockV1().run(predictions=(keypoints, None))[
        "predictions"
    ]

    assert len(merged) == 0
    assert merged.image_metadata["image_dimensions"] == [60, 80]
    assert merged.image_metadata["root_parent_dimensions"] == [120, 160]
    assert merged.image_metadata["root_parent_coordinates"] == [40, 30]


def test_detections_merge_does_not_guess_partially_populated_origin() -> None:
    predictions = sv.Detections(
        xyxy=np.array([[10, 10, 20, 20], [15, 15, 25, 25]]),
        data={
            "parent_id": ["crop", "crop"],
            "root_parent_coordinates": [None, [40, 30]],
        },
    )

    merged = DetectionsMergeBlockV1().run(predictions=predictions)["predictions"]

    assert merged.xyxy.tolist() == [[10, 10, 25, 25]]
    assert "root_parent_coordinates" not in merged.data
    assert "parent_id" not in merged.data


def test_native_detections_merge_leaves_unknown_empty_dimensions_absent() -> None:
    predictions = NativeDetections(
        xyxy=torch.empty((0, 4)),
        confidence=torch.empty((0,)),
        class_id=torch.empty((0,), dtype=torch.long),
        image_metadata={"image_dimensions": [None, None]},
    )

    merged = NativeDetectionsMergeBlockV1().run(predictions=predictions)["predictions"]

    assert len(merged) == 0
    assert merged.image_metadata.get("image_dimensions") is None
