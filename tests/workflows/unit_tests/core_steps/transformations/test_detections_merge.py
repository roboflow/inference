import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.transformations.detections_merge.v1 import (
    DetectionsMergeBlockV1,
    DetectionsMergeManifest,
    calculate_union_bbox,
)


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
