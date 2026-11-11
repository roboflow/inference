from typing import Any

import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError
from roboflow_workflows.core_steps.transformations.detection_offset.v1 import (
    BlockManifest,
    DetectionOffsetBlockV1,
    offset_detections,
)
from roboflow_workflows.execution_engine.entities.base import Batch


@pytest.mark.parametrize(
    "type_alias", ["roboflow_core/detection_offset@v1", "DetectionOffset"]
)
@pytest.mark.parametrize("offset_width_alias", ["offset_width", "offset_x"])
@pytest.mark.parametrize("offset_height_alias", ["offset_height", "offset_y"])
def test_manifest_parsing_when_valid_data_provided(
    type_alias: str,
    offset_width_alias: str,
    offset_height_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "some",
        "predictions": "$steps.some.predictions",
        offset_width_alias: "$inputs.offset_x",
        offset_height_alias: 40,
        "image_metadata": "$steps.some.image",
        "prediction_type": "$steps.some.prediction_type",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type=type_alias,
        name="some",
        predictions="$steps.some.predictions",
        offset_width="$inputs.offset_x",
        offset_height=40,
        image_metadata="$steps.some.image",
        prediction_type="$steps.some.prediction_type",
    )


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("predictions", "invalid"),
        ("offset_width", -1),
        ("offset_height", -1),
    ],
)
def test_manifest_parsing_when_invalid_data_provided(
    field_name: str,
    value: Any,
) -> None:
    # given
    data = {
        "type": "DetectionOffset",
        "name": "some",
        "predictions": "$steps.some.predictions",
        "offset_width": "$inputs.offset_x",
        "offset_height": 40,
        "image_metadata": "$steps.some.image",
    }
    data[field_name] = value

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_offset_detection() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[90, 190, 110, 210]], dtype=np.float64),
        class_id=np.array([1]),
        confidence=np.array([0.5], dtype=np.float64),
        data={
            "detection_id": np.array(["two"]),
            "class_name": np.array(["car"]),
            "parent_id": np.array(["p2"]),
            "image_dimensions": np.array([[640, 640]]),
        },
    )

    # when
    result = offset_detections(
        detections=detections,
        offset_width=50,
        offset_height=100,
    )

    # then
    x1, y1, x2, y2 = result.xyxy[0]
    assert x1 == 65, "Left corner should be moved by 25px to the left"
    assert y1 == 140, "Top corner should be moved by 50px to the top"
    assert x2 == 135, "Right corner should be moved by 25px to the right"
    assert y2 == 260, "Right corner should be moved by 50px to the bottom"
    assert result["parent_id"] == str(
        detections["detection_id"][0]
    ), "Parent id should be set to origin detection id"
    assert result["detection_id"] != str(
        detections["parent_id"][0]
    ), "New detection id (random) must be assigned"


def test_offset_detection_when_larger_than_image() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[90, 190, 110, 210]], dtype=np.float64),
        class_id=np.array([1]),
        confidence=np.array([0.5], dtype=np.float64),
        data={
            "detection_id": np.array(["two"]),
            "class_name": np.array(["car"]),
            "parent_id": np.array(["p2"]),
            "image_dimensions": np.array([[640, 640]]),
        },
    )

    # when
    result = offset_detections(
        detections=detections,
        offset_width=2000,
        offset_height=2000,
    )

    # then
    x1, y1, x2, y2 = result.xyxy[0]
    assert x1 == 0, "Left corner should be at the image border"
    assert y1 == 0, "Top corner should be at the image border"
    assert x2 == 640, "Right corner should be at the image border"
    assert y2 == 640, "Bottom corner should be at the image border"
    assert result["parent_id"] == str(
        detections["detection_id"][0]
    ), "Parent id should be set to origin detection id"
    assert result["detection_id"] != str(
        detections["parent_id"][0]
    ), "New detection id (random) must be assigned"


def test_offset_detection_when_nothing_predicted() -> None:
    # given
    detections = sv.Detections.empty()

    # when
    result = offset_detections(
        detections=detections,
        offset_width=50,
        offset_height=100,
    )

    # then
    assert len(detections) == 0, "Expected empty detections in output"


def test_offset_detection_with_percentage() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[100, 200, 400, 600]], dtype=np.float64),
        class_id=np.array([1]),
        confidence=np.array([0.5], dtype=np.float64),
        data={
            "detection_id": np.array(["three"]),
            "class_name": np.array(["truck"]),
            "parent_id": np.array(["p3"]),
            "image_dimensions": np.array([[640, 640]]),
        },
    )

    # when
    result = offset_detections(
        detections=detections, offset_width=10, offset_height=10, use_percentage=True
    )

    # then
    x1, y1, x2, y2 = result.xyxy[0]
    assert x1 == 85, "Left corner should be moved by 5% of detection width to the left"
    assert y1 == 180, "Top corner should be moved by 5% of detection height to the top"
    assert (
        x2 == 415
    ), "Right corner should be moved by 5% of detection width to the right"
    assert (
        y2 == 620
    ), "Bottom corner should be moved by 5% of detection height to the bottom"


def test_run_offsets_by_percentage_when_percent_units_selected() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[100, 200, 400, 600]], dtype=np.float64),
        class_id=np.array([1]),
        confidence=np.array([0.5], dtype=np.float64),
        data={
            "detection_id": np.array(["three"]),
            "class_name": np.array(["truck"]),
            "parent_id": np.array(["p3"]),
            "image_dimensions": np.array([[640, 640]]),
        },
    )
    block = DetectionOffsetBlockV1()

    # when
    result = block.run(
        predictions=Batch(content=[detections], indices=[(0,)]),
        offset_width=10,
        offset_height=10,
        units="Percent (%)",
    )

    # then
    x1, y1, x2, y2 = result[0]["predictions"].xyxy[0]
    assert x1 == 85, "Left corner should be moved by 5% of detection width to the left"
    assert y1 == 180, "Top corner should be moved by 5% of detection height to the top"
    assert (
        x2 == 415
    ), "Right corner should be moved by 5% of detection width to the right"
    assert (
        y2 == 620
    ), "Bottom corner should be moved by 5% of detection height to the bottom"


def test_run_offsets_by_percentage_when_percent_units_selected_tensor_native() -> None:
    # given - the tensor-native mirror of the numpy percent-mode test above
    torch = pytest.importorskip("torch")
    pytest.importorskip("inference_models")
    from roboflow_workflows.core_steps.transformations.detection_offset.v1_tensor import (
        DetectionOffsetBlockV1 as TensorDetectionOffsetBlockV1,
    )

    from inference_models.models.base.object_detection import (
        Detections as NativeDetections,
    )

    detections = NativeDetections(
        xyxy=torch.tensor([[100.0, 200.0, 400.0, 600.0]], dtype=torch.float64),
        class_id=torch.tensor([1]),
        confidence=torch.tensor([0.5], dtype=torch.float64),
        image_metadata={
            "class_names": {1: "truck"},
            "image_dimensions": [640, 640],
        },
        bboxes_metadata=[
            {"detection_id": "three", "class": "truck", "parent_id": "p3"}
        ],
    )
    block = TensorDetectionOffsetBlockV1()

    # when
    result = block.run(
        predictions=Batch(content=[detections], indices=[(0,)]),
        offset_width=10,
        offset_height=10,
        units="Percent (%)",
    )

    # then
    x1, y1, x2, y2 = result[0]["predictions"].xyxy[0].tolist()
    assert x1 == 85, "Left corner should be moved by 5% of detection width to the left"
    assert y1 == 180, "Top corner should be moved by 5% of detection height to the top"
    assert (
        x2 == 415
    ), "Right corner should be moved by 5% of detection width to the right"
    assert (
        y2 == 620
    ), "Bottom corner should be moved by 5% of detection height to the bottom"


@pytest.mark.parametrize("tensor_native", [False, True], ids=["numpy", "tensor"])
@pytest.mark.parametrize(
    "use_percentage, offset_width, offset_height, top_left, bottom_right",
    [
        (False, 20, 10, [0, 0, 30, 35], [220, 235, 260, 258]),
        (True, 100, 100, [0, 0, 29, 43], [220, 234, 260, 259]),
    ],
    ids=["pixels", "percent"],
)
@pytest.mark.parametrize("image_height, image_width", [(255, 255), (300, 320)])
def test_offset_uint8_boxes_before_clipping(
    tensor_native: bool,
    use_percentage: bool,
    offset_width: int,
    offset_height: int,
    top_left: list[int],
    bottom_right: list[int],
    image_height: int,
    image_width: int,
) -> None:
    boxes = [[2, 3, 20, 30], [230, 240, 250, 253]]
    detection_ids = ["top-left", "bottom-right"]
    expected = [
        top_left,
        [
            bottom_right[0],
            bottom_right[1],
            min(bottom_right[2], image_width),
            min(bottom_right[3], image_height),
        ],
    ]

    if tensor_native:
        torch = pytest.importorskip("torch")
        pytest.importorskip("inference_models")
        from roboflow_workflows.core_steps.transformations.detection_offset.v1_tensor import (
            offset_detections as offset_native_detections,
        )

        from inference_models.models.base.object_detection import (
            Detections as NativeDetections,
        )

        detections = NativeDetections(
            xyxy=torch.tensor(boxes, dtype=torch.uint8),
            class_id=torch.tensor([1, 2]),
            confidence=torch.tensor([0.5, 0.75]),
            image_metadata={"image_dimensions": [image_height, image_width]},
            bboxes_metadata=[
                {"detection_id": detection_id} for detection_id in detection_ids
            ],
        )
        result = offset_native_detections(
            prediction=detections,
            offset_width=offset_width,
            offset_height=offset_height,
            use_percentage=use_percentage,
        )
        assert result.xyxy.device == detections.xyxy.device
        assert result.image_metadata == detections.image_metadata
        assert [row["parent_id"] for row in result.bboxes_metadata] == detection_ids
        assert all(
            row["detection_id"] != original
            for row, original in zip(result.bboxes_metadata, detection_ids)
        )
    else:
        image_dimensions = np.array([[image_height, image_width]] * len(boxes))
        detections = sv.Detections(
            xyxy=np.array(boxes, dtype=np.uint8),
            class_id=np.array([1, 2]),
            confidence=np.array([0.5, 0.75]),
            data={
                "image_dimensions": image_dimensions,
                "detection_id": np.array(detection_ids),
            },
        )
        result = offset_detections(
            detections=detections,
            offset_width=offset_width,
            offset_height=offset_height,
            use_percentage=use_percentage,
        )
        np.testing.assert_array_equal(result["image_dimensions"], image_dimensions)
        assert result["parent_id"].tolist() == detection_ids
        assert all(result["detection_id"] != np.array(detection_ids))

    assert detections.xyxy.tolist() == boxes
    assert result.xyxy.tolist() == expected
    assert result.class_id.tolist() == [1, 2]
    assert result.confidence.tolist() == [0.5, 0.75]
