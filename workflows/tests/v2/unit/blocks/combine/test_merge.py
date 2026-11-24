"""``roboflow_core/detections_merge@v1`` against the V1 numpy block and analytic values."""

import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.transformations.detections_merge.v1 import (
    DetectionsMergeBlockV1,
)
from roboflow_workflows.v2.blocks.combine import merge_rows
from roboflow_workflows.v2.blocks.query.detection_ops import derived_detection_id
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.data.row_identity import row_record

from inference_models.models.base.object_detection import Detections

from .scene import VEHICLES, as_sv, boxes, dense, key_points, rle, view


def test_merged_box_matches_v1_numpy_block() -> None:
    # given
    vehicles = boxes(VEHICLES, namespace="vehicles")
    expected = DetectionsMergeBlockV1().run(
        predictions=as_sv(vehicles), class_name="traffic"
    )["predictions"]

    # when
    merged = merge_rows(vehicles, class_name="traffic")

    # then
    np.testing.assert_array_equal(merged.xyxy.numpy(), expected.xyxy)
    np.testing.assert_array_equal(merged.confidence.numpy(), expected.confidence)
    assert merged.class_id.tolist() == expected.class_id.tolist() == [0]
    assert merged.image_metadata["class_names"] == {0: "traffic"}
    assert expected.data["class_name"].tolist() == ["traffic"]
    assert merged.xyxy.tolist() == [[12.0, 2.0, 58.0, 38.0]]
    assert merged.confidence.tolist() == pytest.approx([0.4])


def test_merged_row_gets_a_fresh_identity_and_keeps_the_frame() -> None:
    # given
    vehicles = boxes(VEHICLES, namespace="vehicles")

    # when
    first = merge_rows(vehicles, class_name="traffic")
    second = merge_rows(vehicles, class_name="traffic")

    # then
    record = row_record(first)
    assert (record.rows, record.ids) == (1, None)
    assert record.namespace not in ("vehicles", row_record(second).namespace)
    assert derived_detection_id(record.namespace, 0) != derived_detection_id(
        row_record(second).namespace, 0
    )
    assert frame_of(first) == frame_of(vehicles)
    assert first.image_metadata["prediction_type"] == "object-detection"
    assert first.bboxes_metadata is None
    assert row_record(vehicles).namespace == "vehicles"
    OBJECT_DETECTION_PREDICTION_KIND.check(first)


@pytest.mark.parametrize("carrier", [dense, rle, view], ids=["dense", "rle", "view"])
def test_instance_inputs_merge_their_boxes_without_reading_masks(carrier) -> None:
    # given
    predictions = carrier(VEHICLES, namespace="vehicles")

    # when
    merged = merge_rows(predictions, class_name="merged_detection")

    # then
    assert isinstance(merged, Detections)
    assert merged.xyxy.tolist() == [[12.0, 2.0, 58.0, 38.0]]
    if carrier is view:
        assert predictions.materialization_count == 0


def test_key_points_merge_their_boxes() -> None:
    # when
    merged = merge_rows(key_points(), class_name="group")

    # then
    assert merged.xyxy.tolist() == [[2.0, 4.0, 38.0, 30.0]]
    assert merged.confidence.tolist() == pytest.approx([0.6])


def test_key_points_without_boxes_are_rejected() -> None:
    # when / then
    with pytest.raises(ValueError, match="no bounding-box Detections"):
        merge_rows(key_points(with_boxes=False), class_name="group")


def test_empty_input_gives_no_rows_on_the_same_frame() -> None:
    # given
    empty = boxes(VEHICLES, namespace="vehicles", rows=[])

    # when
    merged = merge_rows(empty, class_name="traffic")

    # then
    assert merged.xyxy.shape == (0, 4)
    assert merged.class_id.shape == merged.confidence.shape == (0,)
    assert row_record(merged).rows == 0
    assert frame_of(merged) == frame_of(empty)
    v1 = DetectionsMergeBlockV1().run(predictions=as_sv(empty), class_name="traffic")
    assert len(v1["predictions"]) == 0


def test_float_dtype_and_device_follow_the_input() -> None:
    # given
    vehicles = boxes(VEHICLES, namespace=None)
    vehicles = Detections(
        xyxy=vehicles.xyxy.double(),
        class_id=vehicles.class_id,
        confidence=vehicles.confidence.half(),
        image_metadata=vehicles.image_metadata,
    )

    # when
    merged = merge_rows(vehicles, class_name="traffic")

    # then
    assert merged.xyxy.dtype == torch.float64
    assert merged.confidence.dtype == torch.float16
    assert merged.xyxy.device == vehicles.xyxy.device


def test_class_name_must_be_a_string() -> None:
    # when / then
    with pytest.raises(ValueError, match="class_name must be a string"):
        merge_rows(boxes(), class_name=7)
