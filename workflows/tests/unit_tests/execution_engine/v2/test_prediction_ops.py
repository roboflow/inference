"""Strict V2 selection composes with the shared native gather implementation."""

import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.v2.blocks.prediction_ops import (
    select_predictions,
)
from roboflow_workflows.execution_engine.v2.errors import ContractError

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks


def _detections():
    return Detections(
        xyxy=torch.arange(12, dtype=torch.float32).reshape(3, 4),
        class_id=torch.tensor([0, 1, 2]),
        confidence=torch.tensor([0.9, 0.3, 0.8]),
        image_metadata={"parent_id": "crop", "root_parent_id": "frame"},
        bboxes_metadata=[{"detection_id": f"box:{i}"} for i in range(3)],
    )


@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([True, False, True]),
        np.array([True, False, True]),
        [True, False, True],
    ],
)
def test_confidence_selection_preserves_ids_and_does_not_mutate_rows(mask):
    source = _detections()
    selected = select_predictions(source, mask=mask)

    assert selected.class_id.tolist() == [0, 2]
    assert [row["detection_id"] for row in selected.bboxes_metadata] == [
        "box:0",
        "box:2",
    ]
    assert selected.image_metadata is source.image_metadata
    assert selected.xyxy.device == source.xyxy.device
    selected.bboxes_metadata[0]["tracker_id"] = 9
    assert "tracker_id" not in source.bboxes_metadata[0]


@pytest.mark.parametrize("rle", [False, True])
def test_reordering_keeps_dense_or_compressed_masks_aligned(rle):
    boxes = _detections()
    mask = (
        InstancesRLEMasks(image_size=(3, 3), masks=[b"9", b"09", b"135"])
        if rle
        else torch.arange(27).reshape(3, 3, 3) % 2 == 0
    )
    source = InstanceDetections(**vars(boxes), mask=mask)
    selected = select_predictions(source, indices=[2, 0, 2])

    assert selected.class_id.tolist() == [2, 0, 2]
    assert [row["detection_id"] for row in selected.bboxes_metadata] == [
        "box:2",
        "box:0",
        "box:2",
    ]
    if rle:
        assert selected.mask.masks == [b"135", b"9", b"135"]
    else:
        assert torch.equal(selected.mask, mask[[2, 0, 2]])


@pytest.mark.parametrize("with_boxes", [False, True])
def test_keypoint_selection_preserves_slots_covariance_and_detection_confidence(
    with_boxes,
):
    points = KeyPoints(
        xy=torch.arange(18, dtype=torch.float32).reshape(3, 3, 2),
        class_id=torch.tensor([0, 1, 2]),
        confidence=torch.full((3, 3), 0.8),
        covariance=torch.arange(36, dtype=torch.float32).reshape(3, 3, 2, 2),
        detection_confidence=torch.tensor([0.9, 0.3, 0.8]),
        key_points_metadata=[
            {"ids": [0, 4, 9], "detection_id": f"kp:{i}"} for i in range(3)
        ],
    )
    source = (points, _detections() if with_boxes else None)
    selected, boxes = select_predictions(source, indices=torch.tensor([2, 0]))

    assert torch.equal(selected.xy, points.xy[[2, 0]])
    assert torch.equal(selected.covariance, points.covariance[[2, 0]])
    assert torch.equal(
        selected.detection_confidence, points.detection_confidence[[2, 0]]
    )
    assert selected.key_points_metadata[0] == {"ids": [0, 4, 9], "detection_id": "kp:2"}
    assert (boxes is not None) == with_boxes
    if with_boxes:
        assert boxes.class_id.tolist() == [2, 0]


def test_empty_and_identity_selection_preserve_tensor_shape_and_dtype():
    source = _detections()
    identity = select_predictions(source, mask=torch.ones(3, dtype=torch.bool))
    empty = select_predictions(source, mask=torch.zeros(3, dtype=torch.bool))

    assert identity.xyxy is source.xyxy
    assert empty.xyxy.shape == (0, 4)
    assert empty.xyxy.dtype == source.xyxy.dtype
    assert empty.bboxes_metadata == []


@pytest.mark.parametrize(
    "selection, match",
    [
        ({}, "exactly one"),
        ({"mask": [True] * 3, "indices": [0]}, "exactly one"),
        ({"mask": [True]}, "1 entries; expected 3"),
        ({"mask": torch.tensor([1, 0, 1])}, "boolean dtype"),
        ({"mask": torch.ones((3, 1), dtype=torch.bool)}, "one-dimensional"),
        ({"mask": [1, 0, 1]}, "boolean sequence"),
        ({"indices": [-1]}, "outside"),
        ({"indices": [3]}, "outside"),
        ({"indices": [0.0]}, "integer sequence"),
        ({"indices": [True]}, "integer sequence"),
        ({"indices": torch.tensor([0.0])}, "integer dtype"),
        ({"indices": np.array([[0]])}, "one-dimensional"),
    ],
)
def test_invalid_selection_is_rejected_before_gather(selection, match):
    with pytest.raises(ContractError, match=match):
        select_predictions(_detections(), **selection)


def test_unsupported_payload_and_misaligned_rows_fail_usefully():
    with pytest.raises(ContractError, match="supports Detections"):
        select_predictions(torch.zeros(3), indices=[0])

    malformed = _detections()
    malformed.confidence = torch.ones(2)
    with pytest.raises(ContractError, match="confidence"):
        select_predictions(malformed, indices=[0])


def test_only_selected_indices_cross_to_host_for_metadata(monkeypatch):
    source = _detections()
    copied_shapes = []
    original_cpu = torch.Tensor.cpu

    def record_cpu(tensor, *args, **kwargs):
        copied_shapes.append((tuple(tensor.shape), tensor.dtype))
        return original_cpu(tensor, *args, **kwargs)

    def reject_numpy(*args, **kwargs):
        raise AssertionError("Prediction tensors must not be exported to NumPy")

    monkeypatch.setattr(torch.Tensor, "cpu", record_cpu)
    monkeypatch.setattr(torch.Tensor, "numpy", reject_numpy)
    selected = select_predictions(source, mask=torch.tensor([True, False, True]))

    assert copied_shapes == [((2,), torch.int64)]
    assert selected.xyxy.device == source.xyxy.device
