"""Behavioural contract of the tensor-native row-selection helpers
(`take_prediction_by_indices` / `take_prediction_by_mask`): torch masks select
tensor fields on-device with no host round trip, and every entry style
(index list, python mask, numpy mask, torch mask) yields identical results."""

import numpy as np
import pytest
import torch

from inference.core.workflows.core_steps.common.tensor_native import (
    take_prediction_by_indices,
    take_prediction_by_mask,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks


def _detections() -> Detections:
    return Detections(
        xyxy=torch.tensor(
            [[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30]],
            dtype=torch.float32,
        ),
        class_id=torch.tensor([0, 1, 2]),
        confidence=torch.tensor([0.9, 0.5, 0.7]),
        image_metadata={"class_names": {0: "a", 1: "b", 2: "c"}},
        bboxes_metadata=[{"detection_id": f"d{i}"} for i in range(3)],
    )


def _instance_detections(rle: bool = False) -> InstanceDetections:
    base = _detections()
    if rle:
        mask = InstancesRLEMasks(
            image_size=(30, 30),
            masks=[{"size": [30, 30], "counts": f"stub-{i}"} for i in range(3)],
        )
    else:
        dense = torch.zeros((3, 30, 30), dtype=torch.bool)
        for i in range(3):
            dense[i, i * 10 : (i + 1) * 10, i * 10 : (i + 1) * 10] = True
        mask = dense
    return InstanceDetections(
        xyxy=base.xyxy,
        class_id=base.class_id,
        confidence=base.confidence,
        mask=mask,
        image_metadata=base.image_metadata,
        bboxes_metadata=base.bboxes_metadata,
    )


def test_torch_mask_selection_matches_every_other_entry_style() -> None:
    # given
    detections = _instance_detections()
    torch_mask = torch.tensor([True, False, True])

    # when
    by_torch_mask = take_prediction_by_mask(detections, torch_mask)
    by_list_mask = take_prediction_by_mask(detections, [True, False, True])
    by_numpy_mask = take_prediction_by_mask(detections, np.array([True, False, True]))
    by_indices = take_prediction_by_indices(detections, [0, 2])

    # then - all four styles produce identical selections
    for result in (by_torch_mask, by_list_mask, by_numpy_mask, by_indices):
        assert torch.equal(result.xyxy, detections.xyxy[[0, 2]])
        assert torch.equal(result.class_id, torch.tensor([0, 2]))
        assert torch.equal(result.mask, detections.mask[[0, 2]])
        assert [m["detection_id"] for m in result.bboxes_metadata] == ["d0", "d2"]


def test_torch_mask_selection_copies_metadata_dicts() -> None:
    # given
    detections = _detections()

    # when
    result = take_prediction_by_mask(detections, torch.tensor([True, True, False]))
    result.bboxes_metadata[0]["tracker_id"] = 7

    # then - the source metadata must not observe downstream mutation
    assert "tracker_id" not in detections.bboxes_metadata[0]


def test_torch_mask_selection_accepts_model_detections_without_tracker_id() -> None:
    """Model output predating the tracking field is filtered before tracking."""
    detections = _detections()
    if hasattr(detections, "tracker_id"):
        delattr(detections, "tracker_id")

    result = take_prediction_by_mask(detections, torch.tensor([True, False, True]))

    assert torch.equal(result.class_id, torch.tensor([0, 2]))
    assert getattr(result, "tracker_id", None) is None


def test_torch_mask_identity_aliases_tensors() -> None:
    # given
    detections = _instance_detections()

    # when - all-True mask is the identity fast path
    result = take_prediction_by_mask(detections, torch.tensor([True, True, True]))

    # then - tensor fields are aliased, not copied
    assert result.xyxy is detections.xyxy
    assert result.mask is detections.mask


def test_torch_mask_selection_handles_rle_masks() -> None:
    # given
    detections = _instance_detections(rle=True)

    # when
    result = take_prediction_by_mask(detections, torch.tensor([False, True, True]))

    # then - the RLE list follows the surviving rows
    assert [m["counts"] for m in result.mask.masks] == ["stub-1", "stub-2"]


def test_torch_mask_selection_on_keypoint_tuple_preserves_auxiliary_tensors() -> None:
    # given - RF-DETR-shaped keypoints with covariance / detection_confidence
    key_points = KeyPoints(
        xy=torch.arange(3 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2),
        class_id=torch.tensor([0, 1, 2]),
        confidence=torch.full((3, 2), 0.5),
        image_metadata={"class_names": {0: "p"}},
        key_points_metadata=[{"detection_id": f"k{i}"} for i in range(3)],
        covariance=torch.arange(3 * 2 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2, 2),
        detection_confidence=torch.tensor([0.1, 0.2, 0.3]),
    )
    detections = _detections()

    # when
    kp, det = take_prediction_by_mask(
        (key_points, detections), torch.tensor([True, False, True])
    )

    # then - both tuple components sliced consistently, aux tensors in lockstep
    assert torch.equal(kp.xy, key_points.xy[[0, 2]])
    assert torch.equal(kp.covariance, key_points.covariance[[0, 2]])
    assert torch.equal(kp.detection_confidence, torch.tensor([0.1, 0.3]))
    assert torch.equal(det.xyxy, detections.xyxy[[0, 2]])
    assert [m["detection_id"] for m in kp.key_points_metadata] == ["k0", "k2"]


def test_mismatched_length_torch_mask_keeps_legacy_index_semantics() -> None:
    # given - a shorter mask: its nonzero positions are treated as indices
    detections = _detections()

    # when
    result = take_prediction_by_mask(detections, torch.tensor([False, True]))

    # then
    assert torch.equal(result.xyxy, detections.xyxy[[1]])
    assert [m["detection_id"] for m in result.bboxes_metadata] == ["d1"]


def test_index_selection_can_reorder_rows() -> None:
    # given
    detections = _detections()

    # when - masks cannot reorder, index lists can
    result = take_prediction_by_indices(detections, [2, 0])

    # then
    assert torch.equal(result.class_id, torch.tensor([2, 0]))
    assert [m["detection_id"] for m in result.bboxes_metadata] == ["d2", "d0"]


def test_tensor_index_selection_can_reorder_rows() -> None:
    detections = _instance_detections(rle=True)

    result = take_prediction_by_indices(detections, torch.tensor([2, 0]))

    assert torch.equal(result.class_id, torch.tensor([2, 0]))
    assert [m["detection_id"] for m in result.bboxes_metadata] == ["d2", "d0"]
    assert [m["counts"] for m in result.mask.masks] == ["stub-2", "stub-0"]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="device-preservation check needs MPS"
)
def test_torch_mask_selection_stays_on_device() -> None:
    # given - prediction and mask both device-resident
    detections = _detections()
    device = torch.device("mps")
    on_device = Detections(
        xyxy=detections.xyxy.to(device),
        class_id=detections.class_id.to(device),
        confidence=detections.confidence.to(device),
        image_metadata=detections.image_metadata,
        bboxes_metadata=detections.bboxes_metadata,
    )

    # when
    result = take_prediction_by_mask(
        on_device, torch.tensor([True, False, True], device=device)
    )
    reordered = take_prediction_by_indices(
        on_device, torch.tensor([2, 0], device=device)
    )

    # then - results stay on the device
    assert result.xyxy.device.type == "mps"
    assert result.class_id.device.type == "mps"
    assert [m["detection_id"] for m in result.bboxes_metadata] == ["d0", "d2"]
    assert reordered.xyxy.device.type == "mps"
    assert reordered.class_id.cpu().tolist() == [2, 0]
    assert [m["detection_id"] for m in reordered.bboxes_metadata] == ["d2", "d0"]
