import numpy as np
import pytest
import supervision as sv
import torch

from inference.core.workflows.core_steps.sinks.roboflow.dataset_upload import v1_tensor
from inference.core.workflows.execution_engine.constants import (
    IMAGE_DIMENSIONS_KEY,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    POLYGON_KEY_IN_SV_DETECTIONS,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import torch_mask_to_coco_rle


def _instance_detections(mask) -> InstanceDetections:
    return InstanceDetections(
        xyxy=torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.9]),
        mask=mask,
    )


def test_scale_dense_instance_masks_stays_tensor_native(monkeypatch) -> None:
    dense_mask = torch.tensor([[[True, False], [False, True]]])
    detections = _instance_detections(mask=dense_mask)

    def fail_on_numpy_materialization(*args, **kwargs):
        pytest.fail("dense mask scaling must not materialize NumPy masks")

    monkeypatch.setattr(
        v1_tensor, "instance_mask_to_numpy", fail_on_numpy_materialization
    )
    monkeypatch.setattr(sv, "mask_to_polygons", fail_on_numpy_materialization)
    monkeypatch.setattr(sv, "polygon_to_mask", fail_on_numpy_materialization)

    result = v1_tensor._scale_instance_masks(detections=detections, scale=2.0)

    expected = dense_mask.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    assert result.device == dense_mask.device
    assert result.dtype == torch.bool
    assert torch.equal(result, expected)


def test_scale_rle_instance_masks_decodes_without_polygon_round_trip(
    monkeypatch,
) -> None:
    dense_mask = torch.tensor([[True, False], [False, True]])
    rle = torch_mask_to_coco_rle(dense_mask)
    detections = _instance_detections(
        mask=InstancesRLEMasks(image_size=(2, 2), masks=[rle["counts"]])
    )

    def fail_on_polygon_round_trip(*args, **kwargs):
        pytest.fail("RLE mask scaling must not use polygon conversion")

    monkeypatch.setattr(sv, "mask_to_polygons", fail_on_polygon_round_trip)
    monkeypatch.setattr(sv, "polygon_to_mask", fail_on_polygon_round_trip)

    result = v1_tensor._scale_instance_masks(detections=detections, scale=2.0)

    expected = (
        dense_mask.unsqueeze(0).repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    )
    assert result.device == detections.xyxy.device
    assert result.dtype == torch.bool
    assert torch.equal(result, expected)


def test_scale_numeric_metadata_preserves_tensors_and_object_data() -> None:
    device = torch.device("cpu")
    object_metadata = {"source": "camera", "labels": ["cat", "dog"]}
    image_metadata = {
        IMAGE_DIMENSIONS_KEY: torch.tensor([10, 20]),
        "object_metadata": object_metadata,
    }
    bboxes_metadata = [
        {
            KEYPOINTS_XY_KEY_IN_SV_DETECTIONS: torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            POLYGON_KEY_IN_SV_DETECTIONS: np.array([[1, 2], [3, 4]]),
            "object_metadata": object_metadata,
        }
    ]

    scaled_image_metadata = v1_tensor._scale_image_metadata(
        image_metadata=image_metadata,
        scale=1.5,
        device=device,
    )
    scaled_bboxes_metadata = v1_tensor._scale_bboxes_metadata(
        bboxes_metadata=bboxes_metadata,
        detections_number=1,
        scale=1.5,
        device=device,
    )

    assert torch.equal(
        scaled_image_metadata[IMAGE_DIMENSIONS_KEY], torch.tensor([15.0, 30.0])
    )
    assert scaled_image_metadata[IMAGE_DIMENSIONS_KEY].device == device
    assert scaled_image_metadata["object_metadata"] is object_metadata
    assert torch.equal(
        scaled_bboxes_metadata[0][KEYPOINTS_XY_KEY_IN_SV_DETECTIONS],
        torch.tensor([[2.0, 3.0], [4.0, 6.0]]),
    )
    assert torch.equal(
        scaled_bboxes_metadata[0][POLYGON_KEY_IN_SV_DETECTIONS],
        torch.tensor([[2, 3], [4, 6]], dtype=torch.int32),
    )
    assert scaled_bboxes_metadata[0]["object_metadata"] is object_metadata
    assert (
        v1_tensor._scale_numeric_metadata(
            value=object_metadata,
            scale=1.5,
            device=device,
        )
        is object_metadata
    )
