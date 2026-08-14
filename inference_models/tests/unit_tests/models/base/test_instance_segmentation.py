import torch

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks


def test_len_when_no_instances() -> None:
    # given
    detections = InstanceDetections(
        xyxy=torch.zeros((0, 4), dtype=torch.float32),
        class_id=torch.zeros((0,), dtype=torch.long),
        confidence=torch.zeros((0,), dtype=torch.float32),
        mask=torch.zeros((0, 8, 8), dtype=torch.bool),
    )

    # when / then
    assert len(detections) == 0


def test_len_when_multiple_instances_with_dense_mask() -> None:
    # given
    detections = InstanceDetections(
        xyxy=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.float32),
        class_id=torch.tensor([0, 1], dtype=torch.long),
        confidence=torch.tensor([0.3, 0.4], dtype=torch.float32),
        mask=torch.zeros((2, 8, 8), dtype=torch.bool),
    )

    # when / then
    assert len(detections) == 2


def test_len_counts_boxes_regardless_of_mask_representation() -> None:
    # given - len() reads xyxy.shape[0], so the RLE mask representation must not matter
    detections = InstanceDetections(
        xyxy=torch.tensor(
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype=torch.float32
        ),
        class_id=torch.tensor([0, 1, 0], dtype=torch.long),
        confidence=torch.tensor([0.3, 0.4, 0.5], dtype=torch.float32),
        mask=InstancesRLEMasks(image_size=(8, 8), masks=[b"", b"", b""]),
    )

    # when / then
    assert len(detections) == 3
