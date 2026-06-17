import torch

from inference_models.models.base.object_detection import Detections


def test_len_when_no_detections() -> None:
    # given
    detections = Detections(
        xyxy=torch.zeros((0, 4), dtype=torch.float32),
        class_id=torch.zeros((0,), dtype=torch.long),
        confidence=torch.zeros((0,), dtype=torch.float32),
    )

    # when / then
    assert len(detections) == 0


def test_len_when_single_detection() -> None:
    # given
    detections = Detections(
        xyxy=torch.tensor([[0, 1, 2, 3]], dtype=torch.float32),
        class_id=torch.tensor([0], dtype=torch.long),
        confidence=torch.tensor([0.9], dtype=torch.float32),
    )

    # when / then
    assert len(detections) == 1


def test_len_when_multiple_detections() -> None:
    # given
    detections = Detections(
        xyxy=torch.tensor(
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype=torch.float32
        ),
        class_id=torch.tensor([0, 1, 2], dtype=torch.long),
        confidence=torch.tensor([0.3, 0.4, 0.5], dtype=torch.float32),
    )

    # when / then
    assert len(detections) == 3
