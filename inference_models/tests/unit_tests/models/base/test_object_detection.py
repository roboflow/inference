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


def test_tracker_ids_remain_tensor_native_until_iteration() -> None:
    """Tracker IDs stay as tensors while legacy iteration exposes metadata."""
    tracker_ids = torch.tensor([7, 9], dtype=torch.long)
    metadata = [{"source": "a"}, {"source": "b"}]
    detections = Detections(
        xyxy=torch.zeros((2, 4)),
        class_id=torch.zeros(2, dtype=torch.long),
        confidence=torch.ones(2),
        bboxes_metadata=metadata,
        tracker_id=tracker_ids,
    )

    rows = list(detections)

    assert detections.tracker_id is tracker_ids
    assert rows[0][4] == 7
    assert rows[1][5]["tracker_id"] == 9
    assert metadata == [{"source": "a"}, {"source": "b"}]


def test_to_supervision_materializes_first_class_tracker_ids() -> None:
    """The explicit Supervision boundary carries native tracker IDs."""
    detections = Detections(
        xyxy=torch.zeros((2, 4)),
        class_id=torch.zeros(2, dtype=torch.long),
        confidence=torch.ones(2),
        tracker_id=torch.tensor([4, 5]),
    )

    result = detections.to_supervision()

    assert result.tracker_id.tolist() == [4, 5]
