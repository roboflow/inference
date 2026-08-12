import pytest
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


def test_iter_yields_seven_tuple_in_documented_field_order() -> None:
    # given - class_id and confidence are deliberately distinct in dtype and value so
    # that a silent confidence/class_id swap is caught: the sv-mode 6-tuple yields these
    # two fields in the OPPOSITE order, and positional destructures downstream
    # (detection_event_log/v1_tensor, UQL detection/base.py, detections/base.py) depend
    # on this exact native order.
    detections = Detections(
        xyxy=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.float32),
        class_id=torch.tensor([3, 7], dtype=torch.long),
        confidence=torch.tensor([0.9, 0.1], dtype=torch.float32),
        image_metadata={"image": "meta"},
        bboxes_metadata=[
            {"tracker_id": 10, "label": "a"},
            {"tracker_id": 20, "label": "b"},
        ],
    )

    # when
    rows = list(detections)

    # then - order is (xyxy, mask, class_id, confidence, tracker_id, data, metadata)
    assert len(rows) == 2
    for i, row in enumerate(rows):
        assert len(row) == 7, "native __iter__ must yield a 7-tuple"
        xyxy, mask, class_id, confidence, tracker_id, data, metadata = row
        assert torch.equal(xyxy, detections.xyxy[i])  # 0: xyxy
        assert mask is None  # 1: mask (always None for object detection)
        assert torch.equal(class_id, detections.class_id[i])  # 2: class_id
        assert torch.equal(confidence, detections.confidence[i])  # 3: confidence
        assert tracker_id == detections.bboxes_metadata[i]["tracker_id"]  # 4: tracker_id
        assert data == detections.bboxes_metadata[i]  # 5: per-detection data dict
        assert metadata == detections.image_metadata  # 6: per-image metadata dict

    # class_id is the integer label and confidence is the float score - guards directly
    # against a class_id<->confidence positional swap.
    assert rows[0][2].item() == 3
    assert rows[0][3].item() == pytest.approx(0.9)
    assert rows[1][2].item() == 7
    assert rows[1][3].item() == pytest.approx(0.1)


def test_iter_defaults_when_no_per_box_or_image_metadata() -> None:
    # given - neither bboxes_metadata nor image_metadata provided
    detections = Detections(
        xyxy=torch.tensor([[0, 1, 2, 3]], dtype=torch.float32),
        class_id=torch.tensor([5], dtype=torch.long),
        confidence=torch.tensor([0.42], dtype=torch.float32),
    )

    # when
    (row,) = list(detections)

    # then
    xyxy, mask, class_id, confidence, tracker_id, data, metadata = row
    assert mask is None
    assert tracker_id is None  # tracker_id absent -> None
    assert data == {}  # per-detection data defaults to empty dict
    assert metadata == {}  # per-image metadata defaults to empty dict
