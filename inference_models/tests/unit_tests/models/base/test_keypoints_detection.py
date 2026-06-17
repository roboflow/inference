import torch

from inference_models.models.base.keypoints_detection import KeyPoints


def test_len_when_no_instances() -> None:
    # given
    key_points = KeyPoints(
        xy=torch.zeros((0, 17, 2), dtype=torch.float32),
        class_id=torch.zeros((0,), dtype=torch.long),
        confidence=torch.zeros((0, 17), dtype=torch.float32),
    )

    # when / then
    assert len(key_points) == 0


def test_len_when_single_instance() -> None:
    # given
    key_points = KeyPoints(
        xy=torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),  # (1, 3, 2)
        class_id=torch.tensor([0], dtype=torch.long),
        confidence=torch.tensor([[0.9, 0.8, 0.7]]),  # (1, 3)
    )

    # when / then
    assert len(key_points) == 1


def test_len_counts_instances_not_keypoints_per_instance() -> None:
    # given - 2 instances, each with 17 keypoints; len() must be 2 (instance axis),
    # NOT 17 (keypoints-per-instance axis, xy.shape[1]).
    key_points = KeyPoints(
        xy=torch.zeros((2, 17, 2), dtype=torch.float32),
        class_id=torch.tensor([0, 1], dtype=torch.long),
        confidence=torch.zeros((2, 17), dtype=torch.float32),
    )

    # when / then
    assert len(key_points) == 2
