import pytest
import torch
from roboflow_workflows.v2.blocks.fusion.pairwise import box_iou_between

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is not available"
        ),
    ),
]


@pytest.mark.parametrize("device", DEVICES)
def test_iou_of_each_box_in_a_with_each_box_in_b(device: str) -> None:
    boxes_a = torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]], device=device)
    boxes_b = torch.tensor(
        [[0, 0, 10, 10], [5, 0, 15, 10], [100, 100, 110, 110]],
        dtype=torch.float16,
        device=device,
    )

    iou = box_iou_between(boxes_a, boxes_b)

    # Half-shifted box: intersection 50, union 150.
    expected = torch.tensor([[1.0, 1 / 3, 0.0], [0.0, 0.0, 0.0]])
    assert iou.shape == (2, 3)
    assert iou.dtype == torch.float32
    assert iou.device.type == device
    assert torch.allclose(iou.cpu(), expected)


def test_iou_is_not_symmetric_in_shape() -> None:
    boxes_a = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
    boxes_b = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 4.0, 2.0]])

    iou = box_iou_between(boxes_a, boxes_b)
    transposed = box_iou_between(boxes_b, boxes_a)

    assert torch.allclose(iou, torch.tensor([[0.25, 0.5]]))
    assert torch.equal(transposed, iou.T)


def test_zero_area_pairs_have_zero_iou_instead_of_nan() -> None:
    point = torch.tensor([[3.0, 3.0, 3.0, 3.0]])

    iou = box_iou_between(point, point.clone())

    assert torch.equal(iou, torch.zeros((1, 1)))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rows_a, rows_b", [(0, 2), (2, 0), (0, 0)])
def test_empty_sets_give_empty_matrix(device: str, rows_a: int, rows_b: int) -> None:
    boxes_a = torch.zeros((rows_a, 4), device=device)
    boxes_b = torch.ones((rows_b, 4), device=device)

    iou = box_iou_between(boxes_a, boxes_b)

    assert iou.shape == (rows_a, rows_b)
    assert iou.device.type == device


@pytest.mark.parametrize("shape", [(4,), (2, 5), (1, 2, 4)])
def test_rejects_boxes_that_are_not_k_by_4(shape: tuple) -> None:
    with pytest.raises(ValueError, match="boxes_b must have shape"):
        box_iou_between(torch.zeros((1, 4)), torch.zeros(shape))


def test_rejects_boxes_on_different_devices() -> None:
    boxes_meta = torch.zeros((1, 4), device="meta")

    with pytest.raises(ValueError, match="boxes_a is on cpu but boxes_b is on meta"):
        box_iou_between(torch.zeros((1, 4)), boxes_meta)
