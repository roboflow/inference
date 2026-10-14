"""Parity tests for the torch NMM port of the tensor-native detections_stitch
block.

`_oracle_with_nmm` below is a verbatim copy of the block's NMM branch BEFORE the
torch port (dense masks D2H -> sv.Detections.with_nmm on CPU -> re-upload). The
fuzz suite asserts the new `with_nmm` is value-identical to that oracle: same
surviving/merged boxes, class ids, confidences and exact bool-equal masks.
"""

import random
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
import pytest
import roboflow_workflows.core_steps.fusion.detections_stitch.v1_tensor as stitch_module
import supervision as sv
import torch
from roboflow_workflows.core_steps.fusion.detections_stitch.v1_tensor import (
    with_nmm,
    with_nms,
)

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import (
    coco_rle_masks_to_numpy_mask,
    torch_mask_to_coco_rle,
)

TensorNativeDetections = Union[Detections, InstanceDetections]

DEVICES = (
    ["cpu"]
    + (["mps"] if torch.backends.mps.is_available() else [])
    + (["cuda"] if torch.cuda.is_available() else [])
)


def _oracle_with_nmm(
    detections: TensorNativeDetections,
    threshold: float,
) -> TensorNativeDetections:
    """Verbatim copy of the pre-port NMM branch of detections_stitch v1_tensor
    (sv.Detections used as the NMM algorithm, full masks round-tripped through
    host memory). Serves as the behavioral oracle for the torch port."""
    if len(detections) == 0:
        return detections
    is_instance_segmentation = isinstance(detections, InstanceDetections)
    masks = None
    if is_instance_segmentation and detections.mask is not None:
        if isinstance(detections.mask, InstancesRLEMasks):
            masks = coco_rle_masks_to_numpy_mask(detections.mask)
        else:
            masks = detections.mask.detach().to("cpu").numpy().astype(bool)
    nmm_output = sv.Detections(
        xyxy=detections.xyxy.detach().to("cpu").numpy().astype(float),
        confidence=detections.confidence.detach().to("cpu").numpy().astype(float),
        class_id=detections.class_id.detach().to("cpu").numpy().astype(int),
        mask=masks,
    ).with_nmm(threshold=threshold)
    number_of_detections = len(nmm_output)
    device = detections.xyxy.device
    xyxy = torch.as_tensor(
        np.asarray(nmm_output.xyxy), dtype=torch.float32, device=device
    ).reshape(-1, 4)
    class_id = torch.as_tensor(
        np.asarray(nmm_output.class_id), dtype=torch.long, device=device
    )
    confidence = torch.as_tensor(
        np.asarray(nmm_output.confidence), dtype=torch.float32, device=device
    )
    if is_instance_segmentation:
        if nmm_output.mask is not None:
            mask = torch.as_tensor(
                np.asarray(nmm_output.mask), dtype=torch.bool, device=device
            )
        else:
            mask = torch.zeros((number_of_detections, 0, 0), dtype=torch.bool)
        return InstanceDetections(
            xyxy=xyxy,
            class_id=class_id,
            confidence=confidence,
            mask=mask,
            image_metadata=None,
            bboxes_metadata=None,
        )
    return Detections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        image_metadata=None,
        bboxes_metadata=None,
    )


def _make_instance_detections(
    masks: np.ndarray,
    confidence: np.ndarray,
    class_id: np.ndarray,
    xyxy: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> InstanceDetections:
    if xyxy is None:
        xyxy = _boxes_from_masks(masks)
    return InstanceDetections(
        xyxy=torch.as_tensor(xyxy, dtype=torch.float32).to(device),
        class_id=torch.as_tensor(class_id, dtype=torch.long).to(device),
        confidence=torch.as_tensor(confidence, dtype=torch.float32).to(device),
        mask=torch.as_tensor(masks, dtype=torch.bool).to(device),
        image_metadata=None,
        bboxes_metadata=None,
    )


def _boxes_from_masks(masks: np.ndarray) -> np.ndarray:
    boxes = np.zeros((masks.shape[0], 4), dtype=np.float32)
    for index, mask in enumerate(masks):
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        boxes[index] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    return boxes


def _random_blob_mask(
    rng: random.Random, height: int, width: int, kind: str
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if kind == "empty":
        return mask
    if kind == "full":
        mask[:] = True
        return mask
    if kind == "rect":
        x0 = rng.randrange(0, max(1, width - 2))
        y0 = rng.randrange(0, max(1, height - 2))
        x1 = rng.randrange(x0 + 1, width + 1)
        y1 = rng.randrange(y0 + 1, height + 1)
        mask[y0:y1, x0:x1] = True
        return mask
    # "circle"
    cy = rng.uniform(0, height)
    cx = rng.uniform(0, width)
    radius = rng.uniform(2, max(3, min(height, width) / 2))
    yy, xx = np.mgrid[0:height, 0:width]
    mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2] = True
    return mask


def _random_case(
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    height, width = rng.choice([(64, 96), (128, 128), (150, 200), (97, 61), (700, 900)])
    n = rng.choice([1, 2, 3, 5, 8, 12, 20, 30])
    kinds = ["rect", "rect", "circle", "empty", "full"]
    masks = np.zeros((n, height, width), dtype=bool)
    for index in range(n):
        kind = rng.choice(kinds)
        masks[index] = _random_blob_mask(rng, height, width, kind)
        # Sometimes nest: replace with a strict subset of a previous mask.
        if index > 0 and rng.random() < 0.2 and masks[index - 1].any():
            ys, xs = np.where(masks[index - 1])
            y_mid = int(np.median(ys))
            x_mid = int(np.median(xs))
            nested = np.zeros_like(masks[index - 1])
            nested[ys.min() : y_mid + 1, xs.min() : x_mid + 1] = masks[index - 1][
                ys.min() : y_mid + 1, xs.min() : x_mid + 1
            ]
            masks[index] = nested
        # Sometimes duplicate a previous mask exactly (IoU == 1 pairs).
        if index > 0 and rng.random() < 0.1:
            masks[index] = masks[rng.randrange(0, index)]
    confidence = np.array([rng.uniform(0.05, 0.999) for _ in range(n)])
    # Introduce exact-tie confidences occasionally.
    if n > 1 and rng.random() < 0.3:
        confidence = np.round(confidence, 1) + 0.05
    number_of_classes = rng.choice([1, 1, 2, 3])
    class_id = np.array([rng.randrange(0, number_of_classes) for _ in range(n)])
    return masks, confidence, class_id


def _random_sparse_case(
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Large frame with small objects, each detected a few times at jittered
    positions (what overlapping slices produce), some cut by the frame border."""
    rng = random.Random(seed)
    height, width = rng.choice([(360, 640), (480, 854), (720, 1280)])
    yy, xx = np.mgrid[0:height, 0:width]
    masks = []
    for _ in range(rng.randrange(2, 7)):
        cy, cx = rng.uniform(0, height), rng.uniform(0, width)
        ry = rng.uniform(3, 0.08 * height)
        rx = rng.uniform(3, 0.08 * width)
        for _ in range(rng.randrange(1, 5)):
            y = cy + rng.uniform(-1.5, 1.5) * ry
            x = cx + rng.uniform(-1.5, 1.5) * rx
            if rng.random() < 0.5:
                mask = (abs(yy - y) <= ry) & (abs(xx - x) <= rx)
            else:
                mask = ((yy - y) / ry) ** 2 + ((xx - x) / rx) ** 2 <= 1
            masks.append(mask)
    if rng.random() < 0.3:
        masks.append(np.zeros((height, width), dtype=bool))
    rng.shuffle(masks)
    n = len(masks)
    confidence = np.array([rng.uniform(0.05, 0.999) for _ in range(n)])
    number_of_classes = rng.choice([1, 1, 2])
    class_id = np.array([rng.randrange(0, number_of_classes) for _ in range(n)])
    return np.stack(masks), confidence, class_id


def _assert_same_result(
    result: InstanceDetections, expected: InstanceDetections
) -> None:
    assert isinstance(result, InstanceDetections)
    assert len(result) == len(expected)
    assert torch.equal(result.xyxy, expected.xyxy)
    assert torch.equal(result.class_id, expected.class_id)
    assert torch.equal(result.confidence, expected.confidence)
    assert torch.equal(result.mask, expected.mask)


def _forbid_sv_fallback(monkeypatch) -> None:
    def _fail(*args, **kwargs):
        raise AssertionError(
            "torch NMM fast path unexpectedly fell back to the sv implementation"
        )

    monkeypatch.setattr(stitch_module, "_with_nmm_sv", _fail)


@pytest.mark.parametrize("seed", list(range(60)))
@pytest.mark.parametrize("threshold", [0.2, 0.5])
def test_nmm_torch_port_fuzz_matches_sv_oracle(
    seed: int, threshold: float, monkeypatch
) -> None:
    # given
    masks, confidence, class_id = _random_case(seed=seed)
    detections = _make_instance_detections(
        masks=masks, confidence=confidence, class_id=class_id
    )
    expected = _oracle_with_nmm(detections=deepcopy(detections), threshold=threshold)
    _forbid_sv_fallback(monkeypatch)

    # when
    result = with_nmm(detections=deepcopy(detections), threshold=threshold)

    # then
    _assert_same_result(result=result, expected=expected)


@pytest.mark.parametrize("seed", list(range(60, 90)))
@pytest.mark.parametrize("threshold", [0.0, 0.05, 0.3, 0.75, 0.9, 1.0])
def test_nmm_torch_port_fuzz_varied_thresholds(
    seed: int, threshold: float, monkeypatch
) -> None:
    # given
    masks, confidence, class_id = _random_case(seed=seed)
    detections = _make_instance_detections(
        masks=masks, confidence=confidence, class_id=class_id
    )
    expected = _oracle_with_nmm(detections=deepcopy(detections), threshold=threshold)
    _forbid_sv_fallback(monkeypatch)

    # when
    result = with_nmm(detections=deepcopy(detections), threshold=threshold)

    # then
    _assert_same_result(result=result, expected=expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", list(range(90, 115)))
@pytest.mark.parametrize("threshold", [0.05, 0.3, 0.6])
def test_nmm_torch_port_fuzz_sparse_large_frames(
    seed: int, threshold: float, device: str, monkeypatch
) -> None:
    # given
    masks, confidence, class_id = _random_sparse_case(seed=seed)
    detections = _make_instance_detections(
        masks=masks, confidence=confidence, class_id=class_id, device=device
    )
    expected = _oracle_with_nmm(detections=deepcopy(detections), threshold=threshold)
    _forbid_sv_fallback(monkeypatch)

    # when
    result = with_nmm(detections=deepcopy(detections), threshold=threshold)

    # then
    _assert_same_result(result=result, expected=expected)


def test_nmm_torch_port_transitive_union_chain(monkeypatch) -> None:
    # given: A-B and B-C overlap heavily, A-C do not; with a low threshold the
    # growing-union semantics of sv._group_overlapping_masks must chain them.
    height, width = 100, 300
    masks = np.zeros((3, height, width), dtype=bool)
    masks[0, 20:80, 0:120] = True
    masks[1, 20:80, 60:180] = True
    masks[2, 20:80, 120:300] = True
    confidence = np.array([0.9, 0.8, 0.7])
    class_id = np.array([0, 0, 0])
    detections = _make_instance_detections(
        masks=masks, confidence=confidence, class_id=class_id
    )
    for threshold in [0.1, 0.2, 0.4, 0.6]:
        expected = _oracle_with_nmm(
            detections=deepcopy(detections), threshold=threshold
        )
        _forbid_sv_fallback(monkeypatch)

        # when
        result = with_nmm(detections=deepcopy(detections), threshold=threshold)

        # then
        _assert_same_result(result=result, expected=expected)


def test_nmm_torch_port_single_detection_passthrough(monkeypatch) -> None:
    # given
    masks = np.zeros((1, 50, 70), dtype=bool)
    masks[0, 10:30, 20:60] = True
    detections = _make_instance_detections(
        masks=masks, confidence=np.array([0.77]), class_id=np.array([1])
    )
    expected = _oracle_with_nmm(detections=deepcopy(detections), threshold=0.3)
    _forbid_sv_fallback(monkeypatch)

    # when
    result = with_nmm(detections=deepcopy(detections), threshold=0.3)

    # then
    _assert_same_result(result=result, expected=expected)
    assert len(result) == 1


def test_nmm_torch_port_all_empty_masks(monkeypatch) -> None:
    # given: unions are all zero -> IoU 0 -> nothing merges above a positive
    # threshold, every detection survives untouched.
    masks = np.zeros((4, 40, 40), dtype=bool)
    xyxy = np.array(
        [[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30], [1, 1, 2, 2]],
        dtype=np.float32,
    )
    detections = _make_instance_detections(
        masks=masks,
        confidence=np.array([0.9, 0.8, 0.7, 0.6]),
        class_id=np.array([0, 0, 1, 1]),
        xyxy=xyxy,
    )
    expected = _oracle_with_nmm(detections=deepcopy(detections), threshold=0.3)
    _forbid_sv_fallback(monkeypatch)

    # when
    result = with_nmm(detections=deepcopy(detections), threshold=0.3)

    # then
    _assert_same_result(result=result, expected=expected)
    assert len(result) == 4


def test_nmm_torch_port_disjoint_masks_survive(monkeypatch) -> None:
    # given
    masks = np.zeros((3, 60, 60), dtype=bool)
    masks[0, 0:10, 0:10] = True
    masks[1, 20:30, 20:30] = True
    masks[2, 40:50, 40:50] = True
    detections = _make_instance_detections(
        masks=masks,
        confidence=np.array([0.5, 0.6, 0.7]),
        class_id=np.array([0, 0, 0]),
    )
    expected = _oracle_with_nmm(detections=deepcopy(detections), threshold=0.3)
    _forbid_sv_fallback(monkeypatch)

    # when
    result = with_nmm(detections=deepcopy(detections), threshold=0.3)

    # then
    _assert_same_result(result=result, expected=expected)
    assert len(result) == 3


def test_nmm_empty_detections_returned_as_is() -> None:
    # given
    detections = InstanceDetections(
        xyxy=torch.zeros((0, 4), dtype=torch.float32),
        class_id=torch.zeros((0,), dtype=torch.long),
        confidence=torch.zeros((0,), dtype=torch.float32),
        mask=torch.zeros((0, 0, 0), dtype=torch.bool),
        image_metadata=None,
        bboxes_metadata=None,
    )

    # when
    result = with_nmm(detections=detections, threshold=0.3)

    # then
    assert result is detections


def test_nmm_rle_masks_route_through_sv_path_and_match_dense_oracle() -> None:
    # given: RLE-mask inputs keep the pre-port behavior (host-side decode + sv
    # NMM); the result must match the oracle run on the equivalent dense masks.
    masks, confidence, class_id = _random_case(seed=1234)
    dense = _make_instance_detections(
        masks=masks, confidence=confidence, class_id=class_id
    )
    rle_counts = [
        torch_mask_to_coco_rle(torch.as_tensor(mask, dtype=torch.bool))["counts"]
        for mask in masks
    ]
    rle = InstanceDetections(
        xyxy=dense.xyxy.clone(),
        class_id=dense.class_id.clone(),
        confidence=dense.confidence.clone(),
        mask=InstancesRLEMasks(image_size=masks.shape[1:], masks=rle_counts),
        image_metadata=None,
        bboxes_metadata=None,
    )
    expected = _oracle_with_nmm(detections=deepcopy(dense), threshold=0.3)

    # when
    result = with_nmm(detections=rle, threshold=0.3)

    # then
    _assert_same_result(result=result, expected=expected)


def test_nmm_bbox_only_detections_match_oracle() -> None:
    # given: no masks -> the sv box-NMM path is kept verbatim.
    xyxy = np.array(
        [
            [0, 0, 100, 100],
            [10, 10, 110, 110],
            [200, 200, 300, 300],
            [205, 205, 295, 295],
            [400, 0, 500, 80],
        ],
        dtype=np.float32,
    )
    detections = Detections(
        xyxy=torch.as_tensor(xyxy, dtype=torch.float32),
        class_id=torch.as_tensor([0, 0, 1, 1, 0], dtype=torch.long),
        confidence=torch.as_tensor([0.9, 0.85, 0.7, 0.95, 0.5], dtype=torch.float32),
        image_metadata=None,
        bboxes_metadata=None,
    )
    expected = _oracle_with_nmm(detections=deepcopy(detections), threshold=0.3)

    # when
    result = with_nmm(detections=deepcopy(detections), threshold=0.3)

    # then
    assert isinstance(result, Detections)
    assert torch.equal(result.xyxy, expected.xyxy)
    assert torch.equal(result.class_id, expected.class_id)
    assert torch.equal(result.confidence, expected.confidence)


def test_nms_branch_smoke() -> None:
    # given: the NMS branch is untouched by the NMM port — smoke-check that two
    # heavily-overlapping same-class boxes collapse to the higher-confidence one
    # and that surviving mask rows are the original rows.
    masks = np.zeros((3, 50, 50), dtype=bool)
    masks[0, 0:20, 0:20] = True
    masks[1, 1:21, 1:21] = True
    masks[2, 30:45, 30:45] = True
    detections = _make_instance_detections(
        masks=masks,
        confidence=np.array([0.9, 0.6, 0.8]),
        class_id=np.array([0, 0, 1]),
        xyxy=np.array(
            [[0, 0, 20, 20], [1, 1, 21, 21], [30, 30, 45, 45]], dtype=np.float32
        ),
    )

    # when
    result = with_nms(detections=deepcopy(detections), threshold=0.5)

    # then
    assert len(result) == 2
    assert torch.equal(
        result.xyxy,
        torch.as_tensor([[0, 0, 20, 20], [30, 30, 45, 45]], dtype=torch.float32),
    )
    assert torch.equal(result.class_id, torch.as_tensor([0, 1], dtype=torch.long))
    assert torch.equal(
        result.confidence, torch.as_tensor([0.9, 0.8], dtype=torch.float32)
    )
    assert torch.equal(result.mask[0], torch.as_tensor(masks[0], dtype=torch.bool))
    assert torch.equal(result.mask[1], torch.as_tensor(masks[2], dtype=torch.bool))


def test_nmm_torch_port_fractional_boxes_use_float64_areas(monkeypatch) -> None:
    # given: overlapping masks with fractional boxes, where the area-weighted
    # confidence differs between float32 and float64 box areas.
    rng = np.random.default_rng(7)
    masks = np.zeros((4, 64, 64), dtype=bool)
    masks[:, 8:40, 8:40] = True
    masks[3, 8:40, 8:20] = False
    xyxy = rng.uniform(0, 3000, size=(4, 4)).astype(np.float32)
    detections = _make_instance_detections(
        masks=masks,
        confidence=np.array([0.91, 0.62, 0.77, 0.5]),
        class_id=np.zeros(4, dtype=int),
        xyxy=xyxy,
    )
    expected = _oracle_with_nmm(detections=deepcopy(detections), threshold=0.3)
    _forbid_sv_fallback(monkeypatch)

    # when
    result = with_nmm(detections=deepcopy(detections), threshold=0.3)

    # then
    _assert_same_result(result=result, expected=expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("threshold", [0.0, 0.3])
def test_nmm_torch_port_empty_masks_among_overlapping_masks(
    threshold: float, device: str, monkeypatch
) -> None:
    # given: empty masks have no box; they sit between masks that merge over
    # two rounds (3 is absorbed by the union of 0 and 1), one of them is a seed.
    masks = np.zeros((6, 80, 80), dtype=bool)
    masks[0, 10:40, 10:40] = True
    masks[1, 15:45, 15:45] = True
    masks[3, 20:50, 20:50] = True
    detections = _make_instance_detections(
        masks=masks,
        confidence=np.array([0.9, 0.8, 0.95, 0.7, 0.6, 0.5]),
        class_id=np.array([0, 0, 0, 0, 0, 1]),
        xyxy=np.array(
            [
                [10, 10, 40, 40],
                [15, 15, 45, 45],
                [0, 0, 10, 10],
                [20, 20, 50, 50],
                [30, 30, 60, 60],
                [5, 5, 9, 9],
            ],
            dtype=np.float32,
        ),
        device=device,
    )
    expected = _oracle_with_nmm(detections=deepcopy(detections), threshold=threshold)
    _forbid_sv_fallback(monkeypatch)

    # when
    result = with_nmm(detections=deepcopy(detections), threshold=threshold)

    # then
    _assert_same_result(result=result, expected=expected)


@pytest.mark.parametrize("device", DEVICES)
def test_nmm_torch_port_mask_content_outside_detection_box(
    device: str, monkeypatch
) -> None:
    # given: the masks overlap heavily, while the detection boxes neither
    # enclose them nor overlap each other.
    masks = np.zeros((3, 200, 200), dtype=bool)
    masks[0, 100:140, 100:140] = True
    masks[1, 105:145, 105:145] = True
    masks[2, 150:190, 20:60] = True
    detections = _make_instance_detections(
        masks=masks,
        confidence=np.array([0.9, 0.8, 0.7]),
        class_id=np.zeros(3, dtype=int),
        xyxy=np.array(
            [[0, 0, 10, 10], [50, 50, 60, 60], [100, 100, 145, 145]],
            dtype=np.float32,
        ),
        device=device,
    )
    expected = _oracle_with_nmm(detections=deepcopy(detections), threshold=0.3)
    _forbid_sv_fallback(monkeypatch)

    # when
    result = with_nmm(detections=deepcopy(detections), threshold=0.3)

    # then
    _assert_same_result(result=result, expected=expected)
    assert len(result) == 2


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("threshold", [0.0, 1e-6])
def test_nmm_torch_port_overlapping_mask_boxes_without_shared_pixels(
    threshold: float, device: str, monkeypatch
) -> None:
    # given: interleaved rows - the mask boxes overlap almost entirely, the
    # masks share no pixel.
    masks = np.zeros((2, 40, 40), dtype=bool)
    masks[0, 10:30:2, 10:30] = True
    masks[1, 11:30:2, 10:30] = True
    detections = _make_instance_detections(
        masks=masks,
        confidence=np.array([0.9, 0.8]),
        class_id=np.zeros(2, dtype=int),
        device=device,
    )
    expected = _oracle_with_nmm(detections=deepcopy(detections), threshold=threshold)
    _forbid_sv_fallback(monkeypatch)

    # when
    result = with_nmm(detections=deepcopy(detections), threshold=threshold)

    # then
    _assert_same_result(result=result, expected=expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("threshold", [1e-6, 0.25, 0.3])
def test_nmm_torch_port_adjacent_mask_boxes(
    threshold: float, device: str, monkeypatch
) -> None:
    # given: the boxes of masks 0 and 1 share an edge and no pixel. Mask 2
    # overlaps both, so once it is absorbed by 0 the candidate reaches into 1
    # (IoU exactly 0.25), although the box of the seed does not.
    masks = np.zeros((3, 30, 40), dtype=bool)
    masks[0, 5:15, 0:10] = True
    masks[1, 5:15, 10:20] = True
    masks[2, 5:15, 5:15] = True
    detections = _make_instance_detections(
        masks=masks,
        confidence=np.array([0.9, 0.7, 0.8]),
        class_id=np.zeros(3, dtype=int),
        device=device,
    )
    expected = _oracle_with_nmm(detections=deepcopy(detections), threshold=threshold)
    _forbid_sv_fallback(monkeypatch)

    # when
    result = with_nmm(detections=deepcopy(detections), threshold=threshold)

    # then
    _assert_same_result(result=result, expected=expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shared_axis", ["column", "row"])
def test_nmm_torch_port_masks_sharing_only_the_last_line_of_a_box(
    shared_axis: str, device: str, monkeypatch
) -> None:
    # given: the masks share exactly the last column (row) of the first mask's
    # box; the threshold is their IoU, so losing one pixel flips the decision.
    masks = np.zeros((2, 40, 40), dtype=bool)
    masks[0, 0:10, 0:10] = True
    masks[1, 0:10, 9:19] = True
    if shared_axis == "row":
        masks = masks.transpose(0, 2, 1).copy()
    threshold = float(np.float32(10) / np.float32(190))
    detections = _make_instance_detections(
        masks=masks,
        confidence=np.array([0.9, 0.8]),
        class_id=np.zeros(2, dtype=int),
        device=device,
    )
    expected = _oracle_with_nmm(detections=deepcopy(detections), threshold=threshold)
    _forbid_sv_fallback(monkeypatch)

    # when
    result = with_nmm(detections=deepcopy(detections), threshold=threshold)

    # then
    _assert_same_result(result=result, expected=expected)
    assert len(result) == 1


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("counting_budget_bytes", [None, 2 * 9 * 48 * 64])
@pytest.mark.parametrize("threshold", [0.3, 1.0])
def test_nmm_torch_port_all_masks_full_frame(
    threshold: float,
    counting_budget_bytes: Optional[int],
    device: str,
    monkeypatch,
) -> None:
    # given: every mask box is the whole frame, every pair overlaps; the
    # reduced budget splits each gather into copies of two masks.
    masks = np.ones((9, 48, 64), dtype=bool)
    detections = _make_instance_detections(
        masks=masks,
        confidence=np.linspace(0.9, 0.1, 9),
        class_id=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0]),
        device=device,
    )
    expected = _oracle_with_nmm(detections=deepcopy(detections), threshold=threshold)
    if counting_budget_bytes is not None:
        monkeypatch.setattr(
            stitch_module, "_NMM_COUNTING_BUDGET_BYTES", counting_budget_bytes
        )
    _forbid_sv_fallback(monkeypatch)

    # when
    result = with_nmm(detections=deepcopy(detections), threshold=threshold)

    # then
    _assert_same_result(result=result, expected=expected)
    assert len(result) == 2


@pytest.mark.parametrize("device", DEVICES)
def test_nmm_torch_port_counts_rounded_like_sv_at_2_pow_24_pixels(
    device: str, monkeypatch
) -> None:
    # given: exactly 2**24 pixels, the largest frame supervision counts in
    # float32. The areas add up to 2**25 - 3, which float32 rounds to
    # 2**25 - 4, so supervision's IoU is (2**24 - 3) / (2**24 - 1) while the
    # exact one is (2**24 - 3) / 2**24; the threshold lies between the two.
    masks = np.ones((2, 4096, 4096), dtype=bool)
    masks[0, 0, 0] = False
    masks[1, 1, 1:3] = False
    threshold = 1 - 2.5 * 2.0**-24
    detections = _make_instance_detections(
        masks=masks,
        confidence=np.array([0.9, 0.8]),
        class_id=np.zeros(2, dtype=int),
        xyxy=np.array([[0, 0, 4096, 4096]] * 2, dtype=np.float32),
        device=device,
    )
    expected = _oracle_with_nmm(detections=deepcopy(detections), threshold=threshold)
    _forbid_sv_fallback(monkeypatch)

    # when
    result = with_nmm(detections=deepcopy(detections), threshold=threshold)

    # then
    _assert_same_result(result=result, expected=expected)
    assert len(result) == 1


@pytest.mark.parametrize("device", DEVICES)
def test_nmm_torch_port_masks_beyond_float32_exact_range_match_sv_oracle(
    device: str, monkeypatch
) -> None:
    # given: more than 2**24 pixels per mask, where supervision switches its
    # pixel counting from float32 to float64. The first mask has an odd area
    # above 2**24, which float32 would round up to the full frame, turning
    # an IoU just below 1 into exactly 1.
    masks = np.ones((2, 4200, 4200), dtype=bool)
    masks[0, 0, 0] = False
    detections = _make_instance_detections(
        masks=masks,
        confidence=np.array([0.9, 0.8]),
        class_id=np.zeros(2, dtype=int),
        xyxy=np.array([[0, 0, 4200, 4200]] * 2, dtype=np.float32),
        device=device,
    )
    expected = _oracle_with_nmm(detections=deepcopy(detections), threshold=1.0)
    _forbid_sv_fallback(monkeypatch)

    # when
    result = with_nmm(detections=deepcopy(detections), threshold=1.0)

    # then
    _assert_same_result(result=result, expected=expected)
    assert len(result) == 2
