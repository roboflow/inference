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
import supervision as sv
import torch

import inference.core.workflows.core_steps.fusion.detections_stitch.v1_tensor as stitch_module
from inference.core.workflows.core_steps.fusion.detections_stitch.v1_tensor import (
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
) -> InstanceDetections:
    if xyxy is None:
        xyxy = _boxes_from_masks(masks)
    return InstanceDetections(
        xyxy=torch.as_tensor(xyxy, dtype=torch.float32),
        class_id=torch.as_tensor(class_id, dtype=torch.long),
        confidence=torch.as_tensor(confidence, dtype=torch.float32),
        mask=torch.as_tensor(masks, dtype=torch.bool),
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


def test_nmm_chunked_pairwise_intersection_matches_unchunked(monkeypatch) -> None:
    # given: force the tiled matmul path of the pairwise-intersection helper and
    # confirm decisions stay identical (counts are exact either way).
    masks, confidence, class_id = _random_case(seed=4321)
    detections = _make_instance_detections(
        masks=masks, confidence=confidence, class_id=class_id
    )
    expected = _oracle_with_nmm(detections=deepcopy(detections), threshold=0.25)
    monkeypatch.setattr(stitch_module, "_NMM_PAIRWISE_FLOAT_BUDGET_BYTES", 512 * 1024)
    _forbid_sv_fallback(monkeypatch)

    # when
    result = with_nmm(detections=deepcopy(detections), threshold=0.25)

    # then
    _assert_same_result(result=result, expected=expected)
