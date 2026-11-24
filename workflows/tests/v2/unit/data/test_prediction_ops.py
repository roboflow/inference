"""Strict V2 row selection of native predictions: one gather for every field."""

import subprocess
import sys

import numpy as np
import pytest
import torch
from roboflow_workflows.v2.data.prediction_ops import (
    RowSelection,
    select_predictions,
    take_rows,
)
from roboflow_workflows.v2.engine.errors import ContractError

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks


def _detections():
    return Detections(
        xyxy=torch.arange(12, dtype=torch.float32).reshape(3, 4),
        class_id=torch.tensor([0, 1, 2]),
        confidence=torch.tensor([0.9, 0.3, 0.8]),
        image_metadata={"parent_id": "crop", "root_parent_id": "frame"},
        bboxes_metadata=[{"detection_id": f"box:{i}"} for i in range(3)],
    )


@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([True, False, True]),
        np.array([True, False, True]),
        [True, False, True],
    ],
)
def test_confidence_selection_preserves_ids_and_does_not_mutate_rows(mask):
    source = _detections()
    selected = select_predictions(source, mask=mask)

    assert selected.class_id.tolist() == [0, 2]
    assert [row["detection_id"] for row in selected.bboxes_metadata] == [
        "box:0",
        "box:2",
    ]
    assert selected.image_metadata is source.image_metadata
    assert selected.xyxy.device == source.xyxy.device
    selected.bboxes_metadata[0]["tracker_id"] = 9
    assert "tracker_id" not in source.bboxes_metadata[0]


@pytest.mark.parametrize("rle", [False, True])
def test_reordering_keeps_dense_or_compressed_masks_aligned(rle):
    boxes = _detections()
    mask = (
        InstancesRLEMasks(image_size=(3, 3), masks=[b"9", b"09", b"135"])
        if rle
        else torch.arange(27).reshape(3, 3, 3) % 2 == 0
    )
    source = InstanceDetections(**vars(boxes), mask=mask)
    selected = select_predictions(source, indices=[2, 0, 2])

    assert selected.class_id.tolist() == [2, 0, 2]
    assert [row["detection_id"] for row in selected.bboxes_metadata] == [
        "box:2",
        "box:0",
        "box:2",
    ]
    if rle:
        assert selected.mask.masks == [b"135", b"9", b"135"]
    else:
        assert torch.equal(selected.mask, mask[[2, 0, 2]])


@pytest.mark.parametrize("with_boxes", [False, True])
def test_keypoint_selection_preserves_slots_covariance_and_detection_confidence(
    with_boxes,
):
    points = KeyPoints(
        xy=torch.arange(18, dtype=torch.float32).reshape(3, 3, 2),
        class_id=torch.tensor([0, 1, 2]),
        confidence=torch.full((3, 3), 0.8),
        covariance=torch.arange(36, dtype=torch.float32).reshape(3, 3, 2, 2),
        detection_confidence=torch.tensor([0.9, 0.3, 0.8]),
        key_points_metadata=[
            {"ids": [0, 4, 9], "detection_id": f"kp:{i}"} for i in range(3)
        ],
    )
    source = (points, _detections() if with_boxes else None)
    selected, boxes = select_predictions(source, indices=torch.tensor([2, 0]))

    assert torch.equal(selected.xy, points.xy[[2, 0]])
    assert torch.equal(selected.covariance, points.covariance[[2, 0]])
    assert torch.equal(
        selected.detection_confidence, points.detection_confidence[[2, 0]]
    )
    assert selected.key_points_metadata[0] == {"ids": [0, 4, 9], "detection_id": "kp:2"}
    assert (boxes is not None) == with_boxes
    if with_boxes:
        assert boxes.class_id.tolist() == [2, 0]


def test_empty_and_identity_selection_preserve_tensor_shape_and_dtype():
    source = _detections()
    identity = select_predictions(source, mask=torch.ones(3, dtype=torch.bool))
    empty = select_predictions(source, mask=torch.zeros(3, dtype=torch.bool))

    assert identity.xyxy is source.xyxy
    assert empty.xyxy.shape == (0, 4)
    assert empty.xyxy.dtype == source.xyxy.dtype
    assert empty.bboxes_metadata == []


@pytest.mark.parametrize(
    "selection, match",
    [
        ({}, "exactly one"),
        ({"mask": [True] * 3, "indices": [0]}, "exactly one"),
        ({"mask": [True]}, "1 entries; expected 3"),
        ({"mask": torch.tensor([1, 0, 1])}, "boolean dtype"),
        ({"mask": torch.ones((3, 1), dtype=torch.bool)}, "one-dimensional"),
        ({"mask": [1, 0, 1]}, "boolean sequence"),
        ({"indices": [-1]}, "outside"),
        ({"indices": [3]}, "outside"),
        ({"indices": [0.0]}, "integer sequence"),
        ({"indices": [True]}, "integer sequence"),
        ({"indices": torch.tensor([0.0])}, "integer dtype"),
        ({"indices": np.array([[0]])}, "one-dimensional"),
    ],
)
def test_invalid_selection_is_rejected_before_gather(selection, match):
    with pytest.raises(ContractError, match=match):
        select_predictions(_detections(), **selection)


def test_unsupported_payload_and_misaligned_rows_fail_usefully():
    with pytest.raises(ContractError, match="supports Detections"):
        select_predictions(torch.zeros(3), indices=[0])

    malformed = _detections()
    malformed.confidence = torch.ones(2)
    with pytest.raises(ContractError, match="confidence"):
        select_predictions(malformed, indices=[0])


def test_only_selected_indices_cross_to_host_for_metadata(monkeypatch):
    source = _detections()
    copied_shapes = []
    original_cpu = torch.Tensor.cpu

    def record_cpu(tensor, *args, **kwargs):
        copied_shapes.append((tuple(tensor.shape), tensor.dtype))
        return original_cpu(tensor, *args, **kwargs)

    def reject_numpy(*args, **kwargs):
        raise AssertionError("Prediction tensors must not be exported to NumPy")

    monkeypatch.setattr(torch.Tensor, "cpu", record_cpu)
    monkeypatch.setattr(torch.Tensor, "numpy", reject_numpy)
    selected = select_predictions(source, mask=torch.tensor([True, False, True]))

    assert copied_shapes == [((2,), torch.int64)]
    assert selected.xyxy.device == source.xyxy.device


# Aligned gather -------------------------------------------------------------


def _three_rows(*, rle):
    if rle:
        # Distinct opaque COCO counts; selection must move them, never decode.
        mask = InstancesRLEMasks(image_size=(2, 3), masks=[b"6", b"24", b"1112"])
    else:
        mask = torch.zeros((3, 4, 5), dtype=torch.bool)
        mask[0, 1:3, 1:4] = True
        mask[0, 2, 2] = False  # a hole
        mask[1, 0, 0] = True
        mask[2, 3, 4] = True
    return InstanceDetections(
        xyxy=torch.tensor(
            [[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 4.0], [2.0, 0.0, 5.0, 3.0]]
        ),
        class_id=torch.tensor([7, 8, 9], dtype=torch.int32),
        confidence=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float16),
        mask=mask,
        image_metadata={"parent_id": "crop"},
        bboxes_metadata=[{"tracker_id": 10 + i, "name": f"row{i}"} for i in range(3)],
    )


@pytest.mark.parametrize("rle", [False, True])
def test_two_zero_two_then_empty_moves_every_field_together(rle, monkeypatch):
    source = _three_rows(rle=rle)

    picked = select_predictions(source, indices=[2, 0, 2])

    assert picked.xyxy.tolist() == [
        [2.0, 0.0, 5.0, 3.0],
        [0.0, 0.0, 2.0, 2.0],
        [2.0, 0.0, 5.0, 3.0],
    ]
    assert picked.class_id.tolist() == [9, 7, 9]
    assert picked.class_id.dtype == torch.int32
    assert picked.confidence.dtype == torch.float16
    assert [row["tracker_id"] for row in picked.bboxes_metadata] == [12, 10, 12]
    assert picked.bboxes_metadata[0] is not picked.bboxes_metadata[2]
    if rle:
        assert picked.mask.masks == [b"1112", b"6", b"1112"]
        assert picked.mask.image_size == (2, 3)
    else:
        assert torch.equal(picked.mask, source.mask[[2, 0, 2]])
        assert not picked.mask[1, 2, 2]  # the hole survives

    def no_decode(*args, **kwargs):
        raise AssertionError("empty selection must not decode RLE")

    monkeypatch.setattr("pycocotools.mask.decode", no_decode)
    empty = select_predictions(picked, indices=[])

    assert empty.xyxy.shape == (0, 4)
    assert empty.class_id.shape == (0,)
    assert empty.confidence.dtype == torch.float16
    assert empty.bboxes_metadata == []
    if rle:
        assert empty.mask.masks == []
        assert empty.mask.image_size == (2, 3)
    else:
        assert empty.mask.shape == (0, 4, 5)
        assert empty.mask.dtype == torch.bool


def test_two_zero_two_then_empty_keypoints_keep_slots_and_boxes():
    points = KeyPoints(
        xy=torch.arange(18, dtype=torch.float32).reshape(3, 3, 2),
        class_id=torch.tensor([0, 1, 2]),
        confidence=torch.arange(9, dtype=torch.float32).reshape(3, 3),
        covariance=torch.arange(36, dtype=torch.float32).reshape(3, 3, 2, 2),
        detection_confidence=torch.tensor([0.9, 0.3, 0.8]),
        key_points_metadata=[{"id": i} for i in range(3)],
    )

    picked, boxes = select_predictions((points, _detections()), indices=[2, 0, 2])
    empty, empty_boxes = select_predictions((picked, boxes), indices=[])

    assert torch.equal(picked.confidence, points.confidence[[2, 0, 2]])
    assert [row["id"] for row in picked.key_points_metadata] == [2, 0, 2]
    assert boxes.class_id.tolist() == [2, 0, 2]
    assert empty.xy.shape == (0, 3, 2)
    assert empty.covariance.shape == (0, 3, 2, 2)
    assert empty_boxes.xyxy.shape == (0, 4)


def test_strided_fields_and_indices_select_like_contiguous_controls():
    contiguous = _three_rows(rle=False)
    strided = InstanceDetections(
        xyxy=contiguous.xyxy.t().contiguous().t(),
        class_id=torch.tensor([7, 0, 8, 0, 9, 0], dtype=torch.int32)[::2],
        confidence=contiguous.confidence,
        mask=contiguous.mask.transpose(1, 2).contiguous().transpose(1, 2),
        image_metadata=contiguous.image_metadata,
        bboxes_metadata=contiguous.bboxes_metadata,
    )
    indices = torch.tensor([9, 2, 9, 0, 9, 2])[1::2]
    assert not strided.xyxy.is_contiguous()
    assert not strided.class_id.is_contiguous()
    assert not strided.mask.is_contiguous()
    assert not indices.is_contiguous()
    strides_before = strided.xyxy.stride()

    actual = select_predictions(strided, indices=indices)
    expected = select_predictions(contiguous, indices=[2, 0, 2])

    for name in ("xyxy", "class_id", "confidence", "mask"):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name
        assert getattr(actual, name).dtype == getattr(expected, name).dtype
    assert strided.xyxy.stride() == strides_before


def test_device_index_selection_reads_no_values():
    # Meta tensors have no data: any host copy or value read would raise.
    meta = torch.device("meta")
    source = InstanceDetections(
        xyxy=torch.empty((3, 4), device=meta),
        class_id=torch.empty(3, dtype=torch.int64, device=meta),
        confidence=torch.empty(3, device=meta),
        mask=torch.empty((3, 5, 7), dtype=torch.bool, device=meta),
    )

    picked = select_predictions(
        source, indices=torch.empty(4, dtype=torch.int32, device=meta)
    )

    assert picked.xyxy.shape == (4, 4)
    assert picked.mask.shape == (4, 5, 7)
    assert picked.xyxy.device == meta


def test_cpu_index_tensor_without_row_metadata_builds_no_host_list(monkeypatch):
    source = _detections()
    source.bboxes_metadata = None

    def reject_tolist(*args, **kwargs):
        raise AssertionError("tensor-only selection must not build host lists")

    monkeypatch.setattr(torch.Tensor, "tolist", reject_tolist)
    picked = select_predictions(source, indices=torch.tensor([2, 0, 2]))

    assert picked.xyxy.shape == (3, 4)
    assert torch.equal(picked.class_id, torch.tensor([2, 0, 2]))


def test_cpu_index_tensor_out_of_range_is_rejected_with_its_bounds():
    with pytest.raises(ContractError, match=r"\[0, 3\), got \[-1, 2\]"):
        select_predictions(_detections(), indices=torch.tensor([2, -1]))


def test_one_selection_covers_extra_aligned_tensors():
    source = _detections()
    scores = torch.arange(30, dtype=torch.float32).reshape(3, 2, 5)[:, :, ::2]
    selection = RowSelection.of(3, mask=torch.tensor([False, True, True]))

    picked = take_rows(source, selection)

    assert len(selection) == 2
    assert torch.equal(selection.take(scores), scores[[1, 2]])
    assert picked.class_id.tolist() == [1, 2]
    with pytest.raises(ContractError, match="built for 3 rows"):
        take_rows(select_predictions(source, indices=[0]), selection)
    with pytest.raises(ContractError, match="Cannot select 3 rows"):
        selection.take(torch.zeros(2))


def test_selection_module_does_not_import_the_v1_gather():
    code = (
        "import sys\n"
        "import roboflow_workflows.v2.data.prediction_ops\n"
        "assert 'roboflow_workflows.core_steps.common.tensor_native' "
        "not in sys.modules\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr[-2000:]
