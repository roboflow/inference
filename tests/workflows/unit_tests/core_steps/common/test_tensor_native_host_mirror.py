"""Tests for the per-box host mirror of the detection tensors
(``HOST_MIRROR_KEYS`` in ``tensor_native.py``).

``attach_native_detection_metadata`` stores a host-side copy of each box's
``xyxy`` / ``class_id`` / ``confidence`` in ``bboxes_metadata`` (one batched
device->host read per tensor); visualization-phase consumers prefer it over
device reads. These tests prove:

* the mirror values match the tensors for every prediction shape attach handles,
* the implementation is batched (dispatch-op count independent of N, no
  ``.item()`` loops),
* geometry-mutating helpers drop the mirror (staleness contract),
* the wire format is byte-identical with and without the mirror, and the
  deserializers never resurrect it.
"""

import json

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from inference.core.workflows.core_steps.common.deserializers_tensor import (
    deserialize_detections_kind,
)
from inference.core.workflows.core_steps.common.serializers_tensor import (
    serialise_sv_detections,
)
from inference.core.workflows.core_steps.common.tensor_native import (
    HOST_CLASS_ID_KEY,
    HOST_CONFIDENCE_KEY,
    HOST_MIRROR_KEYS,
    HOST_XYXY_KEY,
    attach_native_detection_metadata,
    native_detections_to_root_coordinates,
    read_host_mirror,
    strip_host_mirror_metadata,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

CLASS_NAMES = {0: "cat", 1: "dog", 2: "goggles"}


class _OpAudit(TorchDispatchMode):
    """Record every dispatched aten op (with tensor-argument shapes/dtypes)."""

    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        shapes = []

        def _collect(value):
            if isinstance(value, torch.Tensor):
                shapes.append((tuple(value.shape), str(value.dtype)))
            elif isinstance(value, (list, tuple)):
                for item in value:
                    _collect(item)

        _collect(args)
        _collect(list((kwargs or {}).values()))
        self.ops.append((str(func), shapes))
        return func(*args, **(kwargs or {}))

    def op_names(self):
        return [name for name, _ in self.ops]


def _image(height: int = 48, width: int = 64) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((height, width, 3), dtype=np.uint8),
    )


def _od_detections(n: int = 3, existing_metadata=None) -> Detections:
    xyxy = np.asarray(
        [
            [10.5 + index, 20.25 + index, 110.75 + index, 220.125 + index]
            for index in range(n)
        ],
        dtype=np.float32,
    ).reshape(n, 4)
    return Detections(
        xyxy=torch.tensor(xyxy, dtype=torch.float32),
        class_id=torch.tensor([index % 3 for index in range(n)], dtype=torch.long),
        confidence=torch.tensor(
            np.linspace(0.42, 0.99, max(n, 1))[:n], dtype=torch.float32
        ),
        image_metadata=None,
        bboxes_metadata=existing_metadata,
    )


def _is_detections(n: int = 2) -> InstanceDetections:
    masks = torch.zeros((n, 48, 64), dtype=torch.bool)
    for index in range(n):
        masks[index, 5 + index : 15 + index, 10 : 30 + index] = True
    base = _od_detections(n)
    return InstanceDetections(
        xyxy=base.xyxy,
        class_id=base.class_id,
        confidence=base.confidence,
        mask=masks,
        image_metadata=None,
        bboxes_metadata=None,
    )


def _attach(detections, prediction_type: str = "object-detection"):
    return attach_native_detection_metadata(
        detections=detections,
        image=_image(),
        class_names=CLASS_NAMES,
        prediction_type=prediction_type,
    )


def _assert_mirror_matches_tensors(detections) -> None:
    xyxy = detections.xyxy.detach().cpu().numpy()
    class_id = detections.class_id.detach().cpu().numpy()
    confidence = detections.confidence.detach().cpu().numpy()
    assert detections.bboxes_metadata is not None
    for index, entry in enumerate(detections.bboxes_metadata):
        assert entry[HOST_XYXY_KEY] == pytest.approx(xyxy[index].tolist(), abs=0)
        assert entry[HOST_CLASS_ID_KEY] == int(class_id[index])
        assert entry[HOST_CONFIDENCE_KEY] == float(confidence[index])
        assert isinstance(entry[HOST_XYXY_KEY], list)
        assert len(entry[HOST_XYXY_KEY]) == 4
        assert all(isinstance(value, float) for value in entry[HOST_XYXY_KEY])
        assert isinstance(entry[HOST_CLASS_ID_KEY], int)
        assert isinstance(entry[HOST_CONFIDENCE_KEY], float)


# --------------------------------------------------------------------------- #
# attach: mirror presence and values per prediction shape
# --------------------------------------------------------------------------- #


def test_attach_writes_host_mirror_for_object_detection() -> None:
    # when
    detections = _attach(_od_detections(3))

    # then
    _assert_mirror_matches_tensors(detections)
    for entry in detections.bboxes_metadata:
        assert DETECTION_ID_KEY in entry


def test_attach_writes_host_mirror_for_instance_segmentation() -> None:
    # when
    detections = _attach(_is_detections(2), prediction_type="instance-segmentation")

    # then
    _assert_mirror_matches_tensors(detections)


def test_attach_writes_host_mirror_for_keypoint_bbox_component() -> None:
    # given - the keypoint-detection blocks attach on the bbox component of the
    # (KeyPoints, Detections) tuple, then add per-box keypoint payload keys.
    detections = _attach(_od_detections(2), prediction_type="keypoint-detection")
    key_points = KeyPoints(
        xy=torch.zeros((2, 4, 2), dtype=torch.float32),
        class_id=detections.class_id.clone(),
        confidence=torch.zeros((2, 4), dtype=torch.float32),
        image_metadata=detections.image_metadata,
    )
    for entry in detections.bboxes_metadata:
        entry["keypoints_xy"] = [[1.0, 2.0]]

    # then - the mirror survives the payload write and the tuple keeps it
    prediction = (key_points, detections)
    _assert_mirror_matches_tensors(prediction[1])


def test_attach_preserves_existing_keys_and_overwrites_stale_mirror() -> None:
    # given - metadata carrying a model-set key and a STALE mirror
    existing = [
        {"text": "reading", HOST_XYXY_KEY: [0.0, 0.0, 0.0, 0.0]},
        {"text": "glasses", HOST_CLASS_ID_KEY: 999},
    ]
    detections = _od_detections(2, existing_metadata=existing)

    # when
    detections = _attach(detections)

    # then - custom keys preserved, mirror refreshed from the tensors
    assert [entry["text"] for entry in detections.bboxes_metadata] == [
        "reading",
        "glasses",
    ]
    _assert_mirror_matches_tensors(detections)


def test_attach_empty_prediction_has_no_metadata() -> None:
    # when
    detections = _attach(_od_detections(0))

    # then
    assert detections.bboxes_metadata is None


# --------------------------------------------------------------------------- #
# attach: batched implementation (no per-box .item() loops)
# --------------------------------------------------------------------------- #


def _attach_op_names(n: int):
    detections = _od_detections(n)
    audit = _OpAudit()
    with audit:
        attach_native_detection_metadata(
            detections=detections,
            image=_image(),
            class_names=CLASS_NAMES,
            prediction_type="object-detection",
        )
    return audit.op_names()


def test_attach_reads_are_batched_not_per_box() -> None:
    # when
    ops_small = _attach_op_names(2)
    ops_large = _attach_op_names(16)

    # then - per-box `.item()` would dispatch `_local_scalar_dense` N times and
    # scale the op count with N; the batched implementation is N-independent.
    for ops in (ops_small, ops_large):
        assert not any("_local_scalar_dense" in name for name in ops)
        assert not any("nonzero" in name for name in ops)
        assert sum("_to_copy" in name for name in ops) <= 3
    assert ops_small == ops_large


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_attach_makes_exactly_one_transfer_per_tensor_cross_device() -> None:
    # given
    detections = _od_detections(4)
    detections.xyxy = detections.xyxy.to("mps")
    detections.class_id = detections.class_id.to("mps")
    detections.confidence = detections.confidence.to("mps")

    # when
    audit = _OpAudit()
    with audit:
        attach_native_detection_metadata(
            detections=detections,
            image=_image(),
            class_names=CLASS_NAMES,
            prediction_type="object-detection",
        )

    # then - one device->host copy per tensor, nothing per box
    assert sum("_to_copy" in name for name in audit.op_names()) == 3
    _assert_mirror_matches_tensors(detections)


# --------------------------------------------------------------------------- #
# read_host_mirror / strip_host_mirror_metadata
# --------------------------------------------------------------------------- #


def test_read_host_mirror_assembles_arrays_matching_tensor_reads() -> None:
    # given
    detections = _attach(_od_detections(3))

    # when
    mirror = read_host_mirror(detections.bboxes_metadata, 3)

    # then
    assert mirror is not None
    xyxy, class_id, confidence = mirror
    expected_xyxy = detections.xyxy.detach().cpu().numpy().astype(np.float32)
    expected_class_id = detections.class_id.detach().cpu().numpy().astype(int)
    expected_confidence = (
        detections.confidence.detach().cpu().numpy().astype(np.float32)
    )
    assert np.array_equal(xyxy, expected_xyxy) and xyxy.dtype == expected_xyxy.dtype
    assert (
        np.array_equal(class_id, expected_class_id)
        and class_id.dtype == expected_class_id.dtype
    )
    assert (
        np.array_equal(confidence, expected_confidence)
        and confidence.dtype == expected_confidence.dtype
    )


@pytest.mark.parametrize("missing_key", HOST_MIRROR_KEYS)
def test_read_host_mirror_requires_every_box_to_carry_all_keys(missing_key) -> None:
    # given
    detections = _attach(_od_detections(3))
    del detections.bboxes_metadata[1][missing_key]

    # when / then
    assert read_host_mirror(detections.bboxes_metadata, 3) is None


def test_read_host_mirror_rejects_row_count_mismatch_and_none() -> None:
    detections = _attach(_od_detections(3))
    assert read_host_mirror(detections.bboxes_metadata, 2) is None
    assert read_host_mirror(None, 3) is None
    assert read_host_mirror([], 0) is None


def test_strip_host_mirror_metadata_removes_keys_without_mutating_input() -> None:
    # given
    detections = _attach(_od_detections(2))
    original_entries = detections.bboxes_metadata

    # when
    stripped = strip_host_mirror_metadata(original_entries)

    # then
    assert stripped is not None and len(stripped) == 2
    for entry in stripped:
        assert not any(key in entry for key in HOST_MIRROR_KEYS)
        assert DETECTION_ID_KEY in entry
    # caller-shared dicts are untouched
    for entry in original_entries:
        assert all(key in entry for key in HOST_MIRROR_KEYS)
    assert strip_host_mirror_metadata(None) is None
    # mirror-free entries are reused as-is
    plain = [{"a": 1}]
    assert strip_host_mirror_metadata(plain)[0] is plain[0]


# --------------------------------------------------------------------------- #
# staleness contract: root-coordinate shift drops the mirror
# --------------------------------------------------------------------------- #


def test_root_coordinate_shift_drops_host_mirror() -> None:
    # given - a crop-anchored prediction with a fresh mirror
    detections = _attach(_od_detections(2))
    detections.image_metadata = {
        CLASS_NAMES_KEY: CLASS_NAMES,
        PREDICTION_TYPE_KEY: "object-detection",
        IMAGE_DIMENSIONS_KEY: [48, 64],
        PARENT_ID_KEY: "crop-1",
        PARENT_COORDINATES_KEY: [100, 50],
        PARENT_DIMENSIONS_KEY: [480, 640],
        ROOT_PARENT_ID_KEY: "root",
        ROOT_PARENT_COORDINATES_KEY: [100, 50],
        ROOT_PARENT_DIMENSIONS_KEY: [480, 640],
    }

    # when
    shifted = native_detections_to_root_coordinates(detections)

    # then - xyxy moved, mirror dropped (not stale-carried)
    assert not torch.equal(shifted.xyxy, detections.xyxy)
    for entry in shifted.bboxes_metadata:
        assert not any(key in entry for key in HOST_MIRROR_KEYS)
        assert DETECTION_ID_KEY in entry


# --------------------------------------------------------------------------- #
# wire format: the mirror never reaches serialized output
# --------------------------------------------------------------------------- #


def _assert_no_mirror_key_anywhere(payload) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert key not in HOST_MIRROR_KEYS
            _assert_no_mirror_key_anywhere(value)
    elif isinstance(payload, list):
        for item in payload:
            _assert_no_mirror_key_anywhere(item)


def test_wire_serialization_is_byte_identical_with_and_without_mirror() -> None:
    # given - the same prediction with and without the mirror (identical
    # detection ids, tensors shared)
    mirrored = _attach(_od_detections(3))
    stripped = Detections(
        xyxy=mirrored.xyxy,
        class_id=mirrored.class_id,
        confidence=mirrored.confidence,
        image_metadata=mirrored.image_metadata,
        bboxes_metadata=strip_host_mirror_metadata(mirrored.bboxes_metadata),
    )

    # when
    serialized_mirrored = serialise_sv_detections(mirrored)
    serialized_stripped = serialise_sv_detections(stripped)

    # then - byte-identical JSON, and no mirror key anywhere in the payload
    assert json.dumps(serialized_mirrored) == json.dumps(serialized_stripped)
    _assert_no_mirror_key_anywhere(serialized_mirrored)


def test_wire_serialization_is_byte_identical_for_instance_segmentation() -> None:
    # given
    mirrored = _attach(_is_detections(2), prediction_type="instance-segmentation")
    stripped = InstanceDetections(
        xyxy=mirrored.xyxy,
        class_id=mirrored.class_id,
        confidence=mirrored.confidence,
        mask=mirrored.mask,
        image_metadata=mirrored.image_metadata,
        bboxes_metadata=strip_host_mirror_metadata(mirrored.bboxes_metadata),
    )

    # when
    serialized_mirrored = serialise_sv_detections(mirrored)
    serialized_stripped = serialise_sv_detections(stripped)

    # then
    assert json.dumps(serialized_mirrored) == json.dumps(serialized_stripped)
    _assert_no_mirror_key_anywhere(serialized_mirrored)


def test_deserializer_does_not_resurrect_the_mirror() -> None:
    # given - wire payload produced from a mirrored prediction
    serialized = serialise_sv_detections(_attach(_od_detections(3)))

    # when
    deserialized = deserialize_detections_kind(
        parameter="predictions", detections=serialized
    )

    # then - a deserialized prediction is mirror-less (consumers fall back to
    # tensor reads on it)
    assert deserialized.bboxes_metadata is not None
    for entry in deserialized.bboxes_metadata:
        assert not any(key in entry for key in HOST_MIRROR_KEYS)
