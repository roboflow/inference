"""Parity tests for the tensor-native ByteTracker blocks (v1/v2/v3).

These blocks used to do a genuine GPU->CPU->GPU round trip: the surviving
detections were rebuilt by re-uploading ``sv_tracked.xyxy`` / ``class_id`` /
``confidence`` from numpy back to the device.  ByteTrack only *filters and
reorders* detections (it never mutates coordinates), so the re-upload is pure
waste - the surviving rows can be sliced straight out of the original device
tensors with ``take_prediction_by_indices`` (masks included).

The oracle embedded here (``_oracle_track``) is the ORIGINAL implementation: it
D2H-materialises the input, tags rows with a positional index, runs ByteTrack,
then re-uploads the numpy boxes.  Each test drives the real (new) block and the
oracle over the *same* multi-frame sequence, each with its own fresh
deterministic tracker, and asserts:

* identical surviving rows (by input ``detection_id``) in identical order,
* identical box / class / confidence tensors,
* identical ``tracker_id`` values written into ``bboxes_metadata``,
* for instance-segmentation inputs, masks that follow the SAME index selection
  (the round-tripping oracle drops masks, so masks are checked against the input
  masks sliced by the surviving rows),
* for v3, identical ``new_instances`` / ``already_seen_instances`` splits.

Torch runs on CPU here, which is enough to exercise the selection logic; the
device-transfer win is orthogonal to correctness.
"""

import datetime
from typing import List, Optional, Tuple, Union

import numpy as np
import supervision as sv
import torch

from inference.core.workflows.core_steps.transformations.byte_tracker.v1_tensor import (
    ByteTrackerBlockV1,
)
from inference.core.workflows.core_steps.transformations.byte_tracker.v2_tensor import (
    ByteTrackerBlockV2,
)
from inference.core.workflows.core_steps.transformations.byte_tracker.v3_tensor import (
    ByteTrackerBlockV3,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

_ORACLE_INDEX_KEY = "__oracle_input_index__"


# --------------------------------------------------------------------------- #
# Oracle (the original round-tripping implementation)
# --------------------------------------------------------------------------- #
def _oracle_track(
    tracker: sv.ByteTrack,
    detections: Union[Detections, InstanceDetections],
) -> Tuple[Detections, List[int], np.ndarray]:
    """The pre-change implementation: D2H, run ByteTrack, re-upload numpy boxes.

    Returns the reconstructed plain ``Detections`` (masks dropped, as the old
    block did) plus the surviving input indices and their tracker ids.
    """
    n = int(detections.xyxy.shape[0])
    sv_input = sv.Detections(
        xyxy=detections.xyxy.detach().to("cpu").numpy(),
        class_id=detections.class_id.detach().to("cpu").numpy(),
        confidence=detections.confidence.detach().to("cpu").numpy(),
        data={_ORACLE_INDEX_KEY: np.arange(n, dtype=np.int64)},
    )
    sv_tracked = tracker.update_with_detections(sv_input)
    kept_indices = (
        sv_tracked.data.get(_ORACLE_INDEX_KEY, np.empty((0,), dtype=np.int64))
        if sv_tracked.data
        else np.empty((0,), dtype=np.int64)
    )
    tracker_ids = (
        sv_tracked.tracker_id
        if sv_tracked.tracker_id is not None
        else np.full(len(sv_tracked), -1, dtype=np.int64)
    )
    original_meta = detections.bboxes_metadata or [{} for _ in range(n)]
    new_bboxes_meta: List[dict] = []
    for new_i, orig_i in enumerate(kept_indices):
        base = dict(original_meta[int(orig_i)] or {})
        base["tracker_id"] = int(tracker_ids[new_i])
        new_bboxes_meta.append(base)
    device = detections.xyxy.device
    out = Detections(
        xyxy=torch.from_numpy(sv_tracked.xyxy).to(
            device=device, dtype=detections.xyxy.dtype
        ),
        class_id=torch.from_numpy(sv_tracked.class_id).to(
            device=device, dtype=detections.class_id.dtype
        ),
        confidence=torch.from_numpy(sv_tracked.confidence).to(
            device=device, dtype=detections.confidence.dtype
        ),
        image_metadata=detections.image_metadata,
        bboxes_metadata=new_bboxes_meta if new_bboxes_meta else None,
    )
    return out, [int(i) for i in kept_indices.tolist()], tracker_ids


class _OracleCache:
    """Independent copy of the v3 InstanceCache new/seen bookkeeping."""

    def __init__(self, size: int) -> None:
        from collections import deque

        size = max(1, size)
        self._order = deque(maxlen=size)
        self._seen = set()

    def record(self, tracker_id: int) -> bool:
        in_cache = tracker_id in self._seen
        if not in_cache:
            while len(self._seen) >= self._order.maxlen:
                self._seen.remove(self._order.popleft())
            self._order.append(tracker_id)
            self._seen.add(tracker_id)
        return in_cache


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _meta(video_id: str, frame_number: int, fps: float = 1.0) -> VideoMetadata:
    return VideoMetadata(
        video_identifier=video_id,
        frame_number=frame_number,
        fps=fps,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )


def _wrap_image(metadata: VideoMetadata) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="root"),
        numpy_image=np.zeros((128, 128, 3), dtype=np.uint8),
        video_metadata=metadata,
    )


def _boxes_metadata(count: int, prefix: str) -> List[dict]:
    return [{"detection_id": f"{prefix}-{i}"} for i in range(count)]


def _object_detections(boxes: List[List[float]], prefix: str) -> Detections:
    n = len(boxes)
    return Detections(
        xyxy=torch.tensor(boxes, dtype=torch.float32),
        class_id=torch.arange(n, dtype=torch.long) % 3,
        confidence=torch.full((n,), 0.9, dtype=torch.float32),
        image_metadata={"class_names": {0: "a", 1: "b", 2: "c"}},
        bboxes_metadata=_boxes_metadata(n, prefix),
    )


def _instance_detections(boxes: List[List[float]], prefix: str) -> InstanceDetections:
    n = len(boxes)
    height, width = 40, 40
    # Give every instance a distinct dense mask so the selection is verifiable.
    mask = torch.zeros((n, height, width), dtype=torch.bool)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1 = max(0, min(width - 1, x1))
        x2 = max(x1 + 1, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(y1 + 1, min(height, y2))
        mask[i, y1:y2, x1:x2] = True
    return InstanceDetections(
        xyxy=torch.tensor(boxes, dtype=torch.float32),
        class_id=torch.arange(n, dtype=torch.long) % 3,
        confidence=torch.full((n,), 0.9, dtype=torch.float32),
        mask=mask,
        image_metadata={"class_names": {0: "a", 1: "b", 2: "c"}},
        bboxes_metadata=_boxes_metadata(n, prefix),
    )


def _keypoint_prediction(
    boxes: List[List[float]], prefix: str
) -> Tuple[KeyPoints, Detections]:
    det = _object_detections(boxes, prefix)
    n = len(boxes)
    key_points = KeyPoints(
        xy=torch.zeros((n, 3, 2), dtype=torch.float32),
        class_id=det.class_id.clone(),
        confidence=torch.ones((n, 3), dtype=torch.float32),
        image_metadata={"class_names": {0: "a", 1: "b", 2: "c"}},
    )
    return key_points, det


# A 3-frame sequence: 4 objects persist and drift, one (last) leaves after
# frame 1, and a brand-new object enters on frame 3 (entering / leaving /
# persisting all exercised).
_FRAME_BOXES: List[List[List[float]]] = [
    [[10, 10, 20, 20], [21, 10, 31, 20], [31, 10, 41, 20], [2, 2, 8, 8]],
    [[12, 10, 22, 20], [23, 10, 33, 20], [33, 10, 43, 20]],
    [[14, 10, 24, 20], [25, 10, 35, 20], [35, 10, 45, 20], [1, 30, 9, 38]],
]


def _det_ids(detections: Union[Detections, InstanceDetections]) -> List[str]:
    if detections.bboxes_metadata is None:
        return []
    return [m.get("detection_id") for m in detections.bboxes_metadata]


def _tracker_ids(detections: Union[Detections, InstanceDetections]) -> List[int]:
    if detections.bboxes_metadata is None:
        return []
    return [int(m["tracker_id"]) for m in detections.bboxes_metadata]


def _assert_box_parity(
    produced: Union[Detections, InstanceDetections],
    oracle: Detections,
    input_detections: Union[Detections, InstanceDetections],
    kept_indices: List[int],
) -> None:
    assert torch.equal(produced.xyxy, oracle.xyxy)
    assert torch.equal(produced.class_id, oracle.class_id)
    assert torch.equal(produced.confidence, oracle.confidence)
    assert _tracker_ids(produced) == _tracker_ids(oracle)
    expected_ids = [_det_ids(input_detections)[i] for i in kept_indices]
    assert _det_ids(produced) == expected_ids
    if isinstance(input_detections, InstanceDetections):
        # Old block dropped masks; the new block must carry them, sliced by the
        # SAME surviving rows.
        assert isinstance(produced, InstanceDetections)
        selector = torch.tensor(kept_indices, dtype=torch.long)
        assert torch.equal(produced.mask, input_detections.mask[selector])
    else:
        assert isinstance(produced, Detections)
        assert not isinstance(produced, InstanceDetections)


# --------------------------------------------------------------------------- #
# v1
# --------------------------------------------------------------------------- #
def _run_v1(builder) -> None:
    block = ByteTrackerBlockV1()
    oracle_tracker = sv.ByteTrack(frame_rate=1)
    for frame_number, boxes in enumerate(_FRAME_BOXES):
        detections = builder(boxes, prefix=f"f{frame_number}")
        produced = block.run(
            metadata=_meta("vid", frame_number, fps=1),
            detections=detections,
        )["tracked_detections"]
        oracle_out, kept, _ = _oracle_track(oracle_tracker, detections)
        _assert_box_parity(produced, oracle_out, detections, kept)


def test_v1_object_detection_parity() -> None:
    _run_v1(_object_detections)


def test_v1_instance_segmentation_mask_selection() -> None:
    _run_v1(_instance_detections)


def test_v1_all_surviving_frame_reuses_input_tensor_no_reupload() -> None:
    # Frame 1: every high-confidence detection is activated and matched, so all
    # rows survive in input order -> identity selection returns the ORIGINAL
    # device tensor object (proves no re-upload / no copy).
    block = ByteTrackerBlockV1()
    detections = _object_detections(_FRAME_BOXES[0], prefix="f0")
    produced = block.run(
        metadata=_meta("vid", 0, fps=1),
        detections=detections,
    )["tracked_detections"]
    assert produced.xyxy is detections.xyxy
    assert produced.class_id is detections.class_id
    assert produced.confidence is detections.confidence


# --------------------------------------------------------------------------- #
# v2
# --------------------------------------------------------------------------- #
def _run_v2(builder) -> None:
    block = ByteTrackerBlockV2()
    oracle_tracker = sv.ByteTrack(frame_rate=1)
    for frame_number, boxes in enumerate(_FRAME_BOXES):
        detections = builder(boxes, prefix=f"f{frame_number}")
        produced = block.run(
            image=_wrap_image(_meta("vid", frame_number, fps=1)),
            detections=detections,
        )["tracked_detections"]
        oracle_out, kept, _ = _oracle_track(oracle_tracker, detections)
        _assert_box_parity(produced, oracle_out, detections, kept)


def test_v2_object_detection_parity() -> None:
    _run_v2(_object_detections)


def test_v2_instance_segmentation_mask_selection() -> None:
    _run_v2(_instance_detections)


# --------------------------------------------------------------------------- #
# v3
# --------------------------------------------------------------------------- #
def _run_v3(builder, keypoint_input: bool = False) -> None:
    block = ByteTrackerBlockV3()
    oracle_tracker = sv.ByteTrack(frame_rate=1)
    oracle_cache = _OracleCache(size=16384)
    for frame_number, boxes in enumerate(_FRAME_BOXES):
        raw = builder(boxes, prefix=f"f{frame_number}")
        # For keypoint input the bbox Detections drives tracking / parity.
        bbox_input = raw[1] if keypoint_input else raw
        result = block.run(
            image=_wrap_image(_meta("vid", frame_number, fps=1)),
            detections=raw,
        )
        oracle_out, kept, tracker_ids = _oracle_track(oracle_tracker, bbox_input)
        _assert_box_parity(result["tracked_detections"], oracle_out, bbox_input, kept)
        # new / already-seen split parity.
        not_seen, seen = [], []
        for position, tid in enumerate(tracker_ids.tolist()):
            (seen if oracle_cache.record(int(tid)) else not_seen).append(position)
        assert _tracker_ids(result["new_instances"]) == [
            _tracker_ids(oracle_out)[i] for i in not_seen
        ]
        assert _tracker_ids(result["already_seen_instances"]) == [
            _tracker_ids(oracle_out)[i] for i in seen
        ]
        assert _det_ids(result["new_instances"]) == [
            _det_ids(oracle_out)[i] for i in not_seen
        ]
        assert _det_ids(result["already_seen_instances"]) == [
            _det_ids(oracle_out)[i] for i in seen
        ]


def test_v3_object_detection_parity_and_splits() -> None:
    _run_v3(_object_detections)


def test_v3_instance_segmentation_mask_selection() -> None:
    _run_v3(_instance_detections)


def test_v3_keypoint_input_tracks_on_bbox_and_drops_keypoints() -> None:
    _run_v3(_keypoint_prediction, keypoint_input=True)


def test_v3_first_frame_all_new_none_seen() -> None:
    block = ByteTrackerBlockV3()
    detections = _object_detections(_FRAME_BOXES[0], prefix="f0")
    result = block.run(
        image=_wrap_image(_meta("vid", 0, fps=1)),
        detections=detections,
    )
    tracked_ids = _tracker_ids(result["tracked_detections"])
    assert _tracker_ids(result["new_instances"]) == tracked_ids
    assert len(result["already_seen_instances"]) == 0
