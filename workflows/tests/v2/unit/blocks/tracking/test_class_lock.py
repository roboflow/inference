"""Track class lock: voting, switching, re-attachment, purge and preservation."""

from datetime import datetime

import numpy as np
import pytest
import torch
from pycocotools import mask as coco_mask
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.tracking.base import TrackerCapacityError
from roboflow_workflows.v2.blocks.tracking.track_class_lock.v1 import CLASS_LOCKED_KEY, DEFAULT_SOURCE, TrackClassLockBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

NAMES = {0: "car", 1: "truck", 2: "bus"}
RULES = dict(
    min_votes=2,
    vote_confidence=0.8,
    lead_margin=1,
    switch_after=2,
    state_ttl=300,
    reattach_window=30,
    reattach_iou=0.3,
)


def _image(source="a"):
    image = ImageData.from_tensor(
        torch.zeros((3, 40, 60), dtype=torch.uint8),
        video_metadata=VideoMetadata(
            video_identifier=source,
            fps=30,
            frame_number=0,
            frame_timestamp=datetime(2026, 10, 13),
        ),
    )

    return image


def _rows(*rows, names=NAMES):
    """Detections from (tracker_id, class_id, confidence[, x1]) tuples."""
    xs = [row[3] if len(row) > 3 else 20.0 * index for index, row in enumerate(rows)]
    detections = Detections(
        xyxy=torch.tensor([[x, 0.0, x + 10.0, 10.0] for x in xs]).reshape(-1, 4),
        class_id=torch.tensor([row[1] for row in rows], dtype=torch.int64),
        confidence=torch.tensor([row[2] for row in rows], dtype=torch.float32),
        image_metadata={"class_names": dict(names)} if names is not None else {},
        bboxes_metadata=[
            {} if row[0] is None else {"tracker_id": row[0]} for row in rows
        ]
        or None,
    )

    return detections


def _lock(block, prediction, source="a", **overrides):
    output = block.run(
        image=_image(source), predictions=prediction, **{**RULES, **overrides}
    )["tracked_detections"]

    return output


def _summary(output):
    rows = output.bboxes_metadata or []
    summary = [
        (row.get("class"), row[CLASS_LOCKED_KEY], class_id, round(confidence, 4))
        for row, class_id, confidence in zip(
            rows, output.class_id.tolist(), output.confidence.tolist()
        )
    ]

    return summary


def test_lock_holds_through_a_broken_streak_and_switches_after_a_full_one():
    block = TrackClassLockBlock()
    said = [0, 0, 1, 0, 1, 1]

    outputs = [_summary(_lock(block, _rows((1, cls, 0.9))))[0] for cls in said]

    assert outputs == [
        ("car", False, 0, 0.9),
        ("car", True, 0, 0.9),
        ("car", True, 0, 0.9),
        ("car", True, 0, 0.9),
        ("car", True, 0, 0.9),
        ("truck", True, 1, 0.9),
    ]


def test_locked_confidence_is_mean_of_counted_votes_capped_at_one():
    block = TrackClassLockBlock()
    _lock(block, _rows((1, 0, 0.8)))
    locked = _summary(_lock(block, _rows((1, 0, 1.0))))
    # A later low-confidence frame reports the vote mean, not its own value.
    later = _summary(_lock(block, _rows((1, 0, 0.1))))

    assert locked == [("car", True, 0, 0.9)]
    assert later == [("car", True, 0, 0.9)]


def test_contested_votes_wait_for_the_lead_margin():
    block = TrackClassLockBlock()
    said = [0, 1, 0, 1, 0]

    locks = [
        _summary(_lock(block, _rows((1, cls, 0.9)), lead_margin=2))[0][1]
        for cls in said
    ]

    # 3 car vs 2 truck leads by only 1 after frame 5.
    assert locks == [False] * 5
    assert _summary(_lock(block, _rows((1, 0, 0.9)), lead_margin=2))[0][:2] == (
        "car",
        True,
    )


def test_low_confidence_frames_neither_vote_nor_continue_a_streak():
    block = TrackClassLockBlock()
    for cls, confidence in [(0, 0.5), (0, 0.5), (0, 0.9)]:
        unlocked = _summary(_lock(block, _rows((1, cls, confidence))))
    _lock(block, _rows((1, 0, 0.9)))
    _lock(block, _rows((1, 1, 0.9)))
    _lock(block, _rows((1, 1, 0.5)))
    after_break = _summary(_lock(block, _rows((1, 1, 0.9))))

    assert unlocked == [("car", False, 0, 0.9)]
    # The 0.5 frame reset the truck streak, so one more frame is not enough.
    assert after_break == [("car", True, 0, 0.9)]


def test_new_id_inherits_a_lost_lock_only_with_overlap_inside_the_window():
    block = TrackClassLockBlock()
    for _ in range(2):
        _lock(block, _rows((1, 0, 0.9, 0.0)))
    _lock(block, _rows())  # id 1 lost at frame 3

    far = _summary(_lock(block, _rows((7, 1, 0.9, 40.0))))
    near = _summary(_lock(block, _rows((8, 1, 0.9, 2.0))))

    assert far == [("truck", False, 1, 0.9)]
    assert near == [("car", True, 0, 0.9)]


def test_reattach_window_bounds_the_gap_and_zero_disables_it():
    late = TrackClassLockBlock()
    disabled = TrackClassLockBlock()
    for block in (late, disabled):
        for _ in range(2):
            _lock(block, _rows((1, 0, 0.9, 0.0)))
    for _ in range(3):
        _lock(late, _rows(), reattach_window=2)
    _lock(disabled, _rows(), reattach_window=0)

    after_window = _lock(late, _rows((9, 1, 0.9, 0.0)), reattach_window=2)
    without = _lock(disabled, _rows((9, 1, 0.9, 0.0)), reattach_window=0)

    assert _summary(after_window)[0][:2] == ("truck", False)
    assert _summary(without)[0][:2] == ("truck", False)


def test_state_ttl_purges_unseen_tracks_and_empty_frames_count():
    block = TrackClassLockBlock()
    for _ in range(2):
        _lock(block, _rows((1, 0, 0.9)), state_ttl=2, reattach_window=0)
    for _ in range(3):
        _lock(block, _rows(), state_ttl=2, reattach_window=0)

    returned = _summary(
        _lock(block, _rows((1, 1, 0.9)), state_ttl=2, reattach_window=0)
    )

    assert returned == [("truck", False, 1, 0.9)]


def test_unnamed_rows_vote_under_their_class_id_string():
    block = TrackClassLockBlock()
    for _ in range(2):
        output = _lock(block, _rows((1, 3, 0.9), names=None))

    assert _summary(output) == [("3", True, 3, 0.9)]
    assert output.image_metadata["class_names"] == {3: "3"}


def test_rows_without_tracker_id_pass_unlocked_but_untracked_input_fails():
    block = TrackClassLockBlock()
    for _ in range(2):
        output = _lock(block, _rows((1, 0, 0.9), (None, 1, 0.9)))

    assert _summary(output) == [("car", True, 0, 0.9), ("truck", False, 1, 0.9)]
    with pytest.raises(ValueError, match="no row has a tracker_id"):
        _lock(block, _rows((None, 0, 0.9)))


def test_parameters_apply_from_the_call_that_passes_them():
    block = TrackClassLockBlock()
    first = _lock(block, _rows((1, 0, 0.9)), min_votes=3)
    second = _lock(block, _rows((1, 0, 0.9)), min_votes=3)
    lowered = _lock(block, _rows((1, 0, 0.9)), min_votes=1, vote_confidence=0.95)

    assert [_summary(o)[0][1] for o in (first, second, lowered)] == [
        False,
        False,
        True,
    ]


def test_interleaved_sources_vote_independently():
    block = TrackClassLockBlock()
    for _ in range(2):
        a = _lock(block, _rows((1, 0, 0.9)), source="a")
        b = _lock(block, _rows((1, 2, 0.9)), source="b")

    assert _summary(a) == [("car", True, 0, 0.9)]
    assert _summary(b) == [("bus", True, 2, 0.9)]


def test_capacity_rejects_a_new_source_without_touching_existing_ones():
    block = TrackClassLockBlock()
    _lock(block, _rows((1, 0, 0.9)), max_sources=1)

    with pytest.raises(TrackerCapacityError, match="max_sources=1"):
        _lock(block, _rows((1, 0, 0.9)), source="b", max_sources=1)

    assert _summary(_lock(block, _rows((1, 0, 0.9)), max_sources=1))[0][1] is True
    assert list(block._sources) == ["a"]


def test_plain_call_without_video_identifier_uses_the_default_source():
    block = TrackClassLockBlock()
    for _ in range(2):
        locked = _lock(block, _rows((1, 1, 0.9)), source="")

    assert list(block._sources) == [DEFAULT_SOURCE]
    assert _summary(locked) == [("truck", True, 1, 0.9)]


def test_other_context_failures_are_not_taken_for_a_plain_call(monkeypatch):
    def broken(self):
        raise RuntimeError("context broke")

    monkeypatch.setattr(TrackClassLockBlock, "execution_context", property(broken))

    with pytest.raises(RuntimeError, match="context broke"):
        _lock(TrackClassLockBlock(), _rows((1, 0, 0.9)), source="")


def test_reattach_window_above_state_ttl_fails_in_run():
    with pytest.raises(ValueError, match="must not exceed state_ttl"):
        _lock(TrackClassLockBlock(), _rows((1, 0, 0.9)), reattach_window=5, state_ttl=4)


def _family(name):
    boxes = with_row_identity(
        Detections(
            xyxy=torch.tensor([[0.0, 0, 10, 10], [20, 0, 30, 10]]),
            class_id=torch.tensor([0, 1], dtype=torch.int32),
            confidence=torch.tensor([0.9, 0.95], dtype=torch.float16),
            image_metadata={"class_names": dict(NAMES), "parent_id": "frame"},
            bboxes_metadata=[
                {"tracker_id": 1, "_host_class_id": 0, "custom": "x"},
                {"tracker_id": 2, "custom": "y"},
            ],
        ),
        namespace="rows",
    )
    if name == "boxes":
        return boxes

    masks = torch.zeros((2, 40, 60), dtype=torch.bool)
    masks[0, :10, :10] = True
    masks[1, :10, 20:30] = True
    if name == "keypoints":
        points = KeyPoints(
            xy=boxes.xyxy[:, :2].unsqueeze(1),
            confidence=boxes.confidence.float().unsqueeze(1),
            class_id=boxes.class_id,
            key_points_metadata=[{"point": 0}, {"point": 1}],
        )
        return points, boxes

    if name == "grid":
        geometry = MaskGridGeometry.identity((20, 30))
        return SegmentationView(
            detections=boxes,
            scores=masks[:, ::2, ::2].float(),
            score_type="logits",
            threshold=0.0,
            geometry=geometry,
        )

    mask = masks
    if name == "rle":
        mask = InstancesRLEMasks(
            image_size=(40, 60),
            masks=[
                coco_mask.encode(np.asfortranarray(m.numpy().astype(np.uint8)))[
                    "counts"
                ]
                for m in masks
            ],
        )
    dense = InstanceDetections(
        xyxy=boxes.xyxy,
        class_id=boxes.class_id,
        confidence=boxes.confidence,
        mask=mask,
        image_metadata=boxes.image_metadata,
        bboxes_metadata=boxes.bboxes_metadata,
    )
    if name == "dense_view":
        return SegmentationView.from_dense(dense)

    return dense


def _boxes(prediction):
    if isinstance(prediction, SegmentationView):
        return prediction.detections

    return prediction[1] if isinstance(prediction, tuple) else prediction


@pytest.mark.parametrize(
    "family", ["boxes", "dense", "rle", "grid", "dense_view", "keypoints"]
)
def test_every_family_keeps_rows_masks_identity_and_input(family, monkeypatch):
    def forbidden(self):
        raise AssertionError("full_res must not run")

    monkeypatch.setattr(SegmentationView, "full_res", forbidden)
    block = TrackClassLockBlock()
    _lock(block, _family(family), min_votes=1, lead_margin=0)
    prediction = _family(family)
    before = [dict(row) for row in _boxes(prediction).bboxes_metadata]

    output = _lock(block, prediction, min_votes=1, lead_margin=0)
    boxes = _boxes(output)

    assert type(output) is type(prediction)
    assert boxes.xyxy is _boxes(prediction).xyxy
    assert boxes.class_id.dtype == torch.int32
    assert boxes.confidence.dtype == torch.float16
    assert torch.allclose(
        boxes.confidence.float(), torch.tensor([0.9, 0.95]), atol=1e-3
    )
    assert [row[CLASS_LOCKED_KEY] for row in boxes.bboxes_metadata] == [True, True]
    assert [row["custom"] for row in boxes.bboxes_metadata] == ["x", "y"]
    assert "_host_class_id" not in boxes.bboxes_metadata[0]
    assert row_record(boxes).namespace == "rows"
    assert _boxes(prediction).bboxes_metadata == before
    if isinstance(prediction, SegmentationView):
        assert output.scores is prediction.scores
        assert output.materialization_count == 0
    if isinstance(prediction, InstanceDetections):
        assert output.mask is prediction.mask
    if isinstance(prediction, tuple):
        assert output[0] is prediction[0]


def test_empty_prediction_is_returned_as_given_and_ages_tracks():
    block = TrackClassLockBlock()
    empty = _rows()

    assert _lock(block, empty) is empty
    assert block._sources["a"].frame == 1
