import datetime
from copy import deepcopy

import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.transformations.track_class_lock.v1 import (
    MAX_TRACKED_VIDEOS,
    BlockManifest,
    TrackClassLockBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)

CLASS_IDS = {"cat": 0, "dog": 1}

KNOBS = dict(
    min_votes=10,
    vote_confidence=0.8,
    lead_margin=3,
    switch_after=15,
    state_ttl=300,
    reattach_window=30,
    reattach_iou=0.3,
)


def _image(video_identifier: str = "vid_1") -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((64, 64, 3), dtype=np.uint8),
        video_metadata=VideoMetadata(
            video_identifier=video_identifier,
            frame_number=0,
            fps=30,
            frame_timestamp=datetime.datetime.now(tz=datetime.timezone.utc),
        ),
    )


def _frame(
    class_name: str,
    confidence: float,
    tracker_id: int = 7,
    xyxy: tuple = (10.0, 10.0, 50.0, 50.0),
) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([list(xyxy)]),
        confidence=np.array([confidence]),
        class_id=np.array([CLASS_IDS[class_name]]),
        tracker_id=np.array([tracker_id]),
        data={"class_name": np.array([class_name])},
    )


def _empty_frame() -> sv.Detections:
    return sv.Detections.empty()


def test_track_class_lock_validation_when_valid_manifest_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/track_class_lock@v1",
        "name": "class_lock",
        "image": "$inputs.image",
        "detections": "$steps.byte_tracker.tracked_detections",
        "min_votes": 5,
        "vote_confidence": 0.7,
        "lead_margin": 2,
        "switch_after": 10,
        "state_ttl": 100,
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type="roboflow_core/track_class_lock@v1",
        name="class_lock",
        image="$inputs.image",
        detections="$steps.byte_tracker.tracked_detections",
        min_votes=5,
        vote_confidence=0.7,
        lead_margin=2,
        switch_after=10,
        state_ttl=100,
    )


def test_track_class_lock_raises_when_detections_not_tracked() -> None:
    # given
    block = TrackClassLockBlockV1()
    detections = _frame("cat", 0.9)
    detections.tracker_id = None

    # when / then
    with pytest.raises(ValueError):
        block.run(image=_image(), detections=detections, **KNOBS)


def test_track_class_lock_acquires_lock_after_min_votes() -> None:
    # given
    block = TrackClassLockBlockV1()

    # when
    for _ in range(9):
        result = block.run(image=_image(), detections=_frame("cat", 0.9), **KNOBS)
        assert not result["tracked_detections"].data["class_locked"][0]
    result = block.run(image=_image(), detections=_frame("cat", 0.9), **KNOBS)

    # then
    out = result["tracked_detections"]
    assert out.data["class_locked"][0]
    assert out.data["class_name"][0] == "cat"
    assert out.class_id[0] == CLASS_IDS["cat"]


def test_track_class_lock_ignores_low_confidence_votes() -> None:
    # given
    block = TrackClassLockBlockV1()

    # when - 10 frames below vote_confidence must not produce a lock
    for _ in range(10):
        result = block.run(image=_image(), detections=_frame("cat", 0.5), **KNOBS)

    # then
    assert not result["tracked_detections"].data["class_locked"][0]


def test_track_class_lock_relabels_contrary_frames_while_locked() -> None:
    # given
    block = TrackClassLockBlockV1()
    for _ in range(10):
        block.run(image=_image(), detections=_frame("cat", 0.9), **KNOBS)

    # when - a contrary frame arrives while locked
    result = block.run(image=_image(), detections=_frame("dog", 0.95), **KNOBS)

    # then - it is relabelled to the locked class
    out = result["tracked_detections"]
    assert out.data["class_name"][0] == "cat"
    assert out.class_id[0] == CLASS_IDS["cat"]
    assert out.data["class_locked"][0]


def test_track_class_lock_broken_streak_does_not_switch() -> None:
    # given
    block = TrackClassLockBlockV1()
    for _ in range(10):
        block.run(image=_image(), detections=_frame("cat", 0.9), **KNOBS)

    # when - 14 dog frames, interruption, then 14 more (no 15-streak)
    for _ in range(14):
        block.run(image=_image(), detections=_frame("dog", 0.95), **KNOBS)
    block.run(image=_image(), detections=_frame("cat", 0.9), **KNOBS)
    for _ in range(14):
        result = block.run(image=_image(), detections=_frame("dog", 0.95), **KNOBS)

    # then
    assert result["tracked_detections"].data["class_name"][0] == "cat"


def test_track_class_lock_switches_after_consecutive_streak_with_valid_confidence() -> (
    None
):
    # given
    block = TrackClassLockBlockV1()
    for _ in range(10):
        block.run(image=_image(), detections=_frame("cat", 0.9), **KNOBS)
    # broken streak first - its evidence must NOT leak into the switch
    for _ in range(5):
        block.run(image=_image(), detections=_frame("dog", 0.95), **KNOBS)
    block.run(image=_image(), detections=_frame("cat", 0.9), **KNOBS)

    # when - full consecutive streak
    for _ in range(15):
        result = block.run(image=_image(), detections=_frame("dog", 0.95), **KNOBS)

    # then
    out = result["tracked_detections"]
    assert out.data["class_name"][0] == "dog"
    assert out.class_id[0] == CLASS_IDS["dog"]
    # tallies seeded from the streak only: 15 frames @ 0.95 -> mean of 0.95
    assert 0.0 < float(out.confidence[0]) <= 1.0
    assert abs(float(out.confidence[0]) - 0.95) < 1e-9


def test_track_class_lock_relabels_longer_class_name_without_truncation() -> None:
    # given - locked name is longer than the fixed-width numpy dtype of input
    block = TrackClassLockBlockV1()
    long_name_frame = sv.Detections(
        xyxy=np.array([[10.0, 10.0, 50.0, 50.0]]),
        confidence=np.array([0.9]),
        class_id=np.array([2]),
        tracker_id=np.array([7]),
        data={"class_name": np.array(["bicycle"])},
    )
    for _ in range(10):
        block.run(image=_image(), detections=long_name_frame, **KNOBS)

    # when - contrary frame with a SHORTER class name array dtype
    result = block.run(image=_image(), detections=_frame("cat", 0.95), **KNOBS)

    # then - locked name written back in full
    assert result["tracked_detections"].data["class_name"][0] == "bicycle"


def test_track_class_lock_keeps_state_separate_per_video() -> None:
    # given
    block = TrackClassLockBlockV1()
    for _ in range(10):
        block.run(image=_image("vid_1"), detections=_frame("cat", 0.9), **KNOBS)

    # when - same tracker_id in another video has no votes yet
    result = block.run(image=_image("vid_2"), detections=_frame("dog", 0.9), **KNOBS)

    # then
    assert not result["tracked_detections"].data["class_locked"][0]


def test_track_class_lock_tolerates_empty_frames() -> None:
    # given
    block = TrackClassLockBlockV1()

    # when - empty frames (no detections at all) must not raise
    result = block.run(image=_image(), detections=_empty_frame(), **KNOBS)

    # then
    assert len(result["tracked_detections"]) == 0


def test_track_class_lock_reattaches_lock_after_short_detection_gap() -> None:
    # given - lock acquired on tracker id 7, then the detector misses the
    # object for a few frames and the tracker assigns a NEW id on return
    block = TrackClassLockBlockV1()
    for _ in range(10):
        block.run(image=_image(), detections=_frame("cat", 0.9, tracker_id=7), **KNOBS)
    for _ in range(3):
        block.run(image=_image(), detections=_empty_frame(), **KNOBS)

    # when - same location, new tracker id, contrary class prediction
    result = block.run(
        image=_image(), detections=_frame("dog", 0.9, tracker_id=8), **KNOBS
    )

    # then - lock inherited immediately, including the voting history
    out = result["tracked_detections"]
    assert out.data["class_locked"][0]
    assert out.data["class_name"][0] == "cat"
    assert out.class_id[0] == CLASS_IDS["cat"]
    assert abs(float(out.confidence[0]) - 0.9) < 1e-9


def test_track_class_lock_does_not_reattach_beyond_window() -> None:
    # given
    block = TrackClassLockBlockV1()
    for _ in range(10):
        block.run(image=_image(), detections=_frame("cat", 0.9, tracker_id=7), **KNOBS)
    for _ in range(31):
        block.run(image=_image(), detections=_empty_frame(), **KNOBS)

    # when - gap longer than reattach_window (30)
    result = block.run(
        image=_image(), detections=_frame("dog", 0.9, tracker_id=8), **KNOBS
    )

    # then - new track starts voting from scratch
    assert not result["tracked_detections"].data["class_locked"][0]


def test_track_class_lock_does_not_reattach_far_from_lost_track() -> None:
    # given
    block = TrackClassLockBlockV1()
    for _ in range(10):
        block.run(image=_image(), detections=_frame("cat", 0.9, tracker_id=7), **KNOBS)
    block.run(image=_image(), detections=_empty_frame(), **KNOBS)

    # when - new id appears in a non-overlapping location
    result = block.run(
        image=_image(),
        detections=_frame("dog", 0.9, tracker_id=8, xyxy=(200.0, 200.0, 240.0, 240.0)),
        **KNOBS,
    )

    # then
    assert not result["tracked_detections"].data["class_locked"][0]


def test_track_class_lock_does_not_steal_lock_from_active_track() -> None:
    # given - id 7 locked and STILL present in the frame
    block = TrackClassLockBlockV1()
    for _ in range(10):
        block.run(image=_image(), detections=_frame("cat", 0.9, tracker_id=7), **KNOBS)
    both = sv.Detections(
        xyxy=np.array([[10.0, 10.0, 50.0, 50.0], [12.0, 12.0, 52.0, 52.0]]),
        confidence=np.array([0.9, 0.9]),
        class_id=np.array([CLASS_IDS["cat"], CLASS_IDS["dog"]]),
        tracker_id=np.array([7, 9]),
        data={"class_name": np.array(["cat", "dog"])},
    )

    # when - new id 9 overlaps the still-active id 7
    result = block.run(image=_image(), detections=both, **KNOBS)

    # then - id 7 keeps its lock, id 9 starts fresh
    out = result["tracked_detections"]
    assert out.data["class_locked"][0]
    assert out.data["class_name"][0] == "cat"
    assert not out.data["class_locked"][1]


def test_track_class_lock_reattach_disabled_with_zero_window() -> None:
    # given
    knobs = {**KNOBS, "reattach_window": 0}
    block = TrackClassLockBlockV1()
    for _ in range(10):
        block.run(image=_image(), detections=_frame("cat", 0.9, tracker_id=7), **knobs)
    block.run(image=_image(), detections=_empty_frame(), **knobs)

    # when
    result = block.run(
        image=_image(), detections=_frame("dog", 0.9, tracker_id=8), **knobs
    )

    # then
    assert not result["tracked_detections"].data["class_locked"][0]


def test_track_class_lock_bounds_per_video_state() -> None:
    # given - more distinct video identifiers than the cap (e.g. images
    # produced by Dynamic Crop carry a unique identifier per crop)
    block = TrackClassLockBlockV1()

    # when
    for v in range(MAX_TRACKED_VIDEOS + 50):
        block.run(image=_image(f"vid_{v}"), detections=_frame("cat", 0.9), **KNOBS)

    # then - least-recently-seen video state is evicted, recent ones survive
    assert len(block._per_video_state) == MAX_TRACKED_VIDEOS
    assert "vid_0" not in block._per_video_state
    assert f"vid_{MAX_TRACKED_VIDEOS + 49}" in block._per_video_state


def test_track_class_lock_manifest_rejects_zero_switch_after() -> None:
    # given - switch_after=0 would fire a class switch on the very first
    # qualifying challenger frame, bypassing the stability requirement
    data = {
        "type": "roboflow_core/track_class_lock@v1",
        "name": "class_lock",
        "image": "$inputs.image",
        "detections": "$steps.byte_tracker.tracked_detections",
        "switch_after": 0,
    }

    # when / then
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(data)


def test_track_class_lock_manifest_rejects_reattach_window_beyond_state_ttl() -> None:
    # given - tracks are purged after state_ttl frames, so a reattach_window
    # larger than state_ttl can never fire
    data = {
        "type": "roboflow_core/track_class_lock@v1",
        "name": "class_lock",
        "image": "$inputs.image",
        "detections": "$steps.byte_tracker.tracked_detections",
        "state_ttl": 10,
        "reattach_window": 30,
    }

    # when / then
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(data)


def test_track_class_lock_handles_detections_without_confidence() -> None:
    # given - sv.Detections allows confidence=None; votes fall back to 1.0
    block = TrackClassLockBlockV1()
    frame = _frame("cat", 0.9)
    frame.confidence = None

    # when - locking and writing confidence back must not raise
    for _ in range(10):
        result = block.run(image=_image(), detections=deepcopy(frame), **KNOBS)

    # then
    out = result["tracked_detections"]
    assert out.data["class_locked"][0]
    assert abs(float(out.confidence[0]) - 1.0) < 1e-9


def test_track_class_lock_handles_detections_without_class_id() -> None:
    # given - class identity comes from class_name only
    block = TrackClassLockBlockV1()
    frame = _frame("cat", 0.9)
    frame.class_id = None

    # when - voting, locking and relabelling must not raise
    for _ in range(10):
        result = block.run(image=_image(), detections=deepcopy(frame), **KNOBS)

    # then
    out = result["tracked_detections"]
    assert out.data["class_locked"][0]
    assert out.data["class_name"][0] == "cat"
    assert out.class_id is None


def test_track_class_lock_skips_detections_without_any_class_identity() -> None:
    # given - neither class_name nor class_id present
    block = TrackClassLockBlockV1()
    frame = sv.Detections(
        xyxy=np.array([[10.0, 10.0, 50.0, 50.0]]),
        confidence=np.array([0.9]),
        tracker_id=np.array([7]),
    )

    # when
    for _ in range(10):
        result = block.run(image=_image(), detections=deepcopy(frame), **KNOBS)

    # then - never locks, never crashes
    assert not result["tracked_detections"].data["class_locked"][0]


def test_track_class_lock_does_not_mutate_input_detections() -> None:
    # given
    block = TrackClassLockBlockV1()
    for _ in range(10):
        block.run(image=_image(), detections=_frame("cat", 0.9), **KNOBS)
    contrary = _frame("dog", 0.95)

    # when
    block.run(image=_image(), detections=contrary, **KNOBS)

    # then - input object untouched
    assert contrary.data["class_name"][0] == "dog"
    assert "class_locked" not in contrary.data
