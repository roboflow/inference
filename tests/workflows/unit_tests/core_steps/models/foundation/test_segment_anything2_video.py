"""Unit tests for the SAM2 Video Tracker workflow block."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import supervision as sv
from datetime import datetime

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything2_video.v1 import (
    BlockManifest,
    SegmentAnything2VideoBlockV1,
    _extract_box_prompts,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_frame(
    video_id: str = "stream-0",
    frame_number: int = 0,
    shape=(120, 160, 3),
) -> WorkflowImageData:
    image = np.zeros(shape, dtype=np.uint8)
    metadata = VideoMetadata(
        video_identifier=video_id,
        frame_number=frame_number,
        fps=30,
        frame_timestamp=datetime(2024, 1, 1, 0, 0, frame_number % 60),
    )
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=f"{video_id}:{frame_number}"),
        numpy_image=image,
        video_metadata=metadata,
    )


def _make_box_detections(
    xyxy=((10, 20, 80, 90),),
    class_names=("person",),
    class_ids=(0,),
    confidences=(0.95,),
) -> sv.Detections:
    detections = sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(confidences, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )
    detections.data["class_name"] = np.array(class_names, dtype=object)
    detections.data["detection_id"] = np.array(
        [f"det-{i}" for i in range(len(xyxy))], dtype=object
    )
    return detections


class _FakeVideoModel:
    """Drop-in replacement for ``SegmentAnything2Video`` used in tests.

    Records every call so the test can assert the prompt/track pattern,
    and returns deterministic masks + object ids so we can verify the
    block builds correct ``sv.Detections``.
    """

    def __init__(self):
        self.calls = []  # list of ("prompt"/"track", kwargs)
        self.sessions = {}  # video_id -> list of obj_ids currently active
        self._next_obj_id = 0

    def has_session(self, video_id: str) -> bool:
        return video_id in self.sessions

    def reset_session(self, video_id: str) -> None:
        # Only record the reset if we actually had something to reset —
        # makes the recorded call sequence reflect meaningful transitions.
        if video_id in self.sessions:
            self.sessions.pop(video_id)
            self.calls.append(("reset", {"video_id": video_id}))

    def prompt_and_track(
        self,
        video_id: str,
        frame: np.ndarray,
        boxes_xyxy,
        clear_old_prompts: bool = True,
    ):
        self.calls.append(
            (
                "prompt",
                {
                    "video_id": video_id,
                    "num_boxes": len(boxes_xyxy),
                    "clear_old_prompts": clear_old_prompts,
                },
            )
        )
        n = max(len(boxes_xyxy), 0)
        obj_ids = np.arange(n, dtype=np.int64)
        self.sessions[video_id] = list(obj_ids)
        h, w = frame.shape[:2]
        # Emit one full-frame mask per prompted box, with a distinct pixel
        # in each so _masks_to_sv_detections doesn't drop any as empty.
        masks = np.zeros((n, h, w), dtype=bool)
        for i in range(n):
            masks[i, i, i] = True
        return masks, obj_ids

    def track(self, video_id: str, frame: np.ndarray):
        self.calls.append(("track", {"video_id": video_id}))
        obj_ids = np.asarray(self.sessions[video_id], dtype=np.int64)
        h, w = frame.shape[:2]
        masks = np.zeros((len(obj_ids), h, w), dtype=bool)
        for i in range(len(obj_ids)):
            masks[i, i, i] = True
        return masks, obj_ids


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------


def test_manifest_parses_valid_config():
    data = {
        "type": "roboflow_core/segment_anything_2_video@v1",
        "name": "sam2_video_step",
        "images": "$inputs.image",
        "version": "hiera_tiny",
        "prompt_mode": "first_frame",
    }
    manifest = BlockManifest.model_validate(data)
    assert manifest.type == "roboflow_core/segment_anything_2_video@v1"
    assert manifest.version == "hiera_tiny"
    assert manifest.prompt_mode == "first_frame"
    assert manifest.prompt_interval == 30  # default


def test_manifest_rejects_unknown_prompt_mode():
    data = {
        "type": "roboflow_core/segment_anything_2_video@v1",
        "name": "sam2_video_step",
        "images": "$inputs.image",
        "prompt_mode": "not_a_mode",
    }
    with pytest.raises(Exception):
        BlockManifest.model_validate(data)


# ---------------------------------------------------------------------------
# Remote execution mode is not supported
# ---------------------------------------------------------------------------


def test_block_rejects_remote_execution_mode():
    """Remote mode can't maintain per-video session state across frames."""
    with pytest.raises(NotImplementedError, match="LOCAL workflow step execution"):
        SegmentAnything2VideoBlockV1(
            model_manager=MagicMock(),
            api_key=None,
            step_execution_mode=StepExecutionMode.REMOTE,
        )


# ---------------------------------------------------------------------------
# Per-frame decision logic
# ---------------------------------------------------------------------------


def _make_block_with_fake_model():
    block = SegmentAnything2VideoBlockV1(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    fake = _FakeVideoModel()
    # Short-circuit the lazy model loader so we never touch the real
    # SegmentAnything2Video class (which needs the sam2 package + weights).
    block._video_model = fake
    block._current_model_id = "sam2/hiera_tiny"
    return block, fake


def test_first_frame_mode_prompts_once_then_tracks():
    block, fake = _make_block_with_fake_model()
    boxes = _make_box_detections()

    # Frame 0 — has detections, must prompt.
    result0 = block.run(
        images=[_make_frame(frame_number=0)],
        boxes=[boxes],
        version="hiera_tiny",
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )
    # Frames 1, 2 — detections still arrive but must be ignored.
    result1 = block.run(
        images=[_make_frame(frame_number=1)],
        boxes=[boxes],
        version="hiera_tiny",
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )
    result2 = block.run(
        images=[_make_frame(frame_number=2)],
        boxes=[boxes],
        version="hiera_tiny",
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )

    kinds = [c[0] for c in fake.calls]
    assert kinds == ["prompt", "track", "track"]
    # Output shape: one detection per mask per frame.
    for out in (result0, result1, result2):
        assert len(out) == 1
        dets = out[0]["predictions"]
        assert isinstance(dets, sv.Detections)
        assert len(dets) == 1
        assert dets.tracker_id.tolist() == [0]
        assert dets.data["class_name"].tolist() == ["person"]


def test_every_n_frames_mode_reprompts_at_interval():
    block, fake = _make_block_with_fake_model()
    boxes = _make_box_detections()

    for frame_number in range(8):
        block.run(
            images=[_make_frame(frame_number=frame_number)],
            boxes=[boxes],
            version="hiera_tiny",
            prompt_mode="every_n_frames",
            prompt_interval=3,
            threshold=0.0,
        )

    kinds = [c[0] for c in fake.calls]
    # prompt_interval=3 means "track 3 frames, then re-prompt".
    # Frame 0: prompt (fresh session).  Frames 1-3: track.  Frame 4:
    # re-prompt (3 tracks since last prompt).  Frames 5-7: track.
    assert kinds == [
        "prompt",
        "track",
        "track",
        "track",
        "prompt",
        "track",
        "track",
        "track",
    ]


def test_every_frame_mode_reprompts_every_frame():
    block, fake = _make_block_with_fake_model()
    boxes = _make_box_detections()

    for frame_number in range(3):
        block.run(
            images=[_make_frame(frame_number=frame_number)],
            boxes=[boxes],
            version="hiera_tiny",
            prompt_mode="every_frame",
            prompt_interval=30,
            threshold=0.0,
        )

    kinds = [c[0] for c in fake.calls]
    assert kinds == ["prompt", "prompt", "prompt"]


def test_stream_restart_triggers_session_reset():
    """If frame_number rolls back, we must reset the session to avoid
    blending temporal memory from a previous stream."""
    block, fake = _make_block_with_fake_model()
    boxes = _make_box_detections()

    block.run(
        images=[_make_frame(frame_number=5)],
        boxes=[boxes],
        version="hiera_tiny",
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )
    block.run(
        images=[_make_frame(frame_number=6)],
        boxes=[boxes],
        version="hiera_tiny",
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )
    # Roll back — simulate stream restart.
    block.run(
        images=[_make_frame(frame_number=0)],
        boxes=[boxes],
        version="hiera_tiny",
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )

    kinds = [c[0] for c in fake.calls]
    # prompt (init), track, reset + prompt (rollback).
    assert kinds == ["prompt", "track", "reset", "prompt"]


def test_multiple_video_streams_have_independent_sessions():
    block, fake = _make_block_with_fake_model()
    boxes = _make_box_detections()

    # Stream A — first frame.
    block.run(
        images=[_make_frame(video_id="A", frame_number=0)],
        boxes=[boxes],
        version="hiera_tiny",
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )
    # Stream B — first frame, must also prompt (fresh session).
    block.run(
        images=[_make_frame(video_id="B", frame_number=0)],
        boxes=[boxes],
        version="hiera_tiny",
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )
    # Stream A — frame 1 should track.
    block.run(
        images=[_make_frame(video_id="A", frame_number=1)],
        boxes=[boxes],
        version="hiera_tiny",
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )

    call_video_ids = [(c[0], c[1]["video_id"]) for c in fake.calls]
    assert call_video_ids == [
        ("prompt", "A"),
        ("prompt", "B"),
        ("track", "A"),
    ]


def test_no_boxes_and_no_session_emits_empty_detections():
    block, fake = _make_block_with_fake_model()

    result = block.run(
        images=[_make_frame(frame_number=0)],
        boxes=None,
        version="hiera_tiny",
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )

    assert len(result) == 1
    dets = result[0]["predictions"]
    assert isinstance(dets, sv.Detections)
    assert len(dets) == 0
    # Should not have touched the model at all.
    assert fake.calls == []


# ---------------------------------------------------------------------------
# Box prompt extraction
# ---------------------------------------------------------------------------


def test_extract_box_prompts_returns_coords_and_metadata():
    dets = _make_box_detections(
        xyxy=((10, 20, 100, 200), (50, 60, 150, 160)),
        class_names=("person", "car"),
        class_ids=(0, 1),
        confidences=(0.9, 0.8),
    )
    boxes_xyxy, metas = _extract_box_prompts(dets)
    assert boxes_xyxy == [(10.0, 20.0, 100.0, 200.0), (50.0, 60.0, 150.0, 160.0)]
    assert metas[0]["class_name"] == "person"
    assert metas[1]["class_name"] == "car"
    assert metas[0]["class_id"] == 0
    assert metas[1]["class_id"] == 1


def test_extract_box_prompts_handles_none():
    assert _extract_box_prompts(None) == ([], [])
