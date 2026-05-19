"""Unit tests for the SAM2 Video Tracker workflow block.

The block fetches the actual model via ``inference_models.AutoModel``;
we inject a fake model on the instance so tests don't touch
``inference_models`` or the real SAM2 weights.
"""

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything2_video.v1 import (
    BlockManifest,
    SegmentAnything2VideoBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)

# ---------------------------------------------------------------------------
# Helpers
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


class _FakeStreamingModel:
    """Stand-in for an ``inference_models.SAM2Video`` instance.

    Records each ``prompt`` / ``track`` call and returns deterministic
    masks + obj_ids so the block can build ``sv.Detections``.  The
    returned ``state_dict`` is a simple counter so tests can assert that
    state is threaded correctly across frames.
    """

    def __init__(self):
        self.calls = []
        self._state_counter = 0

    def _next_state(self) -> dict:
        self._state_counter += 1
        return {"_counter": self._state_counter}

    def prompt(
        self, image, bboxes, state_dict=None, clear_old_prompts=True, frame_idx=0
    ):
        self.calls.append(
            (
                "prompt",
                {
                    "num_boxes": len(bboxes),
                    "frame_idx": frame_idx,
                    "had_prior_state": state_dict is not None,
                },
            )
        )
        n = max(len(bboxes), 0)
        obj_ids = np.arange(n, dtype=np.int64)
        h, w = image.shape[:2]
        masks = np.zeros((n, h, w), dtype=bool)
        for i in range(n):
            masks[i, i, i] = True
        return masks, obj_ids, self._next_state()

    def track(self, image, state_dict=None):
        self.calls.append(("track", {"had_prior_state": state_dict is not None}))
        # Re-use the number of objects from the most recent prompt call.
        last_prompt_n = next(
            (c[1]["num_boxes"] for c in reversed(self.calls) if c[0] == "prompt"),
            0,
        )
        obj_ids = np.arange(last_prompt_n, dtype=np.int64)
        h, w = image.shape[:2]
        masks = np.zeros((last_prompt_n, h, w), dtype=bool)
        for i in range(last_prompt_n):
            masks[i, i, i] = True
        return masks, obj_ids, self._next_state()


def _make_block_with_fake_model():
    block = SegmentAnything2VideoBlockV1(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    fake = _FakeStreamingModel()
    block._model = fake
    block._current_model_id = "sam2video/small"
    return block, fake


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_parses_valid_config():
    data = {
        "type": "roboflow_core/segment_anything_2_video@v1",
        "name": "sam2_video_step",
        "images": "$inputs.image",
        "prompt_mode": "first_frame",
    }
    manifest = BlockManifest.model_validate(data)
    assert manifest.type == "roboflow_core/segment_anything_2_video@v1"
    assert manifest.model_id == "sam2video/small"
    assert manifest.prompt_mode == "first_frame"
    assert manifest.prompt_interval == 30


def test_manifest_rejects_unknown_prompt_mode():
    with pytest.raises(Exception):
        BlockManifest.model_validate(
            {
                "type": "roboflow_core/segment_anything_2_video@v1",
                "name": "sam2_video_step",
                "images": "$inputs.image",
                "prompt_mode": "nonsense",
            }
        )


# ---------------------------------------------------------------------------
# Remote execution mode
# ---------------------------------------------------------------------------


def test_block_rejects_remote_execution_mode():
    with pytest.raises(NotImplementedError, match="LOCAL workflow step execution"):
        SegmentAnything2VideoBlockV1(
            model_manager=MagicMock(),
            api_key=None,
            step_execution_mode=StepExecutionMode.REMOTE,
        )


# ---------------------------------------------------------------------------
# Per-frame decision logic
# ---------------------------------------------------------------------------


def test_first_frame_mode_prompts_once_then_tracks():
    block, fake = _make_block_with_fake_model()
    boxes = _make_box_detections()

    results = []
    for frame_number in range(3):
        results.append(
            block.run(
                images=[_make_frame(frame_number=frame_number)],
                boxes=[boxes],
                model_id="sam2video/small",
                prompt_mode="first_frame",
                prompt_interval=30,
                threshold=0.0,
            )
        )

    kinds = [c[0] for c in fake.calls]
    assert kinds == ["prompt", "track", "track"]
    for out in results:
        dets = out[0]["predictions"]
        assert isinstance(dets, sv.Detections)
        assert len(dets) == 1
        assert dets.tracker_id.tolist() == [0]
        assert dets.data["class_name"].tolist() == ["person"]


def test_every_n_frames_mode_reprompts_after_interval():
    block, fake = _make_block_with_fake_model()
    boxes = _make_box_detections()

    for frame_number in range(8):
        block.run(
            images=[_make_frame(frame_number=frame_number)],
            boxes=[boxes],
            model_id="sam2video/small",
            prompt_mode="every_n_frames",
            prompt_interval=3,
            threshold=0.0,
        )

    # prompt_interval=3 means "track 3 frames, then re-prompt".
    kinds = [c[0] for c in fake.calls]
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
            model_id="sam2video/small",
            prompt_mode="every_frame",
            prompt_interval=30,
            threshold=0.0,
        )

    assert [c[0] for c in fake.calls] == ["prompt", "prompt", "prompt"]


def test_stream_restart_triggers_session_reset():
    block, fake = _make_block_with_fake_model()
    boxes = _make_box_detections()

    block.run(
        images=[_make_frame(frame_number=5)],
        boxes=[boxes],
        model_id="sam2video/small",
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )
    block.run(
        images=[_make_frame(frame_number=6)],
        boxes=[boxes],
        model_id="sam2video/small",
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )
    # Rollback to frame 0 — stream restart.
    block.run(
        images=[_make_frame(frame_number=0)],
        boxes=[boxes],
        model_id="sam2video/small",
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )

    kinds = [c[0] for c in fake.calls]
    # prompt (init), track, prompt (rollback — fresh session).
    assert kinds == ["prompt", "track", "prompt"]
    # After a reset, prior state must not leak into the next prompt call.
    assert fake.calls[0][1]["had_prior_state"] is False
    assert fake.calls[2][1]["had_prior_state"] is False


def test_state_dict_threaded_across_track_calls():
    block, fake = _make_block_with_fake_model()
    boxes = _make_box_detections()

    block.run(
        images=[_make_frame(frame_number=0)],
        boxes=[boxes],
        model_id="sam2video/small",
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )
    block.run(
        images=[_make_frame(frame_number=1)],
        boxes=[boxes],
        model_id="sam2video/small",
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )
    block.run(
        images=[_make_frame(frame_number=2)],
        boxes=[boxes],
        model_id="sam2video/small",
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )
    # First prompt had no prior state; the following two track calls
    # must have received the state dict returned from the previous call.
    assert fake.calls[0][1]["had_prior_state"] is False
    assert fake.calls[1][1]["had_prior_state"] is True
    assert fake.calls[2][1]["had_prior_state"] is True


def test_multiple_video_streams_have_independent_sessions():
    block, fake = _make_block_with_fake_model()
    boxes = _make_box_detections()

    block.run(
        images=[_make_frame(video_id="A", frame_number=0)],
        boxes=[boxes],
        model_id="sam2video/small",
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )
    block.run(
        images=[_make_frame(video_id="B", frame_number=0)],
        boxes=[boxes],
        model_id="sam2video/small",
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )
    block.run(
        images=[_make_frame(video_id="A", frame_number=1)],
        boxes=[boxes],
        model_id="sam2video/small",
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )

    kinds = [c[0] for c in fake.calls]
    # A prompts (fresh), B prompts (fresh — own session), A tracks.
    assert kinds == ["prompt", "prompt", "track"]


def test_no_boxes_and_no_session_emits_empty_detections():
    block, fake = _make_block_with_fake_model()

    result = block.run(
        images=[_make_frame(frame_number=0)],
        boxes=None,
        model_id="sam2video/small",
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )

    dets = result[0]["predictions"]
    assert isinstance(dets, sv.Detections)
    assert len(dets) == 0
    assert fake.calls == []
