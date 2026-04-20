"""Unit tests for the SAM3 Video Tracker workflow block."""

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything3_video.v1 import (
    BlockManifest,
    SegmentAnything3VideoBlockV1,
    _normalise_class_names,
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
    """Drop-in replacement for ``SegmentAnything3Video``.

    Accepts either text or box prompts and records every call so the
    test can verify the prompt/track pattern.
    """

    def __init__(self):
        self.calls = []
        self.sessions = {}  # video_id -> list of obj_ids currently active

    def has_session(self, video_id: str) -> bool:
        return video_id in self.sessions

    def reset_session(self, video_id: str) -> None:
        if video_id in self.sessions:
            self.sessions.pop(video_id)
            self.calls.append(("reset", {"video_id": video_id}))

    def prompt_and_track(
        self,
        video_id: str,
        frame: np.ndarray,
        frame_index: int,
        text=None,
        boxes_xyxy=None,
        clear_old_prompts: bool = True,
    ):
        kind = "prompt_text" if text is not None else "prompt_boxes"
        self.calls.append(
            (
                kind,
                {
                    "video_id": video_id,
                    "text": text,
                    "num_boxes": len(boxes_xyxy) if boxes_xyxy else 0,
                    "frame_index": frame_index,
                    "clear_old_prompts": clear_old_prompts,
                },
            )
        )
        n = max(len(boxes_xyxy or []), 1 if text else 0)
        obj_ids = np.arange(n, dtype=np.int64)
        self.sessions[video_id] = list(obj_ids)
        h, w = frame.shape[:2]
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


def _make_block_with_fake_model():
    block = SegmentAnything3VideoBlockV1(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    fake = _FakeVideoModel()
    block._video_model = fake
    block._current_model_id = "sam3/sam3_video"
    return block, fake


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_parses_valid_config():
    data = {
        "type": "roboflow_core/sam3_video@v1",
        "name": "sam3_video_step",
        "images": "$inputs.image",
        "class_names": ["person", "car"],
        "prompt_mode": "first_frame",
    }
    manifest = BlockManifest.model_validate(data)
    assert manifest.type == "roboflow_core/sam3_video@v1"
    assert manifest.class_names == ["person", "car"]
    # Default comes from the Roboflow registry id, not a HF Hub id.
    assert manifest.model_id == "sam3/sam3_video"


def test_manifest_supports_detector_boxes_only():
    data = {
        "type": "roboflow_core/sam3_video@v1",
        "name": "sam3_video_step",
        "images": "$inputs.image",
        "boxes": "$steps.detector.predictions",
        "prompt_mode": "every_n_frames",
        "prompt_interval": 10,
    }
    manifest = BlockManifest.model_validate(data)
    assert manifest.class_names is None
    assert manifest.boxes == "$steps.detector.predictions"


# ---------------------------------------------------------------------------
# Remote execution mode
# ---------------------------------------------------------------------------


def test_block_rejects_remote_execution_mode():
    with pytest.raises(NotImplementedError, match="LOCAL workflow step execution"):
        SegmentAnything3VideoBlockV1(
            model_manager=MagicMock(),
            api_key=None,
            step_execution_mode=StepExecutionMode.REMOTE,
        )


# ---------------------------------------------------------------------------
# Prompt routing
# ---------------------------------------------------------------------------


def test_text_prompts_are_used_when_no_boxes():
    block, fake = _make_block_with_fake_model()

    block.run(
        images=[_make_frame(frame_number=0)],
        model_id="sam3/sam3_video",
        class_names=["person", "car"],
        boxes=None,
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )

    assert len(fake.calls) == 1
    kind, meta = fake.calls[0]
    assert kind == "prompt_text"
    assert meta["text"] == "person, car"


def test_boxes_take_precedence_over_text_when_both_supplied():
    block, fake = _make_block_with_fake_model()
    boxes = _make_box_detections()

    block.run(
        images=[_make_frame(frame_number=0)],
        model_id="sam3/sam3_video",
        class_names=["person"],
        boxes=[boxes],
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )

    assert [c[0] for c in fake.calls] == ["prompt_boxes"]
    assert fake.calls[0][1]["num_boxes"] == 1
    assert fake.calls[0][1]["text"] is None


def test_first_frame_text_prompt_then_tracks_subsequent_frames():
    block, fake = _make_block_with_fake_model()

    for frame_number in range(3):
        block.run(
            images=[_make_frame(frame_number=frame_number)],
            model_id="sam3/sam3_video",
            class_names=["person"],
            boxes=None,
            prompt_mode="first_frame",
            prompt_interval=30,
            threshold=0.0,
        )

    kinds = [c[0] for c in fake.calls]
    assert kinds == ["prompt_text", "track", "track"]


def test_every_frame_mode_reprompts_every_frame_with_boxes():
    block, fake = _make_block_with_fake_model()
    boxes = _make_box_detections()

    for frame_number in range(3):
        block.run(
            images=[_make_frame(frame_number=frame_number)],
            model_id="sam3/sam3_video",
            class_names=None,
            boxes=[boxes],
            prompt_mode="every_frame",
            prompt_interval=30,
            threshold=0.0,
        )

    kinds = [c[0] for c in fake.calls]
    assert kinds == ["prompt_boxes", "prompt_boxes", "prompt_boxes"]


def test_no_prompts_and_no_session_emits_empty():
    block, fake = _make_block_with_fake_model()

    result = block.run(
        images=[_make_frame(frame_number=0)],
        model_id="sam3/sam3_video",
        class_names=None,
        boxes=None,
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )

    assert len(result) == 1
    dets = result[0]["predictions"]
    assert isinstance(dets, sv.Detections)
    assert len(dets) == 0
    assert fake.calls == []


def test_stream_restart_triggers_session_reset():
    block, fake = _make_block_with_fake_model()

    block.run(
        images=[_make_frame(frame_number=5)],
        model_id="sam3/sam3_video",
        class_names=["person"],
        boxes=None,
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )
    block.run(
        images=[_make_frame(frame_number=6)],
        model_id="sam3/sam3_video",
        class_names=["person"],
        boxes=None,
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )
    # Rollback — simulate stream restart.
    block.run(
        images=[_make_frame(frame_number=0)],
        model_id="sam3/sam3_video",
        class_names=["person"],
        boxes=None,
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )

    kinds = [c[0] for c in fake.calls]
    assert kinds == ["prompt_text", "track", "reset", "prompt_text"]


# ---------------------------------------------------------------------------
# class_names normalisation
# ---------------------------------------------------------------------------


def test_normalise_class_names_accepts_list():
    assert _normalise_class_names(["a", "b"]) == ["a", "b"]


def test_normalise_class_names_accepts_comma_string():
    assert _normalise_class_names("a, b,  c") == ["a", "b", "c"]


def test_normalise_class_names_handles_none_and_empty():
    assert _normalise_class_names(None) == []
    assert _normalise_class_names("") == []
    assert _normalise_class_names([]) == []
