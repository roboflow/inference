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
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


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
    def __init__(self):
        self.calls = []
        self._state_counter = 0

    def _next_state(self) -> dict:
        self._state_counter += 1
        return {"_counter": self._state_counter}

    def prompt(
        self,
        image,
        bboxes=None,
        text=None,
        state_dict=None,
        clear_old_prompts=True,
        frame_idx=0,
    ):
        kind = "prompt_text" if text is not None else "prompt_boxes"
        self.calls.append(
            (
                kind,
                {
                    "text": text,
                    "num_boxes": len(bboxes) if bboxes else 0,
                    "frame_idx": frame_idx,
                    "had_prior_state": state_dict is not None,
                },
            )
        )
        n = max(len(bboxes or []), 1 if text else 0)
        obj_ids = np.arange(n, dtype=np.int64)
        h, w = image.shape[:2]
        masks = np.zeros((n, h, w), dtype=bool)
        for i in range(n):
            masks[i, i, i] = True
        return masks, obj_ids, self._next_state()

    def track(self, image, state_dict=None):
        self.calls.append(
            ("track", {"had_prior_state": state_dict is not None})
        )
        last_n = next(
            (
                c[1].get("num_boxes") or (1 if c[1].get("text") else 0)
                for c in reversed(self.calls)
                if c[0] in ("prompt_text", "prompt_boxes")
            ),
            0,
        )
        obj_ids = np.arange(last_n, dtype=np.int64)
        h, w = image.shape[:2]
        masks = np.zeros((last_n, h, w), dtype=bool)
        for i in range(last_n):
            masks[i, i, i] = True
        return masks, obj_ids, self._next_state()


def _make_block_with_fake_model():
    block = SegmentAnything3VideoBlockV1(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    fake = _FakeStreamingModel()
    block._model = fake
    block._current_model_id = "segment-anything-3-rt"
    return block, fake


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_parses_valid_config():
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/sam3_video@v1",
            "name": "sam3_video_step",
            "images": "$inputs.image",
            "class_names": ["person", "car"],
        }
    )
    assert manifest.model_id == "segment-anything-3-rt"
    assert manifest.class_names == ["person", "car"]


def test_manifest_allows_boxes_only():
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/sam3_video@v1",
            "name": "sam3_video_step",
            "images": "$inputs.image",
            "boxes": "$steps.detector.predictions",
        }
    )
    assert manifest.class_names is None


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


def test_text_prompt_used_when_no_boxes_supplied():
    block, fake = _make_block_with_fake_model()

    block.run(
        images=[_make_frame(frame_number=0)],
        model_id="segment-anything-3-rt",
        class_names=["person", "car"],
        boxes=None,
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )

    kind, meta = fake.calls[0]
    assert kind == "prompt_text"
    assert meta["text"] == "person, car"


def test_boxes_take_precedence_over_text_when_both_supplied():
    block, fake = _make_block_with_fake_model()

    block.run(
        images=[_make_frame(frame_number=0)],
        model_id="segment-anything-3-rt",
        class_names=["person"],
        boxes=[_make_box_detections()],
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )

    assert [c[0] for c in fake.calls] == ["prompt_boxes"]
    assert fake.calls[0][1]["text"] is None


def test_first_frame_text_prompt_then_tracks():
    block, fake = _make_block_with_fake_model()

    for frame_number in range(3):
        block.run(
            images=[_make_frame(frame_number=frame_number)],
            model_id="segment-anything-3-rt",
            class_names=["person"],
            boxes=None,
            prompt_mode="first_frame",
            prompt_interval=30,
            threshold=0.0,
        )

    kinds = [c[0] for c in fake.calls]
    assert kinds == ["prompt_text", "track", "track"]


def test_every_frame_mode_reprompts_every_frame():
    block, fake = _make_block_with_fake_model()

    for frame_number in range(3):
        block.run(
            images=[_make_frame(frame_number=frame_number)],
            model_id="segment-anything-3-rt",
            class_names=None,
            boxes=[_make_box_detections()],
            prompt_mode="every_frame",
            prompt_interval=30,
            threshold=0.0,
        )

    assert [c[0] for c in fake.calls] == ["prompt_boxes"] * 3


def test_stream_restart_triggers_reset_and_fresh_prompt():
    block, fake = _make_block_with_fake_model()

    block.run(
        images=[_make_frame(frame_number=5)],
        model_id="segment-anything-3-rt",
        class_names=["person"],
        boxes=None,
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )
    block.run(
        images=[_make_frame(frame_number=6)],
        model_id="segment-anything-3-rt",
        class_names=["person"],
        boxes=None,
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )
    block.run(
        images=[_make_frame(frame_number=0)],
        model_id="segment-anything-3-rt",
        class_names=["person"],
        boxes=None,
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )

    kinds = [c[0] for c in fake.calls]
    assert kinds == ["prompt_text", "track", "prompt_text"]
    # Rollback must clear state — the restart prompt must not carry a prior state_dict.
    assert fake.calls[2][1]["had_prior_state"] is False


def test_no_prompts_and_no_session_emits_empty():
    block, fake = _make_block_with_fake_model()

    result = block.run(
        images=[_make_frame(frame_number=0)],
        model_id="segment-anything-3-rt",
        class_names=None,
        boxes=None,
        prompt_mode="first_frame",
        prompt_interval=30,
        threshold=0.0,
    )

    dets = result[0]["predictions"]
    assert isinstance(dets, sv.Detections)
    assert len(dets) == 0
    assert fake.calls == []
