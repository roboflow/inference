"""Unit tests for ``SAM3TrackerVideo`` (HF transformers visually prompted tracker).

The fakes below mirror the *real* ``Sam3TrackerVideoProcessor``
surface — it has ``add_inputs_to_inference_session`` and
``post_process_masks`` but **no** ``add_text_prompt`` and **no**
``postprocess_outputs`` (those belong to the separate ``Sam3Video``
concept-tracker family).  Hand-rolled fakes (rather than permissive
``MagicMock``s) make any call outside that surface fail loudly.
"""

from typing import List, Optional, Tuple

import numpy as np
import pytest
import torch

from inference_models.errors import ModelRuntimeError
from inference_models.models.common.hf_streaming_video import SESSION_KEY
from inference_models.models.sam3_tracker_video.sam3_tracker_video_hf import (
    SAM3TrackerVideo,
)


class _FakeSession:
    """Mimics ``Sam3TrackerVideoInferenceSession``'s prompt bookkeeping."""

    def __init__(self):
        self.obj_ids: List[int] = []
        self.boxes_added: List[List[float]] = []


class _FakeBatchEncoding:
    def __init__(self, height: int, width: int):
        self.pixel_values = torch.zeros(1, 3, height, width)
        self.original_sizes = [(height, width)]


class _FakeSam3TrackerVideoProcessor:
    """Only the real ``Sam3TrackerVideoProcessor`` surface: ``__call__``,
    ``init_video_session``, ``add_inputs_to_inference_session`` and
    ``post_process_masks``.  Notably there is no ``add_text_prompt``
    and no ``postprocess_outputs``."""

    def __init__(self):
        self.sessions_created = 0
        self.last_init_kwargs: Optional[dict] = None
        self.last_add_inputs_kwargs: Optional[dict] = None

    def __call__(self, images, device=None, return_tensors="pt"):
        height, width = images.shape[:2]
        return _FakeBatchEncoding(height=height, width=width)

    def init_video_session(self, **kwargs):
        self.sessions_created += 1
        self.last_init_kwargs = kwargs
        return _FakeSession()

    def add_inputs_to_inference_session(
        self,
        inference_session,
        frame_idx,
        obj_ids,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        input_masks=None,
        original_size=None,
        clear_old_inputs=True,
    ):
        self.last_add_inputs_kwargs = {
            "frame_idx": frame_idx,
            "obj_ids": obj_ids,
            "input_boxes": input_boxes,
            "original_size": original_size,
        }
        inference_session.obj_ids = list(obj_ids)
        inference_session.boxes_added = input_boxes[0] if input_boxes else []
        return inference_session

    def post_process_masks(self, masks_list, original_sizes, binarize=True):
        height, width = original_sizes[0]
        num_objects = masks_list[0].shape[0]
        return [torch.ones(num_objects, height, width, dtype=torch.bool)]


class _FakeModelOutputs:
    """Mimics ``Sam3TrackerVideoSegmentationOutput`` (same field names
    as SAM2's: ``object_ids`` / ``pred_masks``)."""

    __slots__ = ("object_ids", "pred_masks")

    def __init__(self, object_ids: List[int], height: int, width: int):
        self.object_ids = object_ids
        self.pred_masks = torch.ones(len(object_ids), 1, height, width)


class _FakeSam3TrackerVideoModel:
    def __init__(self):
        self.frames_seen = 0

    def __call__(self, inference_session, frame=None, reverse=False):
        self.frames_seen += 1
        height, width = frame.shape[-2:]
        return _FakeModelOutputs(
            object_ids=list(inference_session.obj_ids),
            height=height,
            width=width,
        )


def _build_model() -> Tuple[
    SAM3TrackerVideo, _FakeSam3TrackerVideoProcessor, _FakeSam3TrackerVideoModel
]:
    processor = _FakeSam3TrackerVideoProcessor()
    model = _FakeSam3TrackerVideoModel()
    tracker = SAM3TrackerVideo(
        model=model,
        processor=processor,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return tracker, processor, model


def test_sam3_tracker_video_rejects_text_prompt():
    tracker, _, _ = _build_model()
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    with pytest.raises(ModelRuntimeError, match="does not support text prompts"):
        tracker.prompt(image=frame, text="person")


def test_sam3_tracker_video_accepts_box_prompts():
    tracker, processor, _ = _build_model()
    frame = np.zeros((32, 48, 3), dtype=np.uint8)

    masks, obj_ids, state = tracker.prompt(
        image=frame,
        bboxes=[(1, 2, 10, 12), (5, 6, 20, 22)],
        clear_old_prompts=True,
    )

    assert masks.shape == (2, 32, 48)
    assert obj_ids.tolist() == [0, 1]
    assert isinstance(state[SESSION_KEY], _FakeSession)
    assert processor.last_add_inputs_kwargs["obj_ids"] == [0, 1]
    # input_boxes use the real 3-level nesting: [image [boxes [coords]]]
    assert processor.last_add_inputs_kwargs["input_boxes"] == [
        [[1.0, 2.0, 10.0, 12.0], [5.0, 6.0, 20.0, 22.0]]
    ]
    assert processor.last_add_inputs_kwargs["original_size"] == (32, 48)


def test_sam3_tracker_video_track_without_state_raises():
    tracker, _, _ = _build_model()
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    with pytest.raises(ModelRuntimeError, match="prompt must be called first"):
        tracker.track(image=frame, state_dict=None)


def test_sam3_tracker_video_track_reuses_session():
    tracker, processor, model = _build_model()
    frame = np.zeros((32, 48, 3), dtype=np.uint8)

    _, _, state = tracker.prompt(image=frame, bboxes=[(1, 2, 10, 12)])
    session = state[SESSION_KEY]

    masks, obj_ids, new_state = tracker.track(image=frame, state_dict=state)

    assert processor.sessions_created == 1
    assert new_state[SESSION_KEY] is session
    assert masks.shape == (1, 32, 48)
    assert obj_ids.tolist() == [0]
    assert model.frames_seen == 2


def test_sam3_tracker_video_prompt_resets_session_by_default():
    tracker, processor, _ = _build_model()
    frame = np.zeros((32, 48, 3), dtype=np.uint8)

    _, _, state = tracker.prompt(image=frame, bboxes=[(1, 2, 10, 12)])
    _, _, new_state = tracker.prompt(
        image=frame, bboxes=[(3, 4, 11, 13)], state_dict=state
    )

    assert processor.sessions_created == 2
    assert new_state[SESSION_KEY] is not state[SESSION_KEY]


def test_sam3_tracker_video_resolves_transformers_tracker_family():
    transformers = pytest.importorskip("transformers")
    model_cls, processor_cls = SAM3TrackerVideo._resolve_transformers_classes()
    assert model_cls is transformers.Sam3TrackerVideoModel
    assert processor_cls is transformers.Sam3TrackerVideoProcessor
    # The concept-tracker surface must NOT leak into this family.
    assert not hasattr(processor_cls, "add_text_prompt")