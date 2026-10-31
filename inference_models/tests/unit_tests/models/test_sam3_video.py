"""Unit tests for ``SAM3Video`` (HF transformers streaming concept tracker).

The fakes below mirror the *real* ``Sam3VideoProcessor`` surface — in
particular it has ``add_text_prompt`` but **no**
``add_inputs_to_inference_session`` (box prompting belongs to the
separate ``Sam3TrackerVideo`` family).  Hand-rolled fakes (rather than
permissive ``MagicMock``s) make any call outside that surface fail
loudly, which is exactly the regression we want to catch.
"""

from typing import List, Optional

import numpy as np
import pytest
import torch

from inference_models.errors import ModelRuntimeError
from inference_models.models.common.hf_streaming_video import SESSION_KEY
from inference_models.models.sam3_video.sam3_video_hf import (
    SAM3Video,
    SAM3VideoFrameResult,
    _normalise_text_prompts,
)


class _FakeSession:
    """Mimics ``Sam3VideoInferenceSession``'s prompt bookkeeping."""

    def __init__(self):
        self.prompts: List[str] = []
        self.frames_processed: int = 0


class _FakeBatchEncoding:
    def __init__(self, height: int, width: int):
        self.pixel_values = torch.zeros(1, 3, height, width)
        self.original_sizes = [(height, width)]


class _FakeSam3VideoProcessor:
    """Only the real ``Sam3VideoProcessor`` surface: ``__call__``,
    ``add_text_prompt``, ``init_video_session``, ``postprocess_outputs``."""

    def __init__(self):
        self.sessions_created = 0
        self.last_init_kwargs: Optional[dict] = None

    def __call__(self, images, device=None, return_tensors="pt"):
        height, width = images.shape[:2]
        return _FakeBatchEncoding(height=height, width=width)

    def init_video_session(self, **kwargs):
        self.sessions_created += 1
        self.last_init_kwargs = kwargs
        return _FakeSession()

    def add_text_prompt(self, inference_session, text):
        if isinstance(text, str):
            text = [text]
        for prompt_text in text:
            if prompt_text not in inference_session.prompts:
                inference_session.prompts.append(prompt_text)
        return inference_session

    def postprocess_outputs(
        self, inference_session, model_outputs, original_sizes=None
    ):
        height, width = original_sizes[0]
        # One object per registered prompt; a second object appears for
        # the first prompt once the session has seen more than one frame
        # (emulating mid-stream detection of a new instance).
        object_ids = list(range(len(inference_session.prompts)))
        prompt_to_obj_ids = {
            prompt: [obj_id] for obj_id, prompt in enumerate(inference_session.prompts)
        }
        if inference_session.frames_processed > 1 and inference_session.prompts:
            new_id = len(object_ids)
            object_ids.append(new_id)
            prompt_to_obj_ids[inference_session.prompts[0]].append(new_id)
        num_objects = len(object_ids)
        return {
            "object_ids": torch.tensor(object_ids, dtype=torch.int64),
            "scores": torch.linspace(0.9, 0.5, steps=num_objects),
            "boxes": torch.zeros(num_objects, 4, dtype=torch.float32),
            "masks": torch.ones(num_objects, height, width, dtype=torch.bool),
            "prompt_to_obj_ids": prompt_to_obj_ids,
        }


class _FakeSam3VideoModel:
    def __call__(self, inference_session, frame):
        inference_session.frames_processed += 1
        return object()  # opaque — consumed only by postprocess_outputs


def _build_model():
    processor = _FakeSam3VideoProcessor()
    sam3 = SAM3Video(
        model=_FakeSam3VideoModel(),
        processor=processor,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return sam3, processor


def _frame(height: int = 32, width: int = 48) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_prompt_with_single_text_returns_frame_result():
    sam3, processor = _build_model()

    result = sam3.prompt(image=_frame(), text="person")

    assert isinstance(result, SAM3VideoFrameResult)
    assert result.masks.shape == (1, 32, 48)
    assert result.masks.dtype == bool
    assert result.object_ids.tolist() == [0]
    assert result.scores.dtype == np.float32
    assert result.boxes.shape == (1, 4)
    assert result.prompt_to_object_ids == {"person": [0]}
    assert isinstance(result.state_dict[SESSION_KEY], _FakeSession)
    assert processor.sessions_created == 1


def test_prompt_with_multiple_texts_registers_each_concept_separately():
    sam3, _ = _build_model()

    result = sam3.prompt(image=_frame(), text=["person", "dog"])

    session = result.state_dict[SESSION_KEY]
    assert session.prompts == ["person", "dog"]
    assert result.prompt_to_object_ids == {"person": [0], "dog": [1]}
    assert result.object_ids.tolist() == [0, 1]


def test_prompt_without_text_raises():
    sam3, _ = _build_model()
    with pytest.raises(ModelRuntimeError, match="text"):
        sam3.prompt(image=_frame(), text=[])


def test_prompt_does_not_accept_boxes():
    # Box prompting belongs to the Sam3TrackerVideo family; the concept
    # tracker's API must not silently accept (and drop) boxes.
    sam3, _ = _build_model()
    with pytest.raises(TypeError):
        sam3.prompt(image=_frame(), text="person", bboxes=[(1, 2, 10, 12)])


def test_track_without_state_raises():
    sam3, _ = _build_model()
    with pytest.raises(ModelRuntimeError, match="prompt must be called first"):
        sam3.track(image=_frame(), state_dict=None)


def test_track_reuses_session_and_surfaces_new_objects():
    sam3, processor = _build_model()

    result = sam3.prompt(image=_frame(), text=["person"])
    result = sam3.track(image=_frame(), state_dict=result.state_dict)
    assert processor.sessions_created == 1

    # The fake reports a new instance of "person" from frame 2 onwards —
    # mid-stream detection must surface without re-prompting.
    assert result.object_ids.tolist() == [0, 1]
    assert result.prompt_to_object_ids == {"person": [0, 1]}
    assert result.masks.shape[0] == 2
    assert result.scores.shape == (2,)


def test_prompt_with_clear_old_prompts_false_extends_session():
    sam3, processor = _build_model()

    result = sam3.prompt(image=_frame(), text=["person"])
    session = result.state_dict[SESSION_KEY]

    result = sam3.prompt(
        image=_frame(),
        text=["dog"],
        state_dict=result.state_dict,
        clear_old_prompts=False,
    )
    assert processor.sessions_created == 1
    assert result.state_dict[SESSION_KEY] is session
    assert session.prompts == ["person", "dog"]


def test_prompt_with_clear_old_prompts_true_starts_fresh_session():
    sam3, processor = _build_model()

    result = sam3.prompt(image=_frame(), text=["person"])
    first_session = result.state_dict[SESSION_KEY]

    result = sam3.prompt(
        image=_frame(),
        text=["dog"],
        state_dict=result.state_dict,
        clear_old_prompts=True,
    )
    assert processor.sessions_created == 2
    assert result.state_dict[SESSION_KEY] is not first_session
    assert result.state_dict[SESSION_KEY].prompts == ["dog"]


def test_prompt_accepts_torch_tensor_frame():
    sam3, _ = _build_model()
    frame = torch.zeros(32, 48, 3, dtype=torch.uint8)
    result = sam3.prompt(image=frame, text="person")
    assert result.masks.shape == (1, 32, 48)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("person", ["person"]),
        (["person", "dog"], ["person", "dog"]),
        (["  person  ", "", None], ["person"]),
        (None, []),
        ([], []),
    ],
)
def test_normalise_text_prompts(raw, expected):
    assert _normalise_text_prompts(raw) == expected
