"""Unit tests for ``SAM2Video`` (HF transformers streaming tracker).

These tests exercise SAM2-specific behaviour — principally that SAM2
rejects text prompts (which are not part of SAM2's prompt vocabulary).
The broader streaming contract lives in ``HFStreamingVideoBase``.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from inference_models.errors import ModelRuntimeError
from inference_models.models.common.hf_streaming_video import SESSION_KEY
from inference_models.models.sam2_video.sam2_video_hf import SAM2Video


def _build_model_with_mocks():
    fake_session = object()
    model = MagicMock(name="Sam2VideoModel")
    processor = MagicMock(name="Sam2VideoProcessor")
    processor.init_video_session.return_value = fake_session

    def _processor_call(images, device=None, return_tensors="pt"):
        h, w = images.shape[:2]
        result = MagicMock()
        result.pixel_values = torch.zeros(1, 3, h, w)
        result.original_sizes = [(h, w)]
        return result

    processor.side_effect = _processor_call
    processor.postprocess_outputs = None

    def _post_process_masks(masks_list, original_sizes, binarize):
        h, w = original_sizes[0]
        return [torch.ones(1, h, w, dtype=torch.bool)]

    processor.post_process_masks.side_effect = _post_process_masks

    model_outputs = MagicMock(spec=["pred_masks", "obj_ids"])
    model_outputs.pred_masks = torch.ones(1, 1, 1, 1)
    model_outputs.obj_ids = [0]
    model.return_value = model_outputs

    sam2 = SAM2Video(
        model=model,
        processor=processor,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return sam2, processor, fake_session


def test_sam2_video_rejects_text_prompt():
    sam2, _, _ = _build_model_with_mocks()
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    with pytest.raises(ModelRuntimeError, match="does not support text prompts"):
        sam2.prompt(image=frame, text="person")


def test_sam2_video_accepts_box_prompt():
    sam2, processor, fake_session = _build_model_with_mocks()
    frame = np.zeros((32, 48, 3), dtype=np.uint8)

    masks, obj_ids, state = sam2.prompt(
        image=frame, bboxes=[(1, 2, 10, 12)], clear_old_prompts=True
    )
    assert masks.shape == (1, 32, 48)
    assert state[SESSION_KEY] is fake_session
    processor.add_inputs_to_inference_session.assert_called_once()


def test_sam2_video_track_without_state_raises():
    sam2, _, _ = _build_model_with_mocks()
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    with pytest.raises(ModelRuntimeError):
        sam2.track(image=frame, state_dict=None)


def test_sam2_video_track_reuses_session():
    sam2, processor, fake_session = _build_model_with_mocks()
    frame = np.zeros((32, 48, 3), dtype=np.uint8)

    _, _, state = sam2.prompt(image=frame, bboxes=[(1, 2, 10, 12)])
    processor.init_video_session.reset_mock()

    _, _, new_state = sam2.track(image=frame, state_dict=state)
    processor.init_video_session.assert_not_called()
    assert new_state[SESSION_KEY] is fake_session
