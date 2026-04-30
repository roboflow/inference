"""Integration tests for ``SAM2Video`` (HF transformers streaming).

SAM2's HF port does not accept text prompts (that's a SAM3 feature);
these tests focus on the bbox-prompt + track flow.
"""

import numpy as np
import pytest
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.errors import ModelRuntimeError
from inference_models.models.sam2_video.sam2_video_hf import SAM2Video


def _translating_square_frames(
    n_frames: int = 4,
    size: int = 256,
    square: int = 60,
    step: int = 8,
) -> list:
    frames = []
    for i in range(n_frames):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        x = 50 + step * i
        y = size // 2 - square // 2
        frame[y : y + square, x : x + square] = 255
        frames.append(frame)
    return frames


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_sam2_video_box_prompt_then_track(sam2_video_package: str) -> None:
    model = SAM2Video.from_pretrained(sam2_video_package, device=DEFAULT_DEVICE)
    frames = _translating_square_frames(n_frames=3)

    masks, obj_ids, state = model.prompt(
        image=frames[0],
        bboxes=[(50, 98, 110, 158)],
        frame_idx=0,
    )
    assert masks.ndim == 3
    assert masks.shape[0] == obj_ids.shape[0]
    assert masks.shape[1:] == frames[0].shape[:2]
    assert isinstance(state, dict)

    for frame in frames[1:]:
        masks, _obj_ids, state = model.track(image=frame, state_dict=state)
        assert masks.shape[1:] == frame.shape[:2]


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_sam2_video_rejects_text_prompts(sam2_video_package: str) -> None:
    model = SAM2Video.from_pretrained(sam2_video_package, device=DEFAULT_DEVICE)
    frame = _translating_square_frames(n_frames=1)[0]
    with pytest.raises(ModelRuntimeError, match="does not support text prompts"):
        model.prompt(image=frame, text="white square")


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_sam2_video_track_without_state_raises(sam2_video_package: str) -> None:
    model = SAM2Video.from_pretrained(sam2_video_package, device=DEFAULT_DEVICE)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    with pytest.raises(ModelRuntimeError):
        model.track(image=frame, state_dict=None)


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_sam2_video_accepts_torch_tensor_input(sam2_video_package: str) -> None:
    model = SAM2Video.from_pretrained(sam2_video_package, device=DEFAULT_DEVICE)
    frame = _translating_square_frames(n_frames=1)[0]
    frame_tensor = torch.from_numpy(frame)

    masks, obj_ids, state = model.prompt(
        image=frame_tensor, bboxes=[(50, 98, 110, 158)]
    )
    assert masks.shape[1:] == frame.shape[:2]
    assert isinstance(state, dict)
