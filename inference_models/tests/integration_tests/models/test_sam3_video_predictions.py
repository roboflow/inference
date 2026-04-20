"""Integration tests for ``SAM3Video`` (HF transformers streaming).

Requires the ``sam3_video_package`` fixture (HF transformers export of
the SAM3 video model) and a transformers install that ships
``Sam3VideoModel`` / ``Sam3VideoProcessor``.
"""

import numpy as np
import pytest
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.errors import ModelRuntimeError
from inference_models.models.sam3_video.sam3_video_hf import SAM3Video


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
def test_sam3_video_text_prompt_then_track(sam3_video_package: str) -> None:
    """Seed a text prompt, then track across subsequent frames.

    Numerical mask contents depend on hardware / transformers version;
    assert only the shape contract and that the state dict is carried
    forward without error.
    """
    model = SAM3Video.from_pretrained(sam3_video_package, device=DEFAULT_DEVICE)
    frames = _translating_square_frames(n_frames=4)

    masks, obj_ids, state = model.prompt(
        image=frames[0],
        text="white square",
        frame_idx=0,
    )
    assert masks.ndim == 3
    assert masks.shape[0] == obj_ids.shape[0]
    assert masks.shape[1:] == frames[0].shape[:2]
    assert isinstance(state, dict)

    for frame in frames[1:]:
        masks, obj_ids, state = model.track(image=frame, state_dict=state)
        assert masks.ndim == 3
        assert masks.shape[0] == obj_ids.shape[0]
        assert masks.shape[1:] == frame.shape[:2]


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_sam3_video_box_prompt_then_track(sam3_video_package: str) -> None:
    model = SAM3Video.from_pretrained(sam3_video_package, device=DEFAULT_DEVICE)
    frames = _translating_square_frames(n_frames=3)

    masks, obj_ids, state = model.prompt(
        image=frames[0],
        bboxes=[(50, 98, 110, 158)],
        frame_idx=0,
    )
    assert masks.shape[1:] == frames[0].shape[:2]
    assert masks.shape[0] == obj_ids.shape[0]

    for frame in frames[1:]:
        masks, _obj_ids, state = model.track(image=frame, state_dict=state)
        assert masks.shape[1:] == frame.shape[:2]


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_sam3_video_track_without_state_raises(sam3_video_package: str) -> None:
    model = SAM3Video.from_pretrained(sam3_video_package, device=DEFAULT_DEVICE)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    with pytest.raises(ModelRuntimeError):
        model.track(image=frame, state_dict=None)


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_sam3_video_accepts_torch_tensor_input(sam3_video_package: str) -> None:
    model = SAM3Video.from_pretrained(sam3_video_package, device=DEFAULT_DEVICE)
    frame = _translating_square_frames(n_frames=1)[0]
    frame_tensor = torch.from_numpy(frame)

    masks, obj_ids, state = model.prompt(image=frame_tensor, text="white square")
    assert masks.shape[1:] == frame.shape[:2]
    assert isinstance(state, dict)
