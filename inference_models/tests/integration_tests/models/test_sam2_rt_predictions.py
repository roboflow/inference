"""Integration tests for the streaming SAM2 camera predictor wrapper.

These tests exercise the real SAM2ForStream model against downloaded
weights.  They require the ``sam2`` (aka ``rf-sam-2``) package to be
installed and a GPU-capable runtime; the fixture
``sam2_rt_package`` downloads the streaming-flavoured package
(containing ``weights.pt`` + ``sam2-rt.yaml``).
"""

import numpy as np
import pytest
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.errors import ModelRuntimeError
from inference_models.models.sam2_rt.sam2_pytorch import SAM2ForStream


def _translating_square_frames(
    n_frames: int = 4,
    size: int = 256,
    square: int = 60,
    step: int = 8,
) -> list:
    """Synthetic video: a white square translating right on black."""
    frames = []
    for i in range(n_frames):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        x = 50 + step * i
        y = size // 2 - square // 2
        frame[y : y + square, x : x + square] = 255
        frames.append(frame)
    return frames


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam2_rt_prompt_returns_non_empty_mask(sam2_rt_package: str) -> None:
    model = SAM2ForStream.from_pretrained(sam2_rt_package, device=DEFAULT_DEVICE)
    frame = _translating_square_frames(n_frames=1)[0]

    masks, obj_ids, state = model.prompt(
        image=frame,
        bboxes=[(50, 98, 110, 158)],
    )

    assert masks.dtype == bool
    assert masks.shape[1:] == frame.shape[:2]
    assert masks.shape[0] == 1
    assert masks[0].any(), "prompting should produce a non-empty mask"
    assert obj_ids.shape == (1,)
    assert isinstance(state, dict)


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam2_rt_track_follows_translating_square(sam2_rt_package: str) -> None:
    """Prompt on frame 0 then track on subsequent frames; the mask
    centroid should move right as the object translates right."""
    model = SAM2ForStream.from_pretrained(sam2_rt_package, device=DEFAULT_DEVICE)
    frames = _translating_square_frames(n_frames=4)

    masks, _, state = model.prompt(
        image=frames[0],
        bboxes=[(50, 98, 110, 158)],
    )
    ys, xs = np.where(masks[0])
    centroids_x = [xs.mean()]

    for frame in frames[1:]:
        masks, _, state = model.track(image=frame, state_dict=state)
        assert masks.shape[0] == 1
        assert masks[0].any()
        ys, xs = np.where(masks[0])
        centroids_x.append(xs.mean())

    assert centroids_x[-1] > centroids_x[0], (
        f"tracked centroid did not move right: {centroids_x}"
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam2_rt_track_without_prompt_raises(sam2_rt_package: str) -> None:
    model = SAM2ForStream.from_pretrained(sam2_rt_package, device=DEFAULT_DEVICE)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    with pytest.raises(ModelRuntimeError):
        model.track(image=frame)


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam2_rt_accepts_torch_tensor_input(sam2_rt_package: str) -> None:
    model = SAM2ForStream.from_pretrained(sam2_rt_package, device=DEFAULT_DEVICE)
    frame = _translating_square_frames(n_frames=1)[0]
    frame_tensor = torch.from_numpy(frame)

    masks, obj_ids, state = model.prompt(
        image=frame_tensor,
        bboxes=[(50, 98, 110, 158)],
    )
    assert masks.shape[0] == 1
    assert obj_ids.shape == (1,)
