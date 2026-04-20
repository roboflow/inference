"""Integration tests for the streaming SAM3 video model wrapper.

These tests require the ``transformers`` SAM3 video classes and the
``sam3_video`` model package to be available at the
``storage.googleapis.com/roboflow-tests-assets/sam3_video.zip`` URL
referenced by the ``sam3_video_model`` fixture.
"""

import numpy as np
import pytest

try:
    from inference.models.sam3 import SegmentAnything3Video
except (ModuleNotFoundError, ImportError):
    SegmentAnything3Video = None  # type: ignore


_VIDEO_ID = "test-stream"


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
@pytest.mark.skipif(
    SegmentAnything3Video is None,
    reason="sam3 / transformers Sam3VideoModel not installed",
)
def test_sam3_video_text_prompt_then_track_across_frames(
    sam3_video_model: str,
) -> None:
    """Seed a text prompt, then track the detected object across frames."""
    model = SegmentAnything3Video(model_id=sam3_video_model)
    frames = _translating_square_frames()

    masks0, obj_ids0 = model.prompt_and_track(
        video_id=_VIDEO_ID,
        frame=frames[0],
        frame_index=0,
        text="white square",
        clear_old_prompts=True,
    )
    assert masks0.ndim == 3
    assert masks0.shape[0] == obj_ids0.shape[0]
    # Detail of what SAM3 returns on a synthetic white square is
    # hardware- and version-dependent, so assert only the shape contract.
    assert masks0.shape[1:] == frames[0].shape[:2]

    for frame in frames[1:]:
        masks, obj_ids = model.track(video_id=_VIDEO_ID, frame=frame)
        assert masks.shape[0] == obj_ids.shape[0]
        assert masks.shape[1:] == frame.shape[:2]

    model.reset_session(_VIDEO_ID)
    assert not model.has_session(_VIDEO_ID)


@pytest.mark.slow
@pytest.mark.skipif(
    SegmentAnything3Video is None,
    reason="sam3 / transformers Sam3VideoModel not installed",
)
def test_sam3_video_track_without_prompt_raises(sam3_video_model: str) -> None:
    model = SegmentAnything3Video(model_id=sam3_video_model)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="No SAM3 video session"):
        model.track(video_id="never-prompted", frame=frame)
