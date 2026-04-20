"""Integration tests for the streaming SAM2 video model wrapper.

These tests run the real SAM2 camera predictor against downloaded
weights.  They are marked ``slow`` and require GPU; on CI they pull
the existing ``sam2_tiny.zip`` package from
``storage.googleapis.com/roboflow-tests-assets``.
"""

import numpy as np
import pytest

try:
    from inference.models.sam2 import SegmentAnything2Video
except ModuleNotFoundError:
    # sam2 package is not installed on this runner.
    SegmentAnything2Video = None  # type: ignore


_VIDEO_ID = "test-stream"


def _translating_square_frames(
    n_frames: int = 4,
    size: int = 256,
    square: int = 60,
    step: int = 8,
) -> list:
    """Synthetic video: a white square on black translating right."""
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
    SegmentAnything2Video is None, reason="sam2 package not installed"
)
def test_sam2_video_prompt_then_track_across_frames(sam2_tiny_model: str) -> None:
    """Seed a session on frame 0 with a bbox, then propagate forward.

    Verifies we get a non-empty mask on every frame and that the
    centroid of the tracked mask moves in the direction we translated
    the synthetic object.
    """
    model = SegmentAnything2Video(model_id=sam2_tiny_model)
    frames = _translating_square_frames()

    # Frame 0 — prompt with a bbox around the square.
    masks0, obj_ids0 = model.prompt_and_track(
        video_id=_VIDEO_ID,
        frame=frames[0],
        boxes_xyxy=[(50.0, 98.0, 110.0, 158.0)],
        clear_old_prompts=True,
    )
    assert masks0.shape[0] == 1
    assert masks0[0].any(), "first frame mask should be non-empty"
    assert obj_ids0.shape == (1,)

    # Record the centroid of frame 0.
    ys0, xs0 = np.where(masks0[0])
    centroid_x0 = xs0.mean()

    # Subsequent frames — propagate.
    centroids_x = [centroid_x0]
    for frame in frames[1:]:
        masks, _ = model.track(video_id=_VIDEO_ID, frame=frame)
        assert masks.shape[0] == 1
        assert masks[0].any(), "tracking mask should be non-empty"
        ys, xs = np.where(masks[0])
        centroids_x.append(xs.mean())

    # Centroid should have moved right as the object translates right.
    assert centroids_x[-1] > centroids_x[0], (
        f"tracked centroid did not move right: {centroids_x}"
    )

    model.reset_session(_VIDEO_ID)
    assert not model.has_session(_VIDEO_ID)


@pytest.mark.slow
@pytest.mark.skipif(
    SegmentAnything2Video is None, reason="sam2 package not installed"
)
def test_sam2_video_track_without_prompt_raises(sam2_tiny_model: str) -> None:
    model = SegmentAnything2Video(model_id=sam2_tiny_model)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="No SAM2 video session"):
        model.track(video_id="never-prompted", frame=frame)


@pytest.mark.slow
@pytest.mark.skipif(
    SegmentAnything2Video is None, reason="sam2 package not installed"
)
def test_sam2_video_separate_streams_have_independent_ids(
    sam2_tiny_model: str,
) -> None:
    """Each ``video_id`` maintains its own obj_id counter."""
    model = SegmentAnything2Video(model_id=sam2_tiny_model)
    frame = _translating_square_frames(n_frames=1)[0]

    masks_a, ids_a = model.prompt_and_track(
        video_id="stream-A",
        frame=frame,
        boxes_xyxy=[(50.0, 98.0, 110.0, 158.0)],
        clear_old_prompts=True,
    )
    masks_b, ids_b = model.prompt_and_track(
        video_id="stream-B",
        frame=frame,
        boxes_xyxy=[(50.0, 98.0, 110.0, 158.0)],
        clear_old_prompts=True,
    )

    assert masks_a.shape[0] == 1 and masks_b.shape[0] == 1
    # Each stream starts its counter at 0 — sessions do not share state.
    assert ids_a.tolist() == [0]
    assert ids_b.tolist() == [0]
