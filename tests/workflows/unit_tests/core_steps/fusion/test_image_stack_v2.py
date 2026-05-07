from datetime import datetime

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.fusion.image_stack.v2 import (
    BlockManifest,
    ImageStackBlockV2,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


def _make_image(
    width: int = 320,
    height: int = 240,
    video_id: str = "cam-1",
    frame_number: int = 0,
    fps: int = 30,
) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=video_id),
        numpy_image=np.random.randint(0, 255, (height, width, 3), dtype=np.uint8),
        video_metadata=VideoMetadata(
            video_identifier=video_id,
            frame_number=frame_number,
            frame_timestamp=datetime.now(),
            fps=fps,
            comes_from_video_file=None,
        ),
    )


# ── Manifest validation ─────────────────────────────────────────────


def test_manifest_defaults() -> None:
    raw = {
        "type": "roboflow_core/image_stack@v2",
        "name": "stack",
        "image": "$inputs.image",
    }
    result = BlockManifest.model_validate(raw)
    assert result.window_seconds == 0.5
    assert result.subsample_fps is None
    assert result.clear is False


def test_manifest_custom_values() -> None:
    raw = {
        "type": "roboflow_core/image_stack@v2",
        "name": "stack",
        "image": "$inputs.image",
        "window_seconds": 1.0,
        "subsample_fps": 5.0,
        "resolution_width": 512,
        "resolution_height": 512,
    }
    result = BlockManifest.model_validate(raw)
    assert result.window_seconds == 1.0
    assert result.subsample_fps == 5.0
    assert result.resolution_width == 512


def test_manifest_rejects_invalid_window_seconds() -> None:
    raw = {
        "type": "roboflow_core/image_stack@v2",
        "name": "stack",
        "image": "$inputs.image",
        "window_seconds": 0.0,
    }
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(raw)


def test_manifest_rejects_invalid_subsample_fps() -> None:
    raw = {
        "type": "roboflow_core/image_stack@v2",
        "name": "stack",
        "image": "$inputs.image",
        "subsample_fps": 0.0,
    }
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(raw)


def test_manifest_subsample_fps_can_be_none() -> None:
    raw = {
        "type": "roboflow_core/image_stack@v2",
        "name": "stack",
        "image": "$inputs.image",
        "subsample_fps": None,
    }
    result = BlockManifest.model_validate(raw)
    assert result.subsample_fps is None


# ── Sampling behavior ────────────────────────────────────────────────


def _run(
    block,
    image,
    *,
    window_seconds=0.5,
    subsample_fps=None,
    clear=False,
):
    return block.run(
        image=image,
        window_seconds=window_seconds,
        subsample_fps=subsample_fps,
        resolution_width=256,
        resolution_height=256,
        clear=clear,
    )


def test_buffer_size_derived_from_source_fps_when_no_subsample() -> None:
    """Without subsample_fps set, the buffer should size to
    ceil(window_seconds * source_fps). At 30 fps source, 0.5s = 15 frames."""
    block = ImageStackBlockV2()
    last_count = 0
    for i in range(60):
        result = _run(
            block,
            _make_image(frame_number=i, fps=30),
            window_seconds=0.5,
        )
        last_count = result["frames_count"]
    assert last_count == 15


def test_buffer_size_with_subsample_fps() -> None:
    """With subsample_fps=10 and window 0.5s, buffer holds ceil(0.5 * 10) = 5
    frames once steady-state."""
    block = ImageStackBlockV2()
    last_count = 0
    for i in range(60):
        result = _run(
            block,
            _make_image(frame_number=i, fps=30),
            window_seconds=0.5,
            subsample_fps=10.0,
        )
        last_count = result["frames_count"]
    assert last_count == 5


def test_subsample_fps_drops_intermediate_frames() -> None:
    """At 30 fps source with subsample_fps=10, only frames at the right
    cadence get sampled (≈ every 3rd source frame)."""
    block = ImageStackBlockV2()
    state_id = "cam-1"
    sampled_indices = []
    prev_t = None
    for i in range(30):
        _run(
            block,
            _make_image(video_id=state_id, frame_number=i, fps=30),
            window_seconds=0.5,
            subsample_fps=10.0,
        )
        t = block._states[state_id].last_sampled_t
        if t != prev_t:
            sampled_indices.append(i)
            prev_t = t

    assert sampled_indices[0] == 0
    gaps = [b - a for a, b in zip(sampled_indices, sampled_indices[1:])]
    assert all(g in (3, 4) for g in gaps), gaps
    # Average gap should be close to 3.33 (= 100ms / 33ms-per-frame).
    assert 3.0 <= sum(gaps) / len(gaps) <= 3.7
    assert 8 <= len(sampled_indices) <= 11


def test_subsample_higher_than_source_is_capped() -> None:
    """If subsample_fps > source_fps, we can't sample faster than frames
    arrive — should silently cap at source_fps."""
    block = ImageStackBlockV2()
    last_count = 0
    for i in range(60):
        result = _run(
            block,
            _make_image(frame_number=i, fps=10),
            window_seconds=1.0,
            subsample_fps=120.0,  # absurdly high
        )
        last_count = result["frames_count"]
    # Source = 10 fps, window = 1.0s → buffer caps at 10 frames.
    assert last_count == 10


def test_clear_empties_buffer() -> None:
    block = ImageStackBlockV2()
    for i in range(20):
        _run(block, _make_image(frame_number=i, fps=30), subsample_fps=10.0)
    result = _run(
        block,
        _make_image(frame_number=20, fps=30),
        subsample_fps=10.0,
        clear=True,
    )
    assert result["frames_count"] == 1


def test_isolated_buffers_per_video() -> None:
    block = ImageStackBlockV2()
    for i in range(20):
        _run(block, _make_image(video_id="cam-A", frame_number=i, fps=30), subsample_fps=10.0)
        _run(block, _make_image(video_id="cam-B", frame_number=i, fps=30), subsample_fps=10.0)
    a_state = block._states["cam-A"]
    b_state = block._states["cam-B"]
    assert a_state is not b_state
    assert len(a_state.frames) == len(b_state.frames) == 5


def test_returned_frames_are_jpeg_bytes() -> None:
    block = ImageStackBlockV2()
    result = _run(block, _make_image(frame_number=0, fps=30))
    frames = result["frames"]
    assert len(frames) == 1
    assert isinstance(frames[0], bytes)
    assert frames[0][:2] == b"\xff\xd8"


def test_falls_back_to_default_fps_when_metadata_missing() -> None:
    """If video_metadata.fps is 0/missing, the block should default to the
    fallback (currently 30 fps) and still process frames."""
    block = ImageStackBlockV2()

    bad_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="x"),
        numpy_image=np.zeros((100, 100, 3), dtype=np.uint8),
        video_metadata=VideoMetadata(
            video_identifier="x",
            frame_number=0,
            frame_timestamp=datetime.now(),
            fps=0,  # invalid
            comes_from_video_file=None,
        ),
    )

    result = _run(block, bad_image)
    assert result["frames_count"] == 1
    # Default fallback fps = 30; with window 0.5s buffer caps at 15.
    # Only one frame in so far, but maxlen should be 15:
    assert block._states["x"].frames.maxlen == 15


def test_buffer_resizes_when_params_change() -> None:
    """If derived buffer size changes mid-stream, the buffer resizes while
    preserving the most recent frames."""
    block = ImageStackBlockV2()
    for i in range(20):
        _run(
            block,
            _make_image(frame_number=i, fps=30),
            window_seconds=0.5,
            subsample_fps=10.0,
        )
    result = _run(
        block,
        _make_image(frame_number=21, fps=30),
        window_seconds=0.2,
        subsample_fps=10.0,
    )
    # ceil(0.2 * 10) = 2
    assert result["frames_count"] <= 2
