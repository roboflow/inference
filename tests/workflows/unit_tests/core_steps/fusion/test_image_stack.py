from datetime import datetime

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.fusion.image_stack.v1 import (
    MAX_RESOLUTION_HEIGHT,
    MAX_RESOLUTION_WIDTH,
    MAX_STACK_SIZE,
    BlockManifest,
    ImageStackBlockV1,
    _compress_frame,
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
) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=video_id),
        numpy_image=np.random.randint(0, 255, (height, width, 3), dtype=np.uint8),
        video_metadata=VideoMetadata(
            video_identifier=video_id,
            frame_number=frame_number,
            frame_timestamp=datetime.now(),
            fps=30,
            comes_from_video_file=None,
        ),
    )


# ── Manifest validation ─────────────────────────────────────────────


def test_manifest_with_defaults() -> None:
    raw = {
        "type": "roboflow_core/image_stack@v1",
        "name": "stack1",
        "image": "$inputs.image",
    }
    result = BlockManifest.model_validate(raw)
    assert result.stack_size == 10
    assert result.resolution_width == MAX_RESOLUTION_WIDTH
    assert result.resolution_height == MAX_RESOLUTION_HEIGHT
    assert result.clear is False


def test_manifest_with_custom_values() -> None:
    raw = {
        "type": "roboflow_core/image_stack@v1",
        "name": "stack1",
        "image": "$inputs.image",
        "stack_size": 5,
        "resolution_width": 640,
        "resolution_height": 480,
        "clear": "$inputs.reset",
    }
    result = BlockManifest.model_validate(raw)
    assert result.stack_size == 5
    assert result.resolution_width == 640
    assert result.resolution_height == 480


def test_manifest_rejects_stack_size_too_large() -> None:
    raw = {
        "type": "roboflow_core/image_stack@v1",
        "name": "stack1",
        "image": "$inputs.image",
        "stack_size": MAX_STACK_SIZE + 1,
    }
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(raw)


def test_manifest_rejects_stack_size_zero() -> None:
    raw = {
        "type": "roboflow_core/image_stack@v1",
        "name": "stack1",
        "image": "$inputs.image",
        "stack_size": 0,
    }
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(raw)


def test_manifest_rejects_resolution_too_large() -> None:
    raw = {
        "type": "roboflow_core/image_stack@v1",
        "name": "stack1",
        "image": "$inputs.image",
        "resolution_width": MAX_RESOLUTION_WIDTH + 1,
    }
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(raw)


def test_manifest_accepts_selectors_for_int_fields() -> None:
    raw = {
        "type": "roboflow_core/image_stack@v1",
        "name": "stack1",
        "image": "$inputs.image",
        "stack_size": "$inputs.stack_size",
        "resolution_width": "$inputs.width",
        "resolution_height": "$inputs.height",
    }
    result = BlockManifest.model_validate(raw)
    assert result.stack_size == "$inputs.stack_size"
    assert result.resolution_width == "$inputs.width"
    assert result.resolution_height == "$inputs.height"


# ── _compress_frame helper ───────────────────────────────────────────


def test_compress_frame_returns_jpeg_bytes() -> None:
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    result = _compress_frame(img, max_width=1920, max_height=1080)
    assert isinstance(result, bytes)
    # JPEG magic bytes
    assert result[:2] == b"\xff\xd8"


def test_compress_frame_downsamples_wide_image() -> None:
    img = np.zeros((1080, 3840, 3), dtype=np.uint8)
    result = _compress_frame(img, max_width=1920, max_height=1080)
    # decode and check dimensions
    decoded = cv2.imdecode(np.frombuffer(result, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] <= 1920
    assert decoded.shape[0] <= 1080


def test_compress_frame_downsamples_tall_image() -> None:
    img = np.zeros((2160, 1000, 3), dtype=np.uint8)
    result = _compress_frame(img, max_width=1920, max_height=1080)
    decoded = cv2.imdecode(np.frombuffer(result, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[0] <= 1080
    assert decoded.shape[1] <= 1920


def test_compress_frame_preserves_small_image_dimensions() -> None:
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    result = _compress_frame(img, max_width=1920, max_height=1080)
    decoded = cv2.imdecode(np.frombuffer(result, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (100, 200)


def test_compress_frame_handles_grayscale() -> None:
    img = np.zeros((100, 200), dtype=np.uint8)
    result = _compress_frame(img, max_width=1920, max_height=1080)
    assert isinstance(result, bytes)
    assert result[:2] == b"\xff\xd8"


def test_compress_frame_handles_rgba() -> None:
    img = np.zeros((100, 200, 4), dtype=np.uint8)
    result = _compress_frame(img, max_width=1920, max_height=1080)
    assert isinstance(result, bytes)
    assert result[:2] == b"\xff\xd8"


# ── Block accumulation ───────────────────────────────────────────────


def test_single_frame_added() -> None:
    block = ImageStackBlockV1()
    result = block.run(
        image=_make_image(),
        stack_size=3,
        resolution_width=1920,
        resolution_height=1080,
        clear=False,
    )
    assert result["frames_count"] == 1
    assert len(result["frames"]) == 1
    assert isinstance(result["frames"][0], bytes)


def test_frames_accumulate_up_to_stack_size() -> None:
    block = ImageStackBlockV1()
    for i in range(5):
        result = block.run(
            image=_make_image(frame_number=i),
            stack_size=3,
            resolution_width=1920,
            resolution_height=1080,
            clear=False,
        )
    assert result["frames_count"] == 3
    assert len(result["frames"]) == 3


def test_newest_frame_is_first() -> None:
    block = ImageStackBlockV1()
    # Add two distinct frames
    img1 = _make_image(width=100, height=100, frame_number=0)
    img2 = _make_image(width=100, height=100, frame_number=1)

    block.run(
        image=img1,
        stack_size=5,
        resolution_width=1920,
        resolution_height=1080,
        clear=False,
    )
    result = block.run(
        image=img2,
        stack_size=5,
        resolution_width=1920,
        resolution_height=1080,
        clear=False,
    )
    # newest (img2) at index 0, oldest (img1) at index 1
    assert len(result["frames"]) == 2
    assert result["frames"][0] != result["frames"][1]


def test_oldest_frame_evicted_on_overflow() -> None:
    block = ImageStackBlockV1()
    frames_added = []
    for i in range(4):
        result = block.run(
            image=_make_image(frame_number=i),
            stack_size=2,
            resolution_width=1920,
            resolution_height=1080,
            clear=False,
        )
        frames_added.append(result["frames"][0])  # newest each time

    # only last 2 frames remain
    assert result["frames_count"] == 2
    assert result["frames"][0] == frames_added[3]
    assert result["frames"][1] == frames_added[2]


# ── Clear ────────────────────────────────────────────────────────────


def test_clear_flushes_buffer_then_adds_current() -> None:
    block = ImageStackBlockV1()
    # fill buffer
    for i in range(3):
        block.run(
            image=_make_image(frame_number=i),
            stack_size=5,
            resolution_width=1920,
            resolution_height=1080,
            clear=False,
        )

    # clear and add one frame
    result = block.run(
        image=_make_image(frame_number=99),
        stack_size=5,
        resolution_width=1920,
        resolution_height=1080,
        clear=True,
    )
    assert result["frames_count"] == 1


def test_clear_false_preserves_buffer() -> None:
    block = ImageStackBlockV1()
    block.run(
        image=_make_image(frame_number=0),
        stack_size=5,
        resolution_width=1920,
        resolution_height=1080,
        clear=False,
    )
    result = block.run(
        image=_make_image(frame_number=1),
        stack_size=5,
        resolution_width=1920,
        resolution_height=1080,
        clear=False,
    )
    assert result["frames_count"] == 2


# ── Per-camera isolation ─────────────────────────────────────────────


def test_buffers_isolated_per_camera() -> None:
    block = ImageStackBlockV1()

    # feed 3 frames to cam-1
    for i in range(3):
        block.run(
            image=_make_image(video_id="cam-1", frame_number=i),
            stack_size=5,
            resolution_width=1920,
            resolution_height=1080,
            clear=False,
        )

    # feed 1 frame to cam-2
    result_cam2 = block.run(
        image=_make_image(video_id="cam-2", frame_number=0),
        stack_size=5,
        resolution_width=1920,
        resolution_height=1080,
        clear=False,
    )
    assert result_cam2["frames_count"] == 1

    # cam-1 still has its own count
    result_cam1 = block.run(
        image=_make_image(video_id="cam-1", frame_number=3),
        stack_size=5,
        resolution_width=1920,
        resolution_height=1080,
        clear=False,
    )
    assert result_cam1["frames_count"] == 4


def test_clear_only_affects_targeted_camera() -> None:
    block = ImageStackBlockV1()

    # fill both cameras
    for i in range(3):
        block.run(
            image=_make_image(video_id="cam-1", frame_number=i),
            stack_size=5,
            resolution_width=1920,
            resolution_height=1080,
            clear=False,
        )
        block.run(
            image=_make_image(video_id="cam-2", frame_number=i),
            stack_size=5,
            resolution_width=1920,
            resolution_height=1080,
            clear=False,
        )

    # clear cam-1 only
    result_cam1 = block.run(
        image=_make_image(video_id="cam-1", frame_number=99),
        stack_size=5,
        resolution_width=1920,
        resolution_height=1080,
        clear=True,
    )
    assert result_cam1["frames_count"] == 1

    # cam-2 unaffected
    result_cam2 = block.run(
        image=_make_image(video_id="cam-2", frame_number=3),
        stack_size=5,
        resolution_width=1920,
        resolution_height=1080,
        clear=False,
    )
    assert result_cam2["frames_count"] == 4


# ── Stack size change mid-stream ─────────────────────────────────────


def test_stack_size_shrink_preserves_newest_frames() -> None:
    block = ImageStackBlockV1()

    # fill with stack_size=3
    for i in range(3):
        block.run(
            image=_make_image(frame_number=i),
            stack_size=3,
            resolution_width=1920,
            resolution_height=1080,
            clear=False,
        )

    # shrink stack_size to 2 — preserves 2 newest + adds current frame
    result = block.run(
        image=_make_image(frame_number=10),
        stack_size=2,
        resolution_width=1920,
        resolution_height=1080,
        clear=False,
    )
    # deque(old_3_items, maxlen=2) keeps 2 newest, then appendleft adds 1 more
    # but maxlen=2 evicts oldest → 2 frames
    assert result["frames_count"] == 2


def test_stack_size_grow_preserves_all_frames() -> None:
    block = ImageStackBlockV1()

    # fill with stack_size=2
    for i in range(2):
        block.run(
            image=_make_image(frame_number=i),
            stack_size=2,
            resolution_width=1920,
            resolution_height=1080,
            clear=False,
        )

    # grow stack_size to 5 — all 2 existing frames preserved + new one
    result = block.run(
        image=_make_image(frame_number=10),
        stack_size=5,
        resolution_width=1920,
        resolution_height=1080,
        clear=False,
    )
    assert result["frames_count"] == 3


# ── Resolution enforcement ───────────────────────────────────────────


def test_custom_resolution_limits_applied() -> None:
    block = ImageStackBlockV1()
    # 800x600 image with 400x300 cap
    result = block.run(
        image=_make_image(width=800, height=600),
        stack_size=3,
        resolution_width=400,
        resolution_height=300,
        clear=False,
    )
    jpeg_bytes = result["frames"][0]
    decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] <= 400
    assert decoded.shape[0] <= 300
