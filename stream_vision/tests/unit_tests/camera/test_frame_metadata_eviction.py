"""A full camera buffer must not strip metadata from its replacement frame."""

from queue import Queue
from unittest.mock import MagicMock

import numpy as np
import pytest
from streamvision.camera.buffer_strategies import BufferFillingStrategy
from streamvision.camera.entities import SourceProperties
from streamvision.camera.video_source import VideoConsumer


@pytest.mark.parametrize(
    "strategy",
    [BufferFillingStrategy.DROP_OLDEST, BufferFillingStrategy.ADAPTIVE_DROP_OLDEST],
)
@pytest.mark.parametrize("declared_fps", [29.97, 0.0, None])
def test_eviction_preserves_declared_and_measured_fps(strategy, declared_fps):
    consumer = VideoConsumer.init(
        buffer_filling_strategy=strategy,
        adaptive_mode_stream_pace_tolerance=0.1,
        adaptive_mode_reader_pace_tolerance=5.0,
        minimum_adaptive_mode_samples=10,
        maximum_adaptive_frames_dropped_in_row=16,
        status_update_handlers=[],
        adaptive_backpressure=True,
    )
    consumer._stream_consumption_pace_monitor = MagicMock(fps=28.5)
    video = MagicMock()
    video.grab.return_value = True
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    video.retrieve.return_value = (True, pixels)
    video.discover_source_properties.return_value = SourceProperties(
        width=2, height=2, total_frames=0, is_file=False, fps=declared_fps
    )
    frames = Queue(maxsize=1)

    def consume():
        assert consumer.consume_frame(
            video=video,
            declared_source_fps=declared_fps,
            is_source_video_file=False,
            buffer=frames,
            frames_buffering_allowed=True,
            source_id=3,
        )

    consume()
    first = frames.get_nowait()
    frames.task_done()
    frames.put(first)
    consume()
    replacement = frames.get_nowait()
    frames.task_done()

    assert first.frame_id == 1
    assert replacement.frame_id == 2
    for frame in (first, replacement):
        assert frame.fps == declared_fps
        assert frame.measured_fps == 28.5
        assert frame.source_id == 3
        assert frame.comes_from_video_file is False
        assert frame.image is pixels
    assert frames.empty()
