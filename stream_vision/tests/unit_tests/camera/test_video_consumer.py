from queue import Queue
from unittest.mock import Mock

import numpy as np
import pytest
import supervision as sv
from streamvision.camera.buffer_strategies import BufferFillingStrategy
from streamvision.camera.video_source import SourceProperties, VideoConsumer


@pytest.mark.parametrize(
    "strategy",
    [BufferFillingStrategy.DROP_OLDEST, BufferFillingStrategy.ADAPTIVE_DROP_OLDEST],
)
def test_replacement_frame_keeps_declared_and_measured_fps(
    strategy: BufferFillingStrategy,
) -> None:
    source = SourceProperties(
        width=8, height=8, total_frames=0, is_file=False, fps=30.0
    )
    video = Mock()
    video.discover_source_properties.return_value = source
    video.grab.return_value = True
    video.retrieve.return_value = (True, np.zeros((8, 8, 3), dtype=np.uint8))
    stream_pace = Mock(spec=sv.FPSMonitor)
    consumer = VideoConsumer(
        buffer_filling_strategy=strategy,
        adaptive_mode_stream_pace_tolerance=0.1,
        adaptive_mode_reader_pace_tolerance=5.0,
        minimum_adaptive_mode_samples=10,
        maximum_adaptive_frames_dropped_in_row=16,
        status_update_handlers=[],
        reader_pace_monitor=sv.FPSMonitor(),
        stream_consumption_pace_monitor=stream_pace,
        decoding_pace_monitor=sv.FPSMonitor(),
        desired_fps=None,
        adaptive_backpressure=True,
    )
    consumer.reset(source_properties=source)
    buffer = Queue(maxsize=1)

    for measured_fps in (24.0, 20.0):
        stream_pace.fps = measured_fps
        assert consumer.consume_frame(
            video=video,
            declared_source_fps=source.fps,
            is_source_video_file=source.is_file,
            buffer=buffer,
            frames_buffering_allowed=True,
        )
        assert buffer.full()

    frame = buffer.get_nowait()
    assert frame.frame_id == 2
    assert (frame.fps, frame.measured_fps) == (30.0, 20.0)
