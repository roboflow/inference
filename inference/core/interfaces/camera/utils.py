import time
from enum import Enum
from typing import Generator, Iterable, Optional, Tuple, Union

import numpy as np
from supervision.utils.video import FPSMonitor

from inference.core.interfaces.camera.entities import (
    FrameID,
    FrameTimestamp,
)
from inference.core.interfaces.camera.video_stream import (
    VideoStream,
)


class FPSLimiterStrategy(Enum):
    DROP = "drop"
    WAIT = "wait"


def get_video_frames_generator(
    stream: Union[VideoStream, str, int],
    max_fps: Optional[float] = None,
) -> Generator[Tuple[FrameTimestamp, FrameID, np.ndarray], None, None]:
    if not issubclass(type(stream), VideoStream):
        stream = VideoStream.init(
            stream_reference=stream,
        )
        stream.start()
    if max_fps is None:
        yield from stream
    limiter_strategy = FPSLimiterStrategy.DROP
    if stream.stream_properties.is_file:
        limiter_strategy = FPSLimiterStrategy.WAIT
    yield from limit_frame_rate(
        frames_generator=stream, max_fps=max_fps, strategy=limiter_strategy
    )


def limit_frame_rate(
    frames_generator: Iterable[Tuple[FrameTimestamp, FrameID, np.ndarray]],
    max_fps: float,
    strategy: FPSLimiterStrategy,
) -> Generator[Tuple[FrameTimestamp, FrameID, np.ndarray], None, None]:
    fps_monitor = FPSMonitor()
    for frame_data in frames_generator:
        fps_monitor.tick()
        current_fps = fps_monitor()
        if current_fps <= max_fps:
            yield frame_data
        if strategy is FPSLimiterStrategy.DROP:
            continue
        delay = 1 / max_fps - 1 / current_fps
        time.sleep(delay)
        yield frame_data
