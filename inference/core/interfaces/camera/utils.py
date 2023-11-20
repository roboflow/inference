import math
import time
from enum import Enum
from typing import Generator, Iterable, Optional, Tuple, Union

import numpy as np
from supervision.utils.video import FPSMonitor

from inference.core.interfaces.camera.entities import (
    FrameID,
    FrameTimestamp,
)
from inference.core.interfaces.camera.video_source import (
    VideoSource,
)


class FPSLimiterStrategy(Enum):
    DROP = "drop"
    WAIT = "wait"


def get_video_frames_generator(
    stream: Union[VideoSource, str, int],
    max_fps: Optional[float] = None,
) -> Generator[Tuple[FrameTimestamp, FrameID, np.ndarray], None, None]:
    if not issubclass(type(stream), VideoSource):
        stream = VideoSource.init(
            stream_reference=stream,
        )
        stream.start()
    if max_fps is None:
        yield from stream
        return None
    limiter_strategy = FPSLimiterStrategy.DROP
    stream_properties = stream.describe_source().stream_properties
    if stream_properties is not None and stream_properties.is_file:
        limiter_strategy = FPSLimiterStrategy.WAIT
    yield from limit_frame_rate(
        frames_generator=stream, max_fps=max_fps, strategy=limiter_strategy
    )


def limit_frame_rate(
    frames_generator: Iterable[Tuple[FrameTimestamp, FrameID, np.ndarray]],
    max_fps: float,
    strategy: FPSLimiterStrategy,
) -> Generator[Tuple[FrameTimestamp, FrameID, np.ndarray], None, None]:
    fps_monitor = FPSMonitor(sample_size=2*math.ceil(max_fps))
    max_single_delay = 1 / max_fps
    for frame_data in frames_generator:
        current_fps = fps_monitor()
        if abs(current_fps) < 1e-5:
            current_fps = max_fps
        if current_fps <= max_fps:
            fps_monitor.tick()
            yield frame_data
            continue
        if strategy is FPSLimiterStrategy.DROP:
            continue
        delay = min((1 / max_fps - 1 / current_fps) * len(fps_monitor.all_timestamps), max_single_delay)
        time.sleep(delay)
        fps_monitor.tick()
        yield frame_data
