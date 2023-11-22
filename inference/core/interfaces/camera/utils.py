import time
from enum import Enum
from typing import Generator, Iterable, Optional, Tuple, Union

import numpy as np

from inference.core.interfaces.camera.entities import FrameID, FrameTimestamp
from inference.core.interfaces.camera.video_source import SourceProperties, VideoSource

MINIMAL_FPS = 0.01


class FPSLimiterStrategy(Enum):
    DROP = "drop"
    WAIT = "wait"


def get_video_frames_generator(
    stream: Union[VideoSource, str, int],
    max_fps: Optional[float] = None,
    limiter_strategy: Optional[FPSLimiterStrategy] = None,
) -> Generator[Tuple[FrameTimestamp, FrameID, np.ndarray], None, None]:
    if not issubclass(type(stream), VideoSource):
        stream = VideoSource.init(
            video_reference=stream,
        )
        stream.start()
    if max_fps is None:
        yield from stream
        return None
    limiter_strategy = resolve_limiter_strategy(
        explicitly_defined_strategy=limiter_strategy,
        source_properties=stream.describe_source().source_properties,
    )
    yield from limit_frame_rate(
        frames_generator=stream, max_fps=max_fps, strategy=limiter_strategy
    )


def resolve_limiter_strategy(
    explicitly_defined_strategy: Optional[FPSLimiterStrategy],
    source_properties: Optional[SourceProperties],
) -> FPSLimiterStrategy:
    if explicitly_defined_strategy is not None:
        return explicitly_defined_strategy
    limiter_strategy = FPSLimiterStrategy.DROP
    if source_properties is not None and source_properties.is_file:
        limiter_strategy = FPSLimiterStrategy.WAIT
    return limiter_strategy


def limit_frame_rate(
    frames_generator: Iterable[Tuple[FrameTimestamp, FrameID, np.ndarray]],
    max_fps: float,
    strategy: FPSLimiterStrategy,
) -> Generator[Tuple[FrameTimestamp, FrameID, np.ndarray], None, None]:
    rate_limiter = RateLimiter(desired_fps=max_fps)
    for frame_data in frames_generator:
        delay = rate_limiter.estimate_next_tick_delay()
        if delay <= 0.0:
            rate_limiter.tick()
            yield frame_data
            continue
        if strategy is FPSLimiterStrategy.WAIT:
            time.sleep(delay)
            rate_limiter.tick()
            yield frame_data


class RateLimiter:
    """
    Implements rate upper-bound rate limiting by ensuring estimate_next_tick_delay()
    to be at min 1 / desired_fps, not letting the client obeying outcomes to exceed
    assumed rate.
    """

    def __init__(self, desired_fps: Union[float, int]):
        self._desired_fps = max(desired_fps, MINIMAL_FPS)
        self._last_tick: Optional[float] = None

    def tick(self) -> None:
        self._last_tick = time.monotonic()

    def estimate_next_tick_delay(self) -> float:
        if self._last_tick is None:
            return 0.0
        desired_delay = 1 / self._desired_fps
        time_since_last_tick = time.monotonic() - self._last_tick
        return max(desired_delay - time_since_last_tick, 0.0)
