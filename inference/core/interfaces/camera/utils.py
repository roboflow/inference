import time
from datetime import datetime, timedelta
from enum import Enum
from threading import Thread
from typing import Callable, Dict, Generator, Iterable, List, Optional, TypeVar, Union

from inference.core import logger
from inference.core.env import RESTART_ATTEMPT_DELAY
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.camera.exceptions import (
    EndOfStreamError,
    SourceConnectionError,
)
from inference.core.interfaces.camera.video_source import (
    SourceProperties,
    StreamState,
    VideoSource,
)

MINIMAL_FPS = 0.01

T = TypeVar("T")


class FPSLimiterStrategy(Enum):
    DROP = "drop"
    WAIT = "wait"


def never_stop() -> bool:
    return False


def log_error(source_id: Optional[int], error: SourceConnectionError) -> None:
    logger.warning(
        f"Could not re-connect to source with id: {source_id}. Error: {error}"
    )


def multiplex_videos(
    videos: List[Union[VideoSource, str, int]],
    max_fps: Optional[Union[float, int]] = None,
    limiter_strategy: FPSLimiterStrategy = FPSLimiterStrategy.DROP,
    batch_collection_timeout: Optional[float] = None,
    reconnect: bool = True,
    should_stop: Callable[[], bool] = never_stop,
    on_reconnection_error: Callable[
        [Optional[int], SourceConnectionError], None
    ] = log_error,
) -> Generator[List[VideoFrame], None, None]:
    generator = _multiplex_videos(
        videos=videos,
        batch_collection_timeout=batch_collection_timeout,
        reconnect=reconnect,
        should_stop=should_stop,
        on_reconnection_error=on_reconnection_error,
    )
    if max_fps is None:
        yield from generator
        return None
    yield from limit_frame_rate(
        frames_generator=generator, max_fps=max_fps, strategy=limiter_strategy
    )


def _multiplex_videos(
    videos: List[Union[VideoSource, str, int]],
    batch_collection_timeout: Optional[float] = None,
    reconnect: bool = True,
    should_stop: Callable[[], bool] = never_stop,
    on_reconnection_error: Callable[
        [Optional[int], SourceConnectionError], None
    ] = log_error,
) -> Generator[List[VideoFrame], None, None]:
    initialised_videos: List[VideoSource] = []
    internally_created_sources: List[VideoSource] = []
    minimal_free_source_id = max(
        v.source_id if v.source_id is not None else -1
        for v in videos
        if issubclass(type(v), VideoSource)
    )
    minimal_free_source_id += 1
    for video in videos:
        if issubclass(type(video), str) or issubclass(type(video), int):
            video = VideoSource.init(
                video_reference=video, source_id=minimal_free_source_id
            )
            minimal_free_source_id += 1
            video.start()
            internally_created_sources.append(video)
        initialised_videos.append(video)
    sources_properties = [
        s.describe_source().source_properties for s in initialised_videos
    ]
    if any(properties is None for properties in sources_properties):
        logger.warning("Could not connect to all sources.")
        return None
    allow_reconnection = [not s.is_file and reconnect for s in sources_properties]
    reconnection_threads: Dict[str, Thread] = {}
    ended_sources = set()
    while len(ended_sources) < len(initialised_videos):
        batch_frames = []
        if batch_collection_timeout is not None:
            batch_timeout_moment = datetime.now() + timedelta(
                seconds=batch_collection_timeout
            )
        else:
            batch_timeout_moment = None
        for video_id, (source, source_should_reconnect) in enumerate(
            zip(initialised_videos, allow_reconnection)
        ):
            if should_stop():
                print("END")
                return None
            batch_time_left = (
                None
                if batch_timeout_moment is None
                else max((batch_timeout_moment - datetime.now()).total_seconds(), 0.0)
            )
            try:
                frame = source.read_frame(timeout=batch_time_left)
                if frame is not None:
                    print(f"Got frame from: {video_id}")
                    batch_frames.append(frame)
                    if video_id in reconnection_threads:
                        reconnection_threads[video_id].join()
                        del reconnection_threads[video_id]
            except EndOfStreamError:
                print(f"Source {video_id} disconnected")
                if source_should_reconnect:
                    print("Staring thread")
                    reconnection_threads[video_id] = Thread(
                        target=attempt_reconnect,
                        args=(source, should_stop, on_reconnection_error),
                    )
                    reconnection_threads[video_id].start()
                else:
                    ended_sources.add(video_id)
        if len(batch_frames) > 0:
            yield batch_frames
    for v in internally_created_sources:
        v.terminate()
    return None


def attempt_reconnect(
    video_source: VideoSource,
    should_stop: Callable[[], bool],
    on_reconnection_error: Callable[[Optional[int], SourceConnectionError], None],
) -> None:
    succeeded = False
    while not should_stop() and not succeeded:
        try:
            video_source.restart()
            succeeded = True
        except SourceConnectionError as error:
            on_reconnection_error(video_source.source_id, error)
            if should_stop():
                return None
            logger.warning(
                f"Could not connect to video source. Retrying in {RESTART_ATTEMPT_DELAY}s..."
            )
            time.sleep(RESTART_ATTEMPT_DELAY)


def get_video_frames_generator(
    video: Union[VideoSource, str, int],
    max_fps: Optional[Union[float, int]] = None,
    limiter_strategy: Optional[FPSLimiterStrategy] = None,
) -> Generator[VideoFrame, None, None]:
    """
    Util function to create a frames generator from `VideoSource` with possibility to
    limit FPS of consumed frames and dictate what to do if frames are produced to fast.

    Args:
        video (Union[VideoSource, str, int]): Either instance of VideoSource or video reference accepted
            by VideoSource.init(...)
        max_fps (Optional[Union[float, int]]): value of maximum FPS rate of generated frames - can be used to limit
            generation frequency
        limiter_strategy (Optional[FPSLimiterStrategy]): strategy used to deal with frames decoding exceeding
            limit of `max_fps`. By default - for files, in the interest of processing all frames -
            generation will be awaited, for streams - frames will be dropped on the floor.
    Returns: generator of `VideoFrame`

    Example:
        ```python
        for frame in get_video_frames_generator(
            video="./some.mp4",
            max_fps=50,
        ):
             pass
        ```
    """
    if issubclass(type(video), str) or issubclass(type(video), int):
        video = VideoSource.init(
            video_reference=video,
        )
        video.start()
    if max_fps is None:
        yield from video
        return None
    limiter_strategy = resolve_limiter_strategy(
        explicitly_defined_strategy=limiter_strategy,
        source_properties=video.describe_source().source_properties,
    )
    yield from limit_frame_rate(
        frames_generator=video, max_fps=max_fps, strategy=limiter_strategy
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
    frames_generator: Iterable[T],
    max_fps: Union[float, int],
    strategy: FPSLimiterStrategy,
) -> Generator[T, None, None]:
    rate_limiter = RateLimiter(desired_fps=max_fps)
    for frame_data in frames_generator:
        delay = rate_limiter.estimate_next_action_delay()
        ticks = 1 if not issubclass(type(frame_data), list) else len(frame_data)
        if delay <= 0.0:
            for _ in range(ticks):
                rate_limiter.tick()
            yield frame_data
            continue
        if strategy is FPSLimiterStrategy.WAIT:
            time.sleep(delay)
            for _ in range(ticks):
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

    def estimate_next_action_delay(self) -> float:
        if self._last_tick is None:
            return 0.0
        desired_delay = 1 / self._desired_fps
        time_since_last_tick = time.monotonic() - self._last_tick
        return max(desired_delay - time_since_last_tick, 0.0)
