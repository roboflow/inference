import time
from copy import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from functools import partial
from threading import Thread
from typing import (
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
)

from inference.core import logger
from inference.core.env import RESTART_ATTEMPT_DELAY
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.camera.exceptions import (
    EndOfStreamError,
    SourceConnectionError,
)
from inference.core.interfaces.camera.video_source import SourceProperties, VideoSource

MINIMAL_FPS = 0.01

T = TypeVar("T")


class FPSLimiterStrategy(Enum):
    DROP = "drop"
    WAIT = "wait"


@dataclass(frozen=True)
class VideoSources:
    all_sources: List[VideoSource]
    allow_reconnection: List[bool]
    managed_sources: List[VideoSource]


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
        from inference.core.interfaces.camera.utils import get_video_frames_generator

        for frame in get_video_frames_generator(
            video="./some.mp4",
            max_fps=50,
        ):
             pass
        ```
    """
    is_managed_source = False
    if issubclass(type(video), str) or issubclass(type(video), int):
        video = VideoSource.init(
            video_reference=video,
        )
        video.start()
        is_managed_source = True
    if max_fps is None:
        yield from video
        if is_managed_source:
            video.terminate(purge_frames_buffer=True)
        return None
    limiter_strategy = resolve_limiter_strategy(
        explicitly_defined_strategy=limiter_strategy,
        source_properties=video.describe_source().source_properties,
    )
    yield from limit_frame_rate(
        frames_generator=video, max_fps=max_fps, strategy=limiter_strategy
    )
    if is_managed_source:
        video.terminate(purge_frames_buffer=True)
    return None


def never_stop() -> bool:
    return False


def log_error(source_id: Optional[int], error: SourceConnectionError) -> None:
    logger.warning(
        f"Could not re-connect to source with id: {source_id}. Error: {error}"
    )


class VideoSourcesManager:
    """
    This class should be treated as internal building block of stream multiplexer - not for external use.
    """

    @classmethod
    def init(
        cls,
        video_sources: VideoSources,
        should_stop: Callable[[], bool],
        on_reconnection_error: Callable[[Optional[int], SourceConnectionError], None],
    ) -> "VideoSourcesManager":
        return cls(
            video_sources=video_sources,
            should_stop=should_stop,
            on_reconnection_error=on_reconnection_error,
        )

    def __init__(
        self,
        video_sources: VideoSources,
        should_stop: Callable[[], bool],
        on_reconnection_error: Callable[[Optional[int], SourceConnectionError], None],
    ):
        self._video_sources = video_sources
        self._reconnection_threads: Dict[int, Thread] = {}
        self._external_should_stop = should_stop
        self._on_reconnection_error = on_reconnection_error
        self._enforce_stop: Dict[int, bool] = {}
        self._ended_sources: Set[int] = set()
        self._threads_to_join: Set[int] = set()
        self._last_batch_yielded_time = datetime.now()

    def retrieve_frames_from_sources(
        self,
        batch_collection_timeout: Optional[float],
    ) -> Optional[List[VideoFrame]]:
        batch_frames = []
        if batch_collection_timeout is not None:
            batch_timeout_moment = self._last_batch_yielded_time + timedelta(
                seconds=batch_collection_timeout
            )
        else:
            batch_timeout_moment = None
        for source_ord, (source, source_should_reconnect) in enumerate(
            zip(self._video_sources.all_sources, self._video_sources.allow_reconnection)
        ):
            if self._external_should_stop():
                self.join_all_reconnection_threads(include_not_finished=True)
                return None
            if self._is_source_inactive(source_ord=source_ord):
                continue
            batch_time_left = (
                None
                if batch_timeout_moment is None
                else max((batch_timeout_moment - datetime.now()).total_seconds(), 0.0)
            )
            try:
                frame = source.read_frame(timeout=batch_time_left)
                if frame is not None:
                    batch_frames.append(frame)
            except EndOfStreamError:
                self._register_end_of_stream(source_ord=source_ord)
        self.join_all_reconnection_threads()
        self._last_batch_yielded_time = datetime.now()
        return batch_frames

    def all_sources_ended(self) -> bool:
        return len(self._ended_sources) >= len(self._video_sources.all_sources)

    def join_all_reconnection_threads(self, include_not_finished: bool = False) -> None:
        for source_ord in copy(self._threads_to_join):
            self._purge_reconnection_thread(source_ord=source_ord)
        if not include_not_finished:
            return None
        for source_ord in list(self._reconnection_threads.keys()):
            self._purge_reconnection_thread(source_ord=source_ord)

    def _is_source_inactive(self, source_ord: int) -> bool:
        return (
            source_ord in self._ended_sources
            or source_ord in self._reconnection_threads
        )

    def _register_end_of_stream(self, source_ord: int) -> None:
        source_should_reconnect = self._video_sources.allow_reconnection[source_ord]
        if source_should_reconnect:
            self._reconnect_source(source_ord=source_ord)
        else:
            self._ended_sources.add(source_ord)

    def _reconnect_source(self, source_ord: int) -> None:
        if source_ord in self._reconnection_threads:
            return None
        self._reconnection_threads[source_ord] = Thread(
            target=_attempt_reconnect,
            args=(
                self._video_sources.all_sources[source_ord],
                partial(self._should_stop, source_ord=source_ord),
                self._on_reconnection_error,
                partial(self._register_thread_to_join, source_ord=source_ord),
                partial(self._register_reconnection_fatal_error, source_ord=source_ord),
            ),
        )
        self._reconnection_threads[source_ord].start()

    def _register_reconnection_fatal_error(self, source_ord: int) -> None:
        self._register_thread_to_join(source_ord=source_ord)
        self._ended_sources.add(source_ord)

    def _register_thread_to_join(self, source_ord: int) -> None:
        self._threads_to_join.add(source_ord)

    def _purge_reconnection_thread(self, source_ord: int) -> None:
        if source_ord not in self._reconnection_threads:
            return None
        self._enforce_stop[source_ord] = True
        self._reconnection_threads[source_ord].join()
        del self._reconnection_threads[source_ord]
        self._enforce_stop[source_ord] = False
        if source_ord in self._threads_to_join:
            self._threads_to_join.remove(source_ord)

    def _should_stop(self, source_ord: int) -> bool:
        if self._external_should_stop():
            return True
        return self._enforce_stop.get(source_ord, False)


def multiplex_videos(
    videos: List[Union[VideoSource, str, int]],
    max_fps: Optional[Union[float, int]] = None,
    limiter_strategy: Optional[FPSLimiterStrategy] = None,
    batch_collection_timeout: Optional[float] = None,
    force_stream_reconnection: bool = True,
    should_stop: Callable[[], bool] = never_stop,
    on_reconnection_error: Callable[
        [Optional[int], SourceConnectionError], None
    ] = log_error,
) -> Generator[List[VideoFrame], None, None]:
    """
    Function that is supposed to provide a generator over frames from multiple video sources. It is capable to
    initialise `VideoSource` from references to video files or streams and grab frames from all the sources -
    each running individual decoding on separate thread. In each cycle it attempts to grab frames from all sources
    (and wait at max `batch_collection_timeout` for whole batch to be collected). If frame from specific source
    cannot be collected in that time - it is simply not included in returned list. If after batch collection list of
    frames is empty - new collection start immediately. Collection does not account for
    sources that lost connectivity (example: streams that went offline). If that does not happen and stream has
    large latency - without reasonable `batch_collection_timeout` it will slow down processing - so please
    set it up in PROD solutions. In case of video streams (not video files) - given that
    `force_stream_reconnection=True` function will attempt to re-connect to disconnected source using background thread,
    not impairing batch frames collection and that source is not going to block frames retrieval even if infinite
    `batch_collection_timeout=None` is set. Similarly, when processing files - video file that is shorter than other
    passed into processing will not block the whole flow after End Of Stream (EOS).

    All sources must be accessible on start - if that's not the case - logic function raises `SourceConnectionError`
    and closes all video sources it opened on it own. Disconnections at later stages are handled by re-connection
    mechanism.

    Args:
        videos (List[Union[VideoSource, str, int]]): List with references to video sources. Elements can be
            pre-initialised `VideoSource` instances, str with stream URI or file location or int representing
            camera device attached to the PC/server running the code.
        max_fps (Optional[Union[float, int]]): Upper-bound of processing speed - to be used when one wants at max
            `max_fps` video frames per second to be yielded from all sources by the generator.
        limiter_strategy (Optional[FPSLimiterStrategy]): strategy used to deal with frames decoding exceeding
            limit of `max_fps`. For video files, in the interest of processing all frames - we recommend WAIT mode,
             for streams - frames should be dropped on the floor with DROP strategy. Not setting the strategy equals
             using automatic mode - WAIT if all sources are files and DROP otherwise
        batch_collection_timeout (Optional[float]): maximum await time to get batch of predictions from all sources.
            `None` means infinite timeout.
        force_stream_reconnection (bool): Flag to decide on reconnection to streams (files are never re-connected)
        should_stop (Callable[[], bool]): external stop signal that is periodically checked - to denote that
            video consumption stopped - make the function to return True
        on_reconnection_error (Callable[[Optional[int], SourceConnectionError], None]): Function that will be
            called whenever source cannot re-connect after disconnection. First parameter is source_id, second
            is connection error instance.

    Returns Generator[List[VideoFrame], None, None]: allowing to iterate through frames from multiple video sources.

    Raises:
        SourceConnectionError: when one or more source is not reachable at start of generation

    Example:
        ```python
        from inference.core.interfaces.camera.utils import multiplex_videos

        for frames in multiplex_videos(videos=["./some.mp4", "./other.mp4"]):
             for frame in frames:
                pass  # do something with frame
        ```
    """
    video_sources = _prepare_video_sources(
        videos=videos, force_stream_reconnection=force_stream_reconnection
    )
    if any(rule is None for rule in video_sources.allow_reconnection):
        logger.warning("Could not connect to all sources.")
        return None
    generator = _multiplex_videos(
        video_sources=video_sources,
        batch_collection_timeout=batch_collection_timeout,
        should_stop=should_stop,
        on_reconnection_error=on_reconnection_error,
    )
    if max_fps is None:
        yield from generator
        return None
    max_fps = max_fps / len(videos)
    if limiter_strategy is None:
        limiter_strategy = negotiate_rate_limiter_strategy_for_multiple_sources(
            video_sources=video_sources.all_sources,
        )
    yield from limit_frame_rate(
        frames_generator=generator, max_fps=max_fps, strategy=limiter_strategy
    )


def _multiplex_videos(
    video_sources: VideoSources,
    batch_collection_timeout: Optional[float],
    should_stop: Callable[[], bool],
    on_reconnection_error: Callable[[Optional[int], SourceConnectionError], None],
) -> Generator[List[VideoFrame], None, None]:
    sources_manager = VideoSourcesManager.init(
        video_sources=video_sources,
        should_stop=should_stop,
        on_reconnection_error=on_reconnection_error,
    )
    while not sources_manager.all_sources_ended():
        batch_frames = sources_manager.retrieve_frames_from_sources(
            batch_collection_timeout=batch_collection_timeout,
        )
        if batch_frames is None:
            break
        if len(batch_frames) > 0:
            yield batch_frames
    sources_manager.join_all_reconnection_threads()
    for video in video_sources.managed_sources:
        video.terminate(wait_on_frames_consumption=False, purge_frames_buffer=True)
    return None


def _prepare_video_sources(
    videos: List[Union[VideoSource, str, int]],
    force_stream_reconnection: bool,
) -> VideoSources:
    all_sources: List[VideoSource] = []
    managed_sources: List[VideoSource] = []
    minimal_free_source_id = _find_free_source_identifier(videos=videos)
    try:
        for video in videos:
            if issubclass(type(video), str) or issubclass(type(video), int):
                video = VideoSource.init(
                    video_reference=video, source_id=minimal_free_source_id
                )
                minimal_free_source_id += 1
                video.start()
                managed_sources.append(video)
            all_sources.append(video)
    except Exception as e:
        for video in managed_sources:
            try:
                video.terminate(
                    wait_on_frames_consumption=False, purge_frames_buffer=True
                )
            except Exception:
                # passing inner termination error
                pass
        raise e
    allow_reconnection = _establish_sources_reconnection_rules(
        all_sources=all_sources,
        force_stream_reconnection=force_stream_reconnection,
    )
    return VideoSources(
        all_sources=all_sources,
        allow_reconnection=allow_reconnection,
        managed_sources=managed_sources,
    )


def _find_free_source_identifier(videos: List[Union[VideoSource, str, int]]) -> int:
    minimal_free_source_id = [
        v.source_id if v.source_id is not None else -1
        for v in videos
        if issubclass(type(v), VideoSource)
    ]
    if len(minimal_free_source_id) == 0:
        minimal_free_source_id = -1
    else:
        minimal_free_source_id = max(minimal_free_source_id)
    minimal_free_source_id += 1
    return minimal_free_source_id


def _establish_sources_reconnection_rules(
    all_sources: List[VideoSource], force_stream_reconnection: bool
) -> List[bool]:
    result = []
    for video_source in all_sources:
        source_properties = video_source.describe_source().source_properties
        if source_properties is None:
            result.append(False)
        else:
            if source_properties.is_reconnectable is False:
                result.append(False)
                continue
            result.append(not source_properties.is_file and force_stream_reconnection)
    return result


def _attempt_reconnect(
    video_source: VideoSource,
    should_stop: Callable[[], bool],
    on_reconnection_failure: Callable[[Optional[int], SourceConnectionError], None],
    on_reconnection_success: Callable[[], None],
    on_fatal_error: Callable[[], None],
) -> None:
    succeeded = False
    while not should_stop() and not succeeded:
        try:
            video_source.restart(wait_on_frames_consumption=False)
            succeeded = True
            on_reconnection_success()
        except SourceConnectionError as error:
            on_reconnection_failure(video_source.source_id, error)
            if should_stop():
                return None
            logger.warning(
                f"Could not connect to video source. Retrying in {RESTART_ATTEMPT_DELAY}s..."
            )
            time.sleep(RESTART_ATTEMPT_DELAY)
        except Exception as error:
            logger.warning(
                f"Fatal error in re-connection to source: {video_source.source_id}. Details: {error}"
            )
            on_fatal_error()
            break


def negotiate_rate_limiter_strategy_for_multiple_sources(
    video_sources: List[VideoSource],
) -> FPSLimiterStrategy:
    source_types_statuses = {
        s.describe_source().source_properties.is_file for s in video_sources
    }
    if len(source_types_statuses) == 2:
        logger.warning(
            f"`InferencePipeline` started with FPS limit rate. Detected both files and video streams as video sources. "
            f"Rate limiter cannot satisfy both - choosing `FPSLimiterStrategy.DROP` which may drop file sources frames "
            f"that would not happen if only video files are submitted into processing."
        )
        return FPSLimiterStrategy.DROP
    if True in source_types_statuses:
        return FPSLimiterStrategy.WAIT
    return FPSLimiterStrategy.DROP


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

    def estimate_next_action_delay(self) -> float:
        if self._last_tick is None:
            return 0.0
        desired_delay = 1 / self._desired_fps
        time_since_last_tick = time.monotonic() - self._last_tick
        return max(desired_delay - time_since_last_tick, 0.0)
