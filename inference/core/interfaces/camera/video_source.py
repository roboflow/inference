import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Callable, List, Optional, Protocol, Tuple, Union

import supervision as sv
import cv2
import numpy as np

from inference.core import logger
from inference.core.interfaces.camera.entities import (
    FrameID,
    FrameTimestamp,
    StatusUpdate,
    UpdateSeverity,
)
from inference.core.interfaces.camera.exceptions import (
    EndOfStreamError,
    SourceConnectionError,
    StreamOperationNotAllowedError,
)

DEFAULT_BUFFER_SIZE = int(os.getenv("VIDEO_SOURCE_BUFFER_SIZE", "64"))
DEFAULT_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE = float(
    os.getenv("VIDEO_SOURCE_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE", "0.1")
)
DEFAULT_ADAPTIVE_MODE_READER_PACE_TOLERANCE = float(
    os.getenv("VIDEO_SOURCE_ADAPTIVE_MODE_READER_PACE_TOLERANCE", "5.0")
)
DEFAULT_MINIMUM_ADAPTIVE_MODE_SAMPLES = int(
    os.getenv("VIDEO_SOURCE_MINIMUM_ADAPTIVE_MODE_SAMPLES", "10")
)
DEFAULT_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW = int(
    os.getenv("VIDEO_SOURCE_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW", "16")
)

STATE_UPDATE_EVENT = "STREAM_STATE_UPDATE"
STREAM_ERROR_EVENT = "STREAM_ERROR"
FRAME_CAPTURED_EVENT = "FRAME_CAPTURED"
FRAME_DROPPED_EVENT = "FRAME_DROPPED"
FRAME_CONSUMED_EVENT = "FRAME_CONSUMED"


class StreamState(Enum):
    NOT_STARTED = "NOT_STARTED"
    INITIALISING = "INITIALISING"
    RESTARTING = "RESTARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    MUTED = "MUTED"
    TERMINATING = "TERMINATING"
    ENDED = "ENDED"
    ERROR = "ERROR"


START_ELIGIBLE_STATES = {StreamState.NOT_STARTED, StreamState.RESTARTING}
PAUSE_ELIGIBLE_STATES = {StreamState.RUNNING}
MUTE_ELIGIBLE_STATES = {StreamState.RUNNING}
RESUME_ELIGIBLE_STATES = {StreamState.PAUSED, StreamState.MUTED}
TERMINATE_ELIGIBLE_STATES = {
    StreamState.MUTED,
    StreamState.RUNNING,
    StreamState.PAUSED,
    StreamState.RESTARTING,
    StreamState.ENDED,
    StreamState.ERROR,
}
RESTART_ELIGIBLE_STATES = {
    StreamState.MUTED,
    StreamState.RUNNING,
    StreamState.PAUSED,
    StreamState.ENDED,
    StreamState.ERROR,
}


class BufferFillingStrategy(Enum):
    WAIT = "WAIT"
    DROP_OLDEST = "DROP_OLDEST"
    ADAPTIVE_DROP_OLDEST = "ADAPTIVE_DROP_OLDEST"
    DROP_LATEST = "DROP_LATEST"
    ADAPTIVE_DROP_LATEST = "ADAPTIVE_DROP_LATEST"


ADAPTIVE_STRATEGIES = {
    BufferFillingStrategy.ADAPTIVE_DROP_LATEST,
    BufferFillingStrategy.ADAPTIVE_DROP_OLDEST,
}


class BufferConsumptionStrategy(Enum):
    LAZY = "LAZY"
    EAGER = "EAGER"


@dataclass(frozen=True)
class SourceProperties:
    width: int
    height: int
    total_frames: int
    is_file: bool
    fps: float


@dataclass(frozen=True)
class SourceMetadata:
    source_properties: Optional[SourceProperties]
    source_reference: str
    buffer_size: int
    state: StreamState
    buffer_filling_strategy: Optional[BufferFillingStrategy]
    buffer_consumption_strategy: Optional[BufferConsumptionStrategy]


class VideoSourceMethod(Protocol):
    def __call__(self, video_source: "VideoSource", *args, **kwargs) -> None:
        ...


def lock_state_transition(
    method: VideoSourceMethod,
) -> Callable[["VideoSource"], None]:
    def locked_executor(video_source: "VideoSource", *args, **kwargs) -> None:
        with video_source._state_change_lock:
            return method(video_source, *args, **kwargs)

    return locked_executor


class VideoSource:
    @classmethod
    def init(
        cls,
        video_reference: Union[str, int],
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        status_update_handlers: Optional[List[Callable[[StatusUpdate], None]]] = None,
        buffer_filling_strategy: Optional[BufferFillingStrategy] = None,
        buffer_consumption_strategy: Optional[BufferConsumptionStrategy] = None,
        adaptive_mode_stream_pace_tolerance: int = DEFAULT_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE,
        adaptive_mode_reader_pace_tolerance: int = DEFAULT_ADAPTIVE_MODE_READER_PACE_TOLERANCE,
        minimum_adaptive_mode_samples: int = DEFAULT_MINIMUM_ADAPTIVE_MODE_SAMPLES,
        maximum_adaptive_frames_dropped_in_row: int = DEFAULT_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW,
    ):
        frames_buffer = Queue(maxsize=buffer_size)
        if status_update_handlers is None:
            status_update_handlers = []
        video_consumer = VideoConsumer(
            buffer_filling_strategy=buffer_filling_strategy,
            adaptive_mode_stream_pace_tolerance=adaptive_mode_stream_pace_tolerance,
            adaptive_mode_reader_pace_tolerance=adaptive_mode_reader_pace_tolerance,
            minimum_adaptive_mode_samples=minimum_adaptive_mode_samples,
            maximum_adaptive_frames_dropped_in_row=maximum_adaptive_frames_dropped_in_row,
            status_update_handlers=status_update_handlers,
        )
        return cls(
            stream_reference=video_reference,
            frames_buffer=frames_buffer,
            status_update_handlers=status_update_handlers,
            buffer_consumption_strategy=buffer_consumption_strategy,
            video_consumer=video_consumer,
        )

    def __init__(
        self,
        stream_reference: Union[str, int],
        frames_buffer: Queue,
        status_update_handlers: List[Callable[[StatusUpdate], None]],
        buffer_consumption_strategy: Optional[BufferConsumptionStrategy],
        video_consumer: "VideoConsumer",
    ):
        self._stream_reference = stream_reference
        self._video: Optional[cv2.VideoCapture] = None
        self._source_properties: Optional[SourceProperties] = None
        self._frames_buffer = frames_buffer
        self._status_update_handlers = status_update_handlers
        self._buffer_consumption_strategy = buffer_consumption_strategy
        self._video_consumer = video_consumer
        self._state = StreamState.NOT_STARTED
        self._playback_allowed = Event()
        self._frames_buffering_allowed = True
        self._stream_consumption_thread: Optional[Thread] = None
        self._state_change_lock = Lock()

    @lock_state_transition
    def restart(self, wait_on_frames_consumption: bool = True) -> None:
        if self._state not in RESTART_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not RESTART stream in state: {self._state}"
            )
        self._restart(wait_on_frames_consumption=wait_on_frames_consumption)

    @lock_state_transition
    def start(self) -> None:
        if self._state not in START_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not START stream in state: {self._state}"
            )
        self._start()

    @lock_state_transition
    def terminate(self, wait_on_frames_consumption: bool = True) -> None:
        if self._state not in TERMINATE_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not TERMINATE stream in state: {self._state}"
            )
        self._terminate(wait_on_frames_consumption=wait_on_frames_consumption)

    @lock_state_transition
    def pause(self) -> None:
        if self._state not in PAUSE_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not PAUSE stream in state: {self._state}"
            )
        self._pause()

    def mute(self) -> None:
        if self._state not in MUTE_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not MUTE stream in state: {self._state}"
            )
        self._mute()

    def resume(self) -> None:
        if self._state not in RESUME_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not RESUME stream in state: {self._state}"
            )
        self._resume()

    def get_state(self) -> StreamState:
        return self._state

    def frame_ready(self) -> bool:
        return not self._frames_buffer.empty()

    def read_frame(self) -> Tuple[FrameTimestamp, FrameID, np.ndarray]:
        if self._buffer_consumption_strategy is BufferConsumptionStrategy.EAGER:
            result = purge_queue(
                queue=self._frames_buffer,
                on_successful_read=self._video_consumer.notify_frame_consumed,
            )
        else:
            result = self._frames_buffer.get()
            self._frames_buffer.task_done()
            self._video_consumer.notify_frame_consumed()
        if result is None:
            raise EndOfStreamError(
                "Attempted to retrieve frame from stream that already ended."
            )
        frame_timestamp, frame_id, frame = result
        send_status_update(
            severity=UpdateSeverity.DEBUG,
            event_type=FRAME_CONSUMED_EVENT,
            payload={"frame_timestamp": frame_timestamp, "frame_id": frame_id},
            status_update_handlers=self._status_update_handlers,
        )
        return frame_timestamp, frame_id, frame

    def describe_source(self) -> SourceMetadata:
        return SourceMetadata(
            source_properties=self._source_properties,
            source_reference=self._stream_reference,
            buffer_size=self._frames_buffer.maxsize,
            state=self._state,
            buffer_filling_strategy=self._video_consumer.buffer_filling_strategy,
            buffer_consumption_strategy=self._buffer_consumption_strategy,
        )

    def _restart(self, wait_on_frames_consumption: bool = True) -> None:
        self._terminate(wait_on_frames_consumption=wait_on_frames_consumption)
        self._change_state(target_state=StreamState.RESTARTING)
        self._playback_allowed = Event()
        self._frames_buffering_allowed = True
        self._video: Optional[cv2.VideoCapture] = None
        self._source_properties: Optional[SourceProperties] = None
        self._start()

    def _start(self) -> None:
        self._change_state(target_state=StreamState.INITIALISING)
        self._video = cv2.VideoCapture(self._stream_reference)
        if not self._video.isOpened():
            self._change_state(target_state=StreamState.ERROR)
            raise SourceConnectionError(
                f"Cannot connect to video source under reference: {self._stream_reference}"
            )
        self._source_properties = discover_source_properties(stream=self._video)
        self._video_consumer.reset(source_properties=self._source_properties)
        if self._source_properties.is_file:
            self._set_file_mode_consumption_strategies()
        else:
            self._set_stream_mode_consumption_strategies()
        self._playback_allowed.set()
        self._stream_consumption_thread = Thread(target=self._consume_video)
        self._stream_consumption_thread.start()

    def _terminate(self, wait_on_frames_consumption: bool) -> None:
        if self._state in RESUME_ELIGIBLE_STATES:
            self._resume()
        self._change_state(target_state=StreamState.TERMINATING)
        self._stream_consumption_thread.join()
        if wait_on_frames_consumption:
            self._frames_buffer.join()
        if self._state is not StreamState.ERROR:
            self._change_state(target_state=StreamState.ENDED)

    def _pause(self) -> None:
        self._playback_allowed.clear()
        self._change_state(target_state=StreamState.PAUSED)

    def _mute(self) -> None:
        self._frames_buffering_allowed = False
        self._change_state(target_state=StreamState.MUTED)

    def _resume(self) -> None:
        previous_state = self._state
        self._change_state(target_state=StreamState.RUNNING)
        if previous_state is StreamState.PAUSED:
            self._video_consumer.reset_stream_consumption_pace()
            self._playback_allowed.set()
        if previous_state is StreamState.MUTED:
            self._frames_buffering_allowed = True

    def _set_file_mode_consumption_strategies(self) -> None:
        if self._buffer_consumption_strategy is None:
            self._buffer_consumption_strategy = BufferConsumptionStrategy.LAZY

    def _set_stream_mode_consumption_strategies(self) -> None:
        if self._buffer_consumption_strategy is None:
            self._buffer_consumption_strategy = BufferConsumptionStrategy.EAGER

    def _consume_video(self) -> None:
        try:
            self._change_state(target_state=StreamState.RUNNING)
            declared_source_fps = None
            if self._source_properties is not None:
                declared_source_fps = self._source_properties.fps
            while self._video.isOpened():
                if self._state is StreamState.TERMINATING:
                    break
                self._playback_allowed.wait()
                success = self._video_consumer.consume_frame(
                    video=self._video,
                    declared_source_fps=declared_source_fps,
                    buffer=self._frames_buffer,
                    frames_buffering_allowed=self._frames_buffering_allowed,
                )
                if not success:
                    break
            self._frames_buffer.put(None)
            self._video.release()
            self._change_state(target_state=StreamState.ENDED)
        except Exception as error:
            self._change_state(target_state=StreamState.ERROR)
            payload = {
                "error_type": error.__class__.__name__,
                "error_message": str(error),
                "error_context": "stream_consumer_thread",
            }
            send_status_update(
                severity=UpdateSeverity.ERROR,
                event_type=STREAM_ERROR_EVENT,
                payload=payload,
                status_update_handlers=self._status_update_handlers,
            )
            logger.exception("Encountered error in video consumption thread")

    def _change_state(self, target_state: StreamState) -> None:
        payload = {
            "previous_state": self._state,
            "new_state": target_state,
        }
        self._state = target_state
        send_status_update(
            severity=UpdateSeverity.INFO,
            event_type=STATE_UPDATE_EVENT,
            payload=payload,
            status_update_handlers=self._status_update_handlers,
        )

    def __iter__(self) -> "VideoSource":
        return self

    def __next__(self) -> Tuple[FrameTimestamp, FrameID, np.ndarray]:
        try:
            return self.read_frame()
        except EndOfStreamError:
            raise StopIteration()


class VideoConsumer:
    def __init__(
        self,
        buffer_filling_strategy: Optional[BufferFillingStrategy],
        adaptive_mode_stream_pace_tolerance: int,
        adaptive_mode_reader_pace_tolerance: int,
        minimum_adaptive_mode_samples: int,
        maximum_adaptive_frames_dropped_in_row: int,
        status_update_handlers: List[Callable[[StatusUpdate], None]],
    ):
        self._buffer_filling_strategy = buffer_filling_strategy
        self._frame_counter = 0
        self._adaptive_mode_stream_pace_tolerance = adaptive_mode_stream_pace_tolerance
        self._adaptive_mode_reader_pace_tolerance = adaptive_mode_reader_pace_tolerance
        self._minimum_adaptive_mode_samples = minimum_adaptive_mode_samples
        self._maximum_adaptive_frames_dropped_in_row = (
            maximum_adaptive_frames_dropped_in_row
        )
        self._adaptive_frames_dropped_in_row = 0
        self._reader_pace_monitor = sv.FPSMonitor(
            sample_size=10 * minimum_adaptive_mode_samples
        )
        self._stream_consumption_pace_monitor = sv.FPSMonitor(
            sample_size=10 * minimum_adaptive_mode_samples
        )
        self._decoding_pace_monitor = sv.FPSMonitor(
            sample_size=10 * minimum_adaptive_mode_samples
        )
        self._status_update_handlers = status_update_handlers

    @property
    def buffer_filling_strategy(self) -> Optional[BufferFillingStrategy]:
        return self._buffer_filling_strategy

    def reset(self, source_properties: SourceProperties) -> None:
        if source_properties.is_file:
            self._set_file_mode_buffering_strategies()
        else:
            self._set_stream_mode_buffering_strategies()
        self._reader_pace_monitor.reset()
        self.reset_stream_consumption_pace()
        self._decoding_pace_monitor.reset()
        self._adaptive_frames_dropped_in_row = 0

    def reset_stream_consumption_pace(self) -> None:
        self._stream_consumption_pace_monitor.reset()

    def notify_frame_consumed(self) -> None:
        self._reader_pace_monitor.tick()

    def consume_frame(
        self,
        video: cv2.VideoCapture,
        declared_source_fps: Optional[float],
        buffer: Queue,
        frames_buffering_allowed: bool,
    ) -> bool:
        frame_timestamp = datetime.now()
        success = video.grab()
        self._stream_consumption_pace_monitor.tick()
        if not success:
            return False
        self._frame_counter += 1
        send_status_update(
            severity=UpdateSeverity.DEBUG,
            event_type=FRAME_CAPTURED_EVENT,
            payload={
                "frame_timestamp": frame_timestamp,
                "frame_id": self._frame_counter,
            },
            status_update_handlers=self._status_update_handlers,
        )
        return self._consume_stream_frame(
            video=video,
            declared_source_fps=declared_source_fps,
            frame_timestamp=frame_timestamp,
            buffer=buffer,
            frames_buffering_allowed=frames_buffering_allowed,
        )

    def _set_file_mode_buffering_strategies(self) -> None:
        if self._buffer_filling_strategy is None:
            self._buffer_filling_strategy = BufferFillingStrategy.WAIT

    def _set_stream_mode_buffering_strategies(self) -> None:
        if self._buffer_filling_strategy is None:
            self._buffer_filling_strategy = BufferFillingStrategy.ADAPTIVE_DROP_OLDEST

    def _consume_stream_frame(
        self,
        video: cv2.VideoCapture,
        declared_source_fps: Optional[float],
        frame_timestamp: datetime,
        buffer: Queue,
        frames_buffering_allowed: bool,
    ) -> bool:
        """
        Returns: boolean flag with success status
        """
        if not frames_buffering_allowed:
            send_frame_drop_update(
                frame_timestamp=frame_timestamp,
                frame_id=self._frame_counter,
                cause="Buffering not allowed at the moment",
                status_update_handlers=self._status_update_handlers,
            )
            return True
        if self._frame_should_be_adaptively_dropped(
            declared_source_fps=declared_source_fps
        ):
            self._adaptive_frames_dropped_in_row += 1
            send_frame_drop_update(
                frame_timestamp=frame_timestamp,
                frame_id=self._frame_counter,
                cause="ADAPTIVE strategy",
                status_update_handlers=self._status_update_handlers,
            )
            return True
        self._adaptive_frames_dropped_in_row = 0
        if (
            not buffer.full()
            or self._buffer_filling_strategy is BufferFillingStrategy.WAIT
        ):
            return self._decode_stream_frame(
                frame_timestamp=frame_timestamp, video=video, buffer=buffer
            )
        if self._buffer_filling_strategy is BufferFillingStrategy.DROP_OLDEST:
            return self._process_stream_frame_dropping_oldest(
                frame_timestamp=frame_timestamp,
                video=video,
                buffer=buffer,
            )
        send_frame_drop_update(
            frame_timestamp=frame_timestamp,
            frame_id=self._frame_counter,
            cause="DROP_LATEST strategy",
            status_update_handlers=self._status_update_handlers,
        )
        return True

    def _frame_should_be_adaptively_dropped(
        self, declared_source_fps: Optional[float]
    ) -> bool:
        if self._buffer_filling_strategy not in ADAPTIVE_STRATEGIES:
            return False
        if (
            self._adaptive_frames_dropped_in_row
            >= self._maximum_adaptive_frames_dropped_in_row
        ):
            return False
        if (
            len(self._stream_consumption_pace_monitor.all_timestamps)
            < self._minimum_adaptive_mode_samples
        ):
            # not enough observations
            return False
        stream_consumption_pace = self._stream_consumption_pace_monitor()
        announced_stream_fps = stream_consumption_pace
        if declared_source_fps is not None and declared_source_fps > 0:
            announced_stream_fps = declared_source_fps
        if (
            announced_stream_fps - stream_consumption_pace
            > self._adaptive_mode_stream_pace_tolerance
        ):
            # cannot keep up with stream emission
            return True
        if (
            len(self._reader_pace_monitor.all_timestamps)
            < self._minimum_adaptive_mode_samples
        ) or (
            len(self._decoding_pace_monitor.all_timestamps)
            < self._minimum_adaptive_mode_samples
        ):
            # not enough observations
            return False
        reader_pace = self._reader_pace_monitor()
        decoding_pace = self._decoding_pace_monitor()
        if decoding_pace - reader_pace > self._adaptive_mode_reader_pace_tolerance:
            # we are too fast for the reader - time to save compute on decoding
            return True
        return False

    def _process_stream_frame_dropping_oldest(
        self,
        frame_timestamp: datetime,
        video: cv2.VideoCapture,
        buffer: Queue,
    ) -> bool:
        drop_single_frame_from_buffer(
            buffer=buffer,
            cause="DROP_OLDEST strategy",
            status_update_handlers=self._status_update_handlers,
        )
        return self._decode_stream_frame(
            frame_timestamp=frame_timestamp, video=video, buffer=buffer
        )

    def _decode_stream_frame(
        self,
        frame_timestamp: datetime,
        video: cv2.VideoCapture,
        buffer: Queue,
    ) -> bool:
        success, frame = video.retrieve()
        self._decoding_pace_monitor.tick()
        if not success:
            return False
        buffer.put((frame_timestamp, self._frame_counter, frame))
        return True


def drop_single_frame_from_buffer(
    buffer: Queue,
    cause: str,
    status_update_handlers: List[Callable[[StatusUpdate], None]],
) -> None:
    try:
        (
            dropped_frame_timestamp,
            dropped_frame_counter,
            _,
        ) = buffer.get_nowait()
        buffer.task_done()
        send_frame_drop_update(
            frame_timestamp=dropped_frame_timestamp,
            frame_id=dropped_frame_counter,
            cause=cause,
            status_update_handlers=status_update_handlers,
        )
    except Empty:
        # buffer may be emptied in the meantime, hence we ignore Empty
        pass


def send_frame_drop_update(
    frame_timestamp: datetime,
    frame_id: int,
    cause: str,
    status_update_handlers: List[Callable[[StatusUpdate], None]],
) -> None:
    send_status_update(
        severity=UpdateSeverity.DEBUG,
        event_type=FRAME_DROPPED_EVENT,
        payload={
            "frame_timestamp": frame_timestamp,
            "frame_id": frame_id,
            "cause": cause,
        },
        status_update_handlers=status_update_handlers,
    )


def send_status_update(
    severity: UpdateSeverity,
    event_type: str,
    payload: dict,
    status_update_handlers: List[Callable[[StatusUpdate], None]],
) -> None:
    status_update = StatusUpdate(
        timestamp=datetime.now(),
        severity=severity,
        event_type=event_type,
        payload=payload,
    )
    for handler in status_update_handlers:
        handler(status_update)


def discover_source_properties(stream: cv2.VideoCapture) -> SourceProperties:
    width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = stream.get(cv2.CAP_PROP_FPS)
    total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    return SourceProperties(
        width=width,
        height=height,
        total_frames=total_frames,
        is_file=total_frames > 0,
        fps=fps,
    )


def purge_queue(
    queue: Queue,
    wait_on_empty: bool = True,
    on_successful_read: Callable[[], None] = lambda: None,
) -> Optional[Any]:
    result = None
    if queue.empty() and wait_on_empty:
        result = queue.get()
        queue.task_done()
        on_successful_read()
    while not queue.empty():
        result = queue.get()
        queue.task_done()
        on_successful_read()
    return result
