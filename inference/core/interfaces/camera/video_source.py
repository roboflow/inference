from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Callable, List, Optional, Tuple, Union

import cv2
import numpy as np

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

DEFAULT_BUFFER_SIZE = 64
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
    DROP_LATEST = "DROP_LATEST"


class BufferConsumptionStrategy(Enum):
    LAZY = "LAZY"
    EAGER = "EAGER"


@dataclass(frozen=True)
class StreamProperties:
    width: int
    height: int
    total_frames: int
    is_file: bool
    fps: float


@dataclass(frozen=True)
class SourceMetadata:
    stream_properties: Optional[StreamProperties]
    source_reference: str
    buffer_size: int
    state: StreamState
    buffer_filling_strategy: Optional[BufferFillingStrategy]
    buffer_consumption_strategy: Optional[BufferConsumptionStrategy]


def lock_state_transition(
    method: Callable[["VideoSource"], None]
) -> Callable[["VideoSource"], None]:
    def locked_executor(video_source: "VideoSource") -> None:
        with video_source._state_change_lock:
            return method(video_source)

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
    ):
        frames_buffer = Queue(maxsize=buffer_size)
        if status_update_handlers is None:
            status_update_handlers = []
        return cls(
            stream_reference=video_reference,
            frames_buffer=frames_buffer,
            status_update_handlers=status_update_handlers,
            buffer_filling_strategy=buffer_filling_strategy,
            buffer_consumption_strategy=buffer_consumption_strategy,
        )

    def __init__(
        self,
        stream_reference: Union[str, int],
        frames_buffer: Queue,
        status_update_handlers: List[Callable[[StatusUpdate], None]],
        buffer_filling_strategy: Optional[BufferFillingStrategy],
        buffer_consumption_strategy: Optional[BufferFillingStrategy],
    ):
        self._stream_reference = stream_reference
        self._stream: Optional[cv2.VideoCapture] = None
        self._stream_properties: Optional[StreamProperties] = None
        self._frames_buffer = frames_buffer
        self._status_update_handlers = status_update_handlers
        self._buffer_filling_strategy = buffer_filling_strategy
        self._frame_counter = 0
        self._buffer_consumption_strategy = buffer_consumption_strategy
        self._state = StreamState.NOT_STARTED
        self._playback_allowed = Event()
        self._frames_buffering_allowed = True
        self._stream_consumption_thread: Optional[Thread] = None
        self._state_change_lock = Lock()

    @lock_state_transition
    def restart(self) -> None:
        if self._state not in RESTART_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not RESTART stream in state: {self._state}"
            )
        self._restart()

    @lock_state_transition
    def start(self) -> None:
        if self._state not in START_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not START stream in state: {self._state}"
            )
        self._start()

    @lock_state_transition
    def terminate(self) -> None:
        if self._state not in TERMINATE_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not TERMINATE stream in state: {self._state}"
            )
        self._terminate()

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
            result = purge_queue(queue=self._frames_buffer)
        else:
            result = self._frames_buffer.get()
            self._frames_buffer.task_done()
        if result is None:
            raise EndOfStreamError(
                "Attempted to retrieve frame from stream that already ended."
            )
        frame_timestamp, frame_id, frame = result
        self._send_status_update(
            severity=UpdateSeverity.DEBUG,
            event_type=FRAME_CONSUMED_EVENT,
            payload={"frame_timestamp": frame_timestamp, "frame_id": frame_id},
        )
        return frame_timestamp, frame_id, frame

    def describe_source(self) -> SourceMetadata:
        return SourceMetadata(
            stream_properties=self._stream_properties,
            source_reference=self._stream_reference,
            buffer_size=self._frames_buffer.maxsize,
            state=self._state,
            buffer_filling_strategy=self._buffer_filling_strategy,
            buffer_consumption_strategy=self._buffer_consumption_strategy,
        )

    def _restart(self) -> None:
        self._terminate()
        self._change_state(target_state=StreamState.RESTARTING)
        self._playback_allowed = Event()
        self._frames_buffering_allowed = True
        self._stream: Optional[cv2.VideoCapture] = None
        self._stream_properties: Optional[StreamProperties] = None
        self._start()

    def _start(self) -> None:
        self._change_state(target_state=StreamState.INITIALISING)
        self._stream = cv2.VideoCapture(self._stream_reference)
        if not self._stream.isOpened():
            self._change_state(target_state=StreamState.ERROR)
            raise SourceConnectionError(
                f"Cannot connect to video source under reference: {self._stream_reference}"
            )
        self._stream_properties = discover_stream_properties(stream=self._stream)
        if self._stream_properties.is_file:
            self._set_file_mode_buffering_strategies()
        else:
            self._set_stream_mode_buffering_strategies()
        self._playback_allowed.set()
        self._stream_consumption_thread = Thread(target=self._consume_stream)
        self._stream_consumption_thread.start()

    def _terminate(self) -> None:
        if self._state in RESUME_ELIGIBLE_STATES:
            self._resume()
        self._change_state(target_state=StreamState.TERMINATING)
        self._stream_consumption_thread.join()
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
            self._playback_allowed.set()
        if previous_state is StreamState.MUTED:
            self._frames_buffering_allowed = True

    def _set_file_mode_buffering_strategies(self) -> None:
        if self._buffer_filling_strategy is None:
            self._buffer_filling_strategy = BufferFillingStrategy.WAIT
        if self._buffer_consumption_strategy is None:
            self._buffer_consumption_strategy = BufferConsumptionStrategy.LAZY

    def _set_stream_mode_buffering_strategies(self) -> None:
        if self._buffer_filling_strategy is None:
            self._buffer_filling_strategy = BufferFillingStrategy.DROP_OLDEST
        if self._buffer_consumption_strategy is None:
            self._buffer_consumption_strategy = BufferConsumptionStrategy.EAGER

    def _consume_stream(self) -> None:
        try:
            self._change_state(target_state=StreamState.RUNNING)
            while self._stream.isOpened():
                if self._state is StreamState.TERMINATING:
                    break
                self._playback_allowed.wait()
                frame_timestamp = datetime.now()
                success = self._stream.grab()
                if not success:
                    break
                self._frame_counter += 1
                self._send_status_update(
                    severity=UpdateSeverity.DEBUG,
                    event_type=FRAME_CAPTURED_EVENT,
                    payload={
                        "frame_timestamp": frame_timestamp,
                        "frame_id": self._frame_counter,
                    },
                )
                self._consume_stream_frame(frame_timestamp=frame_timestamp)
            self._frames_buffer.put(None)
            self._stream.release()
            self._change_state(target_state=StreamState.ENDED)
        except Exception as error:
            self._change_state(target_state=StreamState.ERROR)
            payload = {
                "error_type": error.__class__.__name__,
                "error_message": str(error),
                "error_context": "stream_consumer_thread",
            }
            self._send_status_update(
                severity=UpdateSeverity.ERROR,
                event_type=STREAM_ERROR_EVENT,
                payload=payload,
            )

    def _consume_stream_frame(self, frame_timestamp: datetime) -> bool:
        """
        Returns: boolean flag with success status
        """
        if not self._frames_buffering_allowed:
            self._send_status_update(
                severity=UpdateSeverity.DEBUG,
                event_type=FRAME_DROPPED_EVENT,
                payload={
                    "frame_timestamp": frame_timestamp,
                    "frame_id": self._frame_counter,
                },
            )
            return True
        if (
            not self._frames_buffer.full()
            or self._buffer_filling_strategy is BufferFillingStrategy.WAIT
        ):
            return self._process_stream_frame(frame_timestamp=frame_timestamp)
        if self._buffer_filling_strategy is BufferFillingStrategy.DROP_OLDEST:
            return self._process_stream_frame_dropping_oldest(
                frame_timestamp=frame_timestamp,
            )
        self._send_status_update(
            severity=UpdateSeverity.DEBUG,
            event_type=FRAME_DROPPED_EVENT,
            payload={
                "frame_timestamp": frame_timestamp,
                "frame_id": self._frame_counter,
            },
        )
        return True

    def _process_stream_frame_dropping_oldest(self, frame_timestamp: datetime) -> bool:
        try:
            (
                dropped_frame_timestamp,
                dropped_frame_counter,
                _,
            ) = self._frames_buffer.get_nowait()
            self._frames_buffer.task_done()
            self._send_status_update(
                severity=UpdateSeverity.DEBUG,
                event_type=FRAME_DROPPED_EVENT,
                payload={
                    "frame_timestamp": frame_timestamp,
                    "frame_id": self._frame_counter,
                },
            )
        except Empty:
            # buffer may be emptied in the meantime, hence we ignore Empty
            pass
        return self._process_stream_frame(frame_timestamp=frame_timestamp)

    def _process_stream_frame(self, frame_timestamp: datetime) -> bool:
        success, frame = self._stream.retrieve()
        if not success:
            return False
        self._frames_buffer.put((frame_timestamp, self._frame_counter, frame))
        return True

    def _change_state(self, target_state: StreamState) -> None:
        payload = {
            "previous_state": self._state,
            "new_state": target_state,
        }
        self._state = target_state
        self._send_status_update(
            severity=UpdateSeverity.INFO,
            event_type=STATE_UPDATE_EVENT,
            payload=payload,
        )

    def _send_status_update(
        self, severity: UpdateSeverity, event_type: str, payload: dict
    ) -> None:
        status_update = StatusUpdate(
            timestamp=datetime.now(),
            severity=severity,
            event_type=event_type,
            payload=payload,
        )
        for handler in self._status_update_handlers:
            handler(status_update)

    def __iter__(self) -> "VideoSource":
        return self

    def __next__(self) -> Tuple[FrameTimestamp, FrameID, np.ndarray]:
        try:
            return self.read_frame()
        except EndOfStreamError:
            raise StopIteration()

    def __del__(self) -> None:
        self._frames_buffer.join()


def discover_stream_properties(stream: cv2.VideoCapture) -> StreamProperties:
    width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = stream.get(cv2.CAP_PROP_FPS)
    total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    return StreamProperties(
        width=width,
        height=height,
        total_frames=total_frames,
        is_file=total_frames > 0,
        fps=fps,
    )


def purge_queue(queue: Queue, wait_on_empty: bool = True) -> Optional[Any]:
    result = None
    if queue.empty() and wait_on_empty:
        result = queue.get()
        queue.task_done()
    while not queue.empty():
        result = queue.get()
        queue.task_done()
    return result
