from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from queue import Empty, Queue
from threading import Event, Thread
from typing import Callable, List, Optional, Tuple, Union

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
    StreamOperationNotAllowedError,
    StreamReadNotFeasibleError,
)

DEFAULT_BUFFER_SIZE = 64
STATE_UPDATE_EVENT = "STREAM_STATE_UPDATE"
STREAM_ERROR_EVENT = "STREAM_ERROR"
FRAME_CAPTURED_EVENT = "FRAME_CAPTURED"
FRAME_CONSUMED_EVENT = "FRAME_CONSUMED"


class StreamState(Enum):
    NOT_STARTED = "NOT STARTED"
    INITIALISING = "INITIALISING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    TERMINATING = "TERMINATING"
    ENDED = "ENDED"
    ERROR = "ERROR"


READ_ELIGIBLE_STATES = {
    StreamState.RUNNING,
    StreamState.PAUSED,
    StreamState.TERMINATING,
    StreamState.ENDED,
}
START_ELIGIBLE_STATES = {StreamState.NOT_STARTED}
PAUSE_ELIGIBLE_STATES = {StreamState.PAUSED, StreamState.RUNNING}
RESUME_ELIGIBLE_STATES = {StreamState.PAUSED, StreamState.RUNNING}
TERMINATE_ELIGIBLE_STATES = {
    StreamState.INITIALISING,
    StreamState.RUNNING,
    StreamState.PAUSED,
}


@dataclass(frozen=True)
class StreamProperties:
    width: int
    height: int
    total_frames: int
    is_file: bool
    fps: float


class VideoStream:
    @classmethod
    def init(
        cls,
        stream_reference: Union[str, int],
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        status_update_handlers: Optional[List[Callable[[StatusUpdate], None]]] = None,
    ):
        frames_buffer = Queue(maxsize=buffer_size)
        if status_update_handlers is None:
            status_update_handlers = []
        stream = cv2.VideoCapture(stream_reference)
        return cls(
            stream=stream,
            frames_buffer=frames_buffer,
            status_update_handlers=status_update_handlers,
        )

    def __init__(
        self,
        stream: cv2.VideoCapture,
        frames_buffer: Queue,
        status_update_handlers: List[Callable[[StatusUpdate], None]],
    ):
        self._stream = stream
        self._stream_properties = discover_stream_properties(stream=stream)
        self._frames_buffer = frames_buffer
        self._status_update_handlers = status_update_handlers
        self._state = StreamState.NOT_STARTED
        self._playback_allowed = Event()
        self._stream_consumption_thread: Optional[Thread] = None

    def start(self) -> None:
        if self._state not in START_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not START stream in state: {self._state}"
            )
        self._playback_allowed.set()
        self._stream_consumption_thread = Thread(target=self._consume_stream())
        self._stream_consumption_thread.start()

    def terminate(self) -> None:
        if self._state not in TERMINATE_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not TERMINATE stream in state: {self._state}"
            )
        self._change_state(target_state=StreamState.TERMINATING)
        if self._state is StreamState.PAUSED:
            self.resume()
        self._stream_consumption_thread.join()
        self._frames_buffer.join()
        if self._state is not StreamState.ERROR:
            self._change_state(target_state=StreamState.ENDED)

    def pause(self) -> None:
        if self._state not in PAUSE_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not PAUSE stream in state: {self._state}"
            )
        self._playback_allowed.clear()
        self._change_state(target_state=StreamState.PAUSED)

    def resume(self) -> None:
        if self._state not in RESUME_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not RESUME stream in state: {self._state}"
            )
        self._change_state(target_state=StreamState.RUNNING)
        self._playback_allowed.set()

    def get_state(self) -> StreamState:
        return self._state

    def read_frame(self) -> Tuple[FrameTimestamp, FrameID, np.ndarray]:
        if self._state not in READ_ELIGIBLE_STATES:
            raise StreamReadNotFeasibleError(
                f"Cannot retrieve video frame from stream in state: {self._state}"
            )
        block = True
        if self._state is StreamState.ENDED:
            block = False
        try:
            frame_timestamp, frame_id, frame = self._frames_buffer.get(block=block)
            self._send_status_update(
                severity=UpdateSeverity.DEBUG,
                event_type=FRAME_CONSUMED_EVENT,
                payload={"frame_timestamp": frame_timestamp, "frame_id": frame_id},
            )
        except Empty as error:
            raise EndOfStreamError(
                "Attempted to retrieve frame from stream that already ended."
            ) from error
        self._frames_buffer.task_done()
        return frame_timestamp, frame_id, frame

    @property
    def stream_properties(self) -> StreamProperties:
        return self._stream_properties

    def _consume_stream(self) -> None:
        try:
            self._change_state(target_state=StreamState.RUNNING)
            frame_counter = 0
            while self._stream.isOpened():
                if self._state is StreamState.TERMINATING:
                    break
                self._playback_allowed.wait()
                frame_timestamp = datetime.now()
                success, frame = self._stream.read()
                if not success:
                    break
                frame_counter += 1
                self._send_status_update(
                    severity=UpdateSeverity.DEBUG,
                    event_type=FRAME_CAPTURED_EVENT,
                    payload={
                        "frame_timestamp": frame_timestamp,
                        "frame_id": frame_counter,
                    },
                )
                self._frames_buffer.put((frame_timestamp, frame_counter, frame))
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

    def _change_state(self, target_state: StreamState) -> None:
        payload = {
            "previous_state": self._state,
            "new_state": target_state,
        }
        self._state = target_state
        self._send_status_update(
            severity=UpdateSeverity.DEBUG,
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

    def __iter__(self) -> "VideoStream":
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
