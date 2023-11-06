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
)
from inference.core.interfaces.camera.exceptions import (
    EndOfStreamError,
    StreamOperationNotAllowedError,
    StreamReadNotFeasibleError,
)

DEFAULT_BUFFER_SIZE = 64


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


class VideoStream:
    @classmethod
    def init(
        cls,
        stream_reference: Union[str, int],
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        on_status_update: Optional[List[Callable[[StatusUpdate], None]]] = None,
    ):
        frames_buffer = Queue(maxsize=buffer_size)
        if on_status_update is None:
            on_status_update = []
        stream = cv2.VideoCapture(stream_reference)
        return cls(
            stream=stream,
            frames_buffer=frames_buffer,
            on_status_update=on_status_update,
        )

    def __init__(
        self,
        stream: cv2.VideoCapture,
        frames_buffer: Queue,
        on_status_update: List[Callable[[StatusUpdate], None]],
    ):
        self._stream = stream
        self._frames_buffer = frames_buffer
        self._on_status_update = on_status_update
        self._state = StreamState.NOT_STARTED
        self._resume_event = Event()
        self._stream_consumption_thread: Optional[Thread] = None

    def start(self) -> None:
        if self._state not in START_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not START stream in state: {self._state}"
            )
        self._stream_consumption_thread = Thread(target=self._consume_stream())
        self._stream_consumption_thread.start()

    def terminate(self) -> None:
        if self._state not in TERMINATE_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not TERMINATE stream in state: {self._state}"
            )
        self._state = StreamState.TERMINATING
        if self._state is StreamState.PAUSED:
            self.resume()

    def pause(self) -> None:
        if self._state not in PAUSE_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not PAUSE stream in state: {self._state}"
            )
        self._state = StreamState.PAUSED

    def resume(self) -> None:
        if self._state not in RESUME_ELIGIBLE_STATES:
            raise StreamOperationNotAllowedError(
                f"Could not RESUME stream in state: {self._state}"
            )
        self._state = StreamState.RUNNING
        self._resume_event.set()

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
        except Empty as error:
            raise EndOfStreamError(
                "Attempted to retrieve frame from stream that already ended."
            ) from error
        self._frames_buffer.task_done()
        return frame_timestamp, frame_id, frame

    def _consume_stream(self) -> None:
        self._state = StreamState.RUNNING
        frame_counter = 0
        while self._stream.isOpened():
            frame_timestamp = datetime.now()
            success, frame = self._stream.read()
            frame_counter += 1
            if not success:
                break
            self._frames_buffer.put((frame_timestamp, frame_counter, frame))
        self._state = StreamState.ENDED

    def __del__(self) -> None:
        self._stream.release()
