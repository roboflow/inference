"""Bounded, ordered results and completion accounting for finite local files."""

from collections import deque
from threading import Condition

from inference.core.interfaces.camera.entities import UpdateSeverity
from inference.core.interfaces.camera.video_source import (
    FRAME_CAPTURED_EVENT,
    StreamState,
)
from inference.core.interfaces.stream.watchdog import BasePipelineWatchDog


class FileJobResults(BasePipelineWatchDog):
    def __init__(self, queue_size, frame_stride):
        super().__init__()
        self._condition = Condition()
        self._buffer = deque()
        self._capacity = queue_size
        self._stride = frame_stride
        self._captured = 0
        self._produced = 0
        self._consumed = 0
        self._cancelled = False
        self._state = "running"
        self._error = None

    def on_status_update(self, status_update):
        super().on_status_update(status_update)
        with self._condition:
            if status_update.event_type == FRAME_CAPTURED_EVENT:
                self._captured = status_update.payload["frame_id"]
            if status_update.severity.value >= UpdateSeverity.ERROR.value:
                self._error = "Pipeline reported an error; inspect server logs"

    def on_prediction(self, predictions, video_frame):
        predictions = predictions if isinstance(predictions, list) else [predictions]
        frames = video_frame if isinstance(video_frame, list) else [video_frame]
        with self._condition:
            if self._cancelled:
                return
            if (
                len(predictions) != 1
                or len(frames) != 1
                or predictions[0] is None
                or frames[0] is None
                or frames[0].source_id != 0
                or frames[0].frame_id != 1 + self._produced * self._stride
            ):
                self._error = "Missing or out-of-order file result"
                raise ValueError(self._error)
            while len(self._buffer) >= self._capacity and not self._cancelled:
                self._condition.wait()
            if self._cancelled:
                return
            self._buffer.append((predictions, frames))
            self._produced += 1
            self._condition.notify_all()

    def empty(self):
        with self._condition:
            return not self._buffer

    def consume_prediction(self):
        with self._condition:
            item = self._buffer.popleft()
            self._consumed += 1
            self._condition.notify_all()
            return item

    def cancel(self):
        with self._condition:
            if self._state == "running":
                self._cancelled = True
            self._condition.notify_all()

    def wait_for_completion(self, pipeline):
        try:
            # join waits for inference AND result dispatch, including final outputs.
            pipeline.join()
            report = self.get_report()
            sources = report.sources_metadata
            with self._condition:
                if (
                    len(sources) != 1
                    or sources[0].state != StreamState.ENDED
                    or not sources[0].source_properties.is_file
                ):
                    self._error = "File did not reach a clean end"
                else:
                    declared = sources[0].source_properties.total_frames
                    if declared > 0 and declared != self._captured:
                        self._error = "Decoded frame count differs from file metadata"
                expected = (self._captured + self._stride - 1) // self._stride
                if self._captured == 0 or self._produced != expected:
                    self._error = "File results are incomplete"
                self._state = (
                    "cancelled"
                    if self._cancelled
                    else "failed" if self._error else "completed"
                )
        except Exception:
            with self._condition:
                self._error = "File completion failed; inspect server logs"
                self._state = "failed"

    def snapshot(self):
        with self._condition:
            return dict(
                state=self._state,
                source_frames=self._captured,
                frame_stride=self._stride,
                results_produced=self._produced,
                results_consumed=self._consumed,
                pending_results=len(self._buffer),
                error=self._error,
            )
