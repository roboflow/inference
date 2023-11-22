from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Iterable, List, Optional, TypeVar

import supervision as sv

from inference.core.interfaces.camera.entities import StatusUpdate, UpdateSeverity
from inference.core.interfaces.camera.video_source import SourceMetadata, VideoSource

T = TypeVar("T")

MAX_ERROR_CONTEXT = 64
MAX_LATENCY_CONTEXT = 64
MAX_STREAM_UPDATES_CONTEXT = 128


@dataclass(frozen=True)
class ModelActivityEvent:
    frame_decoding_timestamp: datetime
    event_timestamp: datetime
    frame_id: int


@dataclass(frozen=True)
class LatencyMonitorReport:
    frame_decoding_latency: Optional[float]
    pre_processing_latency: Optional[float]
    inference_latency: Optional[float]
    post_processing_latency: Optional[float]
    model_latency: Optional[float]
    e2e_latency: Optional[float]


@dataclass(frozen=True)
class ErrorDescription:
    context: str
    error_type: str
    error_message: str


@dataclass(frozen=True)
class PipelineStateReport:
    error_traces: List[ErrorDescription]
    video_source_status_updates: List[StatusUpdate]
    latency_report: LatencyMonitorReport
    inference_throughput: float
    source_metadata: Optional[SourceMetadata]


class PipelineWatchDog(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def register_video_source(self, video_source: VideoSource) -> None:
        pass

    @abstractmethod
    def on_video_source_status_update(self, status_update: StatusUpdate) -> None:
        pass

    @abstractmethod
    def on_model_preprocessing_started(
        self, frame_timestamp: datetime, frame_id: int
    ) -> None:
        pass

    @abstractmethod
    def on_model_inference_started(
        self, frame_timestamp: datetime, frame_id: int
    ) -> None:
        pass

    @abstractmethod
    def on_model_postprocessing_started(
        self, frame_timestamp: datetime, frame_id: int
    ) -> None:
        pass

    @abstractmethod
    def on_model_prediction_ready(
        self, frame_timestamp: datetime, frame_id: int
    ) -> None:
        pass

    @abstractmethod
    def on_error(self, context: str, error: Exception) -> None:
        pass

    @abstractmethod
    def get_report(self) -> Optional[PipelineStateReport]:
        pass


class NullPipelineWatchdog(PipelineWatchDog):
    def register_video_source(self, video_source: VideoSource) -> None:
        pass

    def on_video_source_status_update(self, status_update: StatusUpdate) -> None:
        pass

    def on_model_preprocessing_started(
        self, frame_timestamp: datetime, frame_id: int
    ) -> None:
        pass

    def on_model_inference_started(
        self, frame_timestamp: datetime, frame_id: int
    ) -> None:
        pass

    def on_model_postprocessing_started(
        self, frame_timestamp: datetime, frame_id: int
    ) -> None:
        pass

    def on_model_prediction_ready(
        self, frame_timestamp: datetime, frame_id: int
    ) -> None:
        pass

    def on_error(self, context: str, error: Exception) -> None:
        pass

    def get_report(self) -> Optional[PipelineStateReport]:
        return None


class LatencyMonitor:
    def __init__(self):
        self._preprocessing_start_event: Optional[ModelActivityEvent] = None
        self._inference_start_event: Optional[ModelActivityEvent] = None
        self._postprocessing_start_event: Optional[ModelActivityEvent] = None
        self._prediction_ready_event: Optional[ModelActivityEvent] = None
        self._reports: Deque[LatencyMonitorReport] = deque(maxlen=MAX_LATENCY_CONTEXT)

    def register_preprocessing_start(
        self, frame_timestamp: datetime, frame_id: int
    ) -> None:
        self._preprocessing_start_event = ModelActivityEvent(
            event_timestamp=datetime.now(),
            frame_id=frame_id,
            frame_decoding_timestamp=frame_timestamp,
        )

    def register_inference_start(
        self, frame_timestamp: datetime, frame_id: int
    ) -> None:
        self._inference_start_event = ModelActivityEvent(
            event_timestamp=datetime.now(),
            frame_id=frame_id,
            frame_decoding_timestamp=frame_timestamp,
        )

    def register_postprocessing_start(
        self, frame_timestamp: datetime, frame_id: int
    ) -> None:
        self._postprocessing_start_event = ModelActivityEvent(
            event_timestamp=datetime.now(),
            frame_id=frame_id,
            frame_decoding_timestamp=frame_timestamp,
        )

    def register_prediction_ready(
        self, frame_timestamp: datetime, frame_id: int
    ) -> None:
        self._prediction_ready_event = ModelActivityEvent(
            event_timestamp=datetime.now(),
            frame_id=frame_id,
            frame_decoding_timestamp=frame_timestamp,
        )
        self._generate_report()

    def summarise_reports(self) -> LatencyMonitorReport:
        frame_decoding_latencies = [
            r.frame_decoding_latency
            for r in self._reports
            if r.frame_decoding_latency is not None
        ]
        pre_processing_latencies = [
            r.pre_processing_latency
            for r in self._reports
            if r.pre_processing_latency is not None
        ]
        inference_latencies = [
            r.inference_latency
            for r in self._reports
            if r.inference_latency is not None
        ]
        pos_processing_latencies = [
            r.post_processing_latency
            for r in self._reports
            if r.post_processing_latency is not None
        ]
        model_latencies = [
            r.model_latency for r in self._reports if r.model_latency is not None
        ]
        e2e_latencies = [
            r.e2e_latency for r in self._reports if r.e2e_latency is not None
        ]
        return LatencyMonitorReport(
            frame_decoding_latency=safe_average(values=frame_decoding_latencies),
            pre_processing_latency=safe_average(values=pre_processing_latencies),
            inference_latency=safe_average(values=inference_latencies),
            post_processing_latency=safe_average(values=pos_processing_latencies),
            model_latency=safe_average(values=model_latencies),
            e2e_latency=safe_average(values=e2e_latencies),
        )

    def _generate_report(self) -> None:
        (
            frame_decoding_latency,
            pre_processing_latency,
            inference_latency,
            post_processing_latency,
            model_latency,
            e2e_latency,
        ) = (None, None, None, None, None, None)
        if self._preprocessing_start_event is not None:
            frame_decoding_latency = (
                self._preprocessing_start_event.event_timestamp
                - self._preprocessing_start_event.frame_decoding_timestamp
            ).total_seconds()
        if events_compatible(
            [self._preprocessing_start_event, self._inference_start_event]
        ):
            pre_processing_latency = (
                self._inference_start_event.event_timestamp
                - self._preprocessing_start_event.event_timestamp
            ).total_seconds()
        if events_compatible(
            [self._inference_start_event, self._postprocessing_start_event]
        ):
            inference_latency = (
                self._postprocessing_start_event.event_timestamp
                - self._inference_start_event.event_timestamp
            ).total_seconds()
        if events_compatible(
            [self._prediction_ready_event, self._postprocessing_start_event]
        ):
            post_processing_latency = (
                self._prediction_ready_event.event_timestamp
                - self._postprocessing_start_event.event_timestamp
            ).total_seconds()
        if events_compatible(
            [self._preprocessing_start_event, self._prediction_ready_event]
        ):
            model_latency = (
                self._prediction_ready_event.event_timestamp
                - self._preprocessing_start_event.event_timestamp
            ).total_seconds()
        if self._prediction_ready_event is not None:
            e2e_latency = (
                self._prediction_ready_event.event_timestamp
                - self._prediction_ready_event.frame_decoding_timestamp
            ).total_seconds()
        self._reports.append(
            LatencyMonitorReport(
                frame_decoding_latency=frame_decoding_latency,
                pre_processing_latency=pre_processing_latency,
                inference_latency=inference_latency,
                post_processing_latency=post_processing_latency,
                model_latency=model_latency,
                e2e_latency=e2e_latency,
            )
        )


def events_compatible(events: List[Optional[ModelActivityEvent]]) -> bool:
    if not all_not_empty(sequence=events):
        return False
    if len(events) == 0:
        return False
    frame_ids = [e.frame_id for e in events]
    return all(e == frame_ids[0] for e in frame_ids)


def all_not_empty(sequence: Iterable[Optional[T]]) -> bool:
    return all(e is not None for e in sequence)


def safe_average(values: List[float]) -> Optional[float]:
    if len(values) == 0:
        return None
    return sum(values) / len(values)


class BasePipelineWatchDog(PipelineWatchDog):
    def __init__(self):
        super().__init__()
        self._video_source: Optional[VideoSource] = None
        self._inference_throughput_monitor = sv.FPSMonitor()
        self._latency_monitor = LatencyMonitor()
        self._error_traces = deque(maxlen=MAX_ERROR_CONTEXT)
        self._stream_updates = deque(maxlen=MAX_STREAM_UPDATES_CONTEXT)

    def register_video_source(self, video_source: VideoSource) -> None:
        self._video_source = video_source

    def on_video_source_status_update(self, status_update: StatusUpdate) -> None:
        if status_update.severity is UpdateSeverity.DEBUG:
            return None
        self._stream_updates.append(status_update)

    def on_model_preprocessing_started(
        self, frame_timestamp: datetime, frame_id: int
    ) -> None:
        self._latency_monitor.register_preprocessing_start(
            frame_timestamp=frame_timestamp, frame_id=frame_id
        )

    def on_model_inference_started(
        self, frame_timestamp: datetime, frame_id: int
    ) -> None:
        self._latency_monitor.register_inference_start(
            frame_timestamp=frame_timestamp, frame_id=frame_id
        )

    def on_model_postprocessing_started(
        self, frame_timestamp: datetime, frame_id: int
    ) -> None:
        self._latency_monitor.register_postprocessing_start(
            frame_timestamp=frame_timestamp, frame_id=frame_id
        )

    def on_model_prediction_ready(
        self, frame_timestamp: datetime, frame_id: int
    ) -> None:
        self._latency_monitor.register_prediction_ready(
            frame_timestamp=frame_timestamp, frame_id=frame_id
        )
        self._inference_throughput_monitor.tick()

    def on_error(self, context: str, error: Exception) -> None:
        error_description = ErrorDescription(
            context=context,
            error_type=error.__class__.__name__,
            error_message=str(error),
        )
        self._error_traces.append(error_description)

    def get_report(self) -> PipelineStateReport:
        source_metadata = None
        if self._video_source is not None:
            source_metadata = self._video_source.describe_source()
        return PipelineStateReport(
            error_traces=list(self._error_traces),
            video_source_status_updates=list(self._stream_updates),
            latency_report=self._latency_monitor.summarise_reports(),
            inference_throughput=self._inference_throughput_monitor(),
            source_metadata=source_metadata,
        )
