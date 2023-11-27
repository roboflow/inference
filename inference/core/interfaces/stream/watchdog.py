"""
This module contains component intended to use in combination with `InferencePipeline` to ensure
observability. Please consider them internal details of implementation.
"""

from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import Any, Deque, Iterable, List, Optional, TypeVar

import supervision as sv

from inference.core.interfaces.camera.entities import StatusUpdate, UpdateSeverity
from inference.core.interfaces.camera.video_source import VideoSource
from inference.core.interfaces.stream.entities import (
    LatencyMonitorReport,
    ModelActivityEvent,
    PipelineStateReport,
)

T = TypeVar("T")

MAX_LATENCY_CONTEXT = 64
MAX_UPDATES_CONTEXT = 512


class PipelineWatchDog(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def register_video_source(self, video_source: VideoSource) -> None:
        pass

    @abstractmethod
    def on_status_update(self, status_update: StatusUpdate) -> None:
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
    def get_report(self) -> Optional[PipelineStateReport]:
        pass


class NullPipelineWatchdog(PipelineWatchDog):
    def register_video_source(self, video_source: VideoSource) -> None:
        pass

    def on_status_update(self, status_update: StatusUpdate) -> None:
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
        avg_frame_decoding_latency = average_property_values(
            examined_objects=self._reports, property_name="frame_decoding_latency"
        )
        avg_pre_processing_latency = average_property_values(
            examined_objects=self._reports, property_name="pre_processing_latency"
        )
        avg_inference_latency = average_property_values(
            examined_objects=self._reports, property_name="inference_latency"
        )
        avg_pos_processing_latency = average_property_values(
            examined_objects=self._reports, property_name="post_processing_latency"
        )
        avg_model_latency = average_property_values(
            examined_objects=self._reports, property_name="model_latency"
        )
        avg_e2e_latency = average_property_values(
            examined_objects=self._reports, property_name="e2e_latency"
        )
        return LatencyMonitorReport(
            frame_decoding_latency=avg_frame_decoding_latency,
            pre_processing_latency=avg_pre_processing_latency,
            inference_latency=avg_inference_latency,
            post_processing_latency=avg_pos_processing_latency,
            model_latency=avg_model_latency,
            e2e_latency=avg_e2e_latency,
        )

    def _generate_report(self) -> None:
        frame_decoding_latency = None
        if self._preprocessing_start_event is not None:
            frame_decoding_latency = (
                self._preprocessing_start_event.event_timestamp
                - self._preprocessing_start_event.frame_decoding_timestamp
            ).total_seconds()
        event_pairs = [
            (self._preprocessing_start_event, self._inference_start_event),
            (self._inference_start_event, self._postprocessing_start_event),
            (self._postprocessing_start_event, self._prediction_ready_event),
            (self._preprocessing_start_event, self._prediction_ready_event),
        ]
        event_pairs_results = []
        for earlier_event, later_event in event_pairs:
            latency = compute_events_latency(
                earlier_event=earlier_event,
                later_event=later_event,
            )
            event_pairs_results.append(latency)
        (
            pre_processing_latency,
            inference_latency,
            post_processing_latency,
            model_latency,
        ) = event_pairs_results
        e2e_latency = None
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


def average_property_values(
    examined_objects: Iterable, property_name: str
) -> Optional[float]:
    values = get_not_empty_properties(
        examined_objects=examined_objects, property_name=property_name
    )
    return safe_average(values=values)


def get_not_empty_properties(
    examined_objects: Iterable, property_name: str
) -> List[Any]:
    results = [
        getattr(examined_object, property_name, None)
        for examined_object in examined_objects
    ]
    return [e for e in results if e is not None]


def safe_average(values: List[float]) -> Optional[float]:
    if len(values) == 0:
        return None
    return sum(values) / len(values)


def compute_events_latency(
    earlier_event: Optional[ModelActivityEvent],
    later_event: Optional[ModelActivityEvent],
) -> Optional[float]:
    if not are_events_compatible(events=[earlier_event, later_event]):
        return None
    return (later_event.event_timestamp - earlier_event.event_timestamp).total_seconds()


def are_events_compatible(events: List[Optional[ModelActivityEvent]]) -> bool:
    if any(e is None for e in events):
        return False
    if len(events) == 0:
        return False
    frame_ids = [e.frame_id for e in events]
    return all(e == frame_ids[0] for e in frame_ids)


class BasePipelineWatchDog(PipelineWatchDog):
    """
    Implementation to be used from single inference thread, as it keeps
    state assumed to represent status of consecutive stage of prediction process
    in latency monitor.
    """

    def __init__(self):
        super().__init__()
        self._video_source: Optional[VideoSource] = None
        self._inference_throughput_monitor = sv.FPSMonitor()
        self._latency_monitor = LatencyMonitor()
        self._stream_updates = deque(maxlen=MAX_UPDATES_CONTEXT)

    def register_video_source(self, video_source: VideoSource) -> None:
        self._video_source = video_source

    def on_status_update(self, status_update: StatusUpdate) -> None:
        if status_update.severity.value <= UpdateSeverity.DEBUG.value:
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

    def get_report(self) -> PipelineStateReport:
        source_metadata = None
        if self._video_source is not None:
            source_metadata = self._video_source.describe_source()
        return PipelineStateReport(
            video_source_status_updates=list(self._stream_updates),
            latency_report=self._latency_monitor.summarise_reports(),
            inference_throughput=self._inference_throughput_monitor(),
            source_metadata=source_metadata,
        )
