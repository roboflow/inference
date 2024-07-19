"""
This module contains component intended to use in combination with `InferencePipeline` to ensure
observability. Please consider them internal details of implementation.
"""

from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, Iterable, List, Optional, TypeVar

import supervision as sv

from inference.core.interfaces.camera.entities import (
    StatusUpdate,
    UpdateSeverity,
    VideoFrame,
)
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
    def register_video_sources(self, video_sources: List[VideoSource]) -> None:
        pass

    @abstractmethod
    def on_status_update(self, status_update: StatusUpdate) -> None:
        pass

    @abstractmethod
    def on_model_inference_started(
        self,
        frames: List[VideoFrame],
    ) -> None:
        pass

    @abstractmethod
    def on_model_prediction_ready(
        self,
        frames: List[VideoFrame],
    ) -> None:
        pass

    @abstractmethod
    def get_report(self) -> Optional[PipelineStateReport]:
        pass


class NullPipelineWatchdog(PipelineWatchDog):
    def register_video_sources(self, video_sources: VideoSource) -> None:
        pass

    def on_status_update(self, status_update: StatusUpdate) -> None:
        pass

    def on_model_inference_started(self, frames: List[VideoFrame]) -> None:
        pass

    def on_model_prediction_ready(self, frames: List[VideoFrame]) -> None:
        pass

    def get_report(self) -> Optional[PipelineStateReport]:
        return None


class LatencyMonitor:
    def __init__(self, source_id: Optional[int]):
        self._source_id = source_id
        self._inference_start_event: Optional[ModelActivityEvent] = None
        self._prediction_ready_event: Optional[ModelActivityEvent] = None
        self._reports: Deque[LatencyMonitorReport] = deque(maxlen=MAX_LATENCY_CONTEXT)

    def register_inference_start(
        self, frame_timestamp: datetime, frame_id: int
    ) -> None:
        self._inference_start_event = ModelActivityEvent(
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
        avg_inference_latency = average_property_values(
            examined_objects=self._reports, property_name="inference_latency"
        )
        avg_e2e_latency = average_property_values(
            examined_objects=self._reports, property_name="e2e_latency"
        )
        return LatencyMonitorReport(
            source_id=self._source_id,
            frame_decoding_latency=avg_frame_decoding_latency,
            inference_latency=avg_inference_latency,
            e2e_latency=avg_e2e_latency,
        )

    def _generate_report(self) -> None:
        frame_decoding_latency = None
        if self._inference_start_event is not None:
            frame_decoding_latency = (
                self._inference_start_event.event_timestamp
                - self._inference_start_event.frame_decoding_timestamp
            ).total_seconds()
        event_pairs = [
            (self._inference_start_event, self._prediction_ready_event),
        ]
        event_pairs_results = []
        for earlier_event, later_event in event_pairs:
            latency = compute_events_latency(
                earlier_event=earlier_event,
                later_event=later_event,
            )
            event_pairs_results.append(latency)
        (inference_latency,) = event_pairs_results
        e2e_latency = None
        if self._prediction_ready_event is not None:
            e2e_latency = (
                self._prediction_ready_event.event_timestamp
                - self._prediction_ready_event.frame_decoding_timestamp
            ).total_seconds()
        self._reports.append(
            LatencyMonitorReport(
                source_id=self._source_id,
                frame_decoding_latency=frame_decoding_latency,
                inference_latency=inference_latency,
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
        self._video_sources: Optional[List[VideoSource]] = None
        self._inference_throughput_monitor = sv.FPSMonitor()
        self._latency_monitors: Dict[Optional[int], LatencyMonitor] = {}
        self._stream_updates = deque(maxlen=MAX_UPDATES_CONTEXT)

    def register_video_sources(self, video_sources: List[VideoSource]) -> None:
        self._video_sources = video_sources
        for source in video_sources:
            self._latency_monitors[source.source_id] = LatencyMonitor(
                source_id=source.source_id
            )

    def on_status_update(self, status_update: StatusUpdate) -> None:
        if status_update.severity.value <= UpdateSeverity.DEBUG.value:
            return None
        self._stream_updates.append(status_update)

    def on_model_inference_started(self, frames: List[VideoFrame]) -> None:
        for frame in frames:
            self._latency_monitors[frame.source_id].register_inference_start(
                frame_timestamp=frame.frame_timestamp,
                frame_id=frame.frame_id,
            )

    def on_model_prediction_ready(self, frames: List[VideoFrame]) -> None:
        for frame in frames:
            self._latency_monitors[frame.source_id].register_prediction_ready(
                frame_timestamp=frame.frame_timestamp,
                frame_id=frame.frame_id,
            )
            self._inference_throughput_monitor.tick()

    def get_report(self) -> PipelineStateReport:
        sources_metadata = []
        if self._video_sources is not None:
            sources_metadata = [s.describe_source() for s in self._video_sources]
        latency_reports = [
            monitor.summarise_reports() for monitor in self._latency_monitors.values()
        ]
        if hasattr(self._inference_throughput_monitor, "fps"):
            _inference_throughput_fps = self._inference_throughput_monitor.fps
        else:
            _inference_throughput_fps = self._inference_throughput_monitor()
        return PipelineStateReport(
            video_source_status_updates=list(self._stream_updates),
            latency_reports=latency_reports,
            inference_throughput=_inference_throughput_fps,
            sources_metadata=sources_metadata,
        )
