from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import MagicMock

import numpy as np

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.entities import (
    LatencyMonitorReport,
    ModelActivityEvent,
)
from inference.core.interfaces.stream.watchdog import (
    BasePipelineWatchDog,
    are_events_compatible,
    average_property_values,
    compute_events_latency,
)


def assembly_latency_monitor_report(
    frame_decoding_latency: Optional[float],
) -> LatencyMonitorReport:
    return LatencyMonitorReport(
        frame_decoding_latency=frame_decoding_latency,
        inference_latency=0.0,
        e2e_latency=0.0,
    )


def test_average_property_values_when_empty_values_given() -> None:
    # given
    reports = [
        assembly_latency_monitor_report(frame_decoding_latency=None),
        assembly_latency_monitor_report(frame_decoding_latency=None),
    ]

    # when
    result = average_property_values(
        examined_objects=reports, property_name="frame_decoding_latency"
    )

    # then
    assert result is None, "Nothing to average - empty properties"


def test_average_property_values_when_non_empty_values_given() -> None:
    # given
    reports = [
        assembly_latency_monitor_report(frame_decoding_latency=0.1),
        assembly_latency_monitor_report(frame_decoding_latency=0.3),
    ]

    # when
    result = average_property_values(
        examined_objects=reports, property_name="frame_decoding_latency"
    )

    # then
    assert (
        abs(result - 0.2) < 1e-5
    ), "Average of reports should be 0.2, -> (0.1 + 0.3) / 2"


def test_average_property_values_when_empty_and_non_empty_values_given() -> None:
    # given
    reports = [
        assembly_latency_monitor_report(frame_decoding_latency=0.1),
        assembly_latency_monitor_report(frame_decoding_latency=None),
        assembly_latency_monitor_report(frame_decoding_latency=0.3),
        assembly_latency_monitor_report(frame_decoding_latency=None),
        assembly_latency_monitor_report(frame_decoding_latency=None),
    ]

    # when
    result = average_property_values(
        examined_objects=reports, property_name="frame_decoding_latency"
    )

    # then
    assert (
        abs(result - 0.2) < 1e-5
    ), "Average of non-empty reports should be 0.2, -> (0.1 + 0.3) / 2"


def test_are_events_compatible_when_empty_event_given() -> None:
    # given
    events = [
        None,
        ModelActivityEvent(
            frame_decoding_timestamp=datetime.now(),
            event_timestamp=datetime.now(),
            frame_id=1,
        ),
    ]

    # when
    result = are_events_compatible(events=events)

    # then
    assert (
        result is False
    ), "One empty event given (as None) so compatibility cannot be verified"


def test_are_events_compatible_when_no_events_given() -> None:
    # when
    result = are_events_compatible(events=[])

    # then
    assert (
        result is False
    ), "No events given, so compatibility expected not to be checked"


def test_are_events_compatible_when_events_related_to_different_frames_given() -> None:
    # given
    events = [
        ModelActivityEvent(
            frame_decoding_timestamp=datetime.now(),
            event_timestamp=datetime.now(),
            frame_id=1,
        ),
        ModelActivityEvent(
            frame_decoding_timestamp=datetime.now(),
            event_timestamp=datetime.now(),
            frame_id=2,
        ),
    ]

    # when
    result = are_events_compatible(events=events)

    # then
    assert (
        result is False
    ), "Expected events from the different frames not to be compatible"


def test_are_events_compatible_when_events_related_to_the_same_frame_given() -> None:
    # given
    events = [
        ModelActivityEvent(
            frame_decoding_timestamp=datetime.now(),
            event_timestamp=datetime.now(),
            frame_id=1,
        ),
        ModelActivityEvent(
            frame_decoding_timestamp=datetime.now(),
            event_timestamp=datetime.now(),
            frame_id=1,
        ),
    ]

    # when
    result = are_events_compatible(events=events)

    # then
    assert result is True, "Expected events from the same frame to be compatible"


def test_compute_events_latency_when_events_are_compatible() -> None:
    # given
    now = datetime.now()
    then = now + timedelta(milliseconds=38)
    earlier_event = ModelActivityEvent(
        frame_decoding_timestamp=datetime.now(), event_timestamp=now, frame_id=1
    )
    later_event = ModelActivityEvent(
        frame_decoding_timestamp=datetime.now(), event_timestamp=then, frame_id=1
    )

    # when
    result = compute_events_latency(
        earlier_event=earlier_event, later_event=later_event
    )

    # then
    assert abs(result - 0.038) < 1e-5, "Expected latency to be 38ms"


def test_compute_events_latency_when_events_are_not_compatible() -> None:
    # given
    now = datetime.now()
    then = now + timedelta(milliseconds=38)
    earlier_event = ModelActivityEvent(
        frame_decoding_timestamp=datetime.now(), event_timestamp=now, frame_id=1
    )
    later_event = ModelActivityEvent(
        frame_decoding_timestamp=datetime.now(), event_timestamp=then, frame_id=2
    )

    # when
    result = compute_events_latency(
        earlier_event=earlier_event, later_event=later_event
    )

    # then
    assert result is None, "Expected no result when events are not compatible"


def test_base_watchdog_gives_correct_report_when_nothing_registered() -> None:
    # given
    watchdog = BasePipelineWatchDog()

    # when
    result = watchdog.get_report()

    # then
    assert result.video_source_status_updates == [], "No updates emitted to watchdog"
    assert (
        abs(result.inference_throughput) < 1e-5
    ), "Throughput cannot be measured when nothing was registered"
    assert result.latency_reports == [], "Latency report should be empty"
    assert (
        result.sources_metadata == []
    ), "No video source registered, so metadata must be empty"


def test_base_watchdog_gives_correct_report_when_all_events_are_in_series_related_to_the_same_frame() -> (
    None
):
    # given
    watchdog = BasePipelineWatchDog()
    image = np.zeros((192, 168, 3))
    source_mock = MagicMock()
    source_mock.source_id = 0
    source_mock.describe_source.return_value = "METADATA"
    watchdog.register_video_sources(video_sources=[source_mock])
    # when
    first_frame_timestamp = datetime.now()
    watchdog.on_model_inference_started(
        frames=[
            VideoFrame(
                image=image,
                source_id=0,
                frame_id=1,
                frame_timestamp=first_frame_timestamp,
            )
        ]
    )
    watchdog.on_model_prediction_ready(
        frames=[
            VideoFrame(
                image=image,
                source_id=0,
                frame_id=1,
                frame_timestamp=first_frame_timestamp,
            )
        ]
    )
    second_frame_timestamp = datetime.now()
    watchdog.on_model_inference_started(
        frames=[
            VideoFrame(
                image=image,
                source_id=0,
                frame_id=2,
                frame_timestamp=second_frame_timestamp,
            )
        ]
    )
    watchdog.on_model_prediction_ready(
        frames=[
            VideoFrame(
                image=image,
                source_id=0,
                frame_id=2,
                frame_timestamp=second_frame_timestamp,
            )
        ]
    )
    end_of_emission_time = datetime.now()
    result = watchdog.get_report()

    # when
    assert result.video_source_status_updates == [], "No updates emitted to watchdog"
    max_average_e2e_latency = (
        end_of_emission_time - first_frame_timestamp
    ).total_seconds() / 2
    assert (
        len(result.latency_reports) == 1
    ), "Only one video source registered - hence only single latency report is expected"
    assert (
        result.latency_reports[0].e2e_latency <= max_average_e2e_latency
    ), "Latency cannot be larger than time passed in test"
    min_throughput = 2 / (end_of_emission_time - first_frame_timestamp).total_seconds()
    assert (
        result.inference_throughput >= min_throughput
    ), "Throughput cannot be smaller than the one indicated by test time"
    assert (
        len(result.sources_metadata) == 1
    ), "Only one video source registered - hence only single source metadata is expected"
    assert (
        result.sources_metadata[0] == "METADATA"
    ), "Metadata must match mocked video source response"
