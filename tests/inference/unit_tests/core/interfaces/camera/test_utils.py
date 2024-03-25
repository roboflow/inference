import os
import time
from queue import Queue
from typing import Generator, List, Optional, Union
from unittest import mock
from unittest.mock import MagicMock, call

import numpy as np
import pytest
import supervision as sv

from inference.core.interfaces.camera import utils
from inference.core.interfaces.camera.exceptions import SourceConnectionError
from inference.core.interfaces.camera.utils import (
    FPSLimiterStrategy,
    RateLimiter,
    _attempt_reconnect,
    _establish_sources_reconnection_rules,
    _find_free_source_identifier,
    _prepare_video_sources,
    get_video_frames_generator,
    limit_frame_rate,
    multiplex_videos,
    negotiate_rate_limiter_strategy_for_multiple_sources,
    resolve_limiter_strategy,
)
from inference.core.interfaces.camera.video_source import (
    SourceMetadata,
    SourceProperties,
    VideoSource,
)
from inference.core.utils.preprocess import letterbox_image


def test_rate_limiter_when_no_ticks_were_registered() -> None:
    # given
    limiter = RateLimiter(desired_fps=30)

    # when
    result = limiter.estimate_next_action_delay()

    # then
    assert 0 <= result < 1e-5, "First tick should happen immediately"


def test_rate_limiter_when_invalid_fps_registered() -> None:
    # given
    limiter = RateLimiter(desired_fps=-1)

    # when
    limiter.tick()
    result = limiter.estimate_next_action_delay()

    # then
    assert (
        0 < result <= 100
    ), "Default value is 1 tick at 100 s, so delay should be at max 100s"


@mock.patch.object(utils, "time")
def test_rate_limiter_when_next_tick_should_be_executed_immediately(
    time_mock: MagicMock,
) -> None:
    # given
    time_mock.monotonic.side_effect = [100.0, 100.110]
    limiter = RateLimiter(desired_fps=10.0)

    # when
    limiter.tick()
    result = limiter.estimate_next_action_delay()

    # then
    assert (
        0 <= result < 1e-5
    ), "Mock indicate that it last 110ms from .tick(), so its time to take action!"


@mock.patch.object(utils, "time")
def test_rate_limiter_when_next_tick_should_be_delayed(time_mock: MagicMock) -> None:
    # given
    time_mock.monotonic.side_effect = [100.0, 100.05]
    limiter = RateLimiter(desired_fps=10.0)

    # when
    limiter.tick()
    result = limiter.estimate_next_action_delay()

    # then
    assert result >= 0
    assert (
        result - 0.05
    ) < 1e-5, (
        "Mock indicate that it last 50ms from .tick(), so we should rest next 50ms"
    )


def test_limit_frame_rate_when_frames_to_be_dropped_and_stream_is_to_fast() -> None:
    # given
    frames_generator = generate_with_delay(items=10, delay=0.005)

    # when
    results, results_timestamp = [], []
    for result in limit_frame_rate(
        frames_generator=frames_generator, max_fps=100, strategy=FPSLimiterStrategy.DROP
    ):
        results_timestamp.append(time.monotonic())
        results.append(result)

    # then
    timestamp_differences = get_pairs_differences(results_timestamp)
    assert all(
        diff >= 0.01 for diff in timestamp_differences
    ), "Difference between two next frames should be at least 10ms"
    assert (
        0 < len(results) <= 5
    ), "Stream is 200fps, so we should process at most half (5) of the items"


def test_limit_frame_rate_when_frames_to_be_dropped_and_stream_is_to_slow() -> None:
    # given
    frames_generator = generate_with_delay(items=10, delay=0.02)

    # when
    results, results_timestamp = [], []
    for result in limit_frame_rate(
        frames_generator=frames_generator, max_fps=100, strategy=FPSLimiterStrategy.DROP
    ):
        results_timestamp.append(time.monotonic())
        results.append(result)

    # then
    timestamp_differences = get_pairs_differences(results_timestamp)
    assert all(
        diff >= 0.01 for diff in timestamp_differences
    ), "Minimum delay between 100FPS frames should be 0.01s"
    assert results == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ], "As 50FPS stream does not violate 100FPS limit - all frames should have been kept, despite DROP strategy"


def test_limit_frame_rate_when_frames_to_be_awaited_and_stream_is_to_fast() -> None:
    # given
    frames_generator = generate_with_delay(items=10, delay=0.005)

    # when
    results, results_timestamp = [], []
    for result in limit_frame_rate(
        frames_generator=frames_generator,
        max_fps=100,
        strategy=FPSLimiterStrategy.WAIT,
    ):
        results_timestamp.append(time.monotonic())
        results.append(result)

    # then
    timestamp_differences = get_pairs_differences(results_timestamp)
    assert all(
        diff >= 0.01 for diff in timestamp_differences
    ), "Minimum delay between 100FPS frames should be 0.01s"
    assert results == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ], "Despite stream being emitted in 200FPS, WAIT strategy should have enforced all frames to be kept"


def test_limit_frame_rate_when_frames_to_be_awaited_and_stream_is_to_slow() -> None:
    # given
    frames_generator = generate_with_delay(items=10, delay=0.02)

    # when
    results, results_timestamp = [], []
    for result in limit_frame_rate(
        frames_generator=frames_generator,
        max_fps=100,
        strategy=FPSLimiterStrategy.WAIT,
    ):
        results_timestamp.append(time.monotonic())
        results.append(result)

    # then
    timestamp_differences = get_pairs_differences(results_timestamp)
    assert all(
        diff >= 0.01 for diff in timestamp_differences
    ), "Minimum delay between 100FPS frames should be 0.01s"
    assert results == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ], "As 50FPS stream does not violate 100FPS limit - all frames should have been kept, and WAIT strategy should not prevent that"


def test_resolve_limiter_strategy_when_strategy_defined_explicitly() -> None:
    # when
    result = resolve_limiter_strategy(
        explicitly_defined_strategy=FPSLimiterStrategy.WAIT,
        source_properties=None,
    )

    # then
    assert (
        result is FPSLimiterStrategy.WAIT
    ), "Explicit strategy given should override any default behaviour"


def test_resolve_limiter_strategy_when_automatic_choice_to_be_made_without_source_properties() -> (
    None
):
    # when
    result = resolve_limiter_strategy(
        explicitly_defined_strategy=None,
        source_properties=None,
    )

    # then
    assert (
        result is FPSLimiterStrategy.DROP
    ), "Default strategy when `source_properties` not given should be DROP"


def test_resolve_limiter_strategy_when_automatic_choice_to_be_made_against_video_file_source() -> (
    None
):
    # given
    source_properties = SourceProperties(
        width=100, height=100, total_frames=10, is_file=True, fps=25
    )

    # when
    result = resolve_limiter_strategy(
        explicitly_defined_strategy=None,
        source_properties=source_properties,
    )

    # then
    assert (
        result is FPSLimiterStrategy.WAIT
    ), "Automatic strategy for file should be WAIT"


def test_resolve_limiter_strategy_when_automatic_choice_to_be_made_against_video_stream_source() -> (
    None
):
    # given
    source_properties = SourceProperties(
        width=100, height=100, total_frames=-1, is_file=False, fps=25
    )

    # when
    result = resolve_limiter_strategy(
        explicitly_defined_strategy=None,
        source_properties=source_properties,
    )

    # then
    assert (
        result is FPSLimiterStrategy.DROP
    ), "Automatic strategy for on-line stream should be DROP"


@mock.patch.object(utils.VideoSource, "init")
def test_get_video_frames_generator_when_fps_modulation_disabled(
    init_mock: MagicMock,
) -> None:
    # given
    dummy_source = DummyVideoSource(items=10, delay=0.01)
    init_mock.return_value = dummy_source

    # when

    result = list(get_video_frames_generator(video="source-ref"))

    # then
    assert (
        dummy_source.start_called is True
    ), "VideoSource must be started once initialised from source reference"
    assert result == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ], "Without rate limiting, generator must not alter order or completeness of generation"


@mock.patch.object(utils.VideoSource, "init")
def test_get_video_frames_generator_when_fps_modulation_enabled_against_video_file(
    init_mock: MagicMock,
) -> None:
    # given
    source_properties = SourceProperties(
        width=100, height=100, total_frames=10, is_file=True, fps=100
    )
    dummy_source = DummyVideoSource(
        items=10, delay=0.01, source_properties=source_properties
    )
    init_mock.return_value = dummy_source

    # when
    results, results_timestamp = [], []
    for result in get_video_frames_generator(
        video="source-ref",
        max_fps=50,
    ):
        results_timestamp.append(time.monotonic())
        results.append(result)

    # then
    timestamp_differences = get_pairs_differences(results_timestamp)
    assert all(
        diff >= 0.02 for diff in timestamp_differences
    ), "At minimum, 0.02s delay must must be enforced by generator, even if decoding is faster"
    assert results == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ], "For video file, default strategy must enforce in-order processing of all frames"
    assert (
        dummy_source.start_called is True
    ), "VideoSource must be started once initialised from source reference"


@mock.patch.object(utils.VideoSource, "init")
def test_get_video_frames_generator_when_fps_modulation_enabled_against_fast_stream(
    init_mock: MagicMock,
) -> None:
    # given
    source_properties = SourceProperties(
        width=100, height=100, total_frames=-1, is_file=False, fps=100
    )
    dummy_source = DummyVideoSource(
        items=10, delay=0.01, source_properties=source_properties
    )
    init_mock.return_value = dummy_source

    # when
    results, results_timestamp = [], []
    for result in get_video_frames_generator(
        video="source-ref",
        max_fps=50,
    ):
        results_timestamp.append(time.monotonic())
        results.append(result)

    # then
    timestamp_differences = get_pairs_differences(results_timestamp)
    assert all(
        diff >= 0.02 for diff in timestamp_differences
    ), "At minimum, 0.02s delay must be enforced by generator, even if stream is faster"
    assert (
        0 <= len(results) <= 5
    ), "With default strategy being DROP frames that do not fit FPS limit, having 100FPS stream and 50FPS limit we should process at most 50% of frames"
    assert (
        dummy_source.start_called is True
    ), "VideoSource must be started once initialised from source reference"


@mock.patch.object(utils.VideoSource, "init")
def test_get_video_frames_generator_when_fps_modulation_enabled_against_slow_stream(
    init_mock: MagicMock,
) -> None:
    # given
    source_properties = SourceProperties(
        width=100, height=100, total_frames=-1, is_file=False, fps=100
    )
    dummy_source = DummyVideoSource(
        items=10, delay=0.02, source_properties=source_properties
    )
    init_mock.return_value = dummy_source

    # when
    results, results_timestamp = [], []
    for result in get_video_frames_generator(
        video="source-ref",
        max_fps=200,
    ):
        results_timestamp.append(time.monotonic())
        results.append(result)

    # then
    timestamp_differences = get_pairs_differences(results_timestamp)
    assert all(
        diff >= 0.02 for diff in timestamp_differences
    ), "At minimum, 0.02s delay must be enforced by generator, even if stream is slower"
    assert results == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ], "With stream being twice slower that limit of FPS - all frames should have been processed in order"
    assert (
        dummy_source.start_called is True
    ), "VideoSource must be started once initialised from source reference"


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_get_video_frames_generator_against_real_video_without_rate_limit(
    local_video_path: str,
) -> None:
    # when
    results = list(get_video_frames_generator(video=local_video_path))

    # then
    assert (
        len(results) == 431
    ), "This video has 431 frames and all of them should be processed"


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_get_video_frames_generator_against_real_video_with_rate_limit_and_await_strategy(
    local_video_path: str,
) -> None:
    # when
    results, results_timestamp = [], []
    for result in get_video_frames_generator(
        video=local_video_path,
        max_fps=200,
    ):
        results_timestamp.append(time.monotonic())
        results.append(result)
        print(result)

    # then
    assert (
        len(results) == 431
    ), "This video has 431 frames and all of them should be processed"


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_get_video_frames_generator_against_real_video_with_rate_limit_and_drop_strategy(
    local_video_path: str,
) -> None:
    # when
    results, results_timestamp = [], []
    for result in get_video_frames_generator(
        video=local_video_path, max_fps=200, limiter_strategy=FPSLimiterStrategy.DROP
    ):
        results_timestamp.append(time.monotonic())
        results.append(result)

    # then
    timestamp_differences = get_pairs_differences(results_timestamp)
    assert (
        sum(timestamp_differences) / len(timestamp_differences) >= 0.005
    ), "On average, time difference between frames must be at minimum 0.005s to match 200FPS limit"
    assert (
        0 <= len(results) <= 431
    ), "This video has 431 frames and part of them could be dropped, if decoding happens faster than 200FPS"


class DummyVideoSource:
    def __init__(
        self,
        items: int,
        delay: float,
        source_properties: Optional[SourceProperties] = None,
    ):
        self._stream = generate_with_delay(items=items, delay=delay)
        self._source_properties = source_properties
        self.start_called = False

    def start(self) -> None:
        self.start_called = True

    def describe_source(self) -> SourceMetadata:
        metadata_mock = MagicMock()
        metadata_mock.source_properties = self._source_properties
        return metadata_mock

    def terminate(
        self, wait_on_frames_consumption: bool = True, purge_frames_buffer: bool = False
    ) -> None:
        pass

    def __iter__(self) -> "DummyVideoSource":
        return self

    def __next__(self) -> int:
        return next(self._stream)


def generate_with_delay(items: int, delay: float) -> Generator[int, None, None]:
    for i in range(items):
        yield i
        time.sleep(delay)


def get_pairs_differences(values: List[float]) -> List[float]:
    result = []
    for x, y in zip(values, values[1:]):
        result.append(y - x)
    return result


def test_get_differences_when_empty_input_given() -> None:
    # when
    result = get_pairs_differences(values=[])

    # then
    assert (
        len(result) == 0
    ), "For empty values lists - there no between-values differences to be calculated"


def test_get_differences_when_single_element_input_given() -> None:
    # when
    result = get_pairs_differences(values=[1.0])

    # then
    assert (
        len(result) == 0
    ), "For single element list - there no between-values differences to be calculated"


def test_get_differences_when_multi_elements_input_given() -> None:
    # when
    result = get_pairs_differences(values=[1.0, 1.5, 2.5, 4])

    # then
    assert (
        len(result) == 3
    ), "Number of differences between pirs in 4 elements list must be 3"
    assert np.allclose(
        result, [0.5, 1.0, 1.5]
    ), "Result must match differences between values pairs"


@pytest.mark.timeout(90)
@pytest.mark.slow
@pytest.mark.parametrize(
    "max_fps, batch_collection_timeout",
    [(None, None), (None, 0.001), (100, None), (100, 0.001)],
)
def test_multiplex_videos_when_multiple_video_files_provided(
    local_video_path: str,
    max_fps: Optional[Union[float, int]],
    batch_collection_timeout: Optional[float],
) -> None:
    # given
    source_0_frames = []
    source_1_frames = []
    empty_frames = 0

    # when
    for video_frames in multiplex_videos(
        videos=[local_video_path, local_video_path],
        max_fps=max_fps,
        limiter_strategy=FPSLimiterStrategy.WAIT,
        batch_collection_timeout=batch_collection_timeout,
    ):
        for video_frame in video_frames:
            if video_frame.source_id == 0:
                source_0_frames.append(video_frame.frame_id)
            else:
                source_1_frames.append(video_frame.frame_id)
            if np.allclose(
                video_frame.image, np.zeros_like(video_frame.image), atol=5.0
            ):
                empty_frames += 1

    # then
    assert source_0_frames == list(
        range(1, 432)
    ), "Order of video frames abused or not all frames processed for source 0"
    assert source_1_frames == list(
        range(1, 432)
    ), "Order of video frames abused or not all frames processed for source 1"
    assert (
        empty_frames == 0
    ), "Expected not to encounter empty frames in the video (they don't exist in source and may only appear if batch collection is faulty)"


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_multiplex_videos_when_stop_signal_used(local_video_path: str) -> None:
    # given
    stop_state = []

    def should_stop() -> bool:
        return len(stop_state) > 0

    processed_frames = []

    # when
    for video_frames in multiplex_videos(
        videos=[local_video_path, local_video_path],
        should_stop=should_stop,
    ):
        processed_frames.append(video_frames)
        stop_state.append(1)

    # then
    assert (
        len(processed_frames) == 1
    ), "Processing should be stopped immediately after receiving first frames"
    assert len(processed_frames[0]) == 2, "All sources should report first frame"
    assert (
        processed_frames[0][0].source_id == 0
    ), "First video frame should come from source 0"
    assert (
        processed_frames[0][1].source_id == 1
    ), "Second video frame should come from source 1"


@pytest.mark.timeout(5)
def test_attempt_reconnect_when_terminal_error_happens() -> None:
    # given
    video_source = MagicMock()
    video_source.restart.side_effect = Exception()
    on_reconnection_failure, on_reconnection_success, on_fatal_error = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    # when
    _attempt_reconnect(
        video_source=video_source,
        should_stop=lambda: False,
        on_reconnection_failure=on_reconnection_failure,
        on_reconnection_success=on_reconnection_success,
        on_fatal_error=on_fatal_error,
    )

    # then
    on_reconnection_failure.assert_not_called()
    on_reconnection_success.assert_not_called()
    on_fatal_error.assert_called_once()


@pytest.mark.timeout(5)
def test_attempt_reconnect_when_reconnection_succeeds_at_first_attempt() -> None:
    # given
    video_source = MagicMock()
    on_reconnection_failure, on_reconnection_success, on_fatal_error = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    # when
    _attempt_reconnect(
        video_source=video_source,
        should_stop=lambda: False,
        on_reconnection_failure=on_reconnection_failure,
        on_reconnection_success=on_reconnection_success,
        on_fatal_error=on_fatal_error,
    )

    # then
    on_reconnection_failure.assert_not_called()
    on_reconnection_success.assert_called_once()
    on_fatal_error.assert_not_called()


@pytest.mark.timeout(5)
def test_attempt_reconnect_when_reconnection_succeeds_at_nth_attempt() -> None:
    # given
    video_source = MagicMock()
    on_reconnection_failure, on_reconnection_success, on_fatal_error = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    error = SourceConnectionError()
    video_source.restart.side_effect = [error, error, None]

    # when
    _attempt_reconnect(
        video_source=video_source,
        should_stop=lambda: False,
        on_reconnection_failure=on_reconnection_failure,
        on_reconnection_success=on_reconnection_success,
        on_fatal_error=on_fatal_error,
    )

    # then
    on_reconnection_failure.assert_has_calls(
        [call(video_source.source_id, error), call(video_source.source_id, error)]
    )
    on_reconnection_success.assert_called_once()
    on_fatal_error.assert_not_called()


@pytest.mark.timeout(5)
def test_attempt_reconnect_when_stop_signal_from_outside_received() -> None:
    # given
    video_source = MagicMock()
    on_reconnection_failure, on_reconnection_success, on_fatal_error = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    error = SourceConnectionError()
    video_source.restart.side_effect = error
    should_stop_invocations = []

    def should_stop() -> bool:
        should_stop_invocations.append(1)
        return len(should_stop_invocations) > 3

    # when
    _attempt_reconnect(
        video_source=video_source,
        should_stop=should_stop,
        on_reconnection_failure=on_reconnection_failure,
        on_reconnection_success=on_reconnection_success,
        on_fatal_error=on_fatal_error,
    )

    # then
    on_reconnection_failure.assert_called()
    on_reconnection_success.assert_not_called()
    on_fatal_error.assert_not_called()


def test_establish_sources_reconnection_rules_when_reconnection_enforced() -> None:
    # given
    all_sources = [MagicMock(), MagicMock(), MagicMock()]
    all_sources[0].describe_source.return_value.source_properties = (
        None  # source with unknown properties
    )
    all_sources[1].describe_source.return_value.source_properties.is_file = (
        True  # video file
    )
    all_sources[2].describe_source.return_value.source_properties.is_file = (
        False  # stream
    )

    # when
    result = _establish_sources_reconnection_rules(
        all_sources=all_sources,
        force_stream_reconnection=True,
    )

    # then
    assert result == [
        False,
        False,
        True,
    ], "Expected not to reconnect to source without properties and file - only reconnect to stream"


def test_establish_sources_reconnection_rules_when_reconnection_not_enforced() -> None:
    # given
    all_sources = [MagicMock(), MagicMock(), MagicMock()]
    all_sources[0].describe_source.return_value.source_properties = (
        None  # source with unknown properties
    )
    all_sources[1].describe_source.return_value.source_properties.is_file = (
        True  # video file
    )
    all_sources[2].describe_source.return_value.source_properties.is_file = (
        False  # stream
    )

    # when
    result = _establish_sources_reconnection_rules(
        all_sources=all_sources,
        force_stream_reconnection=False,
    )

    # then
    assert result == [
        False,
        False,
        False,
    ], "Expected not to reconnect to source without properties and file - stream reconnection is to be disabled by `force_stream_reconnection` flag"


@pytest.mark.timeout(10)
def test_prepare_video_sources_when_faulty_reference_provided(
    local_video_path: str,
) -> None:
    # when
    with pytest.raises(SourceConnectionError):
        _ = _prepare_video_sources(
            videos=[local_video_path, "invalid/file.mp4"],
            force_stream_reconnection=True,
        )


@pytest.mark.timeout(10)
def test_prepare_video_sources_when_faulty_reference_provided_and_externally_managed_source_should_not_be_terminated(
    local_video_path: str,
) -> None:
    # given
    my_video_source = MagicMock()

    # when
    with pytest.raises(SourceConnectionError):
        _ = _prepare_video_sources(
            videos=[my_video_source, local_video_path, "invalid/file.mp4"],
            force_stream_reconnection=True,
        )

    # then
    my_video_source.terminate.assert_not_called()


@pytest.mark.timeout(5)
def test_prepare_video_sources_when_valid_input_provided(local_video_path: str) -> None:
    # when
    result = _prepare_video_sources(
        videos=[local_video_path, local_video_path],
        force_stream_reconnection=True,
    )

    # then
    try:
        assert len(result.all_sources) == 2, "2 sources should be registered"
        assert (
            len(result.managed_sources) == 2
        ), "2 sources should be registered as managed ones"
        assert result.allow_reconnection == [
            False,
            False,
        ], "Reconnections should not be allowed for video files"
    finally:
        for video in result.all_sources:
            video.terminate(wait_on_frames_consumption=False, purge_frames_buffer=True)


def test_find_free_source_identifier_when_no_initialised_video_sources_provided() -> (
    None
):
    # when
    result = _find_free_source_identifier(videos=["a", 0])

    # then
    assert result == 0


def test_find_free_source_identifier_when_initialised_video_source_provided_without_stream_id() -> (
    None
):
    # given
    video_source = VideoSource(
        stream_reference=0,
        frames_buffer=Queue(),
        status_update_handlers=[],
        buffer_consumption_strategy=None,
        video_consumer=MagicMock(),
        video_source_properties=None,
        source_id=None,
    )

    # when
    result = _find_free_source_identifier(videos=["a", 0, video_source])

    # then
    assert result == 0


def test_find_free_source_identifier_when_initialised_video_source_provided_with_stream_id() -> (
    None
):
    # given
    video_source = VideoSource(
        stream_reference=0,
        frames_buffer=Queue(),
        status_update_handlers=[],
        buffer_consumption_strategy=None,
        video_consumer=MagicMock(),
        video_source_properties=None,
        source_id=37,
    )

    # when
    result = _find_free_source_identifier(videos=["a", 0, video_source])

    # then
    assert result == 38


def test_negotiate_rate_limiter_strategy_for_multiple_sources_when_only_file_sources_provided() -> (
    None
):
    # given
    video_sources = [MagicMock(), MagicMock()]
    video_sources[0].describe_source.return_value.source_properties.is_file = True
    video_sources[1].describe_source.return_value.source_properties.is_file = True

    # when
    result = negotiate_rate_limiter_strategy_for_multiple_sources(
        video_sources=video_sources
    )

    # then
    assert result is FPSLimiterStrategy.WAIT


def test_negotiate_rate_limiter_strategy_for_multiple_sources_when_only_stream_sources_provided() -> (
    None
):
    # given
    video_sources = [MagicMock(), MagicMock()]
    video_sources[0].describe_source.return_value.source_properties.is_file = False
    video_sources[1].describe_source.return_value.source_properties.is_file = False

    # when
    result = negotiate_rate_limiter_strategy_for_multiple_sources(
        video_sources=video_sources
    )

    # then
    assert result is FPSLimiterStrategy.DROP


def test_negotiate_rate_limiter_strategy_for_multiple_sources_when_mixed_sources_provided() -> (
    None
):
    # given
    video_sources = [MagicMock(), MagicMock()]
    video_sources[0].describe_source.return_value.source_properties.is_file = True
    video_sources[1].describe_source.return_value.source_properties.is_file = False

    # when
    result = negotiate_rate_limiter_strategy_for_multiple_sources(
        video_sources=video_sources
    )

    # then
    assert result is FPSLimiterStrategy.DROP
