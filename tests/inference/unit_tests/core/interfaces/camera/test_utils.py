import time
from typing import Generator, Optional, List
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.interfaces.camera.utils import (
    RateLimiter,
    limit_frame_rate,
    FPSLimiterStrategy,
    resolve_limiter_strategy,
    get_video_frames_generator,
)
from inference.core.interfaces.camera import utils
from inference.core.interfaces.camera.video_source import (
    SourceProperties,
    SourceMetadata,
)


def test_rate_limiter_when_no_ticks_were_registered() -> None:
    # given
    limiter = RateLimiter(desired_fps=30)

    # when
    result = limiter.estimate_next_tick_delay()

    # then
    assert result >= 0
    assert result < 1e-5


def test_rate_limiter_when_invalid_fps_registered() -> None:
    # given
    limiter = RateLimiter(desired_fps=-1)

    # when
    limiter.tick()
    result = limiter.estimate_next_tick_delay()

    # then
    assert 0 < result < 100


@mock.patch.object(utils, "time")
def test_rate_limiter_when_next_tick_should_be_executed_immediately(
    time_mock: MagicMock,
) -> None:
    # given
    time_mock.monotonic.side_effect = [100.0, 100.110]
    limiter = RateLimiter(desired_fps=10.0)

    # when
    limiter.tick()
    result = limiter.estimate_next_tick_delay()

    # then
    assert result >= 0
    assert result < 1e-5


@mock.patch.object(utils, "time")
def test_rate_limiter_when_next_tick_should_be_delayed(time_mock: MagicMock) -> None:
    # given
    time_mock.monotonic.side_effect = [100.0, 100.05]
    limiter = RateLimiter(desired_fps=10.0)

    # when
    limiter.tick()
    result = limiter.estimate_next_tick_delay()

    # then
    assert result >= 0
    assert (result - 0.05) < 1e-5


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
    timestamp_differences = get_differences(results_timestamp)
    assert all(diff >= 0.01 for diff in timestamp_differences)
    assert 0 < len(results) <= 5


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
    timestamp_differences = get_differences(results_timestamp)
    assert all(diff >= 0.01 for diff in timestamp_differences)
    assert results == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


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
    timestamp_differences = get_differences(results_timestamp)
    assert all(diff >= 0.01 for diff in timestamp_differences)
    assert results == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


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
    timestamp_differences = get_differences(results_timestamp)
    assert all(diff >= 0.01 for diff in timestamp_differences)
    assert results == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_resolve_limiter_strategy_when_strategy_defined_explicitly() -> None:
    # when
    result = resolve_limiter_strategy(
        explicitly_defined_strategy=FPSLimiterStrategy.WAIT,
        source_properties=None,
    )

    # then
    assert result is FPSLimiterStrategy.WAIT


def test_resolve_limiter_strategy_when_automatic_choice_to_be_made_without_source_properties() -> (
    None
):
    # when
    result = resolve_limiter_strategy(
        explicitly_defined_strategy=None,
        source_properties=None,
    )

    # then
    assert result is FPSLimiterStrategy.DROP


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
    assert result is FPSLimiterStrategy.WAIT


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
    assert result is FPSLimiterStrategy.DROP


@mock.patch.object(utils.VideoSource, "init")
def test_get_video_frames_generator_when_fps_modulation_disabled(
    init_mock: MagicMock,
) -> None:
    # given
    dummy_source = DummyVideoSource(items=10, delay=0.01)
    init_mock.return_value = dummy_source

    # when

    result = list(get_video_frames_generator(stream="source-ref"))

    # then
    assert dummy_source.start_called is True
    assert result == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


@mock.patch.object(utils.VideoSource, "init")
def test_get_video_frames_generator_when_fps_modulation_enabled_against_video_file(
    init_mock: MagicMock,
) -> None:
    # given
    source_properties = SourceProperties(
        width=100, height=100, total_frames=10, is_file=True, fps=25
    )
    dummy_source = DummyVideoSource(
        items=10, delay=0.01, source_properties=source_properties
    )
    init_mock.return_value = dummy_source

    # when
    results, results_timestamp = [], []
    for result in get_video_frames_generator(
        stream="source-ref",
        max_fps=50,
    ):
        results_timestamp.append(time.monotonic())
        results.append(result)

    # then
    timestamp_differences = get_differences(results_timestamp)
    assert all(diff >= 0.02 for diff in timestamp_differences)
    assert results == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert dummy_source.start_called is True


@mock.patch.object(utils.VideoSource, "init")
def test_get_video_frames_generator_when_fps_modulation_enabled_against_fast_stream(
    init_mock: MagicMock,
) -> None:
    # given
    source_properties = SourceProperties(
        width=100, height=100, total_frames=-1, is_file=False, fps=25
    )
    dummy_source = DummyVideoSource(
        items=10, delay=0.01, source_properties=source_properties
    )
    init_mock.return_value = dummy_source

    # when
    results, results_timestamp = [], []
    for result in get_video_frames_generator(
        stream="source-ref",
        max_fps=50,
    ):
        results_timestamp.append(time.monotonic())
        results.append(result)

    # then
    timestamp_differences = get_differences(results_timestamp)
    assert all(diff >= 0.02 for diff in timestamp_differences)
    assert 0 <= len(results) <= 5
    assert dummy_source.start_called is True


@mock.patch.object(utils.VideoSource, "init")
def test_get_video_frames_generator_when_fps_modulation_enabled_against_slow_stream(
    init_mock: MagicMock,
) -> None:
    # given
    dummy_source = DummyVideoSource(items=10, delay=0.02)
    init_mock.return_value = dummy_source

    # when
    results, results_timestamp = [], []
    for result in get_video_frames_generator(
        stream="source-ref",
        max_fps=200,
    ):
        results_timestamp.append(time.monotonic())
        results.append(result)

    # then
    timestamp_differences = get_differences(results_timestamp)
    assert all(diff >= 0.02 for diff in timestamp_differences)
    assert results == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert dummy_source.start_called is True


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_get_video_frames_generator_against_real_video_without_rate_limit(
    local_video_path: str,
) -> None:
    # when
    results = list(get_video_frames_generator(stream=local_video_path))

    # then
    assert len(results) == 431


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_get_video_frames_generator_against_real_video_with_rate_limit_and_await_strategy(
    local_video_path: str,
) -> None:
    # when
    results, results_timestamp = [], []
    for result in get_video_frames_generator(
        stream=local_video_path,
        max_fps=200,
    ):
        results_timestamp.append(time.monotonic())
        results.append(result)

    # then
    timestamp_differences = get_differences(results_timestamp)
    assert all(diff >= 0.005 for diff in timestamp_differences)
    assert len(results) == 431


@pytest.mark.timeout(90)
@pytest.mark.slow
def test_get_video_frames_generator_against_real_video_with_rate_limit_and_drop_strategy(
    local_video_path: str,
) -> None:
    # when
    results, results_timestamp = [], []
    for result in get_video_frames_generator(
        stream=local_video_path, max_fps=200, limiter_strategy=FPSLimiterStrategy.DROP
    ):
        results_timestamp.append(time.monotonic())
        results.append(result)

    # then
    timestamp_differences = get_differences(results_timestamp)
    assert all(diff >= 0.005 for diff in timestamp_differences)
    assert 0 <= len(results) <= 431


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

    def __iter__(self) -> "DummyVideoSource":
        return self

    def __next__(self) -> int:
        return next(self._stream)


def generate_with_delay(items: int, delay: float) -> Generator[int, None, None]:
    for i in range(items):
        yield i
        time.sleep(delay)


def get_differences(values: List[float]) -> List[float]:
    result = []
    for x, y in zip(values, values[1:]):
        result.append(y - x)
    return result


def test_get_differences_when_empty_input_given() -> None:
    # when
    result = get_differences(values=[])

    # then
    assert len(result) == 0


def test_get_differences_when_single_element_input_given() -> None:
    # when
    result = get_differences(values=[1.0])

    # then
    assert len(result) == 0


def test_get_differences_when_multi_elements_input_given() -> None:
    # when
    result = get_differences(values=[1.0, 1.5, 2.5, 4])

    # then
    assert len(result) == 3
    assert np.allclose(result, [0.5, 1.0, 1.5])
