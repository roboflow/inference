import time
from typing import Generator
from unittest import mock
from unittest.mock import MagicMock

from inference.core.interfaces.camera.utils import RateLimiter, limit_frame_rate, FPSLimiterStrategy
from inference.core.interfaces.camera import utils


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
def test_rate_limiter_when_next_tick_should_be_executed_immediately(time_mock: MagicMock) -> None:
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
    result = list(limit_frame_rate(
        frames_generator=frames_generator,
        max_fps=100,
        strategy=FPSLimiterStrategy.DROP
    ))

    # then
    assert 0 < len(result) <= 5


def test_limit_frame_rate_when_frames_to_be_dropped_and_stream_is_to_slow() -> None:
    # given
    frames_generator = generate_with_delay(items=10, delay=0.02)

    # when
    result = list(limit_frame_rate(
        frames_generator=frames_generator,
        max_fps=100,
        strategy=FPSLimiterStrategy.DROP
    ))

    # then
    assert result == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_limit_frame_rate_when_frames_to_be_awaited_and_stream_is_to_fast() -> None:
    # given
    frames_generator = generate_with_delay(items=10, delay=0.005)

    # when
    result = list(limit_frame_rate(
        frames_generator=frames_generator,
        max_fps=100,
        strategy=FPSLimiterStrategy.WAIT
    ))

    # then
    assert result == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_limit_frame_rate_when_frames_to_be_awaited_and_stream_is_to_slow() -> None:
    # given
    frames_generator = generate_with_delay(items=10, delay=0.02)

    # when
    result = list(limit_frame_rate(
        frames_generator=frames_generator,
        max_fps=100,
        strategy=FPSLimiterStrategy.WAIT
    ))

    # then
    assert result == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def generate_with_delay(items: int, delay: float) -> Generator[int, None, None]:
    for i in range(items):
        yield i
        time.sleep(delay)
