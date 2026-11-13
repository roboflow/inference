"""Unit tests for bounded per-request fan-out."""

from __future__ import annotations

import asyncio
import gc
import warnings
from typing import Optional

import pytest

from inference_server.framework.fanout import gather_bounded


class _ConcurrencyProbe:
    def __init__(self) -> None:
        self.running = 0
        self.peak = 0

    async def call(self, value: int) -> int:
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0.01)
        self.running -= 1
        return value


@pytest.mark.asyncio
async def test_gather_bounded_preserves_order_and_results():
    probe = _ConcurrencyProbe()
    out = await gather_bounded(*(probe.call(i) for i in range(6)), limit=2)
    assert out == [0, 1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_gather_bounded_caps_concurrency():
    probe = _ConcurrencyProbe()
    await gather_bounded(*(probe.call(i) for i in range(10)), limit=3)
    assert probe.peak <= 3


@pytest.mark.asyncio
async def test_gather_bounded_runs_all_at_once_below_limit():
    probe = _ConcurrencyProbe()
    await gather_bounded(*(probe.call(i) for i in range(3)), limit=8)
    assert probe.peak == 3


@pytest.mark.asyncio
async def test_gather_bounded_limit_zero_is_unbounded():
    probe = _ConcurrencyProbe()
    await gather_bounded(*(probe.call(i) for i in range(6)), limit=0)
    assert probe.peak == 6


@pytest.mark.asyncio
async def test_gather_bounded_propagates_first_error():
    async def _boom():
        raise ValueError("boom")

    async def _ok():
        return 1

    with pytest.raises(ValueError, match="boom"):
        await gather_bounded(_boom(), _ok(), _ok(), limit=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit, max_started", [(2, 3), (0, 8)])
async def test_gather_bounded_stops_the_batch_after_the_first_error(limit, max_started):
    started: list[int] = []
    finished: list[int] = []

    async def _call(index: int) -> int:
        started.append(index)
        if index == 1:
            raise ValueError("boom")
        await asyncio.sleep(0.05)
        finished.append(index)
        return index

    with pytest.raises(ValueError, match="boom"):
        await gather_bounded(*(_call(i) for i in range(8)), limit=limit)
    await asyncio.sleep(0.2)

    assert finished == []
    assert len(started) <= max_started


class _BatchProbe:
    def __init__(self) -> None:
        self.started: list[int] = []
        self.cleaned: list[int] = []

    async def call(self, index: int, failing: Optional[int] = None) -> int:
        self.started.append(index)
        if index == failing:
            raise ValueError("boom")
        try:
            await asyncio.sleep(10)
        finally:
            await asyncio.sleep(0.01)
            self.cleaned.append(index)
        return index


def _never_awaited_warnings(caught: list) -> list:
    gc.collect()
    return [w for w in caught if "never awaited" in str(w.message)]


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [2, 0])
async def test_gather_bounded_failure_settles_and_closes_every_sibling(limit):
    probe = _BatchProbe()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="boom"):
            await gather_bounded(
                *(probe.call(i, failing=1) for i in range(8)), limit=limit
            )
        assert sorted(probe.cleaned) == [i for i in probe.started if i != 1]
        assert _never_awaited_warnings(caught) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [2, 0])
async def test_gather_bounded_parent_cancellation_settles_and_closes_every_sibling(
    limit,
):
    probe = _BatchProbe()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        task = asyncio.ensure_future(
            gather_bounded(*(probe.call(i) for i in range(8)), limit=limit)
        )
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sorted(probe.cleaned) == sorted(probe.started)
        assert probe.started
        assert _never_awaited_warnings(caught) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [2, 0])
async def test_gather_bounded_raises_the_first_failure_in_completion_order(limit):
    async def _fail(index: int) -> int:
        await asyncio.sleep(0)
        raise ValueError(str(index))

    raised = []
    for _ in range(200):
        with pytest.raises(ValueError) as error:
            await gather_bounded(_fail(0), _fail(1), _fail(2), limit=limit)
        raised.append(str(error.value))

    assert set(raised) == {"0"}


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [2, 0])
async def test_gather_bounded_survives_a_second_cancellation_during_cleanup(limit):
    started: list[int] = []
    cleaning: list[int] = []
    cleaned: list[int] = []

    async def _call(index: int) -> int:
        started.append(index)
        try:
            await asyncio.sleep(10)
        finally:
            cleaning.append(index)
            await asyncio.sleep(0.05)
            cleaned.append(index)
        return index

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        task = asyncio.ensure_future(
            gather_bounded(*(_call(i) for i in range(8)), limit=limit)
        )
        await asyncio.sleep(0.01)
        task.cancel()
        while len(cleaning) < len(started):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert started
        assert sorted(cleaned) == sorted(started)
        assert _never_awaited_warnings(caught) == []


async def _cancel_itself() -> int:
    await asyncio.sleep(0)
    raise asyncio.CancelledError()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [2, 0])
async def test_gather_bounded_child_cancellation_ends_the_batch_promptly(limit):
    probe = _BatchProbe()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(
                gather_bounded(
                    _cancel_itself(), *(probe.call(i) for i in range(1, 8)), limit=limit
                ),
                timeout=2,
            )
        assert sorted(probe.cleaned) == sorted(probe.started)
        assert _never_awaited_warnings(caught) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [2, 0])
async def test_gather_bounded_child_cancellation_wins_over_a_later_error(limit):
    async def _fail_later() -> int:
        await asyncio.sleep(0.01)
        raise ValueError("later")

    with pytest.raises(asyncio.CancelledError):
        await gather_bounded(
            _cancel_itself(), _fail_later(), _fail_later(), limit=limit
        )
