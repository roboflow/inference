"""Unit tests for bounded per-request fan-out."""

from __future__ import annotations

import asyncio

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
