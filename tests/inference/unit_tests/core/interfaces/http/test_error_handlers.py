import pytest

from inference.core.exceptions import RoboflowAPIUsagePausedError
from inference.core.interfaces.http.error_handlers import (
    with_route_exceptions,
    with_route_exceptions_async,
)


def test_with_route_exceptions_when_usage_paused_error_raised():
    @with_route_exceptions
    def my_route():
        raise RoboflowAPIUsagePausedError("usage paused")

    # when
    resp = my_route()

    # then
    assert resp.status_code == 423
    assert "paused" in resp.body.decode().lower()


@pytest.mark.asyncio
async def test_with_route_exceptions_async_when_usage_paused_error_raised():
    @with_route_exceptions_async
    async def my_route():
        raise RoboflowAPIUsagePausedError("usage paused")

    # when
    resp = await my_route()

    # then
    assert resp.status_code == 423
    assert "paused" in resp.body.decode().lower()
