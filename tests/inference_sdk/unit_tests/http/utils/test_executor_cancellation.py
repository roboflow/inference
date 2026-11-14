import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_sdk.http.utils.executors import RequestMethod, make_request_async
from inference_sdk.http.utils.request_building import RequestData


@pytest.mark.asyncio
async def test_cancelling_response_decode_propagates_cancellation() -> None:
    decoding_started = asyncio.Event()

    async def decode_response() -> dict:
        decoding_started.set()
        await asyncio.Event().wait()
        return {}

    response = MagicMock(status=200, headers={})
    response.json = AsyncMock(side_effect=decode_response)
    response.read = AsyncMock(return_value=b"response body")
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=response)
    context.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.get.return_value = context
    request_data = RequestData(
        url="https://example.com/infer",
        request_elements=1,
        headers=None,
        data=None,
        parameters=None,
        payload=None,
        image_scaling_factors=[None],
    )
    task = asyncio.create_task(
        make_request_async(
            request_data=request_data,
            request_method=RequestMethod.GET,
            session=session,
        )
    )
    await asyncio.wait_for(decoding_started.wait(), timeout=1)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    response.read.assert_not_awaited()
    context.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_non_json_response_still_returns_bytes() -> None:
    response = MagicMock(status=200, headers={})
    response.json = AsyncMock(side_effect=ValueError("Not JSON"))
    response.read = AsyncMock(return_value=b"image bytes")
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=response)
    context.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.get.return_value = context
    request_data = RequestData(
        url="https://example.com/infer",
        request_elements=1,
        headers=None,
        data=None,
        parameters=None,
        payload=None,
        image_scaling_factors=[None],
    )

    result = await make_request_async(
        request_data=request_data,
        request_method=RequestMethod.GET,
        session=session,
    )

    assert result == (200, b"image bytes")
