from unittest import mock
from unittest.mock import MagicMock, call

import aiohttp
import pytest
from aiohttp import ClientConnectionError, ClientResponseError
from aioresponses import aioresponses
from requests import HTTPError, Response
from requests_mock import Mocker

from inference_sdk.http.utils import executors
from inference_sdk.http.utils.executors import (
    RequestMethod,
    execute_requests_packages,
    execute_requests_packages_async,
    make_parallel_requests,
    make_parallel_requests_async,
    make_request,
    make_request_async,
)
from inference_sdk.http.utils.request_building import RequestData


@pytest.mark.slow
@mock.patch.object(executors, "requests")
def test_make_request_when_connection_error_occurs_and_does_not_recover(
    requests_mock: MagicMock,
) -> None:
    # given
    requests_mock.get.side_effect = [
        ConnectionError(),
        ConnectionError(),
        ConnectionError("Third"),
    ]
    request_data = RequestData(
        url="https://some.com",
        request_elements=1,
        headers=None,
        data="some",
        parameters=None,
        payload=None,
        image_scaling_factors=[None],
    )

    # when
    with pytest.raises(ConnectionError) as error:
        make_request(request_data=request_data, request_method=RequestMethod.GET)

    # then
    assert (
        str(error.value) == "Third"
    ), "Last error should be re-raised, as retry count is exceeded"


@pytest.mark.slow
@mock.patch.object(executors, "requests")
def test_make_request_when_connection_error_occurs_and_recovers(
    requests_mock: MagicMock,
) -> None:
    # given
    expected_response = Response()
    requests_mock.get.side_effect = [ConnectionError(), expected_response]
    request_data = RequestData(
        url="https://some.com",
        request_elements=1,
        headers=None,
        data="some",
        parameters=None,
        payload=None,
        image_scaling_factors=[None],
    )

    # when
    result = make_request(request_data=request_data, request_method=RequestMethod.GET)

    # then
    assert (
        result is expected_response
    ), "Should return response received after recovery from connection error without exception"


@pytest.mark.slow
def test_make_request_when_retryable_error_occurs_and_does_not_recover(
    requests_mock: Mocker,
) -> None:
    # given
    request_data = RequestData(
        url="https://some.com",
        request_elements=1,
        headers=None,
        data=None,
        parameters=None,
        payload={"some": "value"},
        image_scaling_factors=[None],
    )
    requests_mock.post(
        url="https://some.com",
        response_list=[
            {"status_code": 503},
            {"status_code": 503},
            {"status_code": 503, "json": {"message": "third"}},
        ],
    )

    # when
    result = make_request(request_data=request_data, request_method=RequestMethod.POST)

    # then
    assert result.status_code == 503, "Expected to return last error code"
    assert result.json() == {
        "message": "third"
    }, "Expected to return last error payload"
    assert requests_mock.last_request.json() == {
        "some": "value"
    }, "Request payload expected to be injected"


@pytest.mark.slow
def test_make_request_when_retryable_error_occurs_and_recovers(
    requests_mock: Mocker,
) -> None:
    # given
    request_data = RequestData(
        url="https://some.com",
        request_elements=1,
        headers=None,
        data=None,
        parameters=None,
        payload={"some": "value"},
        image_scaling_factors=[None],
    )
    requests_mock.post(
        url="https://some.com",
        response_list=[
            {"status_code": 503},
            {"json": {"message": "ok"}},
        ],
    )

    # when
    result = make_request(request_data=request_data, request_method=RequestMethod.POST)

    # then
    assert result.status_code == 200, "Expected to return last status code"
    assert result.json() == {
        "message": "ok"
    }, "Expected to return successful request payload"
    assert requests_mock.last_request.json() == {
        "some": "value"
    }, "Request payload expected to be injected"


def test_make_request_when_error_response_should_not_be_retried(
    requests_mock: Mocker,
) -> None:
    # given
    request_data = RequestData(
        url="https://some.com",
        request_elements=1,
        headers=None,
        data=None,
        parameters=None,
        payload={"some": "value"},
        image_scaling_factors=[None],
    )
    requests_mock.post(
        url="https://some.com",
        response_list=[
            {"status_code": 500},
            {"json": {"message": "ok"}},
        ],
    )

    # when
    result = make_request(request_data=request_data, request_method=RequestMethod.POST)

    # then
    assert (
        result.status_code == 500
    ), "Expected to return first status code, as error is not retryable"
    assert len(requests_mock.request_history) == 1, "Only single request should be made"
    assert requests_mock.last_request.json() == {
        "some": "value"
    }, "Request payload expected to be injected"


def test_make_request_when_successful_response_is_expected(
    requests_mock: Mocker,
) -> None:
    # given
    request_data = RequestData(
        url="https://some.com",
        request_elements=1,
        headers={"some": "header"},
        data=None,
        parameters={"a": ["1", "2"], "b": "3"},
        payload={"some": "value"},
        image_scaling_factors=[None],
    )
    requests_mock.post(
        url="https://some.com",
        response_list=[
            {"json": {"message": "ok"}},
        ],
    )

    # when
    result = make_request(request_data=request_data, request_method=RequestMethod.POST)

    # then
    assert result.status_code == 200, "Expected to return first status code"
    assert len(requests_mock.request_history) == 1, "Only single request should be made"
    assert requests_mock.last_request.json() == {
        "some": "value"
    }, "Request payload expected to be injected"
    assert (
        requests_mock.last_request.query == "a=1&a=2&b=3"
    ), "Parameters must be posted according to request specification"


@mock.patch.object(executors, "make_request")
def test_make_parallel_requests(
    make_request_mock: MagicMock,
) -> None:
    # given
    request_data = RequestData(
        url="https://some.com",
        request_elements=1,
        headers={"some": "header"},
        data=None,
        parameters={"a": ["1", "2"], "b": "3"},
        payload={"some": "value"},
        image_scaling_factors=[None],
    )

    # when
    result = make_parallel_requests(
        requests_data=[request_data] * 4,
        request_method=RequestMethod.GET,
    )

    # then
    assert len(result) == 4, "Number of output responses must match number of requests"
    make_request_mock.assert_has_calls(
        [call(request_data, request_method=RequestMethod.GET)] * 4, any_order=True
    ), "Mock of request method must be invoked 4 times with proper parameters"


def test_execute_requests_packages_when_api_call_error_occurs(
    requests_mock: Mocker,
) -> None:
    # given
    request_data = RequestData(
        url="https://some.com",
        request_elements=1,
        headers={"some": "header"},
        data=None,
        parameters={"a": ["1", "2"], "b": "3"},
        payload={"some": "value"},
        image_scaling_factors=[None],
    )
    requests_mock.post(
        url="https://some.com",
        response_list=[
            {"json": {"message": "ok"}},
            {"json": {"message": "ok"}},
            {"json": {"message": "ok"}},
            {"status_code": 500},
        ],
    )

    # when
    with pytest.raises(HTTPError):
        _ = execute_requests_packages(
            requests_data=[request_data] * 4,
            request_method=RequestMethod.POST,
            max_concurrent_requests=2,
        )


def test_execute_requests_packages_when_api_calls_are_successful(
    requests_mock: Mocker,
) -> None:
    # given
    request_data = RequestData(
        url="https://some.com",
        request_elements=1,
        headers={"some": "header"},
        data=None,
        parameters={"a": ["1", "2"], "b": "3"},
        payload={"some": "value"},
        image_scaling_factors=[None],
    )
    requests_mock.post(
        url="https://some.com",
        response_list=[
            {"json": {"message": "ok"}},
            {"json": {"message": "ok"}},
            {"json": {"message": "ok"}},
        ],
    )

    # when
    result = execute_requests_packages(
        requests_data=[request_data] * 3,
        request_method=RequestMethod.POST,
        max_concurrent_requests=2,
    )

    # then
    assert len(result) == 3, "3 requests made - 3 responses are expected"
    assert all(
        r.json() == {"message": "ok"} for r in result
    ), "All responses should be returned with the same, predefined JSON response"


@pytest.mark.asyncio
@pytest.mark.slow
async def test_make_request_async_when_connection_error_occurs_and_does_not_recover() -> (
    None
):
    # given
    request_data = RequestData(
        url="https://some.com",
        request_elements=1,
        headers=None,
        data="some",
        parameters=None,
        payload=None,
        image_scaling_factors=[None],
    )

    with aioresponses() as m:
        async with aiohttp.ClientSession() as session:
            m.get("https://some.com", exception=ClientConnectionError)
            m.get("https://some.com", exception=ClientConnectionError)
            m.get("https://some.com", exception=ClientConnectionError)

            # when
            with pytest.raises(ClientConnectionError):
                _ = await make_request_async(
                    request_data=request_data,
                    request_method=RequestMethod.GET,
                    session=session,
                )


@pytest.mark.asyncio
@pytest.mark.slow
async def test_make_request_async_when_connection_error_occurs_and_does_recover() -> (
    None
):
    # given
    request_data = RequestData(
        url="https://some.com",
        request_elements=1,
        headers=None,
        data="some",
        parameters=None,
        payload=None,
        image_scaling_factors=[None],
    )

    with aioresponses() as m:
        async with aiohttp.ClientSession() as session:
            m.get("https://some.com", exception=ClientConnectionError)
            m.get("https://some.com", payload={"some": "ok"})

            # when
            response = await make_request_async(
                request_data=request_data,
                request_method=RequestMethod.GET,
                session=session,
            )

    # then
    assert response == (
        200,
        {"some": "ok"},
    ), "Expected to return HTTP 200 in second attempt with predefined JSON payload"


@pytest.mark.asyncio
@pytest.mark.slow
async def test_make_request_async_when_retryable_error_occurs_and_does_not_recover() -> (
    None
):
    # given
    request_data = RequestData(
        url="https://some.com",
        request_elements=1,
        headers=None,
        data="some",
        parameters=None,
        payload=None,
        image_scaling_factors=[None],
    )

    with aioresponses() as m:
        async with aiohttp.ClientSession() as session:
            m.get("https://some.com", status=503)
            m.get("https://some.com", status=503)
            m.get("https://some.com", status=503)

            # when
            with pytest.raises(ClientResponseError):
                _ = await make_request_async(
                    request_data=request_data,
                    request_method=RequestMethod.GET,
                    session=session,
                )


@pytest.mark.asyncio
@pytest.mark.slow
async def test_make_request_async_when_retryable_error_occurs_and_does_recover() -> (
    None
):
    # given
    request_data = RequestData(
        url="https://some.com",
        request_elements=1,
        headers=None,
        data="some",
        parameters=None,
        payload=None,
        image_scaling_factors=[None],
    )

    with aioresponses() as m:
        async with aiohttp.ClientSession() as session:
            m.get("https://some.com", status=503)
            m.get("https://some.com", status=200, payload={"status": "ok"})

            # when
            result = await make_request_async(
                request_data=request_data,
                request_method=RequestMethod.GET,
                session=session,
            )

    # then
    assert result == (
        200,
        {"status": "ok"},
    ), "Expected to return HTTP 200 in second attempt with predefined JSON payload"


@pytest.mark.asyncio
async def test_make_request_async_when_non_retryable_error_occurs() -> None:
    # given
    request_data = RequestData(
        url="https://some.com",
        request_elements=1,
        headers=None,
        data="some",
        parameters=None,
        payload=None,
        image_scaling_factors=[None],
    )

    with aioresponses() as m:
        async with aiohttp.ClientSession() as session:
            m.get("https://some.com", status=500)

            # when
            with pytest.raises(ClientResponseError):
                _ = await make_request_async(
                    request_data=request_data,
                    request_method=RequestMethod.GET,
                    session=session,
                )


@pytest.mark.asyncio
async def test_make_request_async_when_request_is_successful() -> None:
    # given
    request_data = RequestData(
        url="https://some.com/",
        request_elements=1,
        headers={"my": "header"},
        data=None,
        parameters=None,
        payload={"some": "data"},
        image_scaling_factors=[None],
    )

    with aioresponses() as m:
        async with aiohttp.ClientSession() as session:
            m.get("https://some.com", status=200, payload={"status": "ok"})

            # when
            result = await make_request_async(
                request_data=request_data,
                request_method=RequestMethod.GET,
                session=session,
            )

        # then
        m.assert_called_with(
            url="https://some.com",
            method="GET",
            headers={"my": "header"},
            json={"some": "data"},
            data=None,
            params=None,
        )
        assert result == (
            200,
            {"status": "ok"},
        ), "Expected to return HTTP 200 in second attempt with predefined JSON payload"


@pytest.mark.asyncio
async def test_make_parallel_requests_async_when_some_request_fails() -> None:
    # given
    request_data = RequestData(
        url="https://some.com",
        request_elements=1,
        headers=None,
        data="some",
        parameters=None,
        payload=None,
        image_scaling_factors=[None],
    )
    with aioresponses() as m:
        m.get("https://some.com", status=200, payload={"status": "ok"})
        m.get("https://some.com", status=200, payload={"status": "ok"})
        m.get("https://some.com", status=500)
        m.get("https://some.com", status=200, payload={"status": "ok"})

        # when
        with pytest.raises(ClientResponseError):
            _ = await make_parallel_requests_async(
                requests_data=[request_data] * 4,
                request_method=RequestMethod.GET,
            )


@pytest.mark.asyncio
async def test_make_parallel_requests_async_when_all_requests_succeed() -> None:
    # given
    request_data = RequestData(
        url="https://some.com",
        request_elements=1,
        headers=None,
        data="some",
        parameters=None,
        payload=None,
        image_scaling_factors=[None],
    )
    with aioresponses() as m:
        m.get("https://some.com", status=200, payload={"status": "ok"})
        m.get("https://some.com", status=200, payload={"status": "ok"})
        m.get("https://some.com", status=200, payload={"status": "ok"})

        # when
        result = await make_parallel_requests_async(
            requests_data=[request_data] * 3,
            request_method=RequestMethod.GET,
        )

    # then
    assert (
        result == [{"status": "ok"}] * 3
    ), "All requests are expected to return predefined result"


@pytest.mark.asyncio
async def test_execute_requests_packages_async_when_some_request_fails() -> None:
    # given
    request_data = RequestData(
        url="https://some.com",
        request_elements=1,
        headers=None,
        data="some",
        parameters=None,
        payload=None,
        image_scaling_factors=[None],
    )
    with aioresponses() as m:
        m.get("https://some.com", status=200, payload={"status": "ok"})
        m.get("https://some.com", status=200, payload={"status": "ok"})
        m.get("https://some.com", status=500)
        m.get("https://some.com", status=200, payload={"status": "ok"})

        # when
        with pytest.raises(ClientResponseError):
            _ = await execute_requests_packages_async(
                requests_data=[request_data] * 4,
                request_method=RequestMethod.GET,
                max_concurrent_requests=2,
            )


@pytest.mark.asyncio
async def test_execute_requests_packages_async_when_all_requests_succeed() -> None:
    # given
    request_data = RequestData(
        url="https://some.com",
        request_elements=1,
        headers=None,
        data="some",
        parameters=None,
        payload=None,
        image_scaling_factors=[None],
    )
    with aioresponses() as m:
        m.get("https://some.com", status=200, payload={"status": "ok"})
        m.get("https://some.com", status=200, payload={"status": "ok"})
        m.get("https://some.com", status=200, payload={"status": "ok"})

        # when
        result = await execute_requests_packages_async(
            requests_data=[request_data] * 3,
            request_method=RequestMethod.GET,
            max_concurrent_requests=2,
        )

    # then
    assert (
        result == [{"status": "ok"}] * 3
    ), "All requests are expected to return predefined result"
