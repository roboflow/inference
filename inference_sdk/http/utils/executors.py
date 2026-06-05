import asyncio
import contextvars
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import partial
from typing import List, Mapping, Optional, Tuple, Union

import aiohttp
import backoff
import requests
from aiohttp import (
    ClientConnectionError,
    ClientResponse,
    ClientResponseError,
    RequestInfo,
)
from requests import Response, Timeout

from inference_sdk.config import (
    EXECUTION_ID_HEADER,
    PROCESSING_TIME_HEADER,
    execution_id,
    remote_processing_times,
)
from inference_sdk.http.errors import RetryError
from inference_sdk.http.utils.iterables import make_batches
from inference_sdk.http.utils.request_building import RequestData
from inference_sdk.http.utils.requests import api_key_safe_raise_for_status

RETRYABLE_STATUS_CODES = {429, 503, 504}
UNKNOWN_MODEL_ID = "unknown"
MODEL_COLD_START_HEADER = "X-Model-Cold-Start"
MODEL_COLD_START_COUNT_HEADER = "X-Model-Cold-Start-Count"
MODEL_LOAD_TIME_HEADER = "X-Model-Load-Time"
MODEL_LOAD_DETAILS_HEADER = "X-Model-Load-Details"
MODEL_ID_HEADER = "X-Model-Id"


class RequestMethod(Enum):
    """Enum for the request method.

    Attributes:
        GET: The GET method.
        POST: The POST method.
    """

    GET = "get"
    POST = "post"


def execute_requests_packages(
    requests_data: List[RequestData],
    request_method: RequestMethod,
    max_concurrent_requests: int,
) -> List[Response]:
    """Execute a list of requests in parallel.

    Args:
        requests_data: The list of requests to execute.
        request_method: The method to use for the requests.
        max_concurrent_requests: The maximum number of concurrent requests.

    Returns:
        The list of responses.
    """
    requests_data_packages = make_batches(
        iterable=requests_data,
        batch_size=max_concurrent_requests,
    )
    results = []
    all_request_data = []
    for requests_data_package in requests_data_packages:
        responses = make_parallel_requests(
            requests_data=requests_data_package,
            request_method=request_method,
        )
        results.extend(responses)
        all_request_data.extend(requests_data_package)
    _collect_remote_processing_times(results, all_request_data)
    for response in results:
        api_key_safe_raise_for_status(response=response)
    return results


def _extract_model_id_from_request_data(request_data: RequestData) -> str:
    if request_data.payload and isinstance(request_data.payload, dict):
        model_id = request_data.payload.get("model_id")
        if model_id:
            return str(model_id)
    try:
        from urllib.parse import urlparse

        path = urlparse(request_data.url).path
        return path.strip("/")
    except Exception:
        return UNKNOWN_MODEL_ID


def _extract_model_ids_from_headers(
    headers: Mapping[str, str],
    request_data: Optional[RequestData] = None,
    fallback_model_id: Optional[str] = None,
) -> List[str]:
    model_ids_header = headers.get(MODEL_ID_HEADER)
    if model_ids_header:
        return [
            model_id.strip()
            for model_id in model_ids_header.split(",")
            if model_id.strip()
        ]
    if (
        request_data is not None
        and request_data.payload
        and isinstance(request_data.payload, dict)
    ):
        model_id = request_data.payload.get("model_id")
        if model_id:
            return [str(model_id)]
    if fallback_model_id not in (None, "", UNKNOWN_MODEL_ID):
        return [str(fallback_model_id)]
    return []


def _parse_model_load_details(
    details_header: str,
) -> Optional[List[Tuple[Optional[str], float]]]:
    try:
        parsed = json.loads(details_header)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    result = []
    for entry in parsed:
        if not isinstance(entry, dict) or "t" not in entry:
            return None
        try:
            load_time = float(entry["t"])
        except (TypeError, ValueError):
            return None
        model_id = entry.get("m")
        model_id = str(model_id) if model_id not in (None, "") else None
        result.append((model_id, load_time))
    return result


def _collect_remote_processing_times(
    responses: List[Response],
    requests_data: List[RequestData],
) -> None:
    if len(responses) != len(requests_data):
        logging.warning(
            "Response count (%d) does not match request count (%d); "
            "only pairing the first %d entries",
            len(responses),
            len(requests_data),
            min(len(responses), len(requests_data)),
        )
    for response, request_data in zip(responses, requests_data):
        collect_remote_processing_metadata_from_headers(
            headers=response.headers,
            request_data=request_data,
        )


def collect_remote_processing_metadata_from_response(
    response: Response,
    model_id: str = UNKNOWN_MODEL_ID,
) -> None:
    collect_remote_processing_metadata_from_headers(
        headers=response.headers,
        fallback_model_id=model_id,
    )


def collect_remote_processing_metadata_from_headers(
    headers: Mapping[str, str],
    request_data: Optional[RequestData] = None,
    fallback_model_id: str = UNKNOWN_MODEL_ID,
) -> None:
    collector = remote_processing_times.get()
    if collector is None:
        return
    pt = headers.get(PROCESSING_TIME_HEADER)
    if pt is not None:
        model_id = (
            _extract_model_id_from_request_data(request_data)
            if request_data is not None
            else fallback_model_id
        )
        try:
            collector.add(float(pt), model_id=model_id)
        except (ValueError, TypeError):
            logging.warning("Malformed %s header value: %r", PROCESSING_TIME_HEADER, pt)
    model_ids = _extract_model_ids_from_headers(
        headers=headers,
        request_data=request_data,
        fallback_model_id=fallback_model_id,
    )
    collector.add_model_ids(model_ids=model_ids)
    if headers.get(MODEL_COLD_START_HEADER, "").lower() != "true":
        return
    details_header = headers.get(MODEL_LOAD_DETAILS_HEADER)
    if details_header:
        parsed_details = _parse_model_load_details(details_header)
        if parsed_details is None:
            logging.warning(
                "Malformed %s header value: %r",
                MODEL_LOAD_DETAILS_HEADER,
                details_header,
            )
        else:
            for entry_model_id, load_time in parsed_details:
                collector.record_cold_start(
                    load_time=load_time,
                    model_id=entry_model_id,
                )
            return
    load_time = 0.0
    load_time_header = headers.get(MODEL_LOAD_TIME_HEADER)
    if load_time_header is not None:
        try:
            load_time = float(load_time_header)
        except (ValueError, TypeError):
            logging.warning(
                "Malformed %s header value: %r",
                MODEL_LOAD_TIME_HEADER,
                load_time_header,
            )
    synthesized_model_id = model_ids[0] if len(model_ids) == 1 else None
    collector.record_cold_start(
        load_time=load_time,
        count=_extract_cold_start_count_from_headers(headers=headers),
        model_id=synthesized_model_id,
    )


def _extract_cold_start_count_from_headers(headers: Mapping[str, str]) -> int:
    count_header = headers.get(MODEL_COLD_START_COUNT_HEADER)
    if count_header is None:
        return 1
    try:
        count = int(count_header)
    except (TypeError, ValueError):
        logging.warning(
            "Malformed %s header value: %r",
            MODEL_COLD_START_COUNT_HEADER,
            count_header,
        )
        return 1
    if count < 1:
        logging.warning(
            "Unexpected %s header value for cold start response: %r",
            MODEL_COLD_START_COUNT_HEADER,
            count_header,
        )
        return 1
    return count


def make_parallel_requests(
    requests_data: List[RequestData],
    request_method: RequestMethod,
) -> List[Response]:
    """Execute a list of requests in parallel.

    Args:
        requests_data: The list of requests to execute.
        request_method: The method to use for the requests.

    Returns:
        The list of responses.
    """
    # Capture the current contextvars snapshot so OTel trace context
    # propagates into the thread pool workers. Must capture here (caller
    # thread) not inside the worker — workers run in fresh threads that
    # don't inherit the caller's context.
    parent_ctx = contextvars.copy_context()
    workers = len(requests_data)
    make_request_closure = partial(make_request, request_method=request_method)

    def _run_in_parent_context(rd: RequestData) -> Response:
        # Restore all context vars from the parent thread snapshot.
        # We iterate and set each var individually since Context.run()
        # is not reentrant.
        tokens = []
        for var in parent_ctx:
            tokens.append((var, var.set(parent_ctx[var])))
        try:
            return make_request_closure(rd)
        finally:
            for var, tok in tokens:
                var.reset(tok)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_run_in_parent_context, requests_data))


@backoff.on_predicate(
    backoff.constant,
    predicate=lambda r: r.status_code in RETRYABLE_STATUS_CODES,
    max_tries=3,
    interval=1,
    backoff_log_level=logging.DEBUG,
    giveup_log_level=logging.DEBUG,
)
@backoff.on_exception(
    backoff.constant,
    exception=ConnectionError,
    max_tries=3,
    interval=1,
    backoff_log_level=logging.DEBUG,
    giveup_log_level=logging.DEBUG,
)
def make_request(request_data: RequestData, request_method: RequestMethod) -> Response:
    """Make a request to the API.

    Args:
        request_data: The request data.
        request_method: The method to use for the request.

    Returns:
        The response from the API.
    """
    method = requests.get if request_method is RequestMethod.GET else requests.post
    return method(
        request_data.url,
        headers=request_data.headers,
        params=request_data.parameters,
        data=request_data.data,
        json=request_data.payload,
    )


async def execute_requests_packages_async(
    requests_data: List[RequestData],
    request_method: RequestMethod,
    max_concurrent_requests: int,
) -> List[Union[dict, bytes]]:
    """Execute a list of requests in parallel asynchronously.

    Args:
        requests_data: The list of requests to execute.
        request_method: The method to use for the requests.
        max_concurrent_requests: The maximum number of concurrent requests.

    Returns:
        The list of responses.
    """
    requests_data_packages = make_batches(
        iterable=requests_data,
        batch_size=max_concurrent_requests,
    )
    results = []
    for requests_data_package in requests_data_packages:
        responses = await make_parallel_requests_async(
            requests_data=requests_data_package,
            request_method=request_method,
        )
        results.extend(responses)
    return results


async def make_parallel_requests_async(
    requests_data: List[RequestData],
    request_method: RequestMethod,
) -> List[Union[dict, bytes]]:
    """Execute a list of requests in parallel asynchronously.

    Args:
        requests_data: The list of requests to execute.
        request_method: The method to use for the requests.

    Returns:
        The list of responses.
    """
    async with aiohttp.ClientSession() as session:
        make_request_closure = partial(
            make_request_async,
            request_method=request_method,
            session=session,
        )
        coroutines = [make_request_closure(data) for data in requests_data]
        responses = list(await asyncio.gather(*coroutines))
        return [r[1] for r in responses]


def raise_client_error(details: dict) -> None:
    """Raise a client error.

    Args:
        details: The details of the error.
    """
    status_code = details["value"][0]
    request_data = details["kwargs"]["request_data"]
    raise ClientResponseError(
        request_info=RequestInfo(
            url=request_data.url,
            method="POST",
            headers={},
        ),
        history=(),
        status=status_code,
    )


@backoff.on_predicate(
    backoff.constant,
    predicate=lambda r: r[0] in RETRYABLE_STATUS_CODES,
    max_tries=3,
    interval=1,
    on_giveup=raise_client_error,
    backoff_log_level=logging.DEBUG,
    giveup_log_level=logging.DEBUG,
)
@backoff.on_exception(
    backoff.constant,
    exception=ClientConnectionError,
    max_tries=3,
    interval=1,
    backoff_log_level=logging.DEBUG,
    giveup_log_level=logging.DEBUG,
)
async def make_request_async(
    request_data: RequestData,
    request_method: RequestMethod,
    session: aiohttp.ClientSession,
) -> Tuple[int, Union[bytes, dict]]:
    """Make a request to the API asynchronously.

    Args:
        request_data: The request data.
        request_method: The method to use for the request.
        session: The session to use for the request.

    Returns:
        The response from the API.
    """
    method = session.get if request_method is RequestMethod.GET else session.post
    parameters_serialised = None
    if request_data.parameters is not None:
        parameters_serialised = {
            name: (
                str(value)
                if not issubclass(type(value), list)
                else [str(e) for e in value]
            )
            for name, value in request_data.parameters.items()
        }
    async with method(
        request_data.url,
        headers=request_data.headers,
        params=parameters_serialised,
        data=request_data.data,
        json=request_data.payload,
    ) as response:
        try:
            response_data = await response.json()
        except:
            response_data = await response.read()
        if response_is_not_retryable_error(response=response):
            response.raise_for_status()
        if response.status == 200:
            collect_remote_processing_metadata_from_headers(
                headers=response.headers,
                request_data=request_data,
            )
        return response.status, response_data


def response_is_not_retryable_error(response: ClientResponse) -> bool:
    """Check if the response is not a retryable error.

    Args:
        response: The response to check.

    Returns:
        True if the response is not a retryable error, False otherwise.
    """
    return response.status != 200 and response.status not in RETRYABLE_STATUS_CODES


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
    backoff_log_level=logging.DEBUG,
    giveup_log_level=logging.DEBUG,
)
def send_post_request(
    url: str,
    payload: dict,
    headers: dict,
    enable_retries: bool,
) -> Response:
    try:
        execution_id_value = execution_id.get()
        if execution_id_value:
            headers = headers.copy()
            headers[EXECUTION_ID_HEADER] = execution_id_value
        response = requests.post(url, json=payload, headers=headers)
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError) as error:
        if enable_retries:
            raise RetryError(
                "Could not connect to the API.", inner_error=error
            ) from error
        raise error
    if enable_retries and response.status_code in RETRYABLE_STATUS_CODES:
        raise RetryError(
            (
                "Transient error in HTTP request - response with status code: "
                f"{response.status_code} received."
            ),
            status_code=response.status_code,
        )
    api_key_safe_raise_for_status(response=response)
    return response
