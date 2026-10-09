"""Per-request execution context of the GCP serverless deployment."""

import hmac
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from inference_sdk.config import (
    INTERNAL_REMOTE_EXEC_REQ_HEADER,
    INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER,
    RemoteProcessingTimeCollector,
    apply_duration_minimum,
    execution_id,
    remote_processing_times,
)
from inference_server import configuration
from inference_server.middlewares.headers import (
    PROCESSING_TIME_HEADER,
    REMOTE_PROCESSING_TIME_HEADER,
    REMOTE_PROCESSING_TIMES_HEADER,
)

REMOTE_COLLECTOR_STATE_KEY = "remote_processing_time_collector"


def _request_header(headers: List[Tuple[bytes, bytes]], name: str) -> Optional[str]:
    wanted = name.lower().encode("latin-1")
    for key, value in headers:
        if key.lower() == wanted:
            return value.decode("latin-1")

    return None


def _is_verified_internal(headers: List[Tuple[bytes, bytes]]) -> bool:
    secret = configuration.ROBOFLOW_INTERNAL_SERVICE_SECRET
    if not secret:
        return False

    supplied = _request_header(headers, INTERNAL_REMOTE_EXEC_REQ_HEADER)
    if supplied is None:
        return False

    verified = hmac.compare_digest(supplied.encode("utf-8"), secret.encode("utf-8"))

    return verified


def _with_headers(
    message: Dict[str, Any], new_headers: Dict[str, str]
) -> Dict[str, Any]:
    replaced = {name.lower().encode("latin-1") for name in new_headers}
    headers = [
        (key, value)
        for key, value in message.get("headers", [])
        if key.lower() not in replaced
    ]
    headers.extend(
        (name.encode("latin-1"), value.encode("latin-1"))
        for name, value in new_headers.items()
    )

    return {**message, "headers": headers}


class ServerlessContextMiddleware:
    """Raw ASGI middleware preparing the serverless context of each request.

    Publishes the execution id, whether the duration minimum applies and a
    remote processing time collector on the ``inference_sdk.config`` context
    variables for the route handler and the usage hook, and stamps the
    response with the execution id, the processing time, the remote
    processing times and the internal-call verification result. The
    variables are reset when the request ends. Applies to every route,
    ``/v2`` included, as the legacy server's middleware does.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = list(scope.get("headers", []))
        execution_id_value = (
            _request_header(headers, configuration.EXECUTION_ID_HEADER)
            or f"{time.time_ns()}_{uuid4().hex[:4]}"
        )
        is_verified_internal = _is_verified_internal(headers)
        collector = RemoteProcessingTimeCollector()
        scope.setdefault("state", {})[REMOTE_COLLECTOR_STATE_KEY] = collector
        started_at = time.time()

        async def _send(message) -> None:
            if message["type"] == "http.response.start":
                message = _with_headers(
                    message,
                    self._response_headers(
                        execution_id_value=execution_id_value,
                        is_verified_internal=is_verified_internal,
                        collector=collector,
                        processing_time=time.time() - started_at,
                    ),
                )
            await send(message)

        execution_token = execution_id.set(execution_id_value)
        duration_token = apply_duration_minimum.set(not is_verified_internal)
        collector_token = remote_processing_times.set(collector)
        try:
            await self.app(scope, receive, _send)
        finally:
            remote_processing_times.reset(collector_token)
            apply_duration_minimum.reset(duration_token)
            execution_id.reset(execution_token)

    @staticmethod
    def _response_headers(
        *,
        execution_id_value: str,
        is_verified_internal: bool,
        collector: RemoteProcessingTimeCollector,
        processing_time: float,
    ) -> Dict[str, str]:
        response_headers = {PROCESSING_TIME_HEADER: str(processing_time)}
        if (
            configuration.WORKFLOWS_REMOTE_EXECUTION_TIME_FORWARDING
            and collector.has_data()
        ):
            total, detail = collector.snapshot_summary()
            response_headers[REMOTE_PROCESSING_TIME_HEADER] = str(total)
            if detail is not None:
                response_headers[REMOTE_PROCESSING_TIMES_HEADER] = detail
        response_headers[configuration.EXECUTION_ID_HEADER] = execution_id_value
        response_headers[INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER] = str(
            is_verified_internal
        ).lower()

        return response_headers
