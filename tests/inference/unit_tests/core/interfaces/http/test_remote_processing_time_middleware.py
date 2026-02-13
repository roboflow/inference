import json
from unittest.mock import patch

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from inference.core.interfaces.http.http_api import (
    REMOTE_PROCESSING_TIME_HEADER,
    REMOTE_PROCESSING_TIMES_HEADER,
    GCPServerlessMiddleware,
)
from inference_sdk.config import (
    PROCESSING_TIME_HEADER,
    RemoteProcessingTimeCollector,
    remote_processing_times,
)


def _endpoint_that_adds_remote_times(request):
    """Simulates a workflow that records remote processing times."""
    collector = remote_processing_times.get()
    if collector is not None:
        collector.add(0.5, model_id="yolov8")
        collector.add(0.3, model_id="clip")
    return PlainTextResponse("OK")


def _endpoint_no_remote_times(request):
    """Simulates a request with no remote calls."""
    return PlainTextResponse("OK")


def _create_app(routes):
    app = Starlette(routes=routes)
    app.add_middleware(GCPServerlessMiddleware)
    return app


@patch(
    "inference.core.interfaces.http.http_api.WORKFLOWS_REMOTE_EXECUTION_TIME_FORWARDING",
    True,
)
class TestGCPServerlessMiddlewareRemoteProcessingTimes:
    def test_adds_remote_processing_time_headers_when_times_collected(self) -> None:
        # given
        app = _create_app(
            [Route("/workflow", endpoint=_endpoint_that_adds_remote_times)]
        )
        client = TestClient(app)

        # when
        response = client.get("/workflow")

        # then
        assert response.status_code == 200
        assert REMOTE_PROCESSING_TIME_HEADER in response.headers
        assert REMOTE_PROCESSING_TIMES_HEADER in response.headers
        total = float(response.headers[REMOTE_PROCESSING_TIME_HEADER])
        assert abs(total - 0.8) < 1e-9
        times = json.loads(response.headers[REMOTE_PROCESSING_TIMES_HEADER])
        assert len(times) == 2
        assert times[0] == {"model_id": "yolov8", "time": 0.5}
        assert times[1] == {"model_id": "clip", "time": 0.3}

    def test_no_remote_headers_when_no_times_collected(self) -> None:
        # given
        app = _create_app(
            [Route("/workflow", endpoint=_endpoint_no_remote_times)]
        )
        client = TestClient(app)

        # when
        response = client.get("/workflow")

        # then
        assert response.status_code == 200
        assert REMOTE_PROCESSING_TIME_HEADER not in response.headers
        assert REMOTE_PROCESSING_TIMES_HEADER not in response.headers

    def test_processing_time_header_always_present(self) -> None:
        # given
        app = _create_app(
            [Route("/workflow", endpoint=_endpoint_no_remote_times)]
        )
        client = TestClient(app)

        # when
        response = client.get("/workflow")

        # then
        assert PROCESSING_TIME_HEADER in response.headers
        assert float(response.headers[PROCESSING_TIME_HEADER]) >= 0


@patch(
    "inference.core.interfaces.http.http_api.WORKFLOWS_REMOTE_EXECUTION_TIME_FORWARDING",
    False,
)
class TestGCPServerlessMiddlewareWithForwardingDisabled:
    def test_no_remote_headers_when_forwarding_disabled(self) -> None:
        # given
        app = _create_app(
            [Route("/workflow", endpoint=_endpoint_that_adds_remote_times)]
        )
        client = TestClient(app)

        # when
        response = client.get("/workflow")

        # then
        assert response.status_code == 200
        assert REMOTE_PROCESSING_TIME_HEADER not in response.headers
        assert REMOTE_PROCESSING_TIMES_HEADER not in response.headers
        # Wall-clock time still present
        assert PROCESSING_TIME_HEADER in response.headers
