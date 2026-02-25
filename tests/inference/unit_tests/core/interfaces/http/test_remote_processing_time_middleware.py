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
    apply_duration_minimum,
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
        assert times[0] == {"m": "yolov8", "t": 0.5}
        assert times[1] == {"m": "clip", "t": 0.3}

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


    def test_omits_detail_header_when_json_exceeds_size_limit(self) -> None:
        # given - endpoint that adds many entries to exceed 4KB
        def _endpoint_many_times(request):
            collector = remote_processing_times.get()
            if collector is not None:
                for i in range(200):
                    collector.add(0.123456789, model_id=f"model_with_long_name_{i:04d}")
            return PlainTextResponse("OK")

        app = _create_app([Route("/workflow", endpoint=_endpoint_many_times)])
        client = TestClient(app)

        # when
        response = client.get("/workflow")

        # then - total header always present, detail header omitted if too large
        assert response.status_code == 200
        assert REMOTE_PROCESSING_TIME_HEADER in response.headers
        assert REMOTE_PROCESSING_TIMES_HEADER not in response.headers


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


def _endpoint_read_duration_minimum(request):
    """Returns the current value of apply_duration_minimum."""
    value = apply_duration_minimum.get()
    return PlainTextResponse(str(value))


@patch(
    "inference.core.interfaces.http.http_api.WORKFLOWS_REMOTE_EXECUTION_TIME_FORWARDING",
    True,
)
class TestApplyDurationMinimumContextVar:
    def test_direct_request_sets_apply_duration_minimum_true(self) -> None:
        """A request without the internal execution header should set
        apply_duration_minimum=True (floor should apply)."""
        app = _create_app(
            [Route("/infer", endpoint=_endpoint_read_duration_minimum)]
        )
        client = TestClient(app)
        response = client.get("/infer")
        assert response.status_code == 200
        assert response.text == "True"

    @patch(
        "inference.core.interfaces.http.http_api.ROBOFLOW_INTERNAL_SERVICE_SECRET",
        "test-secret-123",
    )
    def test_verified_internal_request_sets_apply_duration_minimum_false(self) -> None:
        """A request with a valid internal execution header should set
        apply_duration_minimum=False (floor should NOT apply)."""
        from inference_sdk.config import INTERNAL_REMOTE_EXEC_REQ_HEADER

        app = _create_app(
            [Route("/infer", endpoint=_endpoint_read_duration_minimum)]
        )
        client = TestClient(app)
        response = client.get(
            "/infer",
            headers={INTERNAL_REMOTE_EXEC_REQ_HEADER: "test-secret-123"},
        )
        assert response.status_code == 200
        assert response.text == "False"

    @patch(
        "inference.core.interfaces.http.http_api.ROBOFLOW_INTERNAL_SERVICE_SECRET",
        "test-secret-123",
    )
    def test_wrong_secret_sets_apply_duration_minimum_true(self) -> None:
        """A request with an invalid secret should still set
        apply_duration_minimum=True (floor should apply)."""
        from inference_sdk.config import INTERNAL_REMOTE_EXEC_REQ_HEADER

        app = _create_app(
            [Route("/infer", endpoint=_endpoint_read_duration_minimum)]
        )
        client = TestClient(app)
        response = client.get(
            "/infer",
            headers={INTERNAL_REMOTE_EXEC_REQ_HEADER: "wrong-secret"},
        )
        assert response.status_code == 200
        assert response.text == "True"

    @patch(
        "inference.core.interfaces.http.http_api.ROBOFLOW_INTERNAL_SERVICE_SECRET",
        None,
    )
    def test_no_secret_configured_sets_apply_duration_minimum_true(self) -> None:
        """When ROBOFLOW_INTERNAL_SERVICE_SECRET is not configured,
        apply_duration_minimum should be True regardless of header."""
        from inference_sdk.config import INTERNAL_REMOTE_EXEC_REQ_HEADER

        app = _create_app(
            [Route("/infer", endpoint=_endpoint_read_duration_minimum)]
        )
        client = TestClient(app)
        response = client.get(
            "/infer",
            headers={INTERNAL_REMOTE_EXEC_REQ_HEADER: "some-value"},
        )
        assert response.status_code == 200
        assert response.text == "True"
