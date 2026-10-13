"""Real host API with an invented loopback server; no cloud qualification."""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote_plus, urlsplit

import pytest
from roboflow_workflows.v2.blocks.roboflow_platform.ports import PlatformRequestError
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError

import inference.core.roboflow_api as roboflow_api
from inference.core.utils import url_utils
from inference.roboflow_workflows_plugin.v2 import platform_monitoring_adapter as module
from inference.roboflow_workflows_plugin.v2.credential_boundary import HostOperation
from inference.roboflow_workflows_plugin.v2.platform_monitoring_adapter import (
    RoboflowModelMonitoringAdapter,
)

KEY = "invented-MONITORING-key-A/b+c?= private"
ROWS = [
    {
        "model_id": "project/3",
        "class_name": "cat",
        "confidence": 0.9,
        "inference_id": "original-id",
        "model_type": "object-detection",
    }
]


@pytest.fixture
def loopback(monkeypatch):
    requests, status = [], [200]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._respond({"workspace": "invented-workspace"})

        def do_POST(self):
            self._respond({})

        def _respond(self, response):
            url = urlsplit(self.path)
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else None
            requests.append(
                {
                    "method": self.command,
                    "path": url.path,
                    "query": parse_qs(url.query),
                    "headers": dict(self.headers),
                    "json": json.loads(body) if body else None,
                }
            )
            payload = json.dumps(response).encode()
            self.send_response(status[0])
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        roboflow_api, "API_BASE_URL", f"http://127.0.0.1:{server.server_port}"
    )
    monkeypatch.setattr(roboflow_api, "OFFLINE_MODE", False)
    monkeypatch.setattr(url_utils, "SECURE_GATEWAY", None)
    monkeypatch.setattr(roboflow_api, "ROBOFLOW_API_EXTRA_HEADERS", None)
    roboflow_api.get_roboflow_workspace.cache_clear()
    try:
        yield requests, status
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        roboflow_api.get_roboflow_workspace.cache_clear()


def adapter():
    return RoboflowModelMonitoringAdapter(
        clock=lambda: datetime(2026, 1, 1, 12, 30),
        system_info=lambda: {"platform": "fixture", "hostname": "invented-host"},
    )


def send(host, *, key=None):
    host.send_report(
        api_key=key or Secret(KEY, name="TEST_ONLY"),
        workspace_id="invented-workspace",
        predictions=ROWS,
    )


def test_real_api_workspace_and_report_shape_headers_and_metadata(loopback, caplog):
    requests, _ = loopback
    host = adapter()
    with caplog.at_level(logging.DEBUG):
        assert (
            host.get_workspace(api_key=Secret(KEY, name="TEST_ONLY"))
            == "invented-workspace"
        )
        send(host)
    assert [r["path"] for r in requests] == ["/", "/invented-workspace/inference-stats"]
    assert requests[1]["method"] == "POST"
    assert requests[1]["query"] == {"api_key": [KEY]}
    data = requests[1]["json"]
    assert data == {
        "timestamp": "2026-01-01T12:30:00",
        "source": "workflow",
        "source_info": "ModelMonitoringInferenceAggregatorBlockV1",
        "device_id": module.DEVICE_ID,
        "inference_server_version": module.__version__,
        "platform": "fixture",
        "hostname": "invented-host",
        "inference_results": ROWS,
    }
    expected_headers = roboflow_api.build_roboflow_api_headers()
    for key, value in expected_headers.items():
        assert requests[1]["headers"][key] == value
    assert KEY not in caplog.text
    assert quote_plus(KEY) not in caplog.text
    assert "model monitoring report" in caplog.text


def test_http_failure_is_fixed_and_has_no_original_exception_chain(loopback, caplog):
    requests, status = loopback
    status[0] = 500
    with caplog.at_level(logging.DEBUG), pytest.raises(PlatformRequestError) as caught:
        send(adapter())
    assert len(requests) == 1
    assert "model monitoring report failed" in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert KEY not in str(caught.value) + caplog.text
    assert quote_plus(KEY) not in str(caught.value) + caplog.text


def test_boundary_runs_inside_actual_worker_call_and_remains_thread_local(
    monkeypatch, caplog
):
    thread_ids = []

    def transport(**arguments):
        thread_ids.append(threading.get_ident())
        logging.getLogger("urllib3.connectionpool").warning(
            "request %s", quote_plus(arguments["api_key"]), extra={"credential": KEY}
        )
        raise RuntimeError(KEY)

    monkeypatch.setattr(
        roboflow_api, "send_inference_results_to_model_monitoring", transport
    )
    with caplog.at_level(logging.DEBUG), ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(send, adapter())
        with pytest.raises(PlatformRequestError) as caught:
            future.result()
    assert thread_ids[0] != threading.get_ident()
    assert (
        str(caught.value) == "model monitoring report failed: unexpected host failure"
    )
    assert KEY not in caplog.text
    assert quote_plus(KEY) not in caplog.text
    assert all(not hasattr(record, "credential") for record in caplog.records)
    # The existing emitter must be unchanged outside the adapter context.
    logging.getLogger("urllib3.connectionpool").warning("outside-monitoring-context")
    assert "outside-monitoring-context" in caplog.text


def test_system_metadata_gathering_is_within_the_same_fixed_boundary(
    monkeypatch, caplog
):
    def system_info():
        logging.getLogger("inference").warning("metadata %s", KEY)
        raise ValueError(quote_plus(KEY))

    host = RoboflowModelMonitoringAdapter(system_info=system_info)
    with caplog.at_level(logging.DEBUG), pytest.raises(PlatformRequestError) as caught:
        send(host)
    assert str(caught.value) == "model monitoring report failed: invalid host data"
    assert KEY not in caplog.text + str(caught.value)
    assert quote_plus(KEY) not in caplog.text + str(caught.value)


def test_offline_mode_is_existing_successful_noop(loopback, monkeypatch):
    requests, _ = loopback
    monkeypatch.setattr(roboflow_api, "OFFLINE_MODE", True)
    assert send(adapter()) is None
    assert requests == []


def test_replayed_secret_fails_before_transport(monkeypatch):
    def forbidden(**arguments):
        pytest.fail("transport called with unavailable secret")

    monkeypatch.setattr(
        roboflow_api, "send_inference_results_to_model_monitoring", forbidden
    )
    with pytest.raises(SecretUnavailableError):
        send(adapter(), key=Secret.unavailable("REPLAY"))


def test_report_uses_explicit_monitoring_label_and_preserves_input_rows(monkeypatch):
    calls = []

    def boundary(operation, function, **arguments):
        calls.append(operation)
        arguments.pop("platform")
        return function(**arguments)

    def transport(**arguments):
        arguments["inference_data"]["inference_results"][0]["confidence"] = 0

    monkeypatch.setattr(module, "call_host", boundary)
    monkeypatch.setattr(
        roboflow_api, "send_inference_results_to_model_monitoring", transport
    )
    send(adapter())
    assert calls == [HostOperation.MONITORING]
    assert ROWS[0]["confidence"] == 0.9


def test_existing_api_timeout_and_tls_settings_are_forwarded(monkeypatch):
    calls = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

    def post(**arguments):
        calls.append(arguments)
        return Response()

    monkeypatch.setattr(roboflow_api, "OFFLINE_MODE", False)
    monkeypatch.setattr(roboflow_api.requests, "post", post)
    send(adapter())
    assert calls[0]["timeout"] == roboflow_api.ROBOFLOW_API_REQUEST_TIMEOUT
    assert calls[0]["verify"] == roboflow_api.ROBOFLOW_API_VERIFY_SSL


def test_block_background_runner_calls_real_adapter_boundary_on_worker(
    monkeypatch, caplog
):
    from datetime import timedelta

    from roboflow_workflows.v2.blocks.roboflow_platform.model_monitoring import (
        ModelMonitoringInferenceAggregatorBlock,
    )
    from roboflow_workflows.v2.blocks.sinks.runtime import BackgroundEffects

    caller = threading.get_ident()
    threads = []

    def workspace(**kwargs):
        threads.append(threading.get_ident())
        return "invented-workspace"

    def metadata():
        threads.append(threading.get_ident())
        logging.getLogger("urllib3.connectionpool").warning("metadata %s", KEY)
        return {"fixture": True}

    def transport(**kwargs):
        threads.append(threading.get_ident())
        logging.getLogger("urllib3.connectionpool").warning("upload %s", KEY)
        raise RuntimeError(KEY)

    monkeypatch.setattr(roboflow_api, "get_roboflow_workspace", workspace)
    monkeypatch.setattr(
        roboflow_api, "send_inference_results_to_model_monitoring", transport
    )
    seconds = [0]
    with BackgroundEffects() as effects, caplog.at_level(logging.DEBUG):
        block = ModelMonitoringInferenceAggregatorBlock(
            model_monitoring_port=RoboflowModelMonitoringAdapter(system_info=metadata),
            model_monitoring_clock=lambda: datetime(2026, 1, 1)
            + timedelta(seconds=seconds[0]),
            api_key=KEY,
            sink_effects=effects,
        )
        args = dict(
            predictions=None,
            model_id="fixture/1",
            frequency=1,
            unique_aggregator_key="fixture",
        )
        block.run(**args)
        seconds[0] = 1
        block.run(**args)
        assert effects.drain(5)
        assert effects.stats().failed == 1
        assert len(threads) == 3 and all(thread != caller for thread in threads)
        assert KEY not in caplog.text + repr(effects.failures())
