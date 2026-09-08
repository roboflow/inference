from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from inference.core.interfaces.http import http_api
from inference.core.interfaces.stream_manager.manager_app import entities
from inference.enterprise.stream_management.api import entities as enterprise_entities

TOKEN = "test-only-local-admin-token-0123456789ABCDEF"
PAYLOAD = {
    "video_configuration": {"type": "VideoConfiguration", "video_reference": 0},
    "processing_configuration": {
        "type": "WorkflowConfiguration",
        "workflow_specification": {"version": "1.0", "inputs": [], "steps": []},
    },
}
ROUTES = [
    ("GET", "/list"),
    ("GET", "/victim/status"),
    ("POST", "/initialise"),
    ("POST", "/initialise_webrtc"),
    ("POST", "/victim/pause"),
    ("POST", "/victim/resume"),
    ("POST", "/victim/terminate"),
    ("GET", "/victim/consume"),
]


def make_interface(monkeypatch, root_path="", enabled=True, token=TOKEN):
    monkeypatch.setattr(http_api, "ENABLE_STREAM_API", enabled)
    monkeypatch.setattr(http_api, "STREAM_API_KEY", token)
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(http_api, "LAMBDA", False)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    monkeypatch.setattr(http_api, "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT", None)
    monkeypatch.setattr(http_api, "ALLOW_ORIGINS", ["https://console.example.test"])
    monkeypatch.setattr(http_api, "InferenceInstrumentator", MagicMock())
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )
    manager = AsyncMock()
    for name in [
        "list_pipelines",
        "get_status",
        "initialise_pipeline",
        "initialise_webrtc_pipeline",
        "pause_pipeline",
        "resume_pipeline",
        "terminate_pipeline",
        "consume_pipeline_result",
    ]:
        getattr(manager, name).return_value = {
            "status": "success",
            "context": {},
            "pipelines": ["victim"],
            "report": {},
            "outputs": [],
            "frames_metadata": [],
            "sdp": "test",
            "type": "answer",
        }
    monkeypatch.setattr(
        http_api.StreamManagerClient, "init", MagicMock(return_value=manager)
    )
    models = MagicMock()
    models.pingback = None
    return http_api.HttpInterface(model_manager=models, root_path=root_path), manager


@pytest.mark.parametrize("root_path", ["", "/edge"])
@pytest.mark.parametrize("method,suffix", ROUTES)
def test_every_pipeline_operation_requires_dedicated_admin_token(
    monkeypatch, root_path, method, suffix
):
    interface, manager = make_interface(monkeypatch, root_path)
    with TestClient(interface.app) as client:
        path = root_path + "/inference_pipelines" + suffix
        for headers in [
            {},
            {"X-Stream-API-Key": "wrong"},
            {"Authorization": "Bearer roboflow-key"},
        ]:
            response = client.request(method, path, headers=headers, json=PAYLOAD)
            assert response.status_code == 401
        assert manager.mock_calls == []
        payload = {**PAYLOAD, "webrtc_offer": {"type": "offer", "sdp": "test"}}
        response = client.request(
            method, path, headers={"X-Stream-API-Key": TOKEN}, json=payload
        )
        assert response.status_code == 200, response.text
        assert len(manager.mock_calls) == 1


def test_duplicates_trailing_slash_and_cors_preflight(monkeypatch):
    interface, manager = make_interface(monkeypatch)
    with TestClient(interface.app) as client:
        path = "/inference_pipelines/list"
        response = client.get(
            path, headers=[("X-Stream-API-Key", TOKEN), ("X-Stream-API-Key", TOKEN)]
        )
        assert response.status_code == 401
        assert client.get(path + "/").status_code == 401
        response = client.options(
            path,
            headers={
                "Origin": "https://console.example.test",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-Stream-API-Key",
            },
        )
        assert response.status_code == 200
        assert manager.mock_calls == []


def test_stream_startup_fails_closed_only_when_enabled(monkeypatch):
    with pytest.raises(RuntimeError, match="requires STREAM_API_KEY"):
        make_interface(monkeypatch, enabled=True, token="")
    interface, _ = make_interface(monkeypatch, enabled=False, token="")
    assert not any(
        getattr(route, "path", "").startswith("/inference_pipelines")
        for route in interface.app.routes
    )


@pytest.mark.parametrize("schema", ["manager", "enterprise"])
@pytest.mark.parametrize(
    "reference",
    [
        "videotestsrc ! appsink",
        "udpsrc",
        "tcpserversrc",
        "v4l2src",
        "autovideosink",
        "video.mp4",
        "filesink location=/tmp/scratch",
        "filesink\nlocation=/tmp/scratch",
        "unsupported://host/video",
        "csi://invalid",
        "rtsp://",
        'rtsp://camera/" ! appsink',
        ["rtsp://camera/live", "videotestsrc ! appsink"],
        "rtsp://camera/live ! appsink",
    ],
)
def test_stream_requests_reject_raw_media_launch_syntax(monkeypatch, reference, schema):
    monkeypatch.setattr(entities, "ALLOW_UNSAFE_GSTREAMER_PIPELINES", False)
    monkeypatch.setattr(enterprise_entities, "ALLOW_UNSAFE_GSTREAMER_PIPELINES", False)
    with pytest.raises(ValueError):
        make_video_request(schema, reference)


@pytest.mark.parametrize("schema", ["manager", "enterprise"])
@pytest.mark.parametrize(
    "reference",
    [
        0,
        "/dev/video0",
        "./my video.mp4",
        "csi://0",
        "rtmp://host/live",
        "rtmps://host/live",
        "udp://host:5000",
        "srt://host:5000",
        "rtp://host:5000",
        "tcp://host:5000",
        "rtspt://camera/live",
        "rtspst://camera/live",
        "rtsp://camera/live?mode=fast",
        "https://host/video.mp4?token=x!y",
        ["rtsps://camera/live", 0],
    ],
)
def test_stream_requests_accept_supported_sources(monkeypatch, reference, schema):
    monkeypatch.setattr(entities, "ALLOW_UNSAFE_GSTREAMER_PIPELINES", False)
    monkeypatch.setattr(enterprise_entities, "ALLOW_UNSAFE_GSTREAMER_PIPELINES", False)
    assert make_video_request(schema, reference).video_reference == reference


@pytest.mark.parametrize("schema", ["manager", "enterprise"])
def test_trusted_admin_can_explicitly_allow_raw_media_pipeline(monkeypatch, schema):
    monkeypatch.setattr(entities, "ALLOW_UNSAFE_GSTREAMER_PIPELINES", True)
    monkeypatch.setattr(enterprise_entities, "ALLOW_UNSAFE_GSTREAMER_PIPELINES", True)
    reference = "videotestsrc ! appsink"
    assert make_video_request(schema, reference).video_reference == reference


def test_telemetry_is_installed_before_stream_auth_middleware(monkeypatch):
    installed = []
    monkeypatch.setattr(http_api, "OTEL_TRACING_ENABLED", True)

    def setup(app):
        assert not app.user_middleware
        installed.append(app)

    monkeypatch.setattr(http_api, "setup_telemetry", setup)
    interface, _ = make_interface(monkeypatch)
    assert installed == [interface.app]
    assert any(
        m.cls.__name__ == "StreamAPIAuthMiddleware"
        for m in interface.app.user_middleware
    )


def make_video_request(schema, reference):
    if schema == "manager":
        return entities.VideoConfiguration(
            type="VideoConfiguration", video_reference=reference
        )
    return enterprise_entities.PipelineInitialisationRequest(
        model_id="test/1",
        video_reference=reference,
        sink_configuration={"host": "localhost", "port": 5000},
    )


@pytest.mark.parametrize(
    "token", ["short", "a" * 31, "a" * 257, "a" * 32 + "\n", "invalid token" + "a" * 32]
)
def test_stream_admin_token_requires_bounded_url_safe_secret(monkeypatch, token):
    with pytest.raises(RuntimeError, match="requires STREAM_API_KEY"):
        make_interface(monkeypatch, token=token)
