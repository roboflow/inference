from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from inference.core.interfaces.http import http_api
from inference.core.interfaces.stream_manager.manager_app import entities

PAYLOAD = {
    "video_configuration": {"type": "VideoConfiguration", "video_reference": 0},
    "processing_configuration": {
        "type": "WorkflowConfiguration",
        "workflow_specification": {"version": "1.0", "inputs": [], "steps": []},
    },
}
# The request models every stream initialisation is validated through: the
# video configuration alone, and nested in the regular and WebRTC payloads
# (these carry the checks once made on the removed enterprise request model).
SCHEMAS = ["manager", "initialise_payload", "webrtc_payload"]
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


def make_interface(
    monkeypatch,
    root_path="",
    enabled=True,
    dedicated_workspace=None,
    local_whitelist=None,
):
    monkeypatch.setattr(http_api, "ENABLE_STREAM_API", enabled)
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(http_api, "LAMBDA", False)
    monkeypatch.setattr(
        http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", dedicated_workspace
    )
    monkeypatch.setattr(
        http_api, "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT", local_whitelist
    )
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
def test_pipeline_operations_work_without_workspace_auth_configuration(
    monkeypatch, root_path, method, suffix
):
    interface, manager = make_interface(monkeypatch, root_path)
    payload = {**PAYLOAD, "webrtc_offer": {"type": "offer", "sdp": "test"}}
    with TestClient(interface.app) as client:
        response = client.request(
            method, root_path + "/inference_pipelines" + suffix, json=payload
        )
    assert response.status_code == 200, response.text
    assert len(manager.mock_calls) == 1


@pytest.mark.parametrize("method,suffix", ROUTES)
@pytest.mark.parametrize(
    "auth_config",
    [
        {"dedicated_workspace": "allowed-workspace"},
        {"local_whitelist": ["allowed-workspace"]},
    ],
)
def test_pipeline_operations_retain_configured_workspace_auth(
    monkeypatch, method, suffix, auth_config
):
    interface, manager = make_interface(monkeypatch, **auth_config)
    workspace_lookup = AsyncMock(return_value="denied-workspace")
    monkeypatch.setattr(http_api, "get_roboflow_workspace_async", workspace_lookup)
    payload = {**PAYLOAD, "webrtc_offer": {"type": "offer", "sdp": "test"}}
    with TestClient(interface.app) as client:
        path = "/inference_pipelines" + suffix
        response = client.request(method, path, json=payload)
        assert response.status_code == 401
        workspace_lookup.assert_not_awaited()
        response = client.request(
            method, path, headers={"Authorization": "Bearer denied-key"}, json=payload
        )
        assert response.status_code == 401
        assert manager.mock_calls == []
        workspace_lookup.return_value = "allowed-workspace"
        response = client.request(
            method, path, headers={"Authorization": "Bearer allowed-key"}, json=payload
        )
    assert response.status_code == 200, response.text
    assert len(manager.mock_calls) == 1


def test_disabled_stream_api_does_not_register_pipeline_routes(monkeypatch):
    interface, _ = make_interface(monkeypatch, enabled=False)
    assert not any(
        getattr(route, "path", "").startswith("/inference_pipelines")
        for route in interface.app.routes
    )


@pytest.mark.parametrize("schema", SCHEMAS)
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
    with pytest.raises(ValueError):
        make_video_request(schema, reference)


@pytest.mark.parametrize("schema", SCHEMAS)
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
    assert make_video_request(schema, reference).video_reference == reference


@pytest.mark.parametrize("schema", SCHEMAS)
def test_operator_can_explicitly_allow_raw_media_pipeline(monkeypatch, schema):
    monkeypatch.setattr(entities, "ALLOW_UNSAFE_GSTREAMER_PIPELINES", True)
    reference = "videotestsrc ! appsink"
    assert make_video_request(schema, reference).video_reference == reference


def make_video_request(schema, reference):
    if schema == "manager":
        return entities.VideoConfiguration(
            type="VideoConfiguration", video_reference=reference
        )
    payload = {
        **PAYLOAD,
        "video_configuration": {
            "type": "VideoConfiguration",
            "video_reference": reference,
        },
    }
    if schema == "initialise_payload":
        request = entities.InitialisePipelinePayload.model_validate(payload)
    else:
        request = entities.InitialiseWebRTCPipelinePayload.model_validate(
            {**payload, "webrtc_offer": {"type": "offer", "sdp": "test"}}
        )
    return request.video_configuration


@pytest.mark.parametrize("suffix", ["/initialise", "/initialise_webrtc"])
@pytest.mark.parametrize(
    "payload",
    [
        {
            **PAYLOAD,
            "video_configuration": {
                "type": "VideoConfiguration",
                "video_reference": "videotestsrc ! appsink",
            },
            "webrtc_offer": {"type": "offer", "sdp": "test"},
        },
        None,
    ],
)
def test_pipeline_initialisation_rejects_invalid_payload_before_the_manager(
    monkeypatch, suffix, payload
):
    monkeypatch.setattr(entities, "ALLOW_UNSAFE_GSTREAMER_PIPELINES", False)
    interface, manager = make_interface(monkeypatch)
    with TestClient(interface.app) as client:
        response = client.post("/inference_pipelines" + suffix, json=payload)
    assert response.status_code == 422, response.text
    assert manager.mock_calls == []
