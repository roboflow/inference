from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute, iter_route_contexts
from fastapi.testclient import TestClient
from streamvision.stream_manager.api.entities import (
    CommandContext,
    CommandResponse,
    ConsumePipelineResponse,
    FrameMetadata,
    InferencePipelineStatusResponse,
    InitializeWebRTCPipelineResponse,
    ListPipelinesResponse,
)
from streamvision.stream_manager.api.errors import (
    ConnectivityError,
    ProcessesManagerAuthorisationError,
    ProcessesManagerClientError,
    ProcessesManagerInternalError,
    ProcessesManagerInvalidPayload,
    ProcessesManagerNotFoundError,
    ProcessesManagerOperationError,
)
from streamvision.stream_manager.manager_app.entities import (
    ConsumeResultsPayload,
    InitialisePipelinePayload,
    InitialiseWebRTCPipelinePayload,
)
from streamvision.stream_manager.manager_app.errors import (
    CommunicationProtocolError,
    MalformedHeaderError,
    MalformedPayloadError,
    MessageToBigError,
    TransmissionChannelClosed,
)

from inference_server.streams.router import include_streams_router

UNAUTHORIZED = (
    "Unauthorized access to roboflow API - check API key and make sure the key is "
    "valid for workspace you use. Visit "
    "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
    "to learn how to retrieve one."
)

CONTEXT = CommandContext(request_id="req-1", pipeline_id="pipe-1")
COMMAND = CommandResponse(status="success", context=CONTEXT)
LISTING = ListPipelinesResponse(
    status="success", context=CONTEXT, pipelines=["pipe-1", "pipe-2"]
)
STATUS = InferencePipelineStatusResponse(
    status="success", context=CONTEXT, report={"state": "RUNNING"}
)
CONSUMED = ConsumePipelineResponse(
    status="success",
    context=CONTEXT,
    outputs=[{"predictions": []}, None],
    frames_metadata=[
        FrameMetadata(
            frame_timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            frame_id=7,
            source_id=0,
        ),
        None,
    ],
)
WEBRTC = InitializeWebRTCPipelineResponse(
    status="success", context=CONTEXT, sdp="v=0", type="answer"
)

INITIALISE_PAYLOAD = {
    "video_configuration": {"type": "VideoConfiguration", "video_reference": 0},
    "processing_configuration": {
        "type": "WorkflowConfiguration",
        "workspace_name": "ws",
        "workflow_id": "wf",
    },
}
INITIALISE_WEBRTC_PAYLOAD = {
    **INITIALISE_PAYLOAD,
    "webrtc_offer": {"type": "offer", "sdp": "v=0"},
}

ERROR_ROWS = [
    pytest.param(
        ProcessesManagerInvalidPayload,
        400,
        "ProcessesManagerInvalidPayload",
        id="invalid-payload",
    ),
    pytest.param(
        MalformedPayloadError, 400, "MalformedPayloadError", id="malformed-payload"
    ),
    pytest.param(ProcessesManagerAuthorisationError, 401, None, id="authorisation"),
    pytest.param(
        ProcessesManagerNotFoundError,
        404,
        "ProcessesManagerNotFoundError",
        id="not-found",
    ),
    pytest.param(
        ProcessesManagerInternalError,
        500,
        "ProcessesManagerInternalError",
        id="internal",
    ),
    pytest.param(
        ProcessesManagerOperationError,
        500,
        "ProcessesManagerOperationError",
        id="operation",
    ),
    pytest.param(ConnectivityError, 500, "ConnectivityError", id="connectivity"),
    pytest.param(
        TransmissionChannelClosed,
        500,
        "TransmissionChannelClosed",
        id="channel-closed",
    ),
    pytest.param(MessageToBigError, 400, "MessageToBigError", id="message-too-big"),
    pytest.param(
        MalformedHeaderError, 500, "MalformedHeaderError", id="malformed-header"
    ),
    pytest.param(
        CommunicationProtocolError,
        500,
        "CommunicationProtocolError",
        id="protocol-error",
    ),
    pytest.param(
        ProcessesManagerClientError,
        500,
        "ProcessesManagerClientError",
        id="client-error",
    ),
]


class _FakeStreamManagerClient:
    def __init__(self):
        self.list_pipelines = AsyncMock(return_value=LISTING)
        self.get_status = AsyncMock(return_value=STATUS)
        self.initialise_pipeline = AsyncMock(return_value=COMMAND)
        self.initialise_webrtc_pipeline = AsyncMock(return_value=WEBRTC)
        self.pause_pipeline = AsyncMock(return_value=COMMAND)
        self.resume_pipeline = AsyncMock(return_value=COMMAND)
        self.terminate_pipeline = AsyncMock(return_value=COMMAND)
        self.consume_pipeline_result = AsyncMock(return_value=CONSUMED)


@pytest.fixture
def stream_client():
    return _FakeStreamManagerClient()


@pytest.fixture
def client(stream_client):
    app = FastAPI()
    include_streams_router(app)
    app.state.stream_manager_client = stream_client

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def body_keys_only(monkeypatch):
    monkeypatch.setattr(
        "inference_server.configuration.ALLOW_API_KEY_FROM_HEADERS", False
    )
    monkeypatch.setattr("inference_server.legacy.common.DEFAULT_API_KEY", None)


def _dumped(model):
    return model.model_dump(mode="json")


def test_list_forwards_to_the_client(client, stream_client):
    response = client.get("/inference_pipelines/list")

    assert response.status_code == 200
    assert response.json() == _dumped(LISTING)
    stream_client.list_pipelines.assert_awaited_once_with()


def test_status_forwards_the_pipeline_id(client, stream_client):
    response = client.get("/inference_pipelines/pipe-1/status")

    assert response.status_code == 200
    assert response.json() == _dumped(STATUS)
    stream_client.get_status.assert_awaited_once_with(pipeline_id="pipe-1")


def test_pause_forwards_the_pipeline_id(client, stream_client):
    response = client.post("/inference_pipelines/pipe-1/pause")

    assert response.status_code == 200
    assert response.json() == _dumped(COMMAND)
    stream_client.pause_pipeline.assert_awaited_once_with(pipeline_id="pipe-1")


def test_resume_forwards_the_pipeline_id(client, stream_client):
    response = client.post("/inference_pipelines/pipe-1/resume")

    assert response.status_code == 200
    assert response.json() == _dumped(COMMAND)
    stream_client.resume_pipeline.assert_awaited_once_with(pipeline_id="pipe-1")


def test_terminate_forwards_the_pipeline_id(client, stream_client):
    response = client.post("/inference_pipelines/pipe-1/terminate")

    assert response.status_code == 200
    assert response.json() == _dumped(COMMAND)
    stream_client.terminate_pipeline.assert_awaited_once_with(pipeline_id="pipe-1")


def test_consume_forwards_the_excluded_fields(client, stream_client):
    response = client.request(
        "GET",
        "/inference_pipelines/pipe-1/consume",
        json={"excluded_fields": ["image"]},
    )

    assert response.status_code == 200
    assert response.json() == _dumped(CONSUMED)
    stream_client.consume_pipeline_result.assert_awaited_once_with(
        pipeline_id="pipe-1", excluded_fields=["image"]
    )


def test_consume_without_a_body_excludes_nothing(client, stream_client):
    response = client.get("/inference_pipelines/pipe-1/consume")

    assert response.status_code == 200
    stream_client.consume_pipeline_result.assert_awaited_once_with(
        pipeline_id="pipe-1", excluded_fields=ConsumeResultsPayload().excluded_fields
    )


def test_initialise_forwards_the_payload(client, stream_client, body_keys_only):
    response = client.post(
        "/inference_pipelines/initialise",
        json={**INITIALISE_PAYLOAD, "api_key": "body-key"},
    )

    assert response.status_code == 200
    assert response.json() == _dumped(COMMAND)
    stream_client.initialise_pipeline.assert_awaited_once()
    sent = stream_client.initialise_pipeline.await_args.kwargs["initialisation_request"]
    assert isinstance(sent, InitialisePipelinePayload)
    assert sent.api_key == "body-key"
    assert sent.video_configuration.video_reference == 0
    assert sent.processing_configuration.workflow_id == "wf"


def test_initialise_webrtc_forwards_the_payload(client, stream_client, body_keys_only):
    response = client.post(
        "/inference_pipelines/initialise_webrtc",
        json={**INITIALISE_WEBRTC_PAYLOAD, "api_key": "body-key"},
    )

    assert response.status_code == 200
    assert response.json() == _dumped(WEBRTC)
    sent = stream_client.initialise_webrtc_pipeline.await_args.kwargs[
        "initialisation_request"
    ]
    assert isinstance(sent, InitialiseWebRTCPipelinePayload)
    assert sent.api_key == "body-key"
    assert sent.webrtc_offer.sdp == "v=0"


@pytest.mark.parametrize(
    "path,payload,method_name",
    [
        ("/inference_pipelines/initialise", INITIALISE_PAYLOAD, "initialise_pipeline"),
        (
            "/inference_pipelines/initialise_webrtc",
            INITIALISE_WEBRTC_PAYLOAD,
            "initialise_webrtc_pipeline",
        ),
    ],
)
class TestInitialiseApiKey:
    def test_header_wins_over_the_body(
        self, client, stream_client, monkeypatch, path, payload, method_name
    ):
        monkeypatch.setattr(
            "inference_server.configuration.ALLOW_API_KEY_FROM_HEADERS", True
        )

        response = client.post(
            path,
            json={**payload, "api_key": "body-key"},
            headers={"Authorization": "Bearer header-key"},
        )

        assert response.status_code == 200
        sent = getattr(stream_client, method_name).await_args.kwargs[
            "initialisation_request"
        ]
        assert sent.api_key == "header-key"

    def test_header_is_ignored_when_disallowed(
        self, client, stream_client, monkeypatch, path, payload, method_name
    ):
        monkeypatch.setattr(
            "inference_server.configuration.ALLOW_API_KEY_FROM_HEADERS", False
        )

        response = client.post(
            path,
            json={**payload, "api_key": "body-key"},
            headers={"Authorization": "Bearer header-key"},
        )

        assert response.status_code == 200
        sent = getattr(stream_client, method_name).await_args.kwargs[
            "initialisation_request"
        ]
        assert sent.api_key == "body-key"

    def test_default_key_fills_a_missing_key(
        self, client, stream_client, monkeypatch, path, payload, method_name
    ):
        monkeypatch.setattr("inference_server.legacy.common.DEFAULT_API_KEY", "env-key")

        response = client.post(path, json=payload)

        assert response.status_code == 200
        sent = getattr(stream_client, method_name).await_args.kwargs[
            "initialisation_request"
        ]
        assert sent.api_key == "env-key"


@pytest.mark.parametrize("error_type,status,error_name", ERROR_ROWS)
def test_client_errors_answer_like_legacy(
    client, stream_client, error_type, status, error_name
):
    stream_client.get_status.side_effect = error_type(
        "private", public_message="shown", inner_error=KeyError("k")
    )

    response = client.get("/inference_pipelines/pipe-1/status")

    assert response.status_code == status
    if error_name is None:
        assert response.json() == {"message": UNAUTHORIZED}
    else:
        assert response.json() == {
            "message": "shown",
            "error_type": error_name,
            "inner_error_type": "KeyError",
        }


def test_route_table_matches_legacy():
    app = FastAPI()
    include_streams_router(app)

    table = sorted(
        (context.route.path, method, context.route.summary)
        for context in iter_route_contexts(app.routes)
        if isinstance(context.route, APIRoute)
        for method in context.route.methods - {"HEAD"}
    )

    assert table == [
        (
            "/inference_pipelines/initialise",
            "POST",
            "[EXPERIMENTAL] Starts new InferencePipeline",
        ),
        (
            "/inference_pipelines/initialise_webrtc",
            "POST",
            "[EXPERIMENTAL] Establishes WebRTC peer connection and starts new "
            "InferencePipeline consuming video track",
        ),
        (
            "/inference_pipelines/list",
            "GET",
            "[EXPERIMENTAL] List active InferencePipelines",
        ),
        (
            "/inference_pipelines/{pipeline_id}/consume",
            "GET",
            "[EXPERIMENTAL] Consumes InferencePipeline result",
        ),
        (
            "/inference_pipelines/{pipeline_id}/pause",
            "POST",
            "[EXPERIMENTAL] Pauses the InferencePipeline",
        ),
        (
            "/inference_pipelines/{pipeline_id}/resume",
            "POST",
            "[EXPERIMENTAL] Resumes the InferencePipeline",
        ),
        (
            "/inference_pipelines/{pipeline_id}/status",
            "GET",
            "[EXPERIMENTAL] Get status of InferencePipeline",
        ),
        (
            "/inference_pipelines/{pipeline_id}/terminate",
            "POST",
            "[EXPERIMENTAL] Terminates the InferencePipeline",
        ),
    ]
