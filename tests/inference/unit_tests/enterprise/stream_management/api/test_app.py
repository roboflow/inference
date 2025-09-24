from unittest import mock
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from inference.enterprise.stream_management.api import app
from inference.enterprise.stream_management.api.entities import (
    CommandContext,
    CommandResponse,
    InferencePipelineStatusResponse,
    ListPipelinesResponse,
)
from inference.enterprise.stream_management.api.errors import (
    ConnectivityError,
    ProcessesManagerNotFoundError,
)


@mock.patch.object(app, "STREAM_MANAGER_CLIENT", new_callable=AsyncMock)
def test_list_pipelines_when_communication_with_stream_manager_abused(
    stream_manager_client: AsyncMock,
) -> None:
    # given
    client = TestClient(app.app)
    stream_manager_client.list_pipelines.side_effect = ConnectivityError(
        "Could not connect"
    )

    # when
    response = client.get("/list_pipelines")

    # then
    assert (
        response.status_code == 503
    ), "Status code when connectivity error occurs should be 503"
    assert (
        response.json()["status"] == "failure"
    ), "Failure must be denoted in response payload"
    assert (
        len(response.json()["message"]) > 0
    ), "Message must be denoted in response payload"


@mock.patch.object(app, "STREAM_MANAGER_CLIENT", new_callable=AsyncMock)
def test_list_pipelines_when_communication_with_stream_manager_possible(
    stream_manager_client: AsyncMock,
) -> None:
    # given
    client = TestClient(app.app)
    stream_manager_client.list_pipelines.return_value = ListPipelinesResponse(
        status="success",
        context=CommandContext(request_id="my_request", pipeline_id=None),
        pipelines=["a", "b", "c"],
    )

    # when
    response = client.get("/list_pipelines")

    # then
    assert response.status_code == 200, "Status code for success must be 200"
    assert response.json() == {
        "status": "success",
        "context": {
            "request_id": "my_request",
            "pipeline_id": None,
        },
        "pipelines": ["a", "b", "c"],
    }, "ListPipelinesResponse must be serialised directly to JSON response"


@mock.patch.object(app, "STREAM_MANAGER_CLIENT", new_callable=AsyncMock)
def test_get_pipeline_status_when_pipeline_found(
    stream_manager_client: AsyncMock,
) -> None:
    # given
    client = TestClient(app.app)
    stream_manager_client.get_status.return_value = InferencePipelineStatusResponse(
        status="success",
        context=CommandContext(request_id="my_request", pipeline_id="my_pipeline"),
        report={"my": "report"},  # this is mock data
    )

    # when
    response = client.get("/status/my_pipeline")

    # then
    assert response.status_code == 200, "Status code for success must be 200"
    assert response.json() == {
        "status": "success",
        "context": {
            "request_id": "my_request",
            "pipeline_id": "my_pipeline",
        },
        "report": {"my": "report"},
    }, "InferencePipelineStatusResponse must be serialised directly to JSON response"


@mock.patch.object(app, "STREAM_MANAGER_CLIENT", new_callable=AsyncMock)
def test_get_pipeline_status_when_pipeline_not_found(
    stream_manager_client: AsyncMock,
) -> None:
    # given
    client = TestClient(app.app)
    stream_manager_client.get_status.side_effect = ProcessesManagerNotFoundError(
        "Pipeline not found"
    )

    # when
    response = client.get("/status/my_pipeline")

    # then
    assert response.status_code == 404, "Status code for not found must be 404"
    assert (
        response.json()["status"] == "failure"
    ), "Failure must be denoted in response payload"
    assert (
        len(response.json()["message"]) > 0
    ), "Message must be denoted in response payload"


@mock.patch.object(app, "STREAM_MANAGER_CLIENT", new_callable=AsyncMock)
def test_initialise_pipeline_when_invalid_payload_given(
    stream_manager_client: AsyncMock,
) -> None:
    # given
    client = TestClient(app.app)
    stream_manager_client.initialise_pipeline.return_value = CommandResponse(
        status="success",
        context=CommandContext(request_id="my_request", pipeline_id="my_pipeline"),
    )

    # when
    response = client.post("/initialise")

    # then
    assert (
        response.status_code == 422
    ), "Status code for invalid input entity must be 422"


@mock.patch.object(app, "STREAM_MANAGER_CLIENT", new_callable=AsyncMock)
def test_initialise_pipeline_when_valid_payload_given(
    stream_manager_client: AsyncMock,
) -> None:
    # given
    client = TestClient(app.app)
    stream_manager_client.initialise_pipeline.return_value = CommandResponse(
        status="success",
        context=CommandContext(request_id="my_request", pipeline_id="my_pipeline"),
    )

    # when
    response = client.post(
        "/initialise",
        json={
            "model_id": "some/1",
            "video_reference": "rtsp://some:543",
            "sink_configuration": {
                "type": "udp_sink",
                "host": "127.0.0.1",
                "port": 9090,
            },
            "api_key": "my_api_key",
            "model_configuration": {"type": "object-detection"},
            "active_learning_enabled": True,
            "video_source_properties": {
                "frame_width": 1280,
                "frame_height": 720,
                "fps": 30,
            },
        },
    )

    # then
    assert response.status_code == 200, "Status code for success must be 200"
    assert response.json() == {
        "status": "success",
        "context": {
            "request_id": "my_request",
            "pipeline_id": "my_pipeline",
        },
    }, "CommandResponse must be serialised directly to JSON response"

    actual_request = stream_manager_client.initialise_pipeline.call_args[1][
        "initialisation_request"
    ]
    assert actual_request.video_source_properties["frame_width"] == 1280
    assert actual_request.video_source_properties["frame_height"] == 720
    assert actual_request.video_source_properties["fps"] == 30


@mock.patch.object(app, "STREAM_MANAGER_CLIENT", new_callable=AsyncMock)
def test_initialise_pipeline_when_valid_payload_given_without_api_key(
    stream_manager_client: AsyncMock,
) -> None:
    # given
    client = TestClient(app.app)
    stream_manager_client.initialise_pipeline.return_value = CommandResponse(
        status="success",
        context=CommandContext(request_id="my_request", pipeline_id="my_pipeline"),
    )

    # when
    response = client.post(
        "/initialise",
        json={
            "model_id": "some/1",
            "video_reference": "rtsp://some:543",
            "sink_configuration": {
                "type": "udp_sink",
                "host": "127.0.0.1",
                "port": 9090,
            },
            "model_configuration": {"type": "object-detection"},
            "active_learning_enabled": True,
        },
    )

    # then
    assert response.status_code == 200, "Status code for success must be 200"
    assert response.json() == {
        "status": "success",
        "context": {
            "request_id": "my_request",
            "pipeline_id": "my_pipeline",
        },
    }, "CommandResponse must be serialised directly to JSON response"


@mock.patch.object(app, "STREAM_MANAGER_CLIENT", new_callable=AsyncMock)
def test_pause_pipeline_when_successful_response_expected(
    stream_manager_client: AsyncMock,
) -> None:
    # given
    client = TestClient(app.app)
    stream_manager_client.pause_pipeline.return_value = CommandResponse(
        status="success",
        context=CommandContext(request_id="my_request", pipeline_id="my_pipeline"),
    )

    # when
    response = client.post("/pause/my_pipeline")

    # then
    assert response.status_code == 200, "Status code for success must be 200"
    assert response.json() == {
        "status": "success",
        "context": {
            "request_id": "my_request",
            "pipeline_id": "my_pipeline",
        },
    }, "CommandResponse must be serialised directly to JSON response"


@mock.patch.object(app, "STREAM_MANAGER_CLIENT", new_callable=AsyncMock)
def test_resume_pipeline_when_successful_response_expected(
    stream_manager_client: AsyncMock,
) -> None:
    # given
    client = TestClient(app.app)
    stream_manager_client.resume_pipeline.return_value = CommandResponse(
        status="success",
        context=CommandContext(request_id="my_request", pipeline_id="my_pipeline"),
    )

    # when
    response = client.post("/resume/my_pipeline")

    # then
    assert response.status_code == 200, "Status code for success must be 200"
    assert response.json() == {
        "status": "success",
        "context": {
            "request_id": "my_request",
            "pipeline_id": "my_pipeline",
        },
    }, "CommandResponse must be serialised directly to JSON response"


@mock.patch.object(app, "STREAM_MANAGER_CLIENT", new_callable=AsyncMock)
def test_terminate_pipeline_when_successful_response_expected(
    stream_manager_client: AsyncMock,
) -> None:
    # given
    client = TestClient(app.app)
    stream_manager_client.terminate_pipeline.return_value = CommandResponse(
        status="success",
        context=CommandContext(request_id="my_request", pipeline_id="my_pipeline"),
    )

    # when
    response = client.post("/terminate/my_pipeline")

    # then
    assert response.status_code == 200, "Status code for success must be 200"
    assert response.json() == {
        "status": "success",
        "context": {
            "request_id": "my_request",
            "pipeline_id": "my_pipeline",
        },
    }, "CommandResponse must be serialised directly to JSON response"
