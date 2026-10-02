"""The Roboflow-platform blocks are served by this server, as by `inference`,
and reach the Roboflow API through the server's platform client."""

import inspect
import json
from typing import List
from unittest import mock

import pytest
import requests
import requests_mock
from roboflow_workflows.prototypes.platform_client import RoboflowPlatformClient
from roboflow_workflows.prototypes.platform_errors import (
    RoboflowAPIConnectionError,
    RoboflowAPIForbiddenError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPIRequestError,
    RoboflowAPITimeoutError,
    RoboflowAPIUnsuccessfulRequestError,
)

from inference_server.workflows import host
from inference_server.workflows.errors import with_workflow_errors
from tests.unit_tests.legacy.conftest import FakeGateway

ROBOFLOW_PLATFORM_BLOCKS = {
    "roboflow_core/roboflow_dataset_upload@v1",
    "roboflow_core/roboflow_dataset_upload@v2",
    "roboflow_core/roboflow_custom_metadata@v1",
    "roboflow_core/model_monitoring_inference_aggregator@v1",
    "roboflow_core/roboflow_vision_events@v1",
    "roboflow_core/vision_event_bundle@v1",
    "roboflow_core/visual_search@v1",
    "roboflow_core/visual_search_classifier@v1",
    "roboflow_core/asset_library_attributes@v1",
}
API_URL = "https://api.example.com"


def test_blocks_describe_lists_every_platform_block(legacy_client) -> None:
    response = legacy_client(FakeGateway()).get("/workflows/blocks/describe")

    assert response.status_code == 200, response.text
    blocks = response.json()["blocks"]
    available = {block["manifest_type_identifier"] for block in blocks}
    assert ROBOFLOW_PLATFORM_BLOCKS <= available, ROBOFLOW_PLATFORM_BLOCKS - available
    for block in blocks:
        if block["manifest_type_identifier"] in ROBOFLOW_PLATFORM_BLOCKS:
            assert block["block_source"] == "workflows_core"
            assert block["fully_qualified_block_class_name"].startswith(
                (
                    "inference.core.workflows.core_steps.sinks.roboflow.",
                    "inference.core.workflows.core_steps.integrations.roboflow.",
                )
            )


def test_workflow_with_a_platform_block_compiles(legacy_client) -> None:
    specification = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowParameter", "name": "inference_id"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_custom_metadata@v1",
                "name": "metadata",
                "predictions": "$inputs.inference_id",
                "field_name": "location",
                "field_value": "toronto",
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "error_status",
                "selector": "$steps.metadata.error_status",
            }
        ],
    }

    response = legacy_client(FakeGateway()).post(
        "/workflows/validate", json=specification
    )

    assert response.status_code == 200, response.text


def test_platform_client_implements_every_port_member() -> None:
    port_members = [
        name
        for name, member in vars(RoboflowPlatformClient).items()
        if callable(member) and not name.startswith("_")
    ]
    assert "search_project_images_at_roboflow" in port_members
    for name in port_members:
        port_params = list(
            inspect.signature(getattr(RoboflowPlatformClient, name)).parameters
        )
        real_params = list(
            inspect.signature(
                getattr(host.ServerRoboflowPlatformClient, name)
            ).parameters
        )
        assert port_params == real_params, f"{name}: {port_params} != {real_params}"


def test_host_identity_comes_from_the_server_configuration() -> None:
    with mock.patch.object(host.configuration, "DEVICE_ID", "device-1"):
        assert host.PLATFORM_CLIENT.get_device_id() == "device-1"
    assert host.PLATFORM_CLIENT.get_server_version() == (
        host.configuration.SERVER_VERSION
    )
    assert "hostname" in host.PLATFORM_CLIENT.get_system_info()


class _RecordingPlatformClient(host.ServerRoboflowPlatformClient):
    """The server client with URL wrapping and headers recorded."""

    def __init__(self) -> None:
        self.wrapped: List[str] = []

    def build_api_headers(self, explicit_headers=None):
        return {"x-host": "yes", **(explicit_headers or {})}

    def wrap_url(self, url: str) -> str:
        self.wrapped.append(url)
        return f"https://gateway.example/proxy?url={url}"


def _response(
    status_code: int = 200, payload=None, url: str = API_URL
) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(payload if payload is not None else {}).encode()
    response.url = url
    return response


@pytest.fixture(autouse=True)
def _configured_api(monkeypatch):
    monkeypatch.setattr(host.configuration, "API_BASE_URL", API_URL + "/")
    monkeypatch.setattr(host.configuration, "LEGACY_OFFLINE_MODE", False)
    host.clear_workspace_cache()
    yield
    host.clear_workspace_cache()


def test_raise_for_status_redacts_the_key() -> None:
    response = _response(
        status_code=500, url=f"{API_URL}/x?api_key=abcdefghijk&service_secret=s3cr3t"
    )

    with pytest.raises(requests.HTTPError) as error:
        host._api_key_safe_raise_for_status(response)

    assert "abcdefghijk" not in str(error.value)
    assert "s3cr3t" not in str(error.value)
    assert "api_key=ab***jk" in str(error.value)


def test_model_monitoring_posts_to_the_configured_api() -> None:
    with mock.patch.object(
        host.requests, "post", return_value=_response(payload={})
    ) as post:
        host.PLATFORM_CLIENT.send_inference_results_to_model_monitoring(
            "my-key", "my-workspace", {"source": "workflow"}
        )

    url = post.call_args.kwargs["url"]
    assert url.startswith(f"{API_URL}/my-workspace/inference-stats?")
    assert "api_key=my-key" in url
    assert post.call_args.kwargs["json"] == {"source": "workflow"}
    assert post.call_args.kwargs["headers"] == host.PLATFORM_CLIENT.build_api_headers()


def test_get_workspace_goes_through_headers_and_url_wrapping() -> None:
    client = _RecordingPlatformClient()

    with mock.patch.object(
        host.requests, "get", return_value=_response(payload={"workspace": "ws"})
    ) as get:
        workspace = client.get_roboflow_workspace(api_key="my-key")

    assert workspace == "ws"
    assert client.wrapped == [f"{API_URL}/?api_key=my-key&nocache=true"]
    assert get.call_args.kwargs["url"].startswith("https://gateway.example/proxy?url=")
    assert get.call_args.kwargs["headers"] == {"x-host": "yes"}


@pytest.mark.parametrize("payload", [{}, {"workspace": ""}, {"workspace": "a b"}])
def test_get_workspace_rejects_an_empty_or_malformed_workspace(payload) -> None:
    with mock.patch.object(
        host.requests, "get", return_value=_response(payload=payload)
    ):
        with pytest.raises(RoboflowAPIRequestError):
            _RecordingPlatformClient().get_roboflow_workspace(api_key="my-key")


def test_get_workspace_is_cached_per_api_key() -> None:
    client = _RecordingPlatformClient()

    with mock.patch.object(
        host.requests, "get", return_value=_response(payload={"workspace": "ws"})
    ) as get:
        assert client.get_roboflow_workspace(api_key="key-a") == "ws"
        assert client.get_roboflow_workspace(api_key="key-a") == "ws"
        assert get.call_count == 1
        assert host.PLATFORM_CLIENT.get_roboflow_workspace(api_key="key-a") == "ws"
        assert get.call_count == 1

        assert client.get_roboflow_workspace(api_key="key-b") == "ws"
        assert get.call_count == 2


def test_get_workspace_cache_entry_expires_after_the_ttl() -> None:
    now = [1000.0]
    ttl = host.configuration.WORKSPACE_CACHE_TTL_S

    with mock.patch.object(
        host.requests, "get", return_value=_response(payload={"workspace": "ws"})
    ) as get, mock.patch(
        "inference_server.framework.model_stat.time.monotonic",
        side_effect=lambda: now[0],
    ):
        _RecordingPlatformClient().get_roboflow_workspace(api_key="my-key")
        now[0] += ttl - 1
        _RecordingPlatformClient().get_roboflow_workspace(api_key="my-key")
        assert get.call_count == 1
        now[0] += 2
        _RecordingPlatformClient().get_roboflow_workspace(api_key="my-key")
        assert get.call_count == 2


def test_get_workspace_failures_are_not_cached() -> None:
    with mock.patch.object(
        host.requests,
        "get",
        side_effect=[
            _response(status_code=500),
            _response(payload={}),
            _response(payload={"workspace": "ws"}),
        ],
    ) as get:
        with pytest.raises(RoboflowAPIUnsuccessfulRequestError):
            _RecordingPlatformClient().get_roboflow_workspace(api_key="my-key")
        with pytest.raises(RoboflowAPIRequestError):
            _RecordingPlatformClient().get_roboflow_workspace(api_key="my-key")
        assert (
            _RecordingPlatformClient().get_roboflow_workspace(api_key="my-key") == "ws"
        )
    assert get.call_count == 3


@pytest.mark.parametrize(
    "status_code, error_class",
    [
        (401, RoboflowAPINotAuthorizedError),
        (403, RoboflowAPIForbiddenError),
        (500, RoboflowAPIUnsuccessfulRequestError),
    ],
)
def test_http_errors_map_onto_the_shared_error_classes(
    status_code, error_class
) -> None:
    with mock.patch.object(
        host.requests, "post", return_value=_response(status_code=status_code)
    ):
        with pytest.raises(error_class):
            _RecordingPlatformClient().send_inference_results_to_model_monitoring(
                api_key="my-key", workspace_id="ws", inference_data={}
            )


@pytest.mark.parametrize(
    "transport_error, error_class",
    [
        (requests.exceptions.Timeout(), RoboflowAPITimeoutError),
        (requests.exceptions.ConnectionError(), RoboflowAPIConnectionError),
    ],
)
def test_transport_errors_map_onto_the_shared_error_classes(
    transport_error, error_class
) -> None:
    with mock.patch.object(host.requests, "post", side_effect=transport_error):
        with pytest.raises(error_class):
            _RecordingPlatformClient().batch_update_image_metadata_at_roboflow(
                api_key="my-key", workspace_id="ws", updates=[]
            )


def test_register_image_sends_the_legacy_multipart_upload() -> None:
    client = _RecordingPlatformClient()
    with mock.patch.object(
        host.requests,
        "post",
        return_value=_response(payload={"success": True, "id": "img-1"}),
    ) as post:
        result = client.register_image_at_roboflow(
            api_key="my-key",
            dataset_id="proj",
            local_image_id="local",
            image_bytes=b"jpeg",
            batch_name="batch",
            tags=["a", "b"],
            inference_id="inf-1",
            metadata={"k": "v"},
        )

    assert result == {"success": True, "id": "img-1"}
    assert client.wrapped == [
        f"{API_URL}/dataset/proj/upload?api_key=my-key&batch=batch"
        "&inference_id=inf-1&tag=a&tag=b"
    ]
    assert post.call_args.kwargs["data"] == {
        "name": "local.jpg",
        "metadata": json.dumps({"k": "v"}),
    }
    assert post.call_args.kwargs["files"] == {
        "file": ("imageToUpload", b"jpeg", "image/jpeg")
    }


def test_register_image_rejected_by_the_server_raises() -> None:
    with mock.patch.object(
        host.requests, "post", return_value=_response(payload={"success": False})
    ):
        with pytest.raises(RoboflowAPIUnsuccessfulRequestError):
            _RecordingPlatformClient().register_image_at_roboflow(
                api_key="k",
                dataset_id="p",
                local_image_id="l",
                image_bytes=b"",
                batch_name="b",
            )


def test_annotate_image_posts_text_with_plain_content_type() -> None:
    client = _RecordingPlatformClient()
    with mock.patch.object(
        host.requests, "post", return_value=_response(payload={"success": True})
    ) as post:
        client.annotate_image_at_roboflow(
            api_key="my-key",
            dataset_id="proj",
            local_image_id="local",
            roboflow_image_id="img-1",
            annotation_content="cat",
            annotation_file_type="txt",
        )

    assert client.wrapped == [
        f"{API_URL}/dataset/proj/annotate/img-1?api_key=my-key&name=local.txt"
        "&prediction=true"
    ]
    assert post.call_args.kwargs["data"] == "cat"
    assert post.call_args.kwargs["headers"]["Content-Type"] == "text/plain"


def test_annotate_image_conflict_reports_existing_annotation() -> None:
    with mock.patch.object(
        host.requests, "post", return_value=_response(status_code=409)
    ):
        with pytest.raises(RoboflowAPIUnsuccessfulRequestError, match="already has"):
            _RecordingPlatformClient().annotate_image_at_roboflow(
                api_key="k",
                dataset_id="p",
                local_image_id="l",
                roboflow_image_id="i",
                annotation_content="c",
                annotation_file_type="txt",
            )


def test_update_image_metadata_encodes_the_image_id() -> None:
    client = _RecordingPlatformClient()
    with mock.patch.object(
        host.requests, "post", return_value=_response(payload={"ok": 1})
    ) as post:
        client.update_image_metadata_at_roboflow(
            api_key="my-key",
            workspace_id="ws",
            image_id="a/b",
            metadata={"k": "v"},
            add_tags=["t"],
        )

    assert client.wrapped == [f"{API_URL}/ws/images/a%2Fb/metadata?api_key=my-key"]
    assert post.call_args.kwargs["json"] == {"metadata": {"k": "v"}, "addTags": ["t"]}


def test_custom_metadata_and_search_payloads() -> None:
    client = _RecordingPlatformClient()
    with mock.patch.object(
        host.requests, "post", return_value=_response(payload={"results": []})
    ) as post:
        client.add_custom_metadata(
            api_key="my-key",
            workspace_id="ws",
            inference_ids=["i1"],
            field_name="f",
            field_value="v",
        )
        metadata_call = post.call_args
        client.search_project_images_at_roboflow(
            api_key="my-key",
            workspace="ws",
            project="proj",
            image_base64="b64",
            limit=3,
        )
        search_call = post.call_args

    assert client.wrapped == [
        f"{API_URL}/ws/inference-stats/metadata?api_key=my-key&nocache=true",
        f"{API_URL}/ws/proj/search?api_key=my-key",
    ]
    assert metadata_call.kwargs["json"] == {
        "data": [{"inference_ids": ["i1"], "field_name": "f", "field_value": "v"}]
    }
    assert search_call.kwargs["json"]["limit"] == 3
    assert search_call.kwargs["json"]["fields"][0] == "id"


def test_offline_mode_skips_fire_and_forget_calls_and_refuses_the_rest(
    monkeypatch,
) -> None:
    monkeypatch.setattr(host.configuration, "LEGACY_OFFLINE_MODE", True)
    client = _RecordingPlatformClient()

    with mock.patch.object(host.requests, "post") as post:
        client.add_custom_metadata(
            api_key="k",
            workspace_id="w",
            inference_ids=[],
            field_name="f",
            field_value="v",
        )
        client.send_inference_results_to_model_monitoring(
            api_key="k", workspace_id="w", inference_data={}
        )
        with pytest.raises(RoboflowAPIConnectionError):
            client.register_image_at_roboflow(
                api_key="k",
                dataset_id="p",
                local_image_id="l",
                image_bytes=b"",
                batch_name="b",
            )

    post.assert_not_called()


def test_offline_mode_refuses_an_uncached_workspace_lookup(monkeypatch) -> None:
    monkeypatch.setattr(host.configuration, "LEGACY_OFFLINE_MODE", True)

    with requests_mock.Mocker() as m:
        with pytest.raises(RoboflowAPIConnectionError) as error:
            _RecordingPlatformClient().get_roboflow_workspace(api_key="my-key")

    assert m.call_count == 0
    assert str(error.value) == (
        "Cannot fetch workspace at Roboflow - OFFLINE_MODE is enabled."
    )


def test_offline_mode_still_answers_an_already_cached_workspace(
    monkeypatch,
) -> None:
    with requests_mock.Mocker() as m:
        m.get(requests_mock.ANY, json={"workspace": "ws"})
        assert host.PLATFORM_CLIENT.get_roboflow_workspace(api_key="my-key") == "ws"
        assert m.call_count == 1
        monkeypatch.setattr(host.configuration, "LEGACY_OFFLINE_MODE", True)

        assert host.PLATFORM_CLIENT.get_roboflow_workspace(api_key="my-key") == "ws"
        assert m.call_count == 1


def test_offline_mode_refuses_generic_platform_posts(monkeypatch) -> None:
    monkeypatch.setattr(host.configuration, "LEGACY_OFFLINE_MODE", True)

    with requests_mock.Mocker() as m:
        with pytest.raises(RoboflowAPIConnectionError):
            host.PLATFORM_CLIENT.post("x/y", api_key="k", payload={"a": 1})

    assert m.call_count == 0


def _raw_response(status_code: int, content: bytes) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = content
    response.url = API_URL

    return response


@pytest.mark.parametrize(
    "platform_response, status, message",
    [
        (
            _raw_response(401, b"{}"),
            401,
            "Unauthorized access to roboflow API - check API key and make sure the "
            "key is valid for workspace you use. Visit "
            "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
            "to learn how to retrieve one.",
        ),
        (
            _raw_response(402, b"{}"),
            402,
            "Not enough credits to perform this request. Verify your workspace "
            "billing page.",
        ),
        (
            _raw_response(403, b"{}"),
            403,
            "Unauthorized access to roboflow API - check API key and make sure the "
            "key is valid and have required scopes. Visit "
            "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
            "to learn how to retrieve one.",
        ),
        (
            _raw_response(404, b"{}"),
            404,
            "Requested Roboflow resource not found. Make sure that workspace, "
            "project or model you referred in request exists.",
        ),
        (
            _raw_response(423, b"{}"),
            423,
            "Roboflow API usage is paused. Please contact your workspace "
            "administrator to re-enable api keys.",
        ),
        (
            _raw_response(500, b"{}"),
            502,
            "Internal error. Request to Roboflow API failed.",
        ),
        (
            _raw_response(200, b"not json"),
            502,
            "Internal error. Request to Roboflow API failed.",
        ),
        (
            _raw_response(200, b"{}"),
            502,
            "Internal error. Request to Roboflow API failed.",
        ),
    ],
)
@pytest.mark.asyncio
async def test_workspace_lookup_failure_is_mapped_by_the_workflow_error_decorator(
    platform_response, status, message
) -> None:
    @with_workflow_errors
    async def handler():
        return _RecordingPlatformClient().get_roboflow_workspace(api_key="my-key")

    with mock.patch.object(host.requests, "get", return_value=platform_response):
        response = await handler()

    assert response.status_code == status
    assert json.loads(response.body) == {"message": message}


@pytest.mark.asyncio
async def test_workspace_lookup_without_key_is_mapped_by_the_workflow_error_decorator() -> (
    None
):
    @with_workflow_errors
    async def handler():
        return _RecordingPlatformClient().get_roboflow_workspace(api_key="")

    response = await handler()

    assert response.status_code == 502
    assert json.loads(response.body) == {
        "message": "Internal error. Request to Roboflow API failed."
    }


PROJECT_READS = [
    (
        "get_roboflow_dataset_type",
        f"{API_URL}/ws/proj?api_key=my-key&nocache=true",
        {"project": {"type": "classification"}},
        "classification",
    ),
    (
        "get_roboflow_active_learning_configuration",
        f"{API_URL}/ws/proj/active_learning?api_key=my-key",
        {"enabled": True},
        {"enabled": True},
    ),
    (
        "get_roboflow_labeling_batches",
        f"{API_URL}/ws/proj/batches?api_key=my-key",
        {"batches": [{"name": "b"}]},
        {"batches": [{"name": "b"}]},
    ),
    (
        "get_roboflow_labeling_jobs",
        f"{API_URL}/ws/proj/jobs?api_key=my-key",
        {"jobs": [{"numImages": 2}]},
        {"jobs": [{"numImages": 2}]},
    ),
]
PROJECT_READ_NAMES = [name for name, _, _, _ in PROJECT_READS]


def _read_project(client: host.ServerRoboflowPlatformClient, operation: str):
    return getattr(client, operation)(
        api_key="my-key", workspace_id="ws", dataset_id="proj"
    )


@pytest.mark.parametrize("operation, url, payload, expected_result", PROJECT_READS)
def test_project_read_goes_through_headers_and_url_wrapping(
    operation, url, payload, expected_result
) -> None:
    client = _RecordingPlatformClient()

    with mock.patch.object(
        host.requests, "get", return_value=_response(payload=payload)
    ) as get:
        result = _read_project(client, operation)

    assert result == expected_result
    assert client.wrapped == [url]
    assert get.call_count == 1
    assert get.call_args.kwargs["url"] == f"https://gateway.example/proxy?url={url}"
    assert get.call_args.kwargs["headers"] == {"x-host": "yes"}
    assert get.call_args.kwargs["timeout"] == host.API_REQUEST_TIMEOUT_S
    assert "verify" not in get.call_args.kwargs


@pytest.mark.parametrize("operation", PROJECT_READ_NAMES)
def test_project_read_disables_verification_when_the_switch_is_off(
    monkeypatch, operation
) -> None:
    monkeypatch.setattr(host.configuration, "ROBOFLOW_API_VERIFY_SSL", False)

    with mock.patch.object(
        host.requests, "get", return_value=_response(payload={})
    ) as get:
        _read_project(_RecordingPlatformClient(), operation)

    assert get.call_args.kwargs["verify"] is False


@pytest.mark.parametrize("payload", [{}, {"project": {}}])
def test_dataset_type_defaults_to_object_detection(payload) -> None:
    with mock.patch.object(
        host.requests, "get", return_value=_response(payload=payload)
    ):
        dataset_type = _RecordingPlatformClient().get_roboflow_dataset_type(
            api_key="my-key", workspace_id="ws", dataset_id="proj"
        )

    assert dataset_type == "object-detection"


@pytest.mark.parametrize("operation", PROJECT_READ_NAMES)
@pytest.mark.parametrize(
    "status_code, error_class",
    [
        (401, RoboflowAPINotAuthorizedError),
        (403, RoboflowAPIForbiddenError),
        (404, RoboflowAPINotNotFoundError),
        (500, RoboflowAPIUnsuccessfulRequestError),
    ],
)
def test_project_read_http_errors_map_onto_the_shared_error_classes(
    operation, status_code, error_class
) -> None:
    response = _response(
        status_code=status_code, url=f"{API_URL}/ws/proj?api_key=my-secret-key"
    )

    with mock.patch.object(host.requests, "get", return_value=response):
        with pytest.raises(error_class) as error:
            _read_project(_RecordingPlatformClient(), operation)

    assert "my-secret-key" not in str(error.value)


@pytest.mark.parametrize("operation", PROJECT_READ_NAMES)
@pytest.mark.parametrize(
    "transport_error, error_class",
    [
        (requests.exceptions.Timeout(), RoboflowAPITimeoutError),
        (requests.exceptions.ConnectionError(), RoboflowAPIConnectionError),
    ],
)
def test_project_read_transport_errors_map_onto_the_shared_error_classes(
    operation, transport_error, error_class
) -> None:
    with mock.patch.object(host.requests, "get", side_effect=transport_error):
        with pytest.raises(error_class):
            _read_project(_RecordingPlatformClient(), operation)


@pytest.mark.parametrize("operation", PROJECT_READ_NAMES)
def test_project_read_rejects_a_response_that_is_not_json(operation) -> None:
    response = _response()
    response._content = b"<html>"

    with mock.patch.object(host.requests, "get", return_value=response):
        with pytest.raises(RoboflowAPIRequestError, match="Could not decode JSON"):
            _read_project(_RecordingPlatformClient(), operation)


@pytest.mark.parametrize("operation", PROJECT_READ_NAMES)
def test_project_read_is_refused_in_offline_mode(monkeypatch, operation) -> None:
    monkeypatch.setattr(host.configuration, "LEGACY_OFFLINE_MODE", True)

    with mock.patch.object(host.requests, "get") as get:
        with pytest.raises(RoboflowAPIConnectionError, match="OFFLINE_MODE"):
            _read_project(_RecordingPlatformClient(), operation)

    get.assert_not_called()


@pytest.mark.parametrize("operation", PROJECT_READ_NAMES)
@pytest.mark.parametrize("status_code", [200, 500])
def test_project_read_records_its_duration_under_the_legacy_function_name(
    operation, status_code
) -> None:
    with mock.patch.object(
        host.requests, "get", return_value=_response(status_code=status_code)
    ), mock.patch.object(host.telemetry, "record_api_call") as record_api_call:
        try:
            _read_project(_RecordingPlatformClient(), operation)
        except RoboflowAPIUnsuccessfulRequestError:
            pass

    assert record_api_call.call_count == 1
    assert record_api_call.call_args.args[0] == operation
    assert record_api_call.call_args.args[1] >= 0.0
