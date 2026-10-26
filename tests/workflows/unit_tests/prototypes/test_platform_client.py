"""The Roboflow-platform port, its standalone default and the server adapter.

The adapter assertions are behavioural, with a configured gateway, a controlled
transport and the real header-policy flags - identity checks would not prove
the secure-gateway wrapping, the api-key redaction or the credit/serverless
header policy survived the port.
"""

import inspect
from unittest import mock

import pytest
import requests

from inference.core.workflows.errors import WorkflowEnvironmentConfigurationError
from inference.core.workflows.prototypes.platform_client import (
    OFFLINE_PLATFORM_CLIENT,
    OfflineRoboflowPlatformClient,
    RoboflowPlatformClient,
)


def test_the_shared_offline_instance_is_the_offline_client() -> None:
    assert isinstance(OFFLINE_PLATFORM_CLIENT, OfflineRoboflowPlatformClient)


PORT_MEMBERS = (
    "post",
    "build_api_headers",
    "build_weights_provider_headers",
    "wrap_url",
    "get_roboflow_workspace",
    "add_custom_metadata",
    "register_image_at_roboflow",
    "annotate_image_at_roboflow",
    "update_image_metadata_at_roboflow",
    "batch_update_image_metadata_at_roboflow",
    "search_project_images_at_roboflow",
    "send_inference_results_to_model_monitoring",
    "get_device_id",
    "get_server_version",
    "get_system_info",
)

# The Roboflow API operations of the Roboflow-platform blocks, with arguments
# the offline client must refuse.
PLATFORM_BLOCK_OPERATIONS = {
    "get_roboflow_workspace": {"api_key": "k"},
    "add_custom_metadata": {
        "api_key": "k",
        "workspace_id": "w",
        "inference_ids": ["i"],
        "field_name": "f",
        "field_value": "v",
    },
    "register_image_at_roboflow": {
        "api_key": "k",
        "dataset_id": "d",
        "local_image_id": "l",
        "image_bytes": b"",
        "batch_name": "b",
    },
    "annotate_image_at_roboflow": {
        "api_key": "k",
        "dataset_id": "d",
        "local_image_id": "l",
        "roboflow_image_id": "r",
        "annotation_content": "c",
        "annotation_file_type": "txt",
    },
    "update_image_metadata_at_roboflow": {
        "api_key": "k",
        "workspace_id": "w",
        "image_id": "i",
    },
    "batch_update_image_metadata_at_roboflow": {
        "api_key": "k",
        "workspace_id": "w",
        "updates": [],
    },
    "search_project_images_at_roboflow": {
        "api_key": "k",
        "workspace": "w",
        "project": "p",
        "image_base64": "b64",
        "limit": 1,
    },
    "send_inference_results_to_model_monitoring": {
        "api_key": "k",
        "workspace_id": "w",
        "inference_data": {},
    },
}


def test_port_declares_every_member_blocks_use() -> None:
    for name in PORT_MEMBERS:
        assert hasattr(RoboflowPlatformClient, name)
        assert hasattr(OfflineRoboflowPlatformClient, name)


def test_offline_default_is_not_callable() -> None:
    # steps_initialiser.call_if_callable() invokes any callable registered in
    # REGISTERED_INITIALIZERS with no arguments.
    assert not callable(OFFLINE_PLATFORM_CLIENT)


def test_offline_wrap_url_is_the_identity() -> None:
    url = "https://example.com/a?b=c"
    assert OFFLINE_PLATFORM_CLIENT.wrap_url(url) == url


def test_offline_headers_are_empty() -> None:
    assert OFFLINE_PLATFORM_CLIENT.build_api_headers() == {}
    assert OFFLINE_PLATFORM_CLIENT.build_api_headers(explicit_headers={"X": "1"}) == {
        "X": "1"
    }
    assert OFFLINE_PLATFORM_CLIENT.build_weights_provider_headers() is None


def test_offline_post_raises_an_actionable_error() -> None:
    with pytest.raises(WorkflowEnvironmentConfigurationError) as error:
        OFFLINE_PLATFORM_CLIENT.post(
            endpoint="apiproxy/openai", api_key="k", payload={}
        )
    assert "workflows_core.platform_client" in str(error.value)


@pytest.mark.parametrize("operation", sorted(PLATFORM_BLOCK_OPERATIONS))
def test_offline_platform_block_operations_raise_an_actionable_error(
    operation,
) -> None:
    with pytest.raises(WorkflowEnvironmentConfigurationError) as error:
        getattr(OFFLINE_PLATFORM_CLIENT, operation)(
            **PLATFORM_BLOCK_OPERATIONS[operation]
        )
    assert "workflows_core.platform_client" in str(error.value)


def test_offline_host_identity_is_empty() -> None:
    assert OFFLINE_PLATFORM_CLIENT.get_device_id() is None
    assert OFFLINE_PLATFORM_CLIENT.get_server_version() == "unknown"
    assert OFFLINE_PLATFORM_CLIENT.get_system_info() == {}


def test_server_adapter_satisfies_the_port_signatures() -> None:
    from inference.core.interfaces.roboflow_platform_client import (
        ServerRoboflowPlatformClient,
    )

    for name in PORT_MEMBERS:
        port_params = list(
            inspect.signature(getattr(RoboflowPlatformClient, name)).parameters
        )
        real_params = list(
            inspect.signature(getattr(ServerRoboflowPlatformClient, name)).parameters
        )
        assert port_params == real_params, f"{name}: {port_params} != {real_params}"


def test_server_adapter_wrap_url_really_proxies_through_the_secure_gateway(monkeypatch):
    import inference.core.utils.url_utils as url_utils
    from inference.core.interfaces.roboflow_platform_client import (
        ServerRoboflowPlatformClient,
    )

    monkeypatch.setattr(url_utils, "SECURE_GATEWAY", "gateway.local")
    adapter = ServerRoboflowPlatformClient()
    raw = "https://api.roboflow.com/x?api_key=abcd1234&a=1"
    wrapped = adapter.wrap_url(raw)

    # Measured: urllib.parse.quote(..., safe="~()*!'") encodes the slashes too.
    assert wrapped == (
        "https://gateway.local/proxy?url="
        "https%3A%2F%2Fapi.roboflow.com%2Fx%3Fapi_key%3Dabcd1234%26a%3D1"
    )
    assert wrapped == url_utils.wrap_url(raw)
    assert adapter.wrap_url(wrapped) == wrapped  # idempotent


def test_server_adapter_post_redacts_the_api_key(monkeypatch) -> None:
    """`post_to_roboflow_api` ends in `api_key_safe_raise_for_status`.

    The redaction lands on the INNER `requests.HTTPError` (its message quotes
    the sanitised `response.url`); the outer
    `RoboflowAPIUnsuccessfulRequestError` message is generic by construction
    (`roboflow_api.py:274`). Assert both, so neither can regress.
    """
    import inference.core.roboflow_api as roboflow_api
    from inference.core.interfaces.roboflow_platform_client import (
        ServerRoboflowPlatformClient,
    )

    response = requests.Response()
    response.status_code = 500
    response.url = "https://api.roboflow.com/x?api_key=SECRETKEY123&nocache=true"
    monkeypatch.setattr(roboflow_api, "OFFLINE_MODE", False)
    monkeypatch.setattr(roboflow_api.requests, "post", lambda **kwargs: response)

    with pytest.raises(Exception) as error:
        ServerRoboflowPlatformClient().post(endpoint="x", api_key="SECRETKEY123")

    cause = error.value.__cause__
    assert cause is not None, "the sanitised HTTPError must be chained as the cause"
    assert "SECRETKEY123" not in str(cause)
    assert "api_key=SE***23" in str(cause)
    assert "SECRETKEY123" not in str(error.value)
    assert "SECRETKEY123" not in response.url


def test_server_adapter_header_policy_matches_the_server(monkeypatch) -> None:
    """Credit verification, serverless artefacts, the internal service secret,
    the version header and ROBOFLOW_API_EXTRA_HEADERS merging all have to
    survive the port. Values measured on the unchanged tree.
    """
    import inference.core.roboflow_api as roboflow_api
    from inference.core.interfaces.roboflow_platform_client import (
        ServerRoboflowPlatformClient,
    )
    from inference.core.version import __version__

    adapter = ServerRoboflowPlatformClient()

    monkeypatch.setattr(roboflow_api, "ROBOFLOW_API_EXTRA_HEADERS", None)
    monkeypatch.setattr(roboflow_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(roboflow_api, "ENFORCE_CREDITS_VERIFICATION", False)
    monkeypatch.setattr(roboflow_api, "ROBOFLOW_INTERNAL_SERVICE_SECRET", None)
    assert adapter.build_api_headers() == {
        "X-Roboflow-Inference-Version": __version__,
        "X-Allow-Chunked": "true",
    }
    assert adapter.build_weights_provider_headers() == {
        "X-Roboflow-Inference-Version": __version__,
        "X-Allow-Chunked": "true",
    }

    monkeypatch.setattr(roboflow_api, "GCP_SERVERLESS", True)
    monkeypatch.setattr(roboflow_api, "ENFORCE_CREDITS_VERIFICATION", True)
    monkeypatch.setattr(roboflow_api, "ROBOFLOW_INTERNAL_SERVICE_SECRET", "s3cr3t")
    assert adapter.build_weights_provider_headers() == {
        "x-enforce-internal-artefacts-urls": "true",
        "x-enforce-credits-verification": "true",
        "X-Roboflow-Internal-Service-Secret": "s3cr3t",
        "X-Roboflow-Inference-Version": __version__,
        "X-Allow-Chunked": "true",
    }

    monkeypatch.setattr(roboflow_api, "ROBOFLOW_API_EXTRA_HEADERS", '{"X-Extra": "1"}')
    assert adapter.build_api_headers(explicit_headers={"X-Own": "2"}) == {
        "X-Extra": "1",
        "X-Own": "2",
        "X-Roboflow-Inference-Version": __version__,
        "X-Allow-Chunked": "true",
    }


def test_the_two_api_key_redaction_implementations_agree() -> None:
    # D3: the workflows caller of `api_key_safe_raise_for_status` moves to the
    # plugin, so no copy is made - pin the equivalence that rests on.
    from inference.core.utils.requests import deduct_api_key_from_string as server_impl
    from inference_sdk.http.utils.requests import deduct_api_key_from_string as sdk_impl

    for case in (
        "https://x/y?api_key=abcdefghij&z=1",
        "https://x/y?api_key=ab",
        "https://x/y?service_secret=topsecret&api_key=abcdefghij",
        "no credentials here",
    ):
        assert server_impl(case) == sdk_impl(value=case)


@pytest.mark.parametrize("operation", sorted(PLATFORM_BLOCK_OPERATIONS))
def test_server_adapter_forwards_platform_block_operations_to_roboflow_api(
    operation,
) -> None:
    import inference.core.roboflow_api as roboflow_api
    from inference.core.interfaces.roboflow_platform_client import (
        ServerRoboflowPlatformClient,
    )

    arguments = PLATFORM_BLOCK_OPERATIONS[operation]
    with mock.patch.object(roboflow_api, operation) as forwarded:
        result = getattr(ServerRoboflowPlatformClient(), operation)(**arguments)

    assert result is forwarded.return_value
    forwarded.assert_called_once()
    # Positional arguments are named after the port: `roboflow_api` decorators
    # hide the wrapped signature, and the port mirrors its parameter names.
    port_parameters = [
        name
        for name in inspect.signature(
            getattr(RoboflowPlatformClient, operation)
        ).parameters
        if name != "self"
    ]
    called_with = {
        **dict(zip(port_parameters, forwarded.call_args.args)),
        **forwarded.call_args.kwargs,
    }
    for name, value in arguments.items():
        assert called_with[name] == value, (operation, name)


def test_server_adapter_reports_the_server_identity(monkeypatch) -> None:
    import inference.core.env as env
    import inference.core.managers.metrics as metrics
    from inference.core.interfaces.roboflow_platform_client import (
        ServerRoboflowPlatformClient,
    )
    from inference.core.version import __version__

    monkeypatch.setattr(env, "DEVICE_ID", "device-1")
    monkeypatch.setattr(metrics, "get_system_info", lambda: {"hostname": "h"})
    adapter = ServerRoboflowPlatformClient()

    assert adapter.get_device_id() == "device-1"
    assert adapter.get_server_version() == __version__
    assert adapter.get_system_info() == {"hostname": "h"}
