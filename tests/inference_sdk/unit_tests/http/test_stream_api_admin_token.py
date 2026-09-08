from unittest.mock import MagicMock

import pytest

from inference_sdk.http import client as client_module
from inference_sdk.http.client import InferenceHTTPClient


@pytest.mark.parametrize(
    "api_url", ["http://localhost:9001", "http://localhost:9001/edge/"]
)
@pytest.mark.parametrize("factory", [InferenceHTTPClient, InferenceHTTPClient.init])
def test_admin_token_only_sent_to_pipeline_management(monkeypatch, api_url, factory):
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "name": "inference",
        "version": "test",
        "uuid": "test",
    }
    get = MagicMock(return_value=response)
    post = MagicMock(return_value=response)
    monkeypatch.setattr(client_module.requests, "get", get)
    monkeypatch.setattr(client_module.requests, "post", post)
    client = factory(
        api_url,
        api_key="model-key",
        stream_api_key="test-only-admin-token-0123456789ABCDEFG",
    )
    client.start_inference_pipeline_with_workflow(
        0, workflow_specification={"version": "1.0"}
    )
    client.list_inference_pipelines()
    client.get_inference_pipeline_status("pipeline")
    client.pause_inference_pipeline("pipeline")
    client.resume_inference_pipeline("pipeline")
    client.terminate_inference_pipeline("pipeline")
    client.consume_inference_pipeline_result("pipeline")
    for call in get.call_args_list + post.call_args_list:
        assert "/inference_pipelines/" in call.args[0]
        assert call.args[0].startswith(api_url.rstrip("/") + "/inference_pipelines/")
        assert (
            call.kwargs["headers"]["X-Stream-API-Key"]
            == "test-only-admin-token-0123456789ABCDEFG"
        )
        assert call.kwargs["allow_redirects"] is False
        assert "test-only-admin-token-0123456789ABCDEFG" not in str(
            call.kwargs.get("json")
        )
    get.reset_mock()
    client.get_server_info()
    assert "X-Stream-API-Key" not in get.call_args.kwargs.get("headers", {})


def test_repointed_client_does_not_forward_stream_token(monkeypatch):
    get = MagicMock()
    monkeypatch.setattr(client_module.requests, "get", get)
    client = InferenceHTTPClient(
        "http://localhost:9001",
        stream_api_key="test-only-admin-token-0123456789ABCDEFG",
    )
    client._InferenceHTTPClient__api_url = "http://other.example.test"
    with pytest.raises(ValueError, match="bound to the original"):
        client.list_inference_pipelines()
    get.assert_not_called()
