import numpy as np
import pytest
from requests import Response

from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from inference_sdk.config import WORKFLOW_PREVIEW_HEADER, workflow_is_preview


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize(
    "explicit,inherited", [(True, False), (False, True), (False, False)]
)
def test_workflow_preview_payload(requests_mock, named, explicit, inherited):
    client = InferenceHTTPClient(api_url="http://localhost:9001", api_key="test")
    path = "/workspace/workflows/workflow" if named else "/workflows/run"
    requests_mock.post(f"http://localhost:9001{path}", json={"outputs": []})
    kwargs = (
        {"workspace_name": "workspace", "workflow_id": "workflow"}
        if named
        else {"specification": {}}
    )
    token = workflow_is_preview.set(inherited)
    try:
        client.run_workflow(**kwargs, is_preview=explicit)
    finally:
        workflow_is_preview.reset(token)
    assert requests_mock.last_request.json().get("is_preview", False) is (
        explicit or inherited
    )
    assert (WORKFLOW_PREVIEW_HEADER in requests_mock.last_request.headers) is (
        explicit or inherited
    )
    client.run_workflow(**kwargs)
    assert "is_preview" not in requests_mock.last_request.json()
    assert WORKFLOW_PREVIEW_HEADER not in requests_mock.last_request.headers


@pytest.mark.parametrize("transport", ["legacy", "both", "header"])
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.asyncio
async def test_remote_model_preview_headers_are_scoped(
    transport, async_call, monkeypatch
):
    client = InferenceHTTPClient(api_url="http://localhost:9001", api_key="test")
    client.configure(InferenceConfiguration(api_key_transport=transport))
    captured = []

    def capture(requests_data):
        captured.extend(requests_data)
        response = Response()
        response._content = b'{"predictions": []}'
        return [response]

    async def capture_async(requests_data, **kwargs):
        captured.extend(requests_data)
        return [{"predictions": []}]

    client._execute_infer_from_api_request = capture
    monkeypatch.setattr(
        "inference_sdk.http.client.execute_requests_packages_async", capture_async
    )
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    async def infer():
        if async_call:
            await client.infer_from_api_v0_async(image, model_id="model/1")
        else:
            client.infer_from_api_v0(image, model_id="model/1")

    token = workflow_is_preview.set(True)
    try:
        await infer()
    finally:
        workflow_is_preview.reset(token)
    assert captured[-1].headers[WORKFLOW_PREVIEW_HEADER] == "true"
    await infer()
    assert WORKFLOW_PREVIEW_HEADER not in captured[-1].headers
