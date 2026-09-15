from typing import Any

import numpy as np
import pytest
from aioresponses import aioresponses

from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from inference_sdk.http.entities import model_selection_cache_key
from inference_sdk.http.errors import InvalidParameterError


@pytest.mark.parametrize("acknowledged", [True, False])
def test_v0_selection_requires_server_acknowledgment(requests_mock, acknowledged):
    headers = {"X-Roboflow-Model-Selection": "applied"} if acknowledged else {}
    requests_mock.post(
        "http://server/project/1", json={"predictions": []}, headers=headers
    )
    client = InferenceHTTPClient(api_url="http://server", api_key="key").select_api_v0()
    client.configure(InferenceConfiguration(backend="trt", quantization="fp16"))
    if acknowledged:
        assert client.infer(
            np.zeros((2, 2, 3), dtype=np.uint8), model_id="project/1"
        ) == {"predictions": []}
    else:
        with pytest.raises(InvalidParameterError, match="did not confirm"):
            client.infer(np.zeros((2, 2, 3), dtype=np.uint8), model_id="project/1")
    assert requests_mock.last_request.qs["backend"] == ["trt"]
    assert requests_mock.last_request.qs["quantization"] == ["fp16"]


@pytest.mark.asyncio
@pytest.mark.parametrize("acknowledged", [True, False])
async def test_async_v0_selection_requires_server_acknowledgment(acknowledged):
    import re

    headers = {"X-Roboflow-Model-Selection": "applied"} if acknowledged else {}
    client = InferenceHTTPClient(api_url="http://server", api_key="key").select_api_v0()
    client.configure(InferenceConfiguration(model_package_id="engine-1"))
    with aioresponses() as responses:
        responses.post(
            re.compile(r"http://server/project/1.*"),
            payload={"predictions": []},
            headers=headers,
        )
        if acknowledged:
            assert await client.infer_async(
                np.zeros((2, 2, 3), dtype=np.uint8), model_id="project/1"
            ) == {"predictions": []}
        else:
            with pytest.raises(InvalidParameterError, match="did not confirm"):
                await client.infer_async(
                    np.zeros((2, 2, 3), dtype=np.uint8), model_id="project/1"
                )


def test_v1_description_loads_selected_variant_without_loading_default(requests_mock):
    selectors: dict[str, Any] = {"backend": "trt", "quantization": "fp16"}
    cache_key = model_selection_cache_key("project/1", selectors, "key")
    requests_mock.get(
        "http://server/model/registry",
        json={"models": [{"model_id": "project/1", "task_type": "object-detection"}]},
    )
    requests_mock.post(
        "http://server/model/add",
        json={"models": [{"model_id": cache_key, "task_type": "object-detection"}]},
        headers={"X-Roboflow-Model-Selection": "applied"},
    )
    client = InferenceHTTPClient(api_url="http://server", api_key="key")
    client.configure(InferenceConfiguration(**selectors))
    assert client.get_model_description("project/1").model_id == cache_key
    assert requests_mock.last_request.json() == {
        "model_id": "project/1",
        "api_key": "key",
        **selectors,
    }


@pytest.mark.asyncio
async def test_async_v1_description_loads_selected_variant_without_loading_default():
    selectors: dict[str, Any] = {"model_package_id": "engine-1"}
    cache_key = model_selection_cache_key("project/1", selectors, "key")
    client = InferenceHTTPClient(api_url="http://server", api_key="key")
    client.configure(InferenceConfiguration(**selectors))
    with aioresponses() as responses:
        responses.get(
            "http://server/model/registry?api_key=key",
            payload={
                "models": [{"model_id": "project/1", "task_type": "object-detection"}]
            },
        )
        responses.post(
            "http://server/model/add",
            payload={
                "models": [{"model_id": cache_key, "task_type": "object-detection"}]
            },
            headers={"X-Roboflow-Model-Selection": "applied"},
        )
        assert (
            await client.get_model_description_async("project/1")
        ).model_id == cache_key
        post = [
            calls[0] for key, calls in responses.requests.items() if key[0] == "POST"
        ][0]
        assert post.kwargs["json"] == {
            "model_id": "project/1",
            "api_key": "key",
            **selectors,
        }


def test_unload_selected_variant_never_sends_default_model_id(requests_mock):
    selectors: dict[str, Any] = {"model_package_id": "engine-1"}
    requests_mock.post("http://server/model/remove", json={"models": []})
    client = InferenceHTTPClient(api_url="http://server", api_key="key")
    client.configure(InferenceConfiguration(**selectors))
    client.unload_model("project/1")
    assert requests_mock.last_request.json() == {
        "model_id": model_selection_cache_key("project/1", selectors, "key")
    }
