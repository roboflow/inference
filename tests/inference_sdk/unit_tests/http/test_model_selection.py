from typing import Any

import numpy as np
import pytest
from aioresponses import aioresponses

from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from inference_sdk.http.entities import ApiKeyTransport
from inference_sdk.http.errors import InvalidParameterError, ModelNotInitializedError


def register_response(
    requests_mock, async_responses, method, path, payload, headers=None
):
    url = f"http://server{path}"
    getattr(requests_mock, method)(url, json=payload, headers=headers or {})
    getattr(async_responses, method)(url, payload=payload, headers=headers or {})


async def call_client(client, method, asynchronous, *args, **kwargs):
    if asynchronous:
        return await getattr(client, f"{method}_async")(*args, **kwargs)
    return getattr(client, method)(*args, **kwargs)


def registry_payload(*model_ids, selected_model_id=None):
    payload = {
        "models": [
            {"model_id": model_id, "task_type": "object-detection"}
            for model_id in model_ids
        ]
    }
    if selected_model_id is not None:
        payload["selected_model_id"] = selected_model_id
    return payload


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
    cache_key = "opaque-server-handle"
    requests_mock.get(
        "http://server/model/registry",
        json={"models": [{"model_id": "project/1", "task_type": "object-detection"}]},
    )
    requests_mock.post(
        "http://server/model/add",
        json={
            "models": [{"model_id": cache_key, "task_type": "object-detection"}],
            "selected_model_id": cache_key,
        },
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
    cache_key = "another-server-handle"
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
                "models": [{"model_id": cache_key, "task_type": "object-detection"}],
                "selected_model_id": cache_key,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_selected_unload_requires_a_confirmed_handle_before_sending_request(
    requests_mock, asynchronous
):
    client = InferenceHTTPClient(api_url="http://server", api_key="key")
    client.configure(InferenceConfiguration(model_package_id="engine-1"))
    with aioresponses() as responses:
        register_response(
            requests_mock, responses, "post", "/model/remove", registry_payload()
        )
        with pytest.raises(ModelNotInitializedError, match="Load the selected model"):
            await call_client(client, "unload_model", asynchronous, "project/1")
        assert not requests_mock.called
        assert not responses.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_selection_remembers_server_handles_across_configurations_and_reload(
    requests_mock, asynchronous
):
    client = InferenceHTTPClient(api_url="http://server", api_key="key")
    with aioresponses() as responses:
        for backend, handle in [("onnx", "server-onnx"), ("trt", "server-trt")]:
            client.configure(InferenceConfiguration(backend=backend))
            register_response(
                requests_mock,
                responses,
                "post",
                "/model/add",
                registry_payload(handle, selected_model_id=handle),
                {"X-Roboflow-Model-Selection": "applied"},
            )
            await call_client(client, "load_model", asynchronous, "project/1")

        # Existing handles are selected by the active configuration, without a load.
        for backend, handle in [("onnx", "server-onnx"), ("trt", "server-trt")]:
            client.configure(InferenceConfiguration(backend=backend))
            register_response(
                requests_mock,
                responses,
                "get",
                "/model/registry?api_key=key",
                registry_payload("project/1", "server-onnx", "server-trt"),
            )
            description = await call_client(
                client,
                "get_model_description",
                asynchronous,
                "project/1",
                allow_loading=False,
            )
            assert description.model_id == handle

        # A restarted worker supplies a new handle for the same selection.
        register_response(
            requests_mock,
            responses,
            "get",
            "/model/registry?api_key=key",
            registry_payload("project/1", "unrecognized-handle"),
        )
        register_response(
            requests_mock,
            responses,
            "post",
            "/model/add",
            registry_payload(
                "replacement-handle", selected_model_id="replacement-handle"
            ),
            {"X-Roboflow-Model-Selection": "applied"},
        )
        description = await call_client(
            client, "get_model_description", asynchronous, "project/1"
        )
        assert description.model_id == "replacement-handle"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("method", ["load_model", "get_model_description"])
async def test_selected_load_rejects_missing_server_handle(
    requests_mock, asynchronous, method
):
    client = InferenceHTTPClient(api_url="http://server", api_key="key")
    client.configure(InferenceConfiguration(backend="trt"))
    with aioresponses() as responses:
        register_response(
            requests_mock,
            responses,
            "get",
            "/model/registry?api_key=key",
            registry_payload("project/1"),
        )
        register_response(
            requests_mock,
            responses,
            "post",
            "/model/add",
            registry_payload("project/1"),
            {"X-Roboflow-Model-Selection": "applied"},
        )
        with pytest.raises(InvalidParameterError, match="selected model ID"):
            await call_client(client, method, asynchronous, "project/1")


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("unload_all", [False, True])
async def test_unload_forgets_selected_handle(requests_mock, asynchronous, unload_all):
    from inference_sdk.http.errors import ModelNotInitializedError

    client = InferenceHTTPClient(api_url="http://server", api_key="key")
    client.configure(InferenceConfiguration(model_package_id="engine-1"))
    with aioresponses() as responses:
        register_response(
            requests_mock,
            responses,
            "post",
            "/model/add",
            registry_payload("server-handle", selected_model_id="server-handle"),
            {"X-Roboflow-Model-Selection": "applied"},
        )
        await call_client(client, "load_model", asynchronous, "project/1")
        register_response(
            requests_mock,
            responses,
            "post",
            "/model/clear" if unload_all else "/model/remove",
            registry_payload(),
        )
        if unload_all:
            await call_client(client, "unload_all_models", asynchronous)
        else:
            await call_client(client, "unload_model", asynchronous, "project/1")
        register_response(
            requests_mock,
            responses,
            "get",
            "/model/registry?api_key=key",
            registry_payload("server-handle"),
        )
        with pytest.raises(ModelNotInitializedError):
            await call_client(
                client,
                "get_model_description",
                asynchronous,
                "project/1",
                allow_loading=False,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("transport", [ApiKeyTransport.LEGACY, ApiKeyTransport.HEADER])
async def test_unload_selected_model_uses_confirmed_handle_and_auth(
    requests_mock, asynchronous, transport
):
    client = InferenceHTTPClient(api_url="http://server", api_key="key")
    client.configure(InferenceConfiguration(backend="trt", api_key_transport=transport))
    with aioresponses() as responses:
        register_response(
            requests_mock,
            responses,
            "post",
            "/model/add",
            registry_payload("server-handle", selected_model_id="server-handle"),
            {"X-Roboflow-Model-Selection": "applied"},
        )
        await call_client(client, "load_model", asynchronous, "yolov8n-640")
        register_response(
            requests_mock, responses, "post", "/model/remove", registry_payload()
        )
        await call_client(client, "unload_model", asynchronous, "yolov8n-640")
        if asynchronous:
            request = next(
                calls[0].kwargs
                for (_, url), calls in responses.requests.items()
                if url.path == "/model/remove"
            )
            payload, headers = request["json"], request["headers"]
        else:
            payload, headers = (
                requests_mock.last_request.json(),
                requests_mock.last_request.headers,
            )
        expected = {"model_id": "server-handle"}
        if transport is ApiKeyTransport.LEGACY:
            expected["api_key"] = "key"
        else:
            assert headers["Authorization"] == "Bearer key"
        assert payload == expected
