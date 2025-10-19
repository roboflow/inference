import base64
import json
import os.path
from glob import glob
from io import BytesIO
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
import requests
from aiohttp import ClientConnectionError, ClientResponseError, RequestInfo
from aioresponses import aioresponses
from requests import HTTPError, Request, Response
from requests_mock.mocker import Mocker
from yarl import URL

from inference_sdk.http import client
from inference_sdk.http.client import (
    DEFAULT_HEADERS,
    InferenceHTTPClient,
    _determine_client_downsizing_parameters,
    _determine_client_mode,
    _ensure_model_is_selected,
    wrap_errors,
    wrap_errors_async,
)
from inference_sdk.http.entities import (
    CLASSIFICATION_TASK,
    HTTPClientMode,
    InferenceConfiguration,
    ModelDescription,
    RegisteredModels,
)
from inference_sdk.http.errors import (
    HTTPCallErrorError,
    HTTPClientError,
    InvalidModelIdentifier,
    InvalidParameterError,
    ModelNotSelectedError,
    ModelTaskTypeNotSupportedError,
    WrongClientModeError,
)
from inference_sdk.http.utils import executors


def test_ensure_model_is_selected_when_model_is_selected() -> None:
    # when
    _ensure_model_is_selected(model_id="some/2")


def test_ensure_model_is_selected_when_model_is_not_selected() -> None:
    # when
    with pytest.raises(ModelNotSelectedError):
        _ensure_model_is_selected(model_id=None)


@pytest.mark.parametrize(
    "api_url",
    [
        "https://detect.roboflow.com",
        "https://outline.roboflow.com",
        "https://classify.roboflow.com",
        "https://infer.roboflow.com",
    ],
)
def test_determine_client_mode_when_roboflow_hosted_api_is_used(api_url: str) -> None:
    # when
    result = _determine_client_mode(api_url=api_url)

    # then
    assert result is HTTPClientMode.V0


def test_determine_client_mode_when_self_hosted_api_is_used() -> None:
    # when
    result = _determine_client_mode(api_url="http://127.0.0.1:9090")

    # then
    assert result is HTTPClientMode.V1


def test_determine_client_downsizing_parameters_when_downsizing_is_disabled() -> None:
    # when
    result = _determine_client_downsizing_parameters(
        client_downsizing_disabled=True,
        model_description=ModelDescription(
            model_id="a/1",
            task_type=CLASSIFICATION_TASK,
            input_width=512,
            input_height=512,
        ),
        default_max_input_size=1024,
    )

    # then
    assert result == (None, None)


def test_determine_client_downsizing_parameters_when_model_announce_its_input_dimensions() -> (
    None
):
    # when
    result = _determine_client_downsizing_parameters(
        client_downsizing_disabled=False,
        model_description=ModelDescription(
            model_id="a/1",
            task_type=CLASSIFICATION_TASK,
            input_width=512,
            input_height=384,
        ),
        default_max_input_size=1024,
    )

    # then
    assert result == (384, 512)


def test_determine_client_downsizing_parameters_when_model_does_not_announce_its_input_dimensions() -> (
    None
):
    # when
    result = _determine_client_downsizing_parameters(
        client_downsizing_disabled=False,
        model_description=ModelDescription(
            model_id="a/1",
            task_type=CLASSIFICATION_TASK,
        ),
        default_max_input_size=1024,
    )

    # then
    assert result == (1024, 1024)


def test_wrap_errors_when_no_errors_occurs() -> None:
    # given
    @wrap_errors
    def example(a: int, b: int) -> int:
        return a + b

    # when
    result = example(2, 3)

    # then
    assert result == 5


def test_wrap_errors_when_http_error_occurs() -> None:
    # given
    @wrap_errors
    def example() -> None:
        response = Response()
        response.headers = {"Content-Type": "application/json"}
        response.status_code = 404
        response._content = json.dumps({"message": "Not Found"}).encode("utf-8")
        raise HTTPError(
            request=Request(),
            response=response,
        )

    # when
    with pytest.raises(HTTPCallErrorError) as error:
        example()

    assert error.value.status_code == 404
    assert error.value.api_message == "Not Found"


@pytest.mark.asyncio
async def test_wrap_errors_async_when_http_error_occurs() -> None:
    # given
    @wrap_errors_async
    async def example() -> None:
        raise ClientResponseError(
            request_info=RequestInfo(
                url=URL("https://some.com"),
                method="GET",
                headers={},  # type: ignore
            ),
            history=(),
            status=404,
            message="Not Found",
        )

    # when
    with pytest.raises(HTTPCallErrorError) as error:
        await example()

    assert error.value.status_code == 404
    assert error.value.api_message == "Not Found"


def test_wrap_errors_when_connection_error_occurs() -> None:
    # given
    @wrap_errors
    def example() -> None:
        raise ConnectionError()

    # when
    with pytest.raises(HTTPClientError):
        example()


@pytest.mark.asyncio
async def test_wrap_errors_async_when_connection_error_occurs() -> None:
    # given
    @wrap_errors_async
    async def example() -> None:
        raise ClientConnectionError()

    # when
    with pytest.raises(HTTPClientError):
        await example()


def test_wrap_errors_when_unknown_error_occurs() -> None:
    # given
    @wrap_errors
    def example() -> None:
        raise Exception()

    # when
    with pytest.raises(Exception):
        example()


@pytest.mark.asyncio
async def test_wrap_errors_async_when_unknown_error_occurs() -> None:
    # given
    @wrap_errors_async
    async def example() -> None:
        raise Exception()

    # when
    with pytest.raises(Exception):
        await example()


def test_setting_configuration_statically() -> None:
    # given
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url="https://some.com")
    configuration = InferenceConfiguration(visualize_labels=True, source="source-test")

    # when
    previous_configuration = http_client.inference_configuration
    http_client.configure(inference_configuration=configuration)
    new_configuration = http_client.inference_configuration

    # then
    assert previous_configuration is not configuration
    assert new_configuration is configuration


def test_setting_configuration_with_context_manager() -> None:
    # given
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url="https://some.com")
    configuration = InferenceConfiguration(visualize_labels=True)

    # when
    previous_configuration = http_client.inference_configuration
    with http_client.use_configuration(inference_configuration=configuration):
        new_configuration = http_client.inference_configuration

    # then
    assert previous_configuration is not configuration
    assert new_configuration is configuration
    assert http_client.inference_configuration is previous_configuration


def test_setting_model_statically() -> None:
    # given
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url="https://some.com")

    # when
    previous_model = http_client.selected_model
    http_client.select_model(model_id="some/1")
    new_model = http_client.selected_model

    # then
    assert previous_model is None
    assert new_model is "some/1"


def test_setting_model_with_context_manager() -> None:
    # given
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url="https://some.com")

    # when
    previous_model = http_client.selected_model
    with http_client.use_model(model_id="some/1"):
        new_model = http_client.selected_model

    # then
    assert previous_model is None
    assert new_model is "some/1"
    assert http_client.selected_model is None


def test_setting_mode_statically() -> None:
    # given
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url="https://some.com")

    # when
    previous_mode = http_client.client_mode
    http_client.select_api_v0()
    new_mode = http_client.client_mode

    # then
    assert previous_mode is HTTPClientMode.V1
    assert new_mode is HTTPClientMode.V0


def test_setting_mode_with_context_manager() -> None:
    # given
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url="https://some.com")

    # when
    previous_mode = http_client.client_mode
    with http_client.use_api_v0():
        new_mode = http_client.client_mode

    # then
    assert previous_mode is HTTPClientMode.V1
    assert new_mode is HTTPClientMode.V0
    assert http_client.client_mode is HTTPClientMode.V1


def test_client_unload_all_models_in_v0_mode() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(WrongClientModeError):
        with http_client.use_api_v0():
            _ = http_client.unload_all_models()


@pytest.mark.asyncio
async def test_client_unload_all_models_async_in_v0_mode() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(WrongClientModeError):
        with http_client.use_api_v0():
            _ = await http_client.unload_all_models_async()


def test_client_unload_all_models_when_successful_response_expected(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    requests_mock.post(f"{api_url}/model/clear", json={"models": []})
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    result = http_client.unload_all_models()

    # then
    assert result == RegisteredModels(models=[])


@pytest.mark.asyncio
async def test_client_unload_all_models_async_when_successful_response_expected() -> (
    None
):
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    with aioresponses() as m:
        m.post(f"{api_url}/model/clear", payload={"models": []})

        # when
        result = await http_client.unload_all_models_async()

    # then
    assert result == RegisteredModels(models=[])


def test_client_unload_all_models_when_error_occurs(requests_mock: Mocker) -> None:
    # given
    api_url = "http://some.com"
    requests_mock.post(
        f"{api_url}/model/clear",
        json={"message": "Internal error."},
        status_code=500,
    )
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(HTTPCallErrorError):
        _ = http_client.unload_all_models()


@pytest.mark.asyncio
async def test_client_unload_all_models_async_when_error_occurs() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    with aioresponses() as m:
        m.post(
            f"{api_url}/model/clear",
            payload={"message": "Internal error."},
            status=500,
        )

        # when
        with pytest.raises(HTTPCallErrorError):
            _ = await http_client.unload_all_models_async()


def test_client_unload_single_model_in_v0_mode() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(WrongClientModeError):
        with http_client.use_api_v0():
            _ = http_client.unload_model(model_id="other/1")


@pytest.mark.asyncio
async def test_client_unload_single_model_async_in_v0_mode() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(WrongClientModeError):
        with http_client.use_api_v0():
            _ = await http_client.unload_model_async(model_id="other/1")


def test_client_unload_single_model_when_successful_response_expected(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    requests_mock.post(
        f"{api_url}/model/remove",
        json={"models": [{"model_id": "some/1", "task_type": "classification"}]},
    )
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    result = http_client.unload_model(model_id="some/1")

    # then
    assert result == RegisteredModels(
        models=[ModelDescription(model_id="some/1", task_type="classification")]
    )
    assert requests_mock.last_request.json() == {
        "model_id": "some/1",
    }


def test_client_unload_single_model_when_successful_response_expected_against_alias(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    requests_mock.post(
        f"{api_url}/model/remove",
        json={"models": [{"model_id": "some/1", "task_type": "classification"}]},
    )
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    http_client.select_model(model_id="yolov8n-640")

    # when
    result = http_client.unload_model(model_id="yolov8n-640")

    # then
    assert result == RegisteredModels(
        models=[ModelDescription(model_id="some/1", task_type="classification")]
    )
    assert requests_mock.last_request.json() == {
        "model_id": "coco/3",
    }
    assert (
        http_client.selected_model is None
    ), "Even when alias is in use - selected model should be emptied"


@pytest.mark.asyncio
async def test_client_unload_single_model_async_when_successful_response_expected() -> (
    None
):
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    with aioresponses() as m:
        m.post(
            f"{api_url}/model/remove",
            payload={"models": [{"model_id": "some/1", "task_type": "classification"}]},
        )
        # when
        result = await http_client.unload_model_async(model_id="some/1")

        # then
        assert result == RegisteredModels(
            models=[ModelDescription(model_id="some/1", task_type="classification")]
        )
        m.assert_called_with(
            url=f"{api_url}/model/remove",
            method="POST",
            json={
                "model_id": "some/1",
            },
            headers=DEFAULT_HEADERS,
        )


@pytest.mark.asyncio
async def test_client_unload_single_model_async_when_successful_response_expected_against_alias() -> (
    None
):
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    http_client.select_model(model_id="yolov8n-640")
    with aioresponses() as m:
        m.post(
            f"{api_url}/model/remove",
            payload={"models": [{"model_id": "some/1", "task_type": "classification"}]},
        )
        # when
        result = await http_client.unload_model_async(model_id="yolov8n-640")

        # then
        assert result == RegisteredModels(
            models=[ModelDescription(model_id="some/1", task_type="classification")]
        )
        m.assert_called_with(
            url=f"{api_url}/model/remove",
            method="POST",
            json={
                "model_id": "coco/3",
            },
            headers=DEFAULT_HEADERS,
        )
        assert (
            http_client.selected_model is None
        ), "Even when alias is in use - selected model should be emptied"


def test_client_unload_single_model_when_error_occurs(requests_mock: Mocker) -> None:
    # given
    api_url = "http://some.com"
    requests_mock.post(
        f"{api_url}/model/remove",
        json={"message": "Internal error."},
        status_code=500,
    )
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(HTTPCallErrorError):
        _ = http_client.unload_model(model_id="other/1")
    assert requests_mock.last_request.json() == {
        "model_id": "other/1",
    }


@pytest.mark.asyncio
async def test_client_unload_single_model_async_when_error_occurs() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    with aioresponses() as m:
        m.post(
            f"{api_url}/model/remove",
            payload={"message": "Internal error."},
            status=500,
        )
        # when
        with pytest.raises(HTTPCallErrorError):
            _ = await http_client.unload_model_async(model_id="other/1")
        m.assert_called_with(
            url=f"{api_url}/model/remove",
            method="POST",
            json={
                "model_id": "other/1",
            },
            headers=DEFAULT_HEADERS,
        )


def test_client_load_model_when_successful_response_expected(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    requests_mock.post(
        f"{api_url}/model/add",
        json={"models": [{"model_id": "some/1", "task_type": "classification"}]},
    )
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    result = http_client.load_model(model_id="some/1", set_as_default=True)

    # then
    assert result == RegisteredModels(
        models=[ModelDescription(model_id="some/1", task_type="classification")]
    )
    assert http_client.selected_model == "some/1"
    assert requests_mock.last_request.json() == {
        "model_id": "some/1",
        "api_key": "my-api-key",
    }


def test_client_load_model_when_successful_response_expected_against_alias(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    requests_mock.post(
        f"{api_url}/model/add",
        json={"models": [{"model_id": "coco/3", "task_type": "object-detection"}]},
    )
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    result = http_client.load_model(model_id="yolov8n-640", set_as_default=True)

    # then
    assert result == RegisteredModels(
        models=[ModelDescription(model_id="coco/3", task_type="object-detection")]
    )
    assert http_client.selected_model == "coco/3"
    assert requests_mock.last_request.json() == {
        "model_id": "coco/3",
        "api_key": "my-api-key",
    }


@pytest.mark.asyncio
async def test_client_load_model_async_when_successful_response_expected() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    with aioresponses() as m:
        m.post(
            f"{api_url}/model/add",
            payload={"models": [{"model_id": "some/1", "task_type": "classification"}]},
        )

        # when
        result = await http_client.load_model_async(
            model_id="some/1", set_as_default=True
        )

        # then
        assert result == RegisteredModels(
            models=[ModelDescription(model_id="some/1", task_type="classification")]
        )
        assert http_client.selected_model == "some/1"
        m.assert_called_with(
            url=f"{api_url}/model/add",
            method="POST",
            json={"model_id": "some/1", "api_key": "my-api-key"},
            headers=DEFAULT_HEADERS,
        )


@pytest.mark.asyncio
async def test_client_load_model_async_when_successful_response_expected_against_alias() -> (
    None
):
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    with aioresponses() as m:
        m.post(
            f"{api_url}/model/add",
            payload={
                "models": [{"model_id": "coco/3", "task_type": "object-detection"}]
            },
        )

        # when
        result = await http_client.load_model_async(
            model_id="yolov8n-640", set_as_default=True
        )

        # then
        assert result == RegisteredModels(
            models=[ModelDescription(model_id="coco/3", task_type="object-detection")]
        )
        assert http_client.selected_model == "coco/3"
        m.assert_called_with(
            url=f"{api_url}/model/add",
            method="POST",
            json={"model_id": "coco/3", "api_key": "my-api-key"},
            headers=DEFAULT_HEADERS,
        )


def test_client_load_model_in_v0_mode() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(WrongClientModeError):
        with http_client.use_api_v0():
            _ = http_client.load_model(model_id="other/1")


@pytest.mark.asyncio
async def test_client_load_model_async_in_v0_mode() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(WrongClientModeError):
        with http_client.use_api_v0():
            _ = await http_client.load_model_async(model_id="other/1")


def test_client_load_model_when_unsuccessful_response_expected(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    requests_mock.post(
        f"{api_url}/model/add",
        json={"message": "Internal error."},
        status_code=500,
    )
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(HTTPCallErrorError):
        _ = http_client.load_model(model_id="some/1", set_as_default=True)

    # then
    assert http_client.selected_model is None
    assert requests_mock.last_request.json() == {
        "model_id": "some/1",
        "api_key": "my-api-key",
    }


@pytest.mark.asyncio
async def test_client_load_model_async_when_unsuccessful_response_expected() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    with aioresponses() as m:
        m.post(
            f"{api_url}/model/add",
            payload={"message": "Internal error."},
            status=500,
        )

        # when
        with pytest.raises(HTTPCallErrorError):
            _ = await http_client.load_model_async(
                model_id="some/1", set_as_default=True
            )

        # then
        assert http_client.selected_model is None
        m.assert_called_with(
            url=f"{api_url}/model/add",
            method="POST",
            json={"model_id": "some/1", "api_key": "my-api-key"},
            headers=DEFAULT_HEADERS,
        )


def test_list_loaded_models_in_v0_mode() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(WrongClientModeError):
        with http_client.use_api_v0():
            _ = http_client.list_loaded_models()


@pytest.mark.asyncio
async def test_list_loaded_models_async_in_v0_mode() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(WrongClientModeError):
        with http_client.use_api_v0():
            _ = await http_client.list_loaded_models_async()


def test_list_loaded_models_when_successful_response_expected(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    requests_mock.get(
        f"{api_url}/model/registry",
        json={
            "models": [
                {
                    "model_id": "some/1",
                    "task_type": "classification",
                    "batch_size": "batch",
                }
            ]
        },
    )
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    result = http_client.list_loaded_models()

    # then
    assert result == RegisteredModels(
        models=[
            ModelDescription(
                model_id="some/1", task_type="classification", batch_size="batch"
            )
        ]
    )


@pytest.mark.asyncio
async def test_list_loaded_models_async_when_successful_response_expected() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    with aioresponses() as m:
        m.get(
            f"{api_url}/model/registry?api_key=my-api-key",
            payload={
                "models": [
                    {
                        "model_id": "some/1",
                        "task_type": "classification",
                        "batch_size": "batch",
                    }
                ]
            },
        )

        # when
        result = await http_client.list_loaded_models_async()

        # then
        assert result == RegisteredModels(
            models=[
                ModelDescription(
                    model_id="some/1", task_type="classification", batch_size="batch"
                )
            ]
        )


def test_list_loaded_models_when_unsuccessful_response_expected(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    requests_mock.get(
        f"{api_url}/model/registry",
        json={"message": "Internal error."},
        status_code=500,
    )
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(HTTPCallErrorError):
        _ = http_client.list_loaded_models()


@pytest.mark.asyncio
async def test_list_loaded_models_when_unsuccessful_response_expected() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    with aioresponses() as m:
        m.get(
            f"{api_url}/model/registry?api_key=my-api-key",
            payload={"message": "Internal error."},
            status=500,
        )
        # when
        with pytest.raises(HTTPCallErrorError):
            _ = await http_client.list_loaded_models_async()


def test_get_model_description_in_v0_mode() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(WrongClientModeError):
        with http_client.use_api_v0():
            _ = http_client.get_model_description(model_id="some/1")


@pytest.mark.asyncio
async def test_get_model_description_async_in_v0_mode() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(WrongClientModeError):
        with http_client.use_api_v0():
            _ = await http_client.get_model_description_async(model_id="some/1")


def test_get_model_description_when_model_when_error_occurs_in_model_listing(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    requests_mock.get(
        f"{api_url}/model/registry",
        json={"message": "Internal error."},
        status_code=500,
    )
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(HTTPCallErrorError):
        _ = http_client.get_model_description(model_id="some/1")


@pytest.mark.asyncio
async def test_get_model_description_async_when_model_when_error_occurs_in_model_listing() -> (
    None
):
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    with aioresponses() as m:
        m.get(
            f"{api_url}/model/registry?api_key=my-api-key",
            payload={"message": "Internal error."},
            status=500,
        )
        # when
        with pytest.raises(HTTPCallErrorError):
            _ = await http_client.get_model_description_async(model_id="some/1")


def test_get_model_description_when_model_was_loaded_already(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    requests_mock.get(
        f"{api_url}/model/registry",
        json={"models": [{"model_id": "some/1", "task_type": "classification"}]},
    )
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    result = http_client.get_model_description(model_id="some/1")

    # then
    assert result == ModelDescription(model_id="some/1", task_type="classification")


def test_get_model_description_when_model_was_loaded_already_and_alias_was_resolved(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    requests_mock.get(
        f"{api_url}/model/registry",
        json={"models": [{"model_id": "coco/3", "task_type": "object-detection"}]},
    )
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    result = http_client.get_model_description(model_id="yolov8n-640")

    # then
    assert result == ModelDescription(model_id="coco/3", task_type="object-detection")


@pytest.mark.asyncio
async def test_get_model_description_async_when_model_was_loaded_already() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    with aioresponses() as m:
        m.get(
            f"{api_url}/model/registry?api_key=my-api-key",
            payload={"models": [{"model_id": "some/1", "task_type": "classification"}]},
        )
        # when
        result = await http_client.get_model_description_async(model_id="some/1")

        # then
        assert result == ModelDescription(model_id="some/1", task_type="classification")


@pytest.mark.asyncio
async def test_get_model_description_async_when_model_was_loaded_already_and_alias_was_resolved() -> (
    None
):
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    with aioresponses() as m:
        m.get(
            f"{api_url}/model/registry?api_key=my-api-key",
            payload={
                "models": [{"model_id": "coco/3", "task_type": "object-detection"}]
            },
        )
        # when
        result = await http_client.get_model_description_async(model_id="yolov8n-640")

        # then
        assert result == ModelDescription(
            model_id="coco/3", task_type="object-detection"
        )


def test_get_model_description_when_model_was_not_loaded_before_and_successful_load(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    requests_mock.get(
        f"{api_url}/model/registry",
        [
            {"json": {"models": []}},
        ],
    )
    requests_mock.post(
        f"{api_url}/model/add",
        json={"models": [{"model_id": "some/1", "task_type": "classification"}]},
    )
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    result = http_client.get_model_description(model_id="some/1")

    # then
    assert result == ModelDescription(model_id="some/1", task_type="classification")


def test_get_model_description_when_model_was_not_loaded_before_and_successful_load_with_alias_resolution(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    requests_mock.get(
        f"{api_url}/model/registry",
        [
            {"json": {"models": []}},
        ],
    )
    requests_mock.post(
        f"{api_url}/model/add",
        json={"models": [{"model_id": "coco/3", "task_type": "object-detection"}]},
    )
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    result = http_client.get_model_description(model_id="yolov8n-640")

    # then
    assert result == ModelDescription(model_id="coco/3", task_type="object-detection")


@pytest.mark.asyncio
async def test_get_model_description_async_when_model_was_not_loaded_before_and_successful_load() -> (
    None
):
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    with aioresponses() as m:
        m.get(
            f"{api_url}/model/registry?api_key=my-api-key",
            payload={"models": []},
        )
        m.post(
            f"{api_url}/model/add",
            payload={"models": [{"model_id": "some/1", "task_type": "classification"}]},
        )

        # when
        result = await http_client.get_model_description_async(model_id="some/1")

        # then
        assert result == ModelDescription(model_id="some/1", task_type="classification")


@pytest.mark.asyncio
async def test_get_model_description_async_when_model_was_not_loaded_before_and_successful_load_with_alias_resolution() -> (
    None
):
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    with aioresponses() as m:
        m.get(
            f"{api_url}/model/registry?api_key=my-api-key",
            payload={"models": []},
        )
        m.post(
            f"{api_url}/model/add",
            payload={
                "models": [{"model_id": "coco/3", "task_type": "object-detection"}]
            },
        )

        # when
        result = await http_client.get_model_description_async(model_id="yolov8n-640")

        # then
        assert result == ModelDescription(
            model_id="coco/3", task_type="object-detection"
        )


def test_get_model_description_when_model_was_not_loaded_before_and_unsuccessful_load(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    requests_mock.get(
        f"{api_url}/model/registry",
        [
            {"json": {"models": []}},
        ],
    )
    requests_mock.post(
        f"{api_url}/model/add",
        json={"message": "Internal error."},
        status_code=500,
    )
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(HTTPCallErrorError):
        _ = http_client.get_model_description(model_id="some/1")


@pytest.mark.asyncio
async def test_get_model_description_async_when_model_was_not_loaded_before_and_unsuccessful_load() -> (
    None
):
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    with aioresponses() as m:
        m.get(
            f"{api_url}/model/registry?api_key=my-api-key",
            payload={"models": []},
        )
        m.post(
            f"{api_url}/model/add",
            payload={"message": "Internal error."},
            status=500,
        )

        # when
        with pytest.raises(HTTPCallErrorError):
            _ = await http_client.get_model_description_async(model_id="some/1")


def test_infer_from_api_v0_when_model_not_selected() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(ModelNotSelectedError):
        _ = http_client.infer_from_api_v0(inference_input="https://some/image.jpg")


@pytest.mark.asyncio
async def test_infer_from_api_v0_async_when_model_not_selected() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(ModelNotSelectedError):
        _ = await http_client.infer_from_api_v0_async(
            inference_input="https://some/image.jpg"
        )


def test_infer_from_api_v0_when_model_id_is_invalid() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(InvalidModelIdentifier):
        _ = http_client.infer_from_api_v0(
            inference_input="https://some/image.jpg",
            model_id="invalid",
        )


@pytest.mark.asyncio
async def test_infer_from_api_v0_async_when_model_id_is_invalid() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(InvalidModelIdentifier):
        _ = await http_client.infer_from_api_v0_async(
            inference_input="https://some/image.jpg",
            model_id="invalid",
        )


@mock.patch.object(client, "load_static_inference_input")
@pytest.mark.parametrize("model_id_to_use", ["coco/3", "yolov8n-640"])
def test_infer_from_api_v0_when_request_succeed_for_object_detection_with_batch_request(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
    model_id_to_use: str,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.side_effect = [
        [
            ("base64_image", 0.5),
            ("another_image", None),
        ]
    ]
    configuration = InferenceConfiguration(confidence_threshold=0.5)
    http_client.configure(inference_configuration=configuration)
    requests_mock.post(
        f"{api_url}/coco/3",
        [
            {
                "json": {
                    "image": {"height": 480, "width": 640},
                    "predictions": [
                        {
                            "x": 100.0,
                            "y": 200.0,
                            "width": 200.0,
                            "height": 300.0,
                            "confidence": 0.9,
                            "class": "A",
                        }
                    ],
                },
            },
            {
                "json": {
                    "image": {"height": 480, "width": 640},
                    "predictions": [
                        {
                            "x": 100.0,
                            "y": 200.0,
                            "width": 200.0,
                            "height": 300.0,
                            "confidence": 0.9,
                            "class": "B",
                        }
                    ],
                },
            },
        ],
    )

    # when
    result = http_client.infer_from_api_v0(
        inference_input=["https://some/image.jpg"] * 2,
        model_id=model_id_to_use,
    )

    # then
    assert result == [
        {
            "image": {"height": 960, "width": 1280},
            "predictions": [
                {
                    "x": 200.0,
                    "y": 400.0,
                    "width": 400.0,
                    "height": 600.0,
                    "confidence": 0.9,
                    "class": "A",
                }
            ],
        },
        {
            "image": {"height": 480, "width": 640},
            "predictions": [
                {
                    "x": 100.0,
                    "y": 200.0,
                    "width": 200.0,
                    "height": 300.0,
                    "confidence": 0.9,
                    "class": "B",
                }
            ],
        },
    ]
    assert (
        requests_mock.request_history[0].query
        == "api_key=my-api-key&confidence=0.5&disable_active_learning=false"
    )
    assert (
        requests_mock.request_history[1].query
        == "api_key=my-api-key&confidence=0.5&disable_active_learning=false"
    )


@pytest.mark.asyncio
@mock.patch.object(client, "load_static_inference_input_async")
@pytest.mark.parametrize("model_id_to_use", ["coco/3", "yolov8n-640"])
async def test_infer_from_api_v0_async_when_request_succeed_for_object_detection_with_batch_request(
    load_static_inference_input_mock_async: AsyncMock,
    model_id_to_use: str,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock_async.return_value = [
        ("base64_image", 0.5),
        ("another_image", 0.5),
    ]
    configuration = InferenceConfiguration(confidence_threshold=0.5)
    http_client.configure(inference_configuration=configuration)

    with aioresponses() as m:
        m.post(
            f"{api_url}/coco/3?api_key=my-api-key&confidence=0.5&disable_active_learning=False",
            payload={
                "image": {"height": 480, "width": 640},
                "predictions": [
                    {
                        "x": 100.0,
                        "y": 200.0,
                        "width": 200.0,
                        "height": 300.0,
                        "confidence": 0.9,
                        "class": "A",
                    }
                ],
            },
        )
        m.post(
            f"{api_url}/coco/3?api_key=my-api-key&confidence=0.5&disable_active_learning=False",
            payload={
                "image": {"height": 480, "width": 640},
                "predictions": [
                    {
                        "x": 100.0,
                        "y": 200.0,
                        "width": 200.0,
                        "height": 300.0,
                        "confidence": 0.9,
                        "class": "A",
                    }
                ],
            },
        )

        # when
        result = await http_client.infer_from_api_v0_async(
            inference_input="https://some/image.jpg",
            model_id=model_id_to_use,
        )
        # then
        assert result == [
            {
                "image": {"height": 960, "width": 1280},
                "predictions": [
                    {
                        "x": 200.0,
                        "y": 400.0,
                        "width": 400.0,
                        "height": 600.0,
                        "confidence": 0.9,
                        "class": "A",
                    }
                ],
            },
            {
                "image": {"height": 960, "width": 1280},
                "predictions": [
                    {
                        "x": 200.0,
                        "y": 400.0,
                        "width": 400.0,
                        "height": 600.0,
                        "confidence": 0.9,
                        "class": "A",
                    }
                ],
            },
        ]


@mock.patch.object(client, "load_static_inference_input")
def test_infer_from_api_v0_when_request_succeed_for_object_detection_with_visualisation_and_json(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.return_value = [("base64_image", 0.5)]
    configuration = InferenceConfiguration(
        confidence_threshold=0.5, format="image_and_json"
    )
    http_client.configure(inference_configuration=configuration)
    requests_mock.post(
        f"{api_url}/some/1",
        json={
            "image": {"height": 480, "width": 640},
            "predictions": [
                {
                    "x": 100.0,
                    "y": 200.0,
                    "width": 200.0,
                    "height": 300.0,
                    "confidence": 0.9,
                    "class": "A",
                }
            ],
            "visualization": "aGVsbG8=",
        },
        headers={"content-type": "application/json"},
    )

    # when
    result = http_client.infer_from_api_v0(
        inference_input="https://some/image.jpg", model_id="some/1"
    )

    # then
    assert result == {
        "visualization": "aGVsbG8=",
        "image": {"height": 960, "width": 1280},
        "predictions": [
            {
                "x": 200.0,
                "y": 400.0,
                "width": 400.0,
                "height": 600.0,
                "confidence": 0.9,
                "class": "A",
            }
        ],
    }
    assert (
        requests_mock.last_request.query
        == "api_key=my-api-key&confidence=0.5&format=image_and_json&disable_active_learning=false"
    )


@mock.patch.object(client, "load_static_inference_input")
def test_infer_from_api_v0_when_request_succeed_for_object_detection_with_visualisation(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.return_value = [("base64_image", 0.5)]
    configuration = InferenceConfiguration(confidence_threshold=0.5, format="image")
    http_client.configure(inference_configuration=configuration)
    requests_mock.post(
        f"{api_url}/some/1",
        body=BytesIO(b"data"),
        headers={"content-type": "image/jpeg"},
    )

    # when
    result = http_client.infer_from_api_v0(
        inference_input="https://some/image.jpg", model_id="some/1"
    )

    # then
    assert result == {"visualization": base64.b64encode(b"data").decode("utf-8")}
    assert (
        requests_mock.last_request.query
        == "api_key=my-api-key&confidence=0.5&format=image&disable_active_learning=false"
    )


@pytest.mark.asyncio
@mock.patch.object(client, "load_static_inference_input_async")
async def test_infer_from_api_v0_async_when_request_succeed_for_object_detection_with_visualisation(
    load_static_inference_input_async_mock: AsyncMock,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_async_mock.return_value = [("base64_image", 0.5)]
    configuration = InferenceConfiguration(confidence_threshold=0.5, format="image")
    http_client.configure(inference_configuration=configuration)
    with aioresponses() as m:
        m.post(
            f"{api_url}/some/1?api_key=my-api-key&confidence=0.5&disable_active_learning=False&format=image",
            body=b"data",
            headers={"content-type": "image/jpeg"},
        )
        # when
        result = await http_client.infer_from_api_v0_async(
            inference_input="https://some/image.jpg", model_id="some/1"
        )

        # then
        assert result == {"visualization": base64.b64encode(b"data").decode("utf-8")}


@mock.patch.object(client, "load_static_inference_input")
def test_infer_from_api_v0_when_request_succeed_for_object_detection(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.return_value = [("base64_image", 0.5)]
    configuration = InferenceConfiguration(confidence_threshold=0.5)
    http_client.configure(inference_configuration=configuration)
    requests_mock.post(
        f"{api_url}/some/1",
        json={
            "image": {"height": 480, "width": 640},
            "predictions": [
                {
                    "x": 100.0,
                    "y": 200.0,
                    "width": 200.0,
                    "height": 300.0,
                    "confidence": 0.9,
                    "class": "A",
                }
            ],
        },
    )

    # when
    result = http_client.infer_from_api_v0(
        inference_input="https://some/image.jpg", model_id="some/1"
    )

    # then
    assert result == {
        "image": {"height": 960, "width": 1280},
        "predictions": [
            {
                "x": 200.0,
                "y": 400.0,
                "width": 400.0,
                "height": 600.0,
                "confidence": 0.9,
                "class": "A",
            }
        ],
    }
    assert (
        requests_mock.last_request.query
        == "api_key=my-api-key&confidence=0.5&disable_active_learning=false"
    )


@pytest.mark.asyncio
@mock.patch.object(client, "load_static_inference_input_async")
async def test_infer_from_api_v0_async_when_request_succeed_for_object_detection(
    load_static_inference_input_mock_async: AsyncMock,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock_async.return_value = [("base64_image", 0.5)]
    configuration = InferenceConfiguration(confidence_threshold=0.5)
    http_client.configure(inference_configuration=configuration)

    with aioresponses() as m:
        m.post(
            f"{api_url}/some/1?api_key=my-api-key&confidence=0.5&disable_active_learning=False",
            payload={
                "image": {"height": 480, "width": 640},
                "predictions": [
                    {
                        "x": 100.0,
                        "y": 200.0,
                        "width": 200.0,
                        "height": 300.0,
                        "confidence": 0.9,
                        "class": "A",
                    }
                ],
            },
        )

        # when
        result = await http_client.infer_from_api_v0_async(
            inference_input="https://some/image.jpg", model_id="some/1"
        )

        # then
        assert result == {
            "image": {"height": 960, "width": 1280},
            "predictions": [
                {
                    "x": 200.0,
                    "y": 400.0,
                    "width": 400.0,
                    "height": 600.0,
                    "confidence": 0.9,
                    "class": "A",
                }
            ],
        }


def test_infer_from_api_v1_when_model_id_is_not_selected() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(ModelNotSelectedError):
        _ = http_client.infer_from_api_v1(
            inference_input="https://some/image.jpg",
        )


@pytest.mark.asyncio
async def test_infer_from_api_v1_async_when_model_id_is_not_selected() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(ModelNotSelectedError):
        _ = await http_client.infer_from_api_v1_async(
            inference_input="https://some/image.jpg",
        )


def test_infer_from_api_v1_when_v0_mode_enabled() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(WrongClientModeError):
        with http_client.use_api_v0():
            _ = http_client.infer_from_api_v1(
                inference_input="https://some/image.jpg", model_id="some/1"
            )


@pytest.mark.asyncio
async def test_infer_from_api_v1_async_when_v0_mode_enabled() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(WrongClientModeError):
        with http_client.use_api_v0():
            _ = await http_client.infer_from_api_v1_async(
                inference_input="https://some/image.jpg", model_id="some/1"
            )


def test_infer_from_api_v1_when_task_type_is_not_recognised() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    http_client.get_model_description = MagicMock()
    http_client.get_model_description.return_value = ModelDescription(
        model_id="some/1",
        task_type="unknown",
        input_height=480,
        input_width=640,
    )
    # when
    with pytest.raises(ModelTaskTypeNotSupportedError):
        _ = http_client.infer_from_api_v1(
            inference_input="https://some/image.jpg", model_id="some/1"
        )


@pytest.mark.asyncio
async def test_infer_from_api_v1_async_when_task_type_is_not_recognised() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    http_client.get_model_description_async = AsyncMock()
    http_client.get_model_description_async.return_value = ModelDescription(
        model_id="some/1",
        task_type="unknown",
        input_height=480,
        input_width=640,
    )
    # when
    with pytest.raises(ModelTaskTypeNotSupportedError):
        _ = await http_client.infer_from_api_v1_async(
            inference_input="https://some/image.jpg", model_id="some/1"
        )


@mock.patch.object(client, "load_static_inference_input")
@pytest.mark.parametrize("model_id_to_use", ["coco/3", "yolov8n-640"])
def test_infer_from_api_v1_when_request_succeed_for_object_detection_with_batch_request(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
    model_id_to_use: str,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    http_client.get_model_description = MagicMock()
    http_client.get_model_description.return_value = ModelDescription(
        model_id="coco/3",
        task_type="object-detection",
        input_height=480,
        input_width=640,
    )
    load_static_inference_input_mock.return_value = [
        ("base64_image", 0.5),
        ("another_image", None),
    ]
    configuration = InferenceConfiguration(
        confidence_threshold=0.5, disable_active_learning=True
    )
    http_client.configure(inference_configuration=configuration)
    requests_mock.post(
        f"{api_url}/infer/object_detection",
        [
            {
                "json": {
                    "image": {"height": 480, "width": 640},
                    "predictions": [
                        {
                            "x": 100.0,
                            "y": 200.0,
                            "width": 200.0,
                            "height": 300.0,
                            "confidence": 0.9,
                            "class": "A",
                        }
                    ],
                    "visualization": None,
                },
            },
            {
                "json": {
                    "image": {"height": 480, "width": 640},
                    "predictions": [
                        {
                            "x": 100.0,
                            "y": 200.0,
                            "width": 200.0,
                            "height": 300.0,
                            "confidence": 0.9,
                            "class": "B",
                        }
                    ],
                    "visualization": None,
                },
            },
        ],
    )

    # when
    result = http_client.infer_from_api_v1(
        inference_input="https://some/image.jpg",
        model_id=model_id_to_use,
    )
    # then
    assert result == [
        {
            "image": {"height": 960, "width": 1280},
            "predictions": [
                {
                    "x": 200.0,
                    "y": 400.0,
                    "width": 400.0,
                    "height": 600.0,
                    "confidence": 0.9,
                    "class": "A",
                }
            ],
            "visualization": None,
        },
        {
            "image": {"height": 480, "width": 640},
            "predictions": [
                {
                    "x": 100.0,
                    "y": 200.0,
                    "width": 200.0,
                    "height": 300.0,
                    "confidence": 0.9,
                    "class": "B",
                }
            ],
            "visualization": None,
        },
    ]
    assert requests_mock.request_history[0].json() == {
        "model_id": "coco/3",
        "api_key": "my-api-key",
        "image": {"type": "base64", "value": "base64_image"},
        "visualize_predictions": False,
        "confidence": 0.5,
        "disable_active_learning": True,
    }
    assert requests_mock.request_history[1].json() == {
        "model_id": "coco/3",
        "api_key": "my-api-key",
        "image": {"type": "base64", "value": "another_image"},
        "visualize_predictions": False,
        "confidence": 0.5,
        "disable_active_learning": True,
    }


@pytest.mark.asyncio
@mock.patch.object(client, "load_static_inference_input_async")
@pytest.mark.parametrize("model_id_to_use", ["coco/3", "yolov8n-640"])
async def test_infer_from_api_v1_async_when_request_succeed_for_object_detection_with_batch_request(
    load_static_inference_input_async_mock: MagicMock,
    model_id_to_use: str,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    http_client.get_model_description_async = AsyncMock()
    http_client.get_model_description_async.return_value = ModelDescription(
        model_id="coco/3",
        task_type="object-detection",
        input_height=480,
        input_width=640,
    )
    load_static_inference_input_async_mock.return_value = [
        ("base64_image", 0.5),
        ("another_image", 0.5),
    ]
    configuration = InferenceConfiguration(
        confidence_threshold=0.5, disable_active_learning=True
    )
    http_client.configure(inference_configuration=configuration)

    with aioresponses() as m:
        m.post(
            f"{api_url}/infer/object_detection",
            payload={
                "image": {"height": 480, "width": 640},
                "predictions": [
                    {
                        "x": 100.0,
                        "y": 200.0,
                        "width": 200.0,
                        "height": 300.0,
                        "confidence": 0.9,
                        "class": "A",
                    }
                ],
                "visualization": None,
            },
        )
        m.post(
            f"{api_url}/infer/object_detection",
            payload={
                "image": {"height": 480, "width": 640},
                "predictions": [
                    {
                        "x": 100.0,
                        "y": 200.0,
                        "width": 200.0,
                        "height": 300.0,
                        "confidence": 0.9,
                        "class": "A",
                    }
                ],
                "visualization": None,
            },
        )

        # when
        result = await http_client.infer_from_api_v1_async(
            inference_input="https://some/image.jpg",
            model_id=model_id_to_use,
        )
        # then
        assert result == [
            {
                "image": {"height": 960, "width": 1280},
                "predictions": [
                    {
                        "x": 200.0,
                        "y": 400.0,
                        "width": 400.0,
                        "height": 600.0,
                        "confidence": 0.9,
                        "class": "A",
                    }
                ],
                "visualization": None,
            },
            {
                "image": {"height": 960, "width": 1280},
                "predictions": [
                    {
                        "x": 200.0,
                        "y": 400.0,
                        "width": 400.0,
                        "height": 600.0,
                        "confidence": 0.9,
                        "class": "A",
                    }
                ],
                "visualization": None,
            },
        ]


@mock.patch.object(client, "load_static_inference_input")
def test_infer_from_api_v1_when_request_succeed_for_object_detection_with_visualisation(
    load_static_inference_input_mock: AsyncMock,
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    http_client.get_model_description = MagicMock()
    http_client.get_model_description.return_value = ModelDescription(
        model_id="some/1",
        task_type="object-detection",
        input_height=480,
        input_width=640,
    )
    load_static_inference_input_mock.return_value = [("base64_image", 0.5)]
    configuration = InferenceConfiguration(
        confidence_threshold=0.5, visualize_predictions=True
    )
    http_client.configure(inference_configuration=configuration)
    requests_mock.post(
        f"{api_url}/infer/object_detection",
        json={
            "image": {"height": 480, "width": 640},
            "predictions": [
                {
                    "x": 100.0,
                    "y": 200.0,
                    "width": 200.0,
                    "height": 300.0,
                    "confidence": 0.9,
                    "class": "A",
                }
            ],
            "visualization": "aGVsbG8=",
        },
    )

    # when
    result = http_client.infer_from_api_v1(
        inference_input="https://some/image.jpg", model_id="some/1"
    )

    # then
    assert result == {
        "image": {"height": 960, "width": 1280},
        "predictions": [
            {
                "x": 200.0,
                "y": 400.0,
                "width": 400.0,
                "height": 600.0,
                "confidence": 0.9,
                "class": "A",
            },
        ],
        "visualization": "aGVsbG8=",
    }
    assert requests_mock.request_history[0].json() == {
        "model_id": "some/1",
        "api_key": "my-api-key",
        "image": {"type": "base64", "value": "base64_image"},
        "visualize_predictions": True,
        "confidence": 0.5,
        "disable_active_learning": False,
    }


@pytest.mark.asyncio
@mock.patch.object(client, "load_static_inference_input_async")
async def test_infer_from_api_v1_async_when_request_succeed_for_object_detection_with_visualisation(
    load_static_inference_input_async_mock: AsyncMock,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    http_client.get_model_description_async = AsyncMock()
    http_client.get_model_description_async.return_value = ModelDescription(
        model_id="some/1",
        task_type="object-detection",
        input_height=480,
        input_width=640,
    )
    load_static_inference_input_async_mock.return_value = [("base64_image", 0.5)]
    configuration = InferenceConfiguration(
        confidence_threshold=0.5, visualize_predictions=True
    )
    http_client.configure(inference_configuration=configuration)
    with aioresponses() as m:
        m.post(
            f"{api_url}/infer/object_detection",
            payload={
                "image": {"height": 480, "width": 640},
                "predictions": [
                    {
                        "x": 100.0,
                        "y": 200.0,
                        "width": 200.0,
                        "height": 300.0,
                        "confidence": 0.9,
                        "class": "A",
                    }
                ],
                "visualization": "aGVsbG8=",
            },
        )

        # when
        result = await http_client.infer_from_api_v1_async(
            inference_input="https://some/image.jpg", model_id="some/1"
        )

        # then
        assert result == {
            "image": {"height": 960, "width": 1280},
            "predictions": [
                {
                    "x": 200.0,
                    "y": 400.0,
                    "width": 400.0,
                    "height": 600.0,
                    "confidence": 0.9,
                    "class": "A",
                },
            ],
            "visualization": "aGVsbG8=",
        }


@mock.patch.object(client, "load_static_inference_input")
def test_ocr_image_when_single_image_given_in_v1_mode(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.return_value = [("base64_image", 0.5)]
    requests_mock.post(
        f"{api_url}/doctr/ocr",
        json={
            "response": "Image text 1.",
            "time": 0.33,
        },
    )

    # when
    result = http_client.ocr_image(inference_input="/some/image.jpg")

    # then
    assert result == {
        "response": "Image text 1.",
        "time": 0.33,
    }, "Result must match the value returned by HTTP endpoint"
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key",
        "image": {"type": "base64", "value": "base64_image"},
    }, "Request must contain API key and image encoded in standard format"


@mock.patch.object(client, "load_static_inference_input")
def test_ocr_image_when_trocr_selected(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.return_value = [("base64_image", 0.5)]
    requests_mock.post(
        f"{api_url}/ocr/trocr",
        json={
            "response": "Image text 1.",
            "time": 0.33,
        },
    )

    # when
    result = http_client.ocr_image(inference_input="/some/image.jpg", model="trocr")

    # then
    assert result == {
        "response": "Image text 1.",
        "time": 0.33,
    }, "Result must match the value returned by HTTP endpoint"
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key",
        "image": {"type": "base64", "value": "base64_image"},
    }, "Request must contain API key and image encoded in standard format"


@mock.patch.object(client, "load_static_inference_input")
def test_ocr_image_when_trocr_selected_in_specific_variant(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.return_value = [("base64_image", 0.5)]
    requests_mock.post(
        f"{api_url}/ocr/trocr",
        json={
            "response": "Image text 1.",
            "time": 0.33,
        },
    )

    # when
    result = http_client.ocr_image(
        inference_input="/some/image.jpg", model="trocr", version="trocr-small-printed"
    )

    # then
    assert result == {
        "response": "Image text 1.",
        "time": 0.33,
    }, "Result must match the value returned by HTTP endpoint"
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key",
        "image": {"type": "base64", "value": "base64_image"},
        "trocr_version_id": "trocr-small-printed",
    }, "Request must contain API key and image encoded in standard format"


@pytest.mark.asyncio
@mock.patch.object(client, "load_static_inference_input_async")
async def test_ocr_image_async_when_single_image_given_in_v1_mode(
    load_static_inference_input_async_mock: AsyncMock,
) -> None:
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_async_mock.return_value = [("base64_image", 0.5)]

    with aioresponses() as m:
        m.post(
            f"{api_url}/doctr/ocr",
            payload={
                "response": "Image text 1.",
                "time": 0.33,
            },
        )
        # when
        result = await http_client.ocr_image_async(inference_input="/some/image.jpg")

        # then
        assert result == {
            "response": "Image text 1.",
            "time": 0.33,
        }, "Result must match the value returned by HTTP endpoint"
        m.assert_called_with(
            f"{api_url}/doctr/ocr",
            "POST",
            json={
                "api_key": "my-api-key",
                "image": {"type": "base64", "value": "base64_image"},
            },
            params=None,
            data=None,
            headers={"Content-Type": "application/json"},
        )


@pytest.mark.asyncio
@mock.patch.object(client, "load_static_inference_input_async")
async def test_ocr_image_async_when_trocr_selected(
    load_static_inference_input_async_mock: AsyncMock,
) -> None:
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_async_mock.return_value = [("base64_image", 0.5)]

    with aioresponses() as m:
        m.post(
            f"{api_url}/ocr/trocr",
            payload={
                "response": "Image text 1.",
                "time": 0.33,
            },
        )
        # when
        result = await http_client.ocr_image_async(
            inference_input="/some/image.jpg", model="trocr"
        )

        # then
        assert result == {
            "response": "Image text 1.",
            "time": 0.33,
        }, "Result must match the value returned by HTTP endpoint"
        m.assert_called_with(
            f"{api_url}/ocr/trocr",
            "POST",
            json={
                "api_key": "my-api-key",
                "image": {"type": "base64", "value": "base64_image"},
            },
            params=None,
            data=None,
            headers={"Content-Type": "application/json"},
        )


@pytest.mark.asyncio
@mock.patch.object(client, "load_static_inference_input_async")
async def test_ocr_image_async_when_trocr_selected_in_specific_variant(
    load_static_inference_input_async_mock: AsyncMock,
) -> None:
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_async_mock.return_value = [("base64_image", 0.5)]

    with aioresponses() as m:
        m.post(
            f"{api_url}/ocr/trocr",
            payload={
                "response": "Image text 1.",
                "time": 0.33,
            },
        )
        # when
        result = await http_client.ocr_image_async(
            inference_input="/some/image.jpg",
            model="trocr",
            version="trocr-small-printed",
        )

        # then
        assert result == {
            "response": "Image text 1.",
            "time": 0.33,
        }, "Result must match the value returned by HTTP endpoint"
        m.assert_called_with(
            f"{api_url}/ocr/trocr",
            "POST",
            json={
                "api_key": "my-api-key",
                "image": {"type": "base64", "value": "base64_image"},
                "trocr_version_id": "trocr-small-printed",
            },
            params=None,
            data=None,
            headers={"Content-Type": "application/json"},
        )


@mock.patch.object(client, "load_static_inference_input")
def test_ocr_image_when_single_image_given_in_v0_mode(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    api_url = "https://infer.roboflow.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.return_value = [("base64_image", 0.5)]
    requests_mock.post(
        f"{api_url}/doctr/ocr?api_key=my-api-key",
        json={
            "response": "Image text 1.",
            "time": 0.33,
        },
    )

    # when
    result = http_client.ocr_image(inference_input="/some/image.jpg")

    # then
    assert result == {
        "response": "Image text 1.",
        "time": 0.33,
    }, "Result must match the value returned by HTTP endpoint"
    assert requests_mock.request_history[0].json() == {
        "image": {"type": "base64", "value": "base64_image"},
    }, "Request must image encoded in standard format"


@pytest.mark.asyncio
@mock.patch.object(client, "load_static_inference_input_async")
async def test_ocr_image_async_when_single_image_given_in_v0_mode(
    load_static_inference_input_async_mock: AsyncMock,
) -> None:
    api_url = "https://infer.roboflow.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_async_mock.return_value = [("base64_image", 0.5)]

    with aioresponses() as m:
        m.post(
            f"{api_url}/doctr/ocr?api_key=my-api-key",
            payload={
                "response": "Image text 1.",
                "time": 0.33,
            },
        )

        # when
        result = await http_client.ocr_image_async(inference_input="/some/image.jpg")

        # then
        assert result == {
            "response": "Image text 1.",
            "time": 0.33,
        }, "Result must match the value returned by HTTP endpoint"
        m.assert_called_with(
            f"{api_url}/doctr/ocr?api_key=my-api-key",
            "POST",
            json={
                "image": {"type": "base64", "value": "base64_image"},
            },
            params=None,
            data=None,
            headers={"Content-Type": "application/json"},
        )


@mock.patch.object(client, "load_static_inference_input")
def test_ocr_image_when_multiple_images_given(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.return_value = [
        ("base64_image_1", 0.5),
        ("base64_image_2", 0.6),
    ]
    requests_mock.post(
        f"{api_url}/doctr/ocr",
        response_list=[
            {
                "json": {
                    "response": "Image text 1.",
                    "time": 0.33,
                }
            },
            {
                "json": {
                    "response": "Image text 2.",
                    "time": 0.33,
                }
            },
        ],
    )

    # when
    result = http_client.ocr_image(inference_input=["/some/image.jpg"] * 2)

    # then
    assert result == [
        {
            "response": "Image text 1.",
            "time": 0.33,
        },
        {
            "response": "Image text 2.",
            "time": 0.33,
        },
    ], "Result must match the value returned by HTTP endpoint"
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key",
        "image": {"type": "base64", "value": "base64_image_1"},
    }, "First request must contain API key and first image encoded in standard format"
    assert requests_mock.request_history[1].json() == {
        "api_key": "my-api-key",
        "image": {"type": "base64", "value": "base64_image_2"},
    }, "Second request must contain API key and second image encoded in standard format"


@pytest.mark.asyncio
@mock.patch.object(client, "load_static_inference_input_async")
async def test_ocr_image_async_when_multiple_images_given(
    load_static_inference_input_async: AsyncMock,
) -> None:
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_async.return_value = [
        ("base64_image_1", 0.5),
        ("base64_image_2", 0.6),
    ]

    with aioresponses() as m:
        m.post(
            f"{api_url}/doctr/ocr",
            payload={
                "response": "Image text 1.",
                "time": 0.33,
            },
        )
        m.post(
            f"{api_url}/doctr/ocr",
            payload={
                "response": "Image text 1.",
                "time": 0.33,
            },
        )

        # when
        result = await http_client.ocr_image_async(
            inference_input=["/some/image.jpg"] * 2
        )

        # then
        assert result == [
            {
                "response": "Image text 1.",
                "time": 0.33,
            },
            {
                "response": "Image text 1.",
                "time": 0.33,
            },
        ], "Result must match the value returned by HTTP endpoint"


@mock.patch.object(client, "load_static_inference_input")
def test_ocr_image_when_faulty_response_returned(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.return_value = [("base64_image", 0.5)]
    requests_mock.post(
        f"{api_url}/doctr/ocr",
        json={
            "message": "Cannot load DocTR model.",
        },
        status_code=500,
    )

    with pytest.raises(HTTPCallErrorError):
        _ = http_client.ocr_image(inference_input="/some/image.jpg")


@pytest.mark.asyncio
@mock.patch.object(client, "load_static_inference_input_async")
async def test_ocr_image_async_hen_faulty_response_returned(
    load_static_inference_input_async_mock: MagicMock,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_async_mock.return_value = [("base64_image", 0.5)]

    with aioresponses() as m:
        m.post(
            f"{api_url}/doctr/ocr",
            payload={"message": "Cannot load DocTR model."},
            status=500,
        )

        # when
        with pytest.raises(HTTPCallErrorError):
            _ = await http_client.ocr_image_async(inference_input="/some/image.jpg")


def test_detect_gazes_in_v0_mode() -> None:
    # given
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url="http://some.com")
    http_client.select_api_v0()

    # when
    with pytest.raises(WrongClientModeError):
        _ = http_client.detect_gazes(
            inference_input="https://some.com/image.jpg",
        )


@pytest.mark.asyncio
async def test_detect_gazes_async_in_v0_mode() -> None:
    # given
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url="http://some.com")
    http_client.select_api_v0()

    # when
    with pytest.raises(WrongClientModeError):
        _ = await http_client.detect_gazes_async(
            inference_input="https://some.com/image.jpg",
        )


@mock.patch.object(client, "load_static_inference_input")
def test_detect_gazes_when_single_image_given(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.return_value = [("base64_image", 0.5)]
    expected_prediction = {
        "predictions": [
            {
                "face": {
                    "x": 272.0,
                    "y": 112.0,
                    "width": 92.0,
                    "height": 92.0,
                    "confidence": 0.9473056197166443,
                    "class": "face",
                    "class_confidence": None,
                    "class_id": 0,
                    "tracker_id": None,
                    "landmarks": [
                        {"x": 252.0, "y": 90.0},
                        {"x": 295.0, "y": 90.0},
                        {"x": 275.0, "y": 111.0},
                        {"x": 274.0, "y": 130.0},
                        {"x": 225.0, "y": 99.0},
                        {"x": 316.0, "y": 101.0},
                    ],
                },
                "yaw": -0.060329124331474304,
                "pitch": -0.012491557747125626,
            }
        ],
        "time": 0.22586208400025498,
        "time_face_det": None,
        "time_gaze_det": None,
    }
    requests_mock.post(
        f"{api_url}/gaze/gaze_detection",
        json=expected_prediction,
    )

    # when
    result = http_client.detect_gazes(inference_input="/some/image.jpg")

    # then
    assert (
        result == expected_prediction
    ), "Result must match the value returned by HTTP endpoint"
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key",
        "image": {"type": "base64", "value": "base64_image"},
    }, "Request must contain API key and image encoded in standard format"


@pytest.mark.asyncio
@mock.patch.object(client, "load_static_inference_input_async")
async def test_detect_gazes_async_when_single_image_given(
    load_static_inference_input_async_mock: MagicMock,
) -> None:
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_async_mock.return_value = [("base64_image", 0.5)]
    expected_prediction = {
        "predictions": [
            {
                "face": {
                    "x": 272.0,
                    "y": 112.0,
                    "width": 92.0,
                    "height": 92.0,
                    "confidence": 0.9473056197166443,
                    "class": "face",
                    "class_confidence": None,
                    "class_id": 0,
                    "tracker_id": None,
                    "landmarks": [
                        {"x": 252.0, "y": 90.0},
                        {"x": 295.0, "y": 90.0},
                        {"x": 275.0, "y": 111.0},
                        {"x": 274.0, "y": 130.0},
                        {"x": 225.0, "y": 99.0},
                        {"x": 316.0, "y": 101.0},
                    ],
                },
                "yaw": -0.060329124331474304,
                "pitch": -0.012491557747125626,
            }
        ],
        "time": 0.22586208400025498,
        "time_face_det": None,
        "time_gaze_det": None,
    }

    with aioresponses() as m:
        m.post(
            f"{api_url}/gaze/gaze_detection",
            payload=expected_prediction,
        )

        # when
        result = await http_client.detect_gazes_async(inference_input="/some/image.jpg")

        # then
        assert (
            result == expected_prediction
        ), "Result must match the value returned by HTTP endpoint"
        m.assert_called_with(
            f"{api_url}/gaze/gaze_detection",
            "POST",
            params=None,
            data=None,
            json={
                "api_key": "my-api-key",
                "image": {"type": "base64", "value": "base64_image"},
            },
            headers={"Content-Type": "application/json"},
        )


@mock.patch.object(client, "load_static_inference_input")
def test_detect_gazes_when_faulty_response_returned(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.return_value = [("base64_image", 0.5)]
    requests_mock.post(
        f"{api_url}/gaze/gaze_detection",
        json={
            "message": "Cannot load gaze model.",
        },
        status_code=500,
    )

    with pytest.raises(HTTPCallErrorError):
        _ = http_client.detect_gazes(inference_input="/some/image.jpg")


@pytest.mark.asyncio
@mock.patch.object(client, "load_static_inference_input_async")
async def test_detect_gazes_when_faulty_response_returned(
    load_static_inference_input_async_mock: AsyncMock,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_async_mock.return_value = [("base64_image", 0.5)]

    with aioresponses() as m:
        m.post(
            f"{api_url}/gaze/gaze_detection",
            payload={
                "message": "Cannot load gaze model.",
            },
            status=500,
        )

        # when
        with pytest.raises(HTTPCallErrorError):
            _ = await http_client.detect_gazes_async(inference_input="/some/image.jpg")


@mock.patch.object(client, "load_static_inference_input")
def test_get_clip_image_embeddings_when_single_image_given_in_v1_mode(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.return_value = [("base64_image", 0.5)]
    expected_prediction = {
        "frame_id": None,
        "time": 0.05899370899714995,
        "embeddings": [
            [
                0.38750073313713074,
                -0.1737658828496933,
                -0.6624148488044739,
                0.129795640707016,
                0.10291421413421631,
                0.42692098021507263,
                -0.07305282354354858,
                0.030459187924861908,
            ]
        ],
    }
    requests_mock.post(
        f"{api_url}/clip/embed_image",
        json=expected_prediction,
    )

    # when
    result = http_client.get_clip_image_embeddings(inference_input="/some/image.jpg")

    # then
    assert (
        result == expected_prediction
    ), "Result must match the value returned by HTTP endpoint"
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key",
        "image": {"type": "base64", "value": "base64_image"},
    }, "Request must contain API key and image encoded in standard format"


@pytest.mark.asyncio
@mock.patch.object(client, "load_static_inference_input_async")
async def test_get_clip_image_embeddings_async_when_single_image_given_in_v1_mode(
    load_static_inference_input_async_mock: AsyncMock,
) -> None:
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_async_mock.return_value = [("base64_image", 0.5)]
    expected_prediction = {
        "frame_id": None,
        "time": 0.05899370899714995,
        "embeddings": [
            [
                0.38750073313713074,
                -0.1737658828496933,
                -0.6624148488044739,
                0.129795640707016,
                0.10291421413421631,
                0.42692098021507263,
                -0.07305282354354858,
                0.030459187924861908,
            ]
        ],
    }

    with aioresponses() as m:
        m.post(
            f"{api_url}/clip/embed_image",
            payload=expected_prediction,
        )

        # when
        result = await http_client.get_clip_image_embeddings_async(
            inference_input="/some/image.jpg"
        )

        # then
        assert (
            result == expected_prediction
        ), "Result must match the value returned by HTTP endpoint"
        m.assert_called_with(
            f"{api_url}/clip/embed_image",
            "POST",
            params=None,
            data=None,
            json={
                "api_key": "my-api-key",
                "image": {"type": "base64", "value": "base64_image"},
            },
            headers={"Content-Type": "application/json"},
        )


@mock.patch.object(client, "load_static_inference_input")
def test_get_clip_image_embeddings_when_single_image_given_in_v0_mode(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    api_url = "https://infer.roboflow.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.return_value = [("base64_image", 0.5)]
    expected_prediction = {
        "frame_id": None,
        "time": 0.05899370899714995,
        "embeddings": [
            [
                0.38750073313713074,
                -0.1737658828496933,
                -0.6624148488044739,
                0.129795640707016,
                0.10291421413421631,
                0.42692098021507263,
                -0.07305282354354858,
                0.030459187924861908,
            ]
        ],
    }
    requests_mock.post(
        f"{api_url}/clip/embed_image?api_key=my-api-key",
        json=expected_prediction,
    )

    # when
    result = http_client.get_clip_image_embeddings(
        inference_input="/some/image.jpg", clip_version="ViT-B-32"
    )

    # then
    assert (
        result == expected_prediction
    ), "Result must match the value returned by HTTP endpoint"
    assert requests_mock.request_history[0].json() == {
        "image": {"type": "base64", "value": "base64_image"},
        "clip_version_id": "ViT-B-32",
    }, "Request must contain image encoded in standard format"


@pytest.mark.asyncio
@mock.patch.object(client, "load_static_inference_input_async")
async def test_get_clip_image_embeddings_async_when_single_image_given_in_v0_mode(
    load_static_inference_input_async_mock: AsyncMock,
) -> None:
    api_url = "https://infer.roboflow.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_async_mock.return_value = [("base64_image", 0.5)]
    expected_prediction = {
        "frame_id": None,
        "time": 0.05899370899714995,
        "embeddings": [
            [
                0.38750073313713074,
                -0.1737658828496933,
                -0.6624148488044739,
                0.129795640707016,
                0.10291421413421631,
                0.42692098021507263,
                -0.07305282354354858,
                0.030459187924861908,
            ]
        ],
    }

    with aioresponses() as m:
        m.post(
            f"{api_url}/clip/embed_image?api_key=my-api-key",
            payload=expected_prediction,
        )

        # when
        result = await http_client.get_clip_image_embeddings_async(
            inference_input="/some/image.jpg",
            clip_version="ViT-B-32",
        )

        # then
        assert (
            result == expected_prediction
        ), "Result must match the value returned by HTTP endpoint"
        m.assert_called_with(
            f"{api_url}/clip/embed_image?api_key=my-api-key",
            "POST",
            params=None,
            data=None,
            json={
                "image": {"type": "base64", "value": "base64_image"},
                "clip_version_id": "ViT-B-32",
            },
            headers={"Content-Type": "application/json"},
        )


@mock.patch.object(client, "load_static_inference_input")
def test_get_clip_image_embeddings_when_faulty_response_returned(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.return_value = [("base64_image", 0.5)]
    requests_mock.post(
        f"{api_url}/clip/embed_image",
        json={
            "message": "Cannot load Clip model.",
        },
        status_code=500,
    )

    with pytest.raises(HTTPCallErrorError):
        _ = http_client.get_clip_image_embeddings(inference_input="/some/image.jpg")


@pytest.mark.asyncio
@mock.patch.object(client, "load_static_inference_input_async")
async def test_get_clip_image_embeddings_when_faulty_response_returned(
    load_static_inference_input_async_mock: AsyncMock,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_async_mock.return_value = [("base64_image", 0.5)]

    with aioresponses() as m:
        m.post(
            f"{api_url}/clip/embed_image",
            payload={
                "message": "Cannot load Clip model.",
            },
            status=500,
        )

        with pytest.raises(HTTPCallErrorError):
            _ = await http_client.get_clip_image_embeddings_async(
                inference_input="/some/image.jpg"
            )


def test_get_clip_text_embeddings_when_single_text_given(
    requests_mock: Mocker,
) -> None:
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    expected_prediction = {
        "frame_id": None,
        "time": 0.05899370899714995,
        "embeddings": [
            [
                0.38750073313713074,
                -0.1737658828496933,
                -0.6624148488044739,
                0.129795640707016,
                0.10291421413421631,
                0.42692098021507263,
                -0.07305282354354858,
                0.030459187924861908,
            ]
        ],
    }
    requests_mock.post(
        f"{api_url}/clip/embed_text",
        json=expected_prediction,
    )

    # when
    result = http_client.get_clip_text_embeddings(text="some", clip_version="ViT-B-32")

    # then
    assert (
        result == expected_prediction
    ), "Result must match the value returned by HTTP endpoint"
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key",
        "text": "some",
        "clip_version_id": "ViT-B-32",
    }, "Request must contain API key and text"


@pytest.mark.asyncio
async def test_get_clip_text_embeddings_async_when_single_text_given() -> None:
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    expected_prediction = {
        "frame_id": None,
        "time": 0.05899370899714995,
        "embeddings": [
            [
                0.38750073313713074,
                -0.1737658828496933,
                -0.6624148488044739,
                0.129795640707016,
                0.10291421413421631,
                0.42692098021507263,
                -0.07305282354354858,
                0.030459187924861908,
            ]
        ],
    }

    with aioresponses() as m:
        m.post(
            f"{api_url}/clip/embed_text",
            payload=expected_prediction,
        )

        # when
        result = await http_client.get_clip_text_embeddings_async(
            text="some", clip_version="ViT-B-32"
        )

        # then
        assert (
            result == expected_prediction
        ), "Result must match the value returned by HTTP endpoint"
        m.assert_called_with(
            f"{api_url}/clip/embed_text",
            "POST",
            data=None,
            json={
                "api_key": "my-api-key",
                "text": "some",
                "clip_version_id": "ViT-B-32",
            },
            headers={"Content-Type": "application/json"},
        )


def test_get_clip_text_embeddings_when_faulty_response_returned(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.post(
        f"{api_url}/clip/embed_text",
        json={
            "message": "Cannot load Clip model.",
        },
        status_code=500,
    )

    with pytest.raises(HTTPCallErrorError):
        _ = http_client.get_clip_text_embeddings(text="some")


@pytest.mark.asyncio
async def test_get_clip_text_embeddings_async_when_faulty_response_returned() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    with aioresponses() as m:
        m.post(
            f"{api_url}/clip/embed_text",
            payload={
                "message": "Cannot load Clip model.",
            },
            status=500,
        )

        with pytest.raises(HTTPCallErrorError):
            _ = await http_client.get_clip_text_embeddings_async(text="some")


def test_clip_compare_when_invalid_subject_given() -> None:
    # given
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url="http://some.com")

    # when
    with pytest.raises(InvalidParameterError):
        _ = http_client.clip_compare(
            subject="/some/image.jpg", prompt=["dog", "house"], subject_type="unknown"
        )


@pytest.mark.asyncio
async def test_clip_compare_async_when_invalid_subject_given() -> None:
    # given
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url="http://some.com")

    # when
    with pytest.raises(InvalidParameterError):
        _ = await http_client.clip_compare_async(
            subject="/some/image.jpg", prompt=["dog", "house"], subject_type="unknown"
        )


def test_clip_compare_when_invalid_prompt_given() -> None:
    # given
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url="http://some.com")

    # when
    with pytest.raises(InvalidParameterError):
        _ = http_client.clip_compare(
            subject="/some/image.jpg", prompt=["dog", "house"], prompt_type="unknown"
        )


@pytest.mark.asyncio
async def test_clip_compare_async_when_invalid_prompt_given() -> None:
    # given
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url="http://some.com")

    # when
    with pytest.raises(InvalidParameterError):
        _ = await http_client.clip_compare_async(
            subject="/some/image.jpg", prompt=["dog", "house"], prompt_type="unknown"
        )


def test_clip_compare_when_both_prompt_and_subject_are_texts(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.post(
        f"{api_url}/clip/compare",
        json={
            "frame_id": None,
            "time": 0.1435863340011565,
            "similarity": [0.8963012099266052, 0.8830886483192444],
        },
    )

    # when
    result = http_client.clip_compare(
        subject="some",
        prompt=["dog", "house"],
        subject_type="text",
        prompt_type="text",
        clip_version="ViT-B-32",
    )

    # then
    assert result == {
        "frame_id": None,
        "time": 0.1435863340011565,
        "similarity": [0.8963012099266052, 0.8830886483192444],
    }, "Result must match the value returned by HTTP endpoint"
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key",
        "subject": "some",
        "prompt": ["dog", "house"],
        "prompt_type": "text",
        "subject_type": "text",
        "clip_version_id": "ViT-B-32",
    }, "Request must contain API key, subject and prompt types as text, exact values of subject and list of prompt values"


@pytest.mark.asyncio
async def test_clip_compare_async_when_both_prompt_and_subject_are_texts() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    with aioresponses() as m:
        m.post(
            f"{api_url}/clip/compare",
            payload={
                "frame_id": None,
                "time": 0.1435863340011565,
                "similarity": [0.8963012099266052, 0.8830886483192444],
            },
        )
        # when
        result = await http_client.clip_compare_async(
            subject="some",
            prompt=["dog", "house"],
            subject_type="text",
            prompt_type="text",
            clip_version="ViT-B-32",
        )

        # then
        assert result == {
            "frame_id": None,
            "time": 0.1435863340011565,
            "similarity": [0.8963012099266052, 0.8830886483192444],
        }, "Result must match the value returned by HTTP endpoint"
        m.assert_called_with(
            f"{api_url}/clip/compare",
            "POST",
            json={
                "api_key": "my-api-key",
                "subject": "some",
                "prompt": ["dog", "house"],
                "prompt_type": "text",
                "subject_type": "text",
                "clip_version_id": "ViT-B-32",
            },
            headers={"Content-Type": "application/json"},
        )


@mock.patch.object(client, "load_static_inference_input")
def test_clip_compare_when_mixed_input_is_given(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.side_effect = [[("base64_image_1", 0.5)]]
    requests_mock.post(
        f"{api_url}/clip/compare",
        json={
            "frame_id": None,
            "time": 0.1435863340011565,
            "similarity": [0.8963012099266052, 0.8830886483192444],
        },
    )

    # when
    result = http_client.clip_compare(
        subject="/some/image.jpg",
        prompt=["dog", "house"],
    )

    # then
    assert result == {
        "frame_id": None,
        "time": 0.1435863340011565,
        "similarity": [0.8963012099266052, 0.8830886483192444],
    }, "Result must match the value returned by HTTP endpoint"
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key",
        "subject": {"type": "base64", "value": "base64_image_1"},
        "prompt": ["dog", "house"],
        "prompt_type": "text",
        "subject_type": "image",
    }, "Request must contain API key, subject and prompt types as text, exact values of subject and list of prompt values"


@pytest.mark.asyncio
@mock.patch.object(client, "load_static_inference_input_async")
async def test_clip_compare_when_mixed_input_is_given(
    load_static_inference_input_async_mock: AsyncMock,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_async_mock.side_effect = [[("base64_image_1", 0.5)]]

    with aioresponses() as m:
        m.post(
            f"{api_url}/clip/compare",
            payload={
                "frame_id": None,
                "time": 0.1435863340011565,
                "similarity": [0.8963012099266052, 0.8830886483192444],
            },
        )

        # when
        result = await http_client.clip_compare_async(
            subject="/some/image.jpg",
            prompt=["dog", "house"],
        )

        # then
        assert result == {
            "frame_id": None,
            "time": 0.1435863340011565,
            "similarity": [0.8963012099266052, 0.8830886483192444],
        }, "Result must match the value returned by HTTP endpoint"
        m.assert_called_with(
            f"{api_url}/clip/compare",
            "POST",
            json={
                "api_key": "my-api-key",
                "subject": {"type": "base64", "value": "base64_image_1"},
                "prompt": ["dog", "house"],
                "prompt_type": "text",
                "subject_type": "image",
            },
            headers={"Content-Type": "application/json"},
        )


@mock.patch.object(client, "load_static_inference_input")
def test_clip_compare_when_both_prompt_and_subject_are_images(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.side_effect = [
        [("base64_image_1", 0.5)],
        [("base64_image_2", 0.5), ("base64_image_3", 0.5)],
    ]
    requests_mock.post(
        f"{api_url}/clip/compare",
        json={
            "frame_id": None,
            "time": 0.1435863340011565,
            "similarity": [0.8963012099266052, 0.8830886483192444],
        },
    )

    # when
    result = http_client.clip_compare(
        subject="/some/image_1.jpg",
        prompt=["/some/image_2.jpg", "/some/image_3.jpg"],
        subject_type="image",
        prompt_type="image",
    )

    # then
    assert result == {
        "frame_id": None,
        "time": 0.1435863340011565,
        "similarity": [0.8963012099266052, 0.8830886483192444],
    }, "Result must match the value returned by HTTP endpoint"
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key",
        "subject": {"type": "base64", "value": "base64_image_1"},
        "prompt": [
            {"type": "base64", "value": "base64_image_2"},
            {"type": "base64", "value": "base64_image_3"},
        ],
        "prompt_type": "image",
        "subject_type": "image",
    }, "Request must contain API key, subject and prompt types as image, and encoded image - image 1 as subject, images 2 and 3 as prompt"


@pytest.mark.asyncio
@mock.patch.object(client, "load_static_inference_input_async")
async def test_clip_compare_when_both_prompt_and_subject_are_images(
    load_static_inference_input_async_mock: MagicMock,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_async_mock.side_effect = [
        [("base64_image_1", 0.5)],
        [("base64_image_2", 0.5), ("base64_image_3", 0.5)],
    ]

    with aioresponses() as m:
        m.post(
            f"{api_url}/clip/compare",
            payload={
                "frame_id": None,
                "time": 0.1435863340011565,
                "similarity": [0.8963012099266052, 0.8830886483192444],
            },
        )

        # when
        result = await http_client.clip_compare_async(
            subject="/some/image_1.jpg",
            prompt=["/some/image_2.jpg", "/some/image_3.jpg"],
            subject_type="image",
            prompt_type="image",
        )

        # then
        assert result == {
            "frame_id": None,
            "time": 0.1435863340011565,
            "similarity": [0.8963012099266052, 0.8830886483192444],
        }, "Result must match the value returned by HTTP endpoint"
        m.assert_called_with(
            f"{api_url}/clip/compare",
            "POST",
            json={
                "api_key": "my-api-key",
                "subject": {"type": "base64", "value": "base64_image_1"},
                "prompt": [
                    {"type": "base64", "value": "base64_image_2"},
                    {"type": "base64", "value": "base64_image_3"},
                ],
                "prompt_type": "image",
                "subject_type": "image",
            },
            headers={"Content-Type": "application/json"},
        )


def test_clip_compare_when_faulty_response_returned(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.post(
        f"{api_url}/clip/compare",
        json={
            "message": "Cannot load Clip model.",
        },
        status_code=500,
    )

    with pytest.raises(HTTPCallErrorError):
        _ = http_client.clip_compare(
            subject="some", prompt=["dog", "house"], subject_type="text"
        )


@pytest.mark.asyncio
async def test_clip_compare_when_faulty_response_returned() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    with aioresponses() as m:
        m.post(
            f"{api_url}/clip/compare",
            payload={
                "message": "Cannot load Clip model.",
            },
            status=500,
        )

        # when
        with pytest.raises(HTTPCallErrorError):
            _ = await http_client.clip_compare_async(
                subject="some", prompt=["dog", "house"], subject_type="text"
            )


@pytest.mark.parametrize(
    "legacy_endpoints, endpoint_to_use, parameter_name",
    [
        (True, "/infer/workflows/my_workspace/my_workflow", "workflow_name"),
        (False, "/my_workspace/workflows/my_workflow", "workflow_id"),
    ],
)
def test_infer_from_workflow_when_v0_mode_used(
    requests_mock: Mocker,
    legacy_endpoints: bool,
    endpoint_to_use: str,
    parameter_name: str,
) -> None:
    # given
    api_url = "http://infer.roboflow.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.post(
        f"{api_url}{endpoint_to_use}",
        json={
            "outputs": [{"some": 3, "other": [1, {"a": "b"}]}],
        },
    )
    method = (
        http_client.infer_from_workflow
        if legacy_endpoints
        else http_client.run_workflow
    )

    # when
    result = method(
        workspace_name="my_workspace",
        **{parameter_name: "my_workflow"},
    )

    # then
    assert result == [
        {"some": 3, "other": [1, {"a": "b"}]}
    ], "Response from API must be properly decoded"
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key",
        "inputs": {},
        "use_cache": True,
        "enable_profiling": False,
    }, "Request payload must contain api key and inputs"


@pytest.mark.parametrize(
    "legacy_endpoints",
    [(True,), (False,)],
)
@mock.patch.object(executors, "requests")
def test_infer_from_workflow_when_connection_error_to_be_retried_successfully(
    requests_mock: MagicMock,
    legacy_endpoints: bool,
) -> None:
    # given
    api_url = "http://infer.roboflow.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    valid_response = Response()
    valid_response.status_code = 200
    valid_response._content = json.dumps(
        {"outputs": [{"some": 3, "other": [1, {"a": "b"}]}]}
    ).encode("utf-8")
    requests_mock.exceptions.ConnectionError = requests.exceptions.ConnectionError
    requests_mock.post.side_effect = [ConnectionError, valid_response]
    method = (
        http_client.infer_from_workflow
        if legacy_endpoints
        else http_client.run_workflow
    )

    # when
    result = method(
        workspace_name="my_workspace",
        workflow_name="my_workflow",
    )

    # then
    assert result == [
        {"some": 3, "other": [1, {"a": "b"}]}
    ], "Response from API must be properly decoded"


@pytest.mark.parametrize(
    "legacy_endpoints",
    [(True,), (False,)],
)
@mock.patch.object(executors, "requests")
def test_infer_from_workflow_when_connection_error_cannot_be_retried_successfully(
    requests_mock: MagicMock,
    legacy_endpoints: bool,
) -> None:
    # given
    api_url = "http://infer.roboflow.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.exceptions.ConnectionError = requests.exceptions.ConnectionError
    requests_mock.post.side_effect = ConnectionError
    method = (
        http_client.infer_from_workflow
        if legacy_endpoints
        else http_client.run_workflow
    )

    # when
    with pytest.raises(HTTPClientError):
        _ = method(
            workspace_name="my_workspace",
            workflow_name="my_workflow",
        )


@pytest.mark.parametrize(
    "legacy_endpoints",
    [(True,), (False,)],
)
@mock.patch.object(executors, "requests")
def test_infer_from_workflow_when_connection_error_happens_and_retries_disabled(
    requests_mock: MagicMock,
    legacy_endpoints: bool,
) -> None:
    # given
    api_url = "http://infer.roboflow.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    config = InferenceConfiguration(workflow_run_retries_enabled=False)
    http_client.configure(config)
    requests_mock.exceptions.ConnectionError = requests.exceptions.ConnectionError
    valid_response = Response()
    valid_response.url = api_url
    valid_response.status_code = 200
    valid_response._content = json.dumps(
        {"outputs": [{"some": 3, "other": [1, {"a": "b"}]}]}
    ).encode("utf-8")
    requests_mock.exceptions.ConnectionError = requests.exceptions.ConnectionError
    requests_mock.post.side_effect = [ConnectionError, valid_response]
    method = (
        http_client.infer_from_workflow
        if legacy_endpoints
        else http_client.run_workflow
    )

    # when
    with pytest.raises(HTTPClientError):
        _ = method(
            workspace_name="my_workspace",
            workflow_name="my_workflow",
        )


@pytest.mark.parametrize(
    "legacy_endpoints",
    [(True,), (False,)],
)
@mock.patch.object(executors, "requests")
def test_infer_from_workflow_when_transient_error_to_be_retried_successfully(
    requests_mock: MagicMock,
    legacy_endpoints: bool,
) -> None:
    # given
    api_url = "http://infer.roboflow.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    invalid_response = Response()
    invalid_response.status_code = 503
    valid_response = Response()
    valid_response.status_code = 200
    valid_response._content = json.dumps(
        {"outputs": [{"some": 3, "other": [1, {"a": "b"}]}]}
    ).encode("utf-8")
    requests_mock.exceptions.ConnectionError = requests.exceptions.ConnectionError
    requests_mock.post.side_effect = [
        invalid_response,
        invalid_response,
        valid_response,
    ]
    method = (
        http_client.infer_from_workflow
        if legacy_endpoints
        else http_client.run_workflow
    )

    # when
    result = method(
        workspace_name="my_workspace",
        workflow_name="my_workflow",
    )

    # then
    assert result == [
        {"some": 3, "other": [1, {"a": "b"}]}
    ], "Response from API must be properly decoded"


@pytest.mark.parametrize(
    "legacy_endpoints",
    [(True,), (False,)],
)
@mock.patch.object(executors, "requests")
def test_infer_from_workflow_when_transient_error_happens_with_retries_disabled(
    requests_mock: MagicMock,
    legacy_endpoints: bool,
) -> None:
    # given
    api_url = "http://infer.roboflow.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    config = InferenceConfiguration(workflow_run_retries_enabled=False)
    http_client.configure(config)
    invalid_response = Response()
    invalid_response.url = api_url
    invalid_response.status_code = 503
    valid_response = Response()
    valid_response.url = api_url
    valid_response.status_code = 200
    valid_response._content = json.dumps(
        {"outputs": [{"some": 3, "other": [1, {"a": "b"}]}]}
    ).encode("utf-8")
    requests_mock.exceptions.ConnectionError = requests.exceptions.ConnectionError
    requests_mock.post.side_effect = [
        invalid_response,
        invalid_response,
        valid_response,
    ]
    method = (
        http_client.infer_from_workflow
        if legacy_endpoints
        else http_client.run_workflow
    )

    # when
    with pytest.raises(HTTPCallErrorError):
        _ = method(
            workspace_name="my_workspace",
            workflow_name="my_workflow",
        )


@pytest.mark.parametrize(
    "legacy_endpoints",
    [(True,), (False,)],
)
@mock.patch.object(executors, "requests")
def test_infer_from_workflow_when_transient_error_cannot_be_retried_successfully(
    requests_mock: MagicMock,
    legacy_endpoints: bool,
) -> None:
    # given
    api_url = "http://infer.roboflow.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    invalid_response = Response()
    invalid_response.status_code = 503
    requests_mock.post.side_effect = [invalid_response] * 3
    requests_mock.exceptions.ConnectionError = requests.exceptions.ConnectionError
    method = (
        http_client.infer_from_workflow
        if legacy_endpoints
        else http_client.run_workflow
    )

    # when
    with pytest.raises(HTTPCallErrorError):
        _ = method(
            workspace_name="my_workspace",
            workflow_name="my_workflow",
        )


@pytest.mark.parametrize(
    "legacy_endpoints",
    [(True,), (False,)],
)
@mock.patch.object(executors, "requests")
def test_infer_from_workflow_when_non_transient_error_occurred(
    requests_mock: MagicMock,
    legacy_endpoints: bool,
) -> None:
    # given
    api_url = "http://infer.roboflow.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    invalid_response = Response()
    invalid_response.url = api_url
    invalid_response.status_code = 500
    requests_mock.post.return_value = invalid_response
    requests_mock.exceptions.ConnectionError = requests.exceptions.ConnectionError
    method = (
        http_client.infer_from_workflow
        if legacy_endpoints
        else http_client.run_workflow
    )

    # when
    with pytest.raises(HTTPCallErrorError):
        _ = method(
            workspace_name="my_workspace",
            workflow_name="my_workflow",
        )


@pytest.mark.parametrize(
    "legacy_endpoints, endpoint_to_use, parameter_name",
    [
        (True, "/infer/workflows/my_workspace/my_workflow", "workflow_name"),
        (False, "/my_workspace/workflows/my_workflow", "workflow_id"),
    ],
)
def test_infer_from_workflow_when_no_parameters_given(
    requests_mock: Mocker,
    legacy_endpoints: bool,
    endpoint_to_use: str,
    parameter_name: str,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.post(
        f"{api_url}{endpoint_to_use}",
        json={
            "outputs": [{"some": 3}],
        },
    )
    method = (
        http_client.infer_from_workflow
        if legacy_endpoints
        else http_client.run_workflow
    )

    # when
    result = method(
        workspace_name="my_workspace",
        **{parameter_name: "my_workflow"},
    )

    # then
    assert result == [{"some": 3}], "Response from API must be properly decoded"
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key",
        "inputs": {},
        "use_cache": True,
        "enable_profiling": False,
    }, "Request payload must contain api key and inputs"


@mock.patch.object(client, "load_nested_batches_of_inference_input")
@pytest.mark.parametrize(
    "legacy_endpoints, endpoint_to_use, parameter_name",
    [
        (True, "/infer/workflows/my_workspace/my_workflow", "workflow_name"),
        (False, "/my_workspace/workflows/my_workflow", "workflow_id"),
    ],
)
def test_infer_from_workflow_when_parameters_and_excluded_fields_given(
    load_nested_batches_of_inference_input_mock: MagicMock,
    requests_mock: Mocker,
    legacy_endpoints: bool,
    endpoint_to_use: str,
    parameter_name: str,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.post(
        f"{api_url}{endpoint_to_use}",
        json={
            "outputs": [{"some": 3}],
        },
    )
    load_nested_batches_of_inference_input_mock.side_effect = [
        ("base64_image_1", 0.5),
        [("base64_image_2", 0.5), ("base64_image_3", 0.5)],
    ]
    method = (
        http_client.infer_from_workflow
        if legacy_endpoints
        else http_client.run_workflow
    )

    # when
    result = method(
        workspace_name="my_workspace",
        images={"image_1": "https://...", "image_2": ["https://...", "https://..."]},
        parameters={
            "some": 10,
        },
        excluded_fields=["some"],
        **{parameter_name: "my_workflow"},
    )

    # then
    assert result == [{"some": 3}], "Response from API must be properly decoded"
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key",
        "use_cache": True,
        "enable_profiling": False,
        "inputs": {
            "image_1": {
                "type": "base64",
                "value": "base64_image_1",
            },
            "image_2": [
                {
                    "type": "base64",
                    "value": "base64_image_2",
                },
                {
                    "type": "base64",
                    "value": "base64_image_3",
                },
            ],
            "some": 10,
        },
        "excluded_fields": ["some"],
    }, "Request payload must contain api key and inputs"


@mock.patch.object(client, "load_nested_batches_of_inference_input")
@pytest.mark.parametrize(
    "legacy_endpoints, endpoint_to_use, parameter_name",
    [
        (True, "/infer/workflows/my_workspace/my_workflow", "workflow_name"),
        (False, "/my_workspace/workflows/my_workflow", "workflow_id"),
    ],
)
def test_infer_from_workflow_when_usage_of_cache_disabled(
    load_nested_batches_of_inference_input_mock: MagicMock,
    requests_mock: Mocker,
    legacy_endpoints: bool,
    endpoint_to_use: str,
    parameter_name: str,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.post(
        f"{api_url}{endpoint_to_use}",
        json={
            "outputs": [{"some": 3}],
        },
    )
    load_nested_batches_of_inference_input_mock.side_effect = [
        ("base64_image_1", 0.5),
        [("base64_image_2", 0.5), ("base64_image_3", 0.5)],
    ]
    method = (
        http_client.infer_from_workflow
        if legacy_endpoints
        else http_client.run_workflow
    )

    # when
    result = method(
        workspace_name="my_workspace",
        images={"image_1": "https://...", "image_2": ["https://...", "https://..."]},
        use_cache=False,
        **{parameter_name: "my_workflow"},
    )

    # then
    assert result == [{"some": 3}], "Response from API must be properly decoded"
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key",
        "use_cache": False,
        "enable_profiling": False,
        "inputs": {
            "image_1": {
                "type": "base64",
                "value": "base64_image_1",
            },
            "image_2": [
                {
                    "type": "base64",
                    "value": "base64_image_2",
                },
                {
                    "type": "base64",
                    "value": "base64_image_3",
                },
            ],
        },
    }, "Request payload must contain api key, inputs and no cache flag"


@mock.patch.object(client, "load_nested_batches_of_inference_input")
@pytest.mark.parametrize(
    "legacy_endpoints, endpoint_to_use, parameter_name",
    [
        (True, "/infer/workflows/my_workspace/my_workflow", "workflow_name"),
        (False, "/my_workspace/workflows/my_workflow", "workflow_id"),
    ],
)
def test_infer_from_workflow_when_usage_of_profiler_enabled(
    load_nested_batches_of_inference_input_mock: MagicMock,
    requests_mock: Mocker,
    legacy_endpoints: bool,
    endpoint_to_use: str,
    parameter_name: str,
    empty_directory: str,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url).configure(
        inference_configuration=InferenceConfiguration(
            profiling_directory=empty_directory
        )
    )
    requests_mock.post(
        f"{api_url}{endpoint_to_use}",
        json={"outputs": [{"some": 3}], "profiler_trace": [{"my": "trace"}]},
    )
    load_nested_batches_of_inference_input_mock.side_effect = [
        ("base64_image_1", 0.5),
        [("base64_image_2", 0.5), ("base64_image_3", 0.5)],
    ]
    method = (
        http_client.infer_from_workflow
        if legacy_endpoints
        else http_client.run_workflow
    )

    # when
    result = method(
        workspace_name="my_workspace",
        images={"image_1": "https://...", "image_2": ["https://...", "https://..."]},
        enable_profiling=True,
        **{parameter_name: "my_workflow"},
    )

    # then
    assert result == [{"some": 3}], "Response from API must be properly decoded"
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key",
        "use_cache": True,
        "enable_profiling": True,
        "inputs": {
            "image_1": {
                "type": "base64",
                "value": "base64_image_1",
            },
            "image_2": [
                {
                    "type": "base64",
                    "value": "base64_image_2",
                },
                {
                    "type": "base64",
                    "value": "base64_image_3",
                },
            ],
        },
    }, "Request payload must contain api key, inputs and no cache flag"
    json_files_in_profiling_directory = glob(os.path.join(empty_directory, "*.json"))
    assert (
        len(json_files_in_profiling_directory) == 1
    ), "Expected to find one JSON file with profiler trace"
    with open(json_files_in_profiling_directory[0], "r") as f:
        data = json.load(f)
    assert data == [{"my": "trace"}], "Trace content must be fully saved"


@mock.patch.object(client, "load_nested_batches_of_inference_input")
@pytest.mark.parametrize(
    "legacy_endpoints, endpoint_to_use, parameter_name",
    [
        (True, "/infer/workflows/my_workspace/my_workflow", "workflow_name"),
        (False, "/my_workspace/workflows/my_workflow", "workflow_id"),
    ],
)
def test_infer_from_workflow_when_nested_batch_of_inputs_provided(
    load_nested_batches_of_inference_input_mock: MagicMock,
    requests_mock: Mocker,
    legacy_endpoints: bool,
    endpoint_to_use: str,
    parameter_name: str,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.post(
        f"{api_url}{endpoint_to_use}",
        json={
            "outputs": [{"some": 3}],
        },
    )
    load_nested_batches_of_inference_input_mock.side_effect = [
        [
            [("base64_image_1", 0.5), ("base64_image_2", 0.5)],
            [("base64_image_3", 0.5), ("base64_image_4", 0.5), ("base64_image_5", 0.5)],
            [("base64_image_6", 0.5)],
        ],
    ]
    method = (
        http_client.infer_from_workflow
        if legacy_endpoints
        else http_client.run_workflow
    )

    # when
    result = method(
        workspace_name="my_workspace",
        images={"image_1": [["1", "2"], ["3", "4", "5"], ["6"]]},
        parameters={"batch_oriented_param": [["a", "b"], ["c", "d", "e"], ["f"]]},
        **{parameter_name: "my_workflow"},
    )

    # then
    assert result == [{"some": 3}], "Response from API must be properly decoded"
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key",
        "use_cache": True,
        "enable_profiling": False,
        "inputs": {
            "image_1": [
                [
                    {"type": "base64", "value": "base64_image_1"},
                    {"type": "base64", "value": "base64_image_2"},
                ],
                [
                    {"type": "base64", "value": "base64_image_3"},
                    {"type": "base64", "value": "base64_image_4"},
                    {"type": "base64", "value": "base64_image_5"},
                ],
                [
                    {"type": "base64", "value": "base64_image_6"},
                ],
            ],
            "batch_oriented_param": [
                ["a", "b"],
                ["c", "d", "e"],
                ["f"],
            ],
        },
    }, "Request payload must contain api key, inputs and no cache flag"


@pytest.mark.parametrize(
    "legacy_endpoints, endpoint_to_use, parameter_name",
    [
        (True, "/infer/workflows/my_workspace/my_workflow", "workflow_name"),
        (False, "/my_workspace/workflows/my_workflow", "workflow_id"),
    ],
)
def test_infer_from_workflow_when_faulty_response_given(
    requests_mock: Mocker,
    legacy_endpoints: bool,
    endpoint_to_use: str,
    parameter_name: str,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.post(
        f"{api_url}{endpoint_to_use}",
        json={"message": "some"},
        status_code=500,
    )
    method = (
        http_client.infer_from_workflow
        if legacy_endpoints
        else http_client.run_workflow
    )

    # when
    with pytest.raises(HTTPCallErrorError):
        _ = method(
            workspace_name="my_workspace",
            **{parameter_name: "my_workflow"},
        )


def test_infer_from_workflow_when_neither_workflow_name_nor_specs_given() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(InvalidParameterError):
        _ = http_client.infer_from_workflow()


def test_infer_from_workflow_when_both_workflow_name_and_specs_given() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(InvalidParameterError):
        _ = http_client.infer_from_workflow(
            workspace_name="my_workspace",
            workflow_name="some",
            specification={"some": "specs"},
        )


@mock.patch.object(client, "load_nested_batches_of_inference_input")
@pytest.mark.parametrize(
    "legacy_endpoints, endpoint_to_use",
    [(True, "/infer/workflows"), (False, "/workflows/run")],
)
def test_infer_from_workflow_when_custom_workflow_with_both_parameters_and_excluded_fields_given(
    load_nested_batches_of_inference_input_mock: MagicMock,
    requests_mock: Mocker,
    legacy_endpoints: bool,
    endpoint_to_use: str,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.post(
        f"{api_url}{endpoint_to_use}",
        json={
            "outputs": [{"some": 3}],
        },
    )
    load_nested_batches_of_inference_input_mock.side_effect = [
        ("base64_image_1", 0.5),
        [("base64_image_2", 0.5), ("base64_image_3", 0.5)],
    ]
    method = (
        http_client.infer_from_workflow
        if legacy_endpoints
        else http_client.run_workflow
    )

    # when
    result = method(
        specification={"my": "specification"},
        images={"image_1": "https://...", "image_2": ["https://...", "https://..."]},
        parameters={
            "some": 10,
        },
        excluded_fields=["some"],
    )

    # then
    assert result == [{"some": 3}], "Response from API must be properly decoded"
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key",
        "use_cache": True,
        "enable_profiling": False,
        "specification": {"my": "specification"},
        "inputs": {
            "image_1": {
                "type": "base64",
                "value": "base64_image_1",
            },
            "image_2": [
                {
                    "type": "base64",
                    "value": "base64_image_2",
                },
                {
                    "type": "base64",
                    "value": "base64_image_3",
                },
            ],
            "some": 10,
        },
        "excluded_fields": ["some"],
    }, "Request payload must contain api key and inputs"


def test_list_inference_pipelines(requests_mock: Mocker) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.get(
        f"{api_url}/inference_pipelines/list",
        json={
            "status": "success",
            "context": {
                "request_id": "52f5df39-b7de-4a56-8c42-b979d365cfa0",
                "pipeline_id": None,
            },
            "pipelines": ["acd62146-edca-4253-8eeb-40c88906cd70"],
        },
    )

    # when
    result = http_client.list_inference_pipelines()

    # then
    assert result == {
        "status": "success",
        "context": {
            "request_id": "52f5df39-b7de-4a56-8c42-b979d365cfa0",
            "pipeline_id": None,
        },
        "pipelines": ["acd62146-edca-4253-8eeb-40c88906cd70"],
    }
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key"
    }, "Expected payload to contain API key"


def test_list_inference_pipelines_on_auth_error(requests_mock: Mocker) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.get(
        f"{api_url}/inference_pipelines/list",
        status_code=401,
    )

    # when
    with pytest.raises(HTTPCallErrorError):
        _ = http_client.list_inference_pipelines()


def test_get_inference_pipeline_status(requests_mock: Mocker) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.get(
        f"{api_url}/inference_pipelines/my-pipeline/status",
        json={
            "status": "success",
        },
    )

    # when
    result = http_client.get_inference_pipeline_status(pipeline_id="my-pipeline")

    # then
    assert result == {
        "status": "success",
    }
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key"
    }, "Expected payload to contain API key"


def test_get_inference_pipeline_status_when_pipeline_id_empty(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(InvalidParameterError):
        _ = http_client.get_inference_pipeline_status(pipeline_id="")


def test_get_inference_pipeline_status_when_pipeline_id_not_found(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.get(
        f"{api_url}/inference_pipelines/my-pipeline/status",
        status_code=404,
    )

    # when
    with pytest.raises(HTTPCallErrorError):
        _ = http_client.get_inference_pipeline_status(pipeline_id="my-pipeline")


def test_pause_inference_pipeline(requests_mock: Mocker) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.post(
        f"{api_url}/inference_pipelines/my-pipeline/pause",
        json={
            "status": "success",
        },
    )

    # when
    result = http_client.pause_inference_pipeline(pipeline_id="my-pipeline")

    # then
    assert result == {
        "status": "success",
    }
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key"
    }, "Expected payload to contain API key"


def test_pause_inference_pipeline_when_pipeline_id_empty() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(InvalidParameterError):
        _ = http_client.pause_inference_pipeline(pipeline_id="")


def test_pause_inference_pipeline_when_pipeline_id_not_found(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.post(
        f"{api_url}/inference_pipelines/my-pipeline/pause",
        status_code=404,
    )

    # when
    with pytest.raises(HTTPCallErrorError):
        _ = http_client.pause_inference_pipeline(pipeline_id="my-pipeline")


def test_resume_inference_pipeline(requests_mock: Mocker) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.post(
        f"{api_url}/inference_pipelines/my-pipeline/resume",
        json={
            "status": "success",
        },
    )

    # when
    result = http_client.resume_inference_pipeline(pipeline_id="my-pipeline")

    # then
    assert result == {
        "status": "success",
    }
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key"
    }, "Expected payload to contain API key"


def test_resume_inference_pipeline_when_pipeline_id_empty() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(InvalidParameterError):
        _ = http_client.resume_inference_pipeline(pipeline_id="")


def test_resume_inference_pipeline_when_pipeline_id_not_found(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.post(
        f"{api_url}/inference_pipelines/my-pipeline/resume",
        status_code=404,
    )

    # when
    with pytest.raises(HTTPCallErrorError):
        _ = http_client.resume_inference_pipeline(pipeline_id="my-pipeline")


def test_terminate_inference_pipeline(requests_mock: Mocker) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.post(
        f"{api_url}/inference_pipelines/my-pipeline/terminate",
        json={
            "status": "success",
        },
    )

    # when
    result = http_client.terminate_inference_pipeline(pipeline_id="my-pipeline")

    # then
    assert result == {
        "status": "success",
    }
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key"
    }, "Expected payload to contain API key"


def test_terminate_inference_pipeline_when_pipeline_id_empty() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(InvalidParameterError):
        _ = http_client.terminate_inference_pipeline(pipeline_id="")


def test_terminate_inference_pipeline_when_pipeline_id_not_found(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.post(
        f"{api_url}/inference_pipelines/my-pipeline/terminate",
        status_code=404,
    )

    # when
    with pytest.raises(HTTPCallErrorError):
        _ = http_client.terminate_inference_pipeline(pipeline_id="my-pipeline")


def test_consume_inference_pipeline_result(requests_mock: Mocker) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.get(
        f"{api_url}/inference_pipelines/my-pipeline/consume",
        json={
            "status": "success",
        },
    )

    # when
    result = http_client.consume_inference_pipeline_result(
        pipeline_id="my-pipeline",
        excluded_fields=["a"],
    )

    # then
    assert result == {
        "status": "success",
    }
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key",
        "excluded_fields": ["a"],
    }, "Expected payload to contain API key"


def test_consume_inference_pipeline_result_when_pipeline_id_empty() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(InvalidParameterError):
        _ = http_client.consume_inference_pipeline_result(pipeline_id="")


def test_consume_inference_pipeline_result_when_pipeline_id_not_found(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.get(
        f"{api_url}/inference_pipelines/my-pipeline/consume",
        status_code=404,
    )

    # when
    with pytest.raises(HTTPCallErrorError):
        _ = http_client.consume_inference_pipeline_result(pipeline_id="my-pipeline")


def test_start_inference_pipeline_with_workflow_when_configuration_does_not_specify_workflow() -> (
    None
):
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(InvalidParameterError):
        http_client.start_inference_pipeline_with_workflow(
            video_reference="rtsp://some/stream"
        )


def test_start_inference_pipeline_with_workflow_when_configuration_does_over_specify_workflow() -> (
    None
):
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(InvalidParameterError):
        http_client.start_inference_pipeline_with_workflow(
            video_reference="rtsp://some/stream",
            workflow_specification={},
            workspace_name="some",
            workflow_id="some",
        )


def test_start_inference_pipeline_with_workflow_when_configuration_is_valid(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    requests_mock.post(
        f"{api_url}/inference_pipelines/initialise",
        json={
            "status": "success",
        },
    )

    # when
    result = http_client.start_inference_pipeline_with_workflow(
        video_reference="rtsp://some/stream",
        workspace_name="some",
        workflow_id="other",
    )

    # then
    assert result == {
        "status": "success",
    }
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key",
        "video_configuration": {
            "type": "VideoConfiguration",
            "video_reference": "rtsp://some/stream",
            "max_fps": None,
            "source_buffer_filling_strategy": "DROP_OLDEST",
            "source_buffer_consumption_strategy": "EAGER",
            "video_source_properties": None,
            "batch_collection_timeout": None,
        },
        "processing_configuration": {
            "type": "WorkflowConfiguration",
            "workflow_specification": None,
            "workspace_name": "some",
            "workflow_id": "other",
            "image_input_name": "image",
            "workflows_parameters": None,
            "workflows_thread_pool_workers": 4,
            "cancel_thread_pool_tasks_on_exit": True,
            "video_metadata_input_name": "video_metadata",
        },
        "sink_configuration": {
            "type": "MemorySinkConfiguration",
            "results_buffer_size": 64,
        },
    }
