import base64
import json
from io import BytesIO
from unittest import mock
from unittest.mock import MagicMock

import pytest
from requests import HTTPError, Request, Response
from requests_mock.mocker import Mocker

from inference_sdk.http import client
from inference_sdk.http.client import (
    InferenceHTTPClient,
    _determine_client_downsizing_parameters,
    _determine_client_mode,
    _ensure_model_is_selected,
    wrap_errors,
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


def test_ensure_model_is_selected_when_model_is_selected() -> None:
    # when
    _ensure_model_is_selected(model_id="some/2")


def test_ensure_model_is_selected_when_model_is_not_selected() -> None:
    # when
    with pytest.raises(ModelNotSelectedError):
        _ensure_model_is_selected(model_id=None)


@pytest.mark.parametrize(
    "api_url", ["https://detections.roboflow.com", "inference.roboflow.com"]
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


def test_wrap_errors_when_connection_error_occurs() -> None:
    # given
    @wrap_errors
    def example() -> None:
        raise ConnectionError()

    # when
    with pytest.raises(HTTPClientError):
        example()


def test_wrap_errors_when_unknown_error_occurs() -> None:
    # given
    @wrap_errors
    def example() -> None:
        raise Exception()

    # when
    with pytest.raises(Exception):
        example()


def test_setting_configuration_statically() -> None:
    # given
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url="https://some.com")
    configuration = InferenceConfiguration(visualize_labels=True)

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


def test_client_unload_single_model_in_v0_mode() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(WrongClientModeError):
        with http_client.use_api_v0():
            _ = http_client.unload_model(model_id="other/1")


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
    result = http_client.unload_model(model_id="other/1")

    # then
    assert result == RegisteredModels(
        models=[ModelDescription(model_id="some/1", task_type="classification")]
    )
    assert requests_mock.last_request.json() == {
        "model_id": "other/1",
    }


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


def test_client_load_model_in_v0_mode() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(WrongClientModeError):
        with http_client.use_api_v0():
            _ = http_client.load_model(model_id="other/1")


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


def test_list_loaded_models_in_v0_mode() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(WrongClientModeError):
        with http_client.use_api_v0():
            _ = http_client.list_loaded_models()


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


def test_get_model_description_in_v0_mode() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(WrongClientModeError):
        with http_client.use_api_v0():
            _ = http_client.get_model_description(model_id="some/1")


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


def test_get_model_description_when_model_was_not_loaded_before_and_successful_load(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    requests_mock.get(
        f"{api_url}/model/registry",
        [
            {"json": {"models": []}},
            {
                "json": {
                    "models": [{"model_id": "some/1", "task_type": "classification"}]
                }
            },
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


def test_get_model_description_when_model_was_not_loaded_before_and_unsuccessful_load(
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    requests_mock.get(
        f"{api_url}/model/registry",
        [
            {"json": {"models": []}},
            {
                "json": {
                    "models": [{"model_id": "some/1", "task_type": "classification"}]
                }
            },
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


def test_infer_from_api_v0_when_model_not_selected() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(ModelNotSelectedError):
        _ = http_client.infer_from_api_v0(inference_input="https://some/image.jpg")


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
    assert requests_mock.last_request.query == "api_key=my-api-key&confidence=0.5"


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


@mock.patch.object(client, "load_static_inference_input")
def test_infer_from_api_v0_when_request_succeed_for_object_detection_with_batch_request(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.return_value = [
        ("base64_image", 0.5),
        ("another_image", None),
    ]
    configuration = InferenceConfiguration(confidence_threshold=0.5)
    http_client.configure(inference_configuration=configuration)
    requests_mock.post(
        f"{api_url}/some/1",
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
        inference_input="https://some/image.jpg", model_id="some/1"
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


def test_infer_from_api_v1_when_model_id_is_not_selected() -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)

    # when
    with pytest.raises(ModelNotSelectedError):
        _ = http_client.infer_from_api_v1(
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


@mock.patch.object(client, "load_static_inference_input")
def test_infer_from_api_v1_when_request_succeed_for_object_detection_with_batch_request(
    load_static_inference_input_mock: MagicMock,
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
        inference_input="https://some/image.jpg", model_id="some/1"
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
        "model_id": "some/1",
        "api_key": "my-api-key",
        "image": {"type": "base64", "value": "base64_image"},
        "visualize_predictions": False,
        "confidence": 0.5,
        "disable_active_learning": True,
    }
    assert requests_mock.request_history[1].json() == {
        "model_id": "some/1",
        "api_key": "my-api-key",
        "image": {"type": "base64", "value": "another_image"},
        "visualize_predictions": False,
        "confidence": 0.5,
        "disable_active_learning": True,
    }


@mock.patch.object(client, "load_static_inference_input")
def test_infer_from_api_v1_when_request_succeed_for_object_detection_with_visualisation(
    load_static_inference_input_mock: MagicMock,
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


def test_prompt_cogvlm_in_v0_mode() -> None:
    # given
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url="http://some.com")
    http_client.select_api_v0()

    # when
    with pytest.raises(WrongClientModeError):
        _ = http_client.prompt_cogvlm(
            visual_prompt="https://some.com/image.jpg",
            text_prompt="What is the content of that picture?",
        )


@mock.patch.object(client, "load_static_inference_input")
def test_prompt_cogvlm_when_successful_response_is_returned(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.return_value = [("base64_image", 0.5)]
    requests_mock.post(
        f"{api_url}/llm/cogvlm",
        json={
            "response": "The image portrays a confident and happy man, possibly in a professional setting.",
            "time": 12.274745374999952,
        },
    )

    # when
    result = http_client.prompt_cogvlm(
        visual_prompt="/some/image.jpg",
        text_prompt="What is the topic of that picture?",
        chat_history=[("A", "B")],
    )

    # then
    assert result == {
        "response": "The image portrays a confident and happy man, possibly in a professional setting.",
        "time": 12.274745374999952,
    }, "Result must match the value returned by HTTP endpoint"
    assert requests_mock.request_history[0].json() == {
        "model_id": "cogvlm",
        "api_key": "my-api-key",
        "image": {"type": "base64", "value": "base64_image"},
        "prompt": "What is the topic of that picture?",
        "history": [["A", "B"]],
    }, "Request must contain API key, model id, prompt, chat history and image encoded in standard format"


@mock.patch.object(client, "load_static_inference_input")
def test_prompt_cogvlm_when_unsuccessful_response_is_returned(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.return_value = [("base64_image", 0.5)]
    requests_mock.post(
        f"{api_url}/llm/cogvlm",
        json={
            "message": "Cannot load CogVLM.",
        },
        status_code=500,
    )

    with pytest.raises(HTTPCallErrorError):
        _ = http_client.prompt_cogvlm(
            visual_prompt="/some/image.jpg",
            text_prompt="What is the topic of that picture?",
            chat_history=[("A", "B")],
        )


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


def test_detect_gazes_in_v0_mode() -> None:
    # given
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url="http://some.com")
    http_client.select_api_v0()

    # when
    with pytest.raises(WrongClientModeError):
        _ = http_client.detect_gazes(
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
    result = http_client.get_clip_image_embeddings(inference_input="/some/image.jpg")

    # then
    assert (
        result == expected_prediction
    ), "Result must match the value returned by HTTP endpoint"
    assert requests_mock.request_history[0].json() == {
        "image": {"type": "base64", "value": "base64_image"},
    }, "Request must contain image encoded in standard format"


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


def test_get_clip_text_embeddings_when_single_image_given(
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
    result = http_client.get_clip_text_embeddings(text="some")

    # then
    assert (
        result == expected_prediction
    ), "Result must match the value returned by HTTP endpoint"
    assert requests_mock.request_history[0].json() == {
        "api_key": "my-api-key",
        "text": "some",
    }, "Request must contain API key and text"


@mock.patch.object(client, "load_static_inference_input")
def test_get_clip_text_embeddings_when_faulty_response_returned(
    load_static_inference_input_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    # given
    api_url = "http://some.com"
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url=api_url)
    load_static_inference_input_mock.return_value = [("base64_image", 0.5)]
    requests_mock.post(
        f"{api_url}/clip/embed_text",
        json={
            "message": "Cannot load Clip model.",
        },
        status_code=500,
    )

    with pytest.raises(HTTPCallErrorError):
        _ = http_client.get_clip_text_embeddings(text="some")


def test_clip_compare_when_invalid_subject_given() -> None:
    # given
    http_client = InferenceHTTPClient(api_key="my-api-key", api_url="http://some.com")

    # when
    with pytest.raises(InvalidParameterError):
        _ = http_client.clip_compare(
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
    }, "Request must contain API key, subject and prompt types as text, exact values of subject and list of prompt values"


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
