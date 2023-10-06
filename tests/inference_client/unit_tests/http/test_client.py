import base64
import json
from io import BytesIO
from unittest import mock
from unittest.mock import MagicMock

import pytest
from requests import HTTPError, Request, Response
from requests_mock.mocker import Mocker

from inference_client.http import client
from inference_client.http.client import (
    _ensure_model_is_selected,
    _determine_client_mode,
    _determine_client_downsizing_parameters,
    wrap_errors,
    InferenceHTTPClient,
)
from inference_client.http.entities import (
    HTTPClientMode,
    ModelDescription,
    CLASSIFICATION_TASK,
    RegisteredModels,
    InferenceConfiguration,
)
from inference_client.http.errors import (
    ModelNotSelectedError,
    HTTPCallErrorError,
    HTTPClientError,
    InvalidModelIdentifier,
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
    assert requests_mock.request_history[0].query == "api_key=my-api-key&confidence=0.5"
    assert requests_mock.request_history[1].query == "api_key=my-api-key&confidence=0.5"


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
        == "api_key=my-api-key&confidence=0.5&format=image"
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
    assert requests_mock.last_request.query == "api_key=my-api-key&confidence=0.5"


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
    configuration = InferenceConfiguration(confidence_threshold=0.5)
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
    }
    assert requests_mock.request_history[1].json() == {
        "model_id": "some/1",
        "api_key": "my-api-key",
        "image": {"type": "base64", "value": "another_image"},
        "visualize_predictions": False,
        "confidence": 0.5,
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
    }
