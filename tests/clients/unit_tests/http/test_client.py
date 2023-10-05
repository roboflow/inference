import json

import pytest
from requests import HTTPError, Request, Response

from clients.http.client import (
    _ensure_model_is_selected,
    _determine_client_mode,
    _determine_client_downsizing_parameters,
    wrap_errors,
)
from clients.http.entities import HTTPClientMode, ModelDescription, CLASSIFICATION_TASK
from clients.http.errors import (
    ModelNotSelectedError,
    HTTPCallErrorError,
    HTTPClientError,
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
