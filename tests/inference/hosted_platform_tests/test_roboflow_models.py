import pytest
import requests

from inference_sdk import InferenceHTTPClient
from inference_sdk.http.errors import HTTPCallErrorError
from tests.inference.hosted_platform_tests.conftest import ROBOFLOW_API_KEY

IMAGE_URL = "https://media.roboflow.com/inference/dog.jpeg"


def test_infer_from_object_detection_model_without_api_key(
    object_detection_service_url: str,
    detection_model_id: str,
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/{detection_model_id}",
        params={
            "image": IMAGE_URL,
        }
    )

    # then
    assert response.status_code == 401, "Expected to see unauthorised error"


def test_infer_from_object_detection_model_with_invalid_api_key(
    object_detection_service_url: str,
    detection_model_id: str,
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/{detection_model_id}",
        params={
            "image": IMAGE_URL,
            "api_key": "invalid",
        }
    )

    # then
    assert response.status_code == 403, "Expected to see unauthorised error"


def test_infer_from_object_detection_model_with_invalid_model_id(
    object_detection_service_url: str,
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/some/38",
        params={
            "image": IMAGE_URL,
            "api_key": ROBOFLOW_API_KEY,
        }
    )

    # then
    assert response.status_code == 403, "Expected to see unauthorised error, as there is no such model in workspace"


def test_infer_from_object_detection_model_when_valid_response_expected(
    object_detection_service_url: str,
    detection_model_id: str,
) -> None:
    # given
    client = InferenceHTTPClient(
        api_url=object_detection_service_url,
        api_key=ROBOFLOW_API_KEY,
    ).select_api_v0()

    # when
    response = client.infer(IMAGE_URL, model_id=detection_model_id)

    # then
    assert isinstance(response, dict), "Expected dict as response"
    assert set(response.keys()) == {"image", "predictions", "time", "inference_id"}, "Expected all required keys to be provided in response"


def test_infer_from_instance_segmentation_model_without_api_key(
    instance_segmentation_service_url: str,
    segmentation_model_id: str,
) -> None:
    # when
    response = requests.post(
        f"{instance_segmentation_service_url}/{segmentation_model_id}",
        params={
            "image": IMAGE_URL,
        }
    )

    # then
    assert response.status_code == 401, "Expected to see unauthorised error"


def test_infer_from_instance_segmentation_model_with_invalid_api_key(
    instance_segmentation_service_url: str,
    segmentation_model_id: str,
) -> None:
    # when
    response = requests.post(
        f"{instance_segmentation_service_url}/{segmentation_model_id}",
        params={
            "image": IMAGE_URL,
            "api_key": "invalid",
        }
    )

    # then
    assert response.status_code == 403, "Expected to see unauthorised error"


def test_infer_from_instance_segmentation_model_with_invalid_model_id(
    instance_segmentation_service_url: str,
) -> None:
    # when
    response = requests.post(
        f"{instance_segmentation_service_url}/some/38",
        params={
            "image": IMAGE_URL,
            "api_key": ROBOFLOW_API_KEY,
        }
    )

    # then
    assert response.status_code == 403, "Expected to see unauthorised error, as there is no such model in workspace"


def test_infer_from_classification_model_without_api_key(
    classification_service_url: str,
    classification_model_id: str,
) -> None:
    # when
    response = requests.post(
        f"{classification_service_url}/{classification_model_id}",
        params={
            "image": IMAGE_URL,
        }
    )

    # then
    assert response.status_code == 401, "Expected to see unauthorised error"


def test_infer_from_classification_model_with_invalid_api_key(
    classification_service_url: str,
    classification_model_id: str,
) -> None:
    # when
    response = requests.post(
        f"{classification_service_url}/{classification_model_id}",
        params={
            "image": IMAGE_URL,
            "api_key": "invalid",
        }
    )

    # then
    assert response.status_code == 403, "Expected to see unauthorised error"


def test_infer_from_classification_model_with_invalid_model_id(
    classification_service_url: str,
) -> None:
    # when
    response = requests.post(
        f"{classification_service_url}/some/38",
        params={
            "image": IMAGE_URL,
            "api_key": ROBOFLOW_API_KEY,
        }
    )

    # then
    assert response.status_code == 403, "Expected to see unauthorised error, as there is no such model in workspace"


def test_infer_from_core_model_without_api_key(
    core_models_service_url: str,
) -> None:
    client = InferenceHTTPClient(
        api_url=core_models_service_url,
    ).select_api_v0()

    # when
    with pytest.raises(HTTPCallErrorError) as error:
        _ = client.ocr_image(IMAGE_URL)

    # then
    assert error.value.status_code == 403, "Expected to see unauthorised error"


def test_infer_from_core_model_with_invalid_api_key(
    core_models_service_url: str,
) -> None:
    client = InferenceHTTPClient(
        api_url=core_models_service_url,
        api_key="invalid"
    ).select_api_v0()

    # when
    with pytest.raises(HTTPCallErrorError) as error:
        _ = client.ocr_image(IMAGE_URL)

    # then
    assert error.value.status_code == 403, "Expected to see unauthorised error"
