import pytest
import requests

from inference_sdk import InferenceHTTPClient
from inference_sdk.http.errors import HTTPCallErrorError
from tests.inference.hosted_platform_tests.conftest import (
    IMAGE_URL,
    ROBOFLOW_API_KEY,
    PlatformEnvironment,
)

EXPECTED_AUTH_ERROR_FOR_ENVIRONMENT = {
    PlatformEnvironment.ROBOFLOW_STAGING_LAMBDA: 403,
    PlatformEnvironment.ROBOFLOW_PLATFORM_LAMBDA: 403,
    PlatformEnvironment.ROBOFLOW_STAGING_SERVERLESS: 401,
    PlatformEnvironment.ROBOFLOW_PLATFORM_SERVERLESS: 401,
    PlatformEnvironment.ROBOFLOW_STAGING_LOCALHOST: 403,
    PlatformEnvironment.ROBOFLOW_PLATFORM_LOCALHOST: 403,
}


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_core_model_without_api_key(
    core_models_service_url: str,
    platform_environment: PlatformEnvironment,
) -> None:
    # given
    client = InferenceHTTPClient(
        api_url=core_models_service_url,
    ).select_api_v0()

    # when
    with pytest.raises(HTTPCallErrorError) as error:
        _ = client.ocr_image(IMAGE_URL)

    # then
    assert (
        error.value.status_code
        == EXPECTED_AUTH_ERROR_FOR_ENVIRONMENT[platform_environment]
    ), "Expected to see unauthorised error"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_core_model_with_invalid_api_key(
    core_models_service_url: str,
    platform_environment: PlatformEnvironment,
) -> None:
    # given
    client = InferenceHTTPClient(
        api_url=core_models_service_url, api_key="invalid"
    ).select_api_v0()

    # when
    with pytest.raises(HTTPCallErrorError) as error:
        _ = client.ocr_image(IMAGE_URL)

    # then
    assert (
        error.value.status_code
        == EXPECTED_AUTH_ERROR_FOR_ENVIRONMENT[platform_environment]
    ), "Expected to see unauthorised error"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_ocr_model_when_valid_input_given(
    core_models_service_url: str,
) -> None:
    # given
    client = InferenceHTTPClient(
        api_url=core_models_service_url,
        api_key=ROBOFLOW_API_KEY,
    ).select_api_v0()

    # when
    result = client.ocr_image(IMAGE_URL)

    # then
    assert isinstance(result, dict), "Expected dict as response"
    assert set(result.keys()) == {
        "result",
        "time",
        "parent_id",
    }, "Expected all fields to be present in output"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_easy_ocr_model_when_valid_input_given(
    core_models_service_url: str,
) -> None:
    # given
    client = InferenceHTTPClient(
        api_url=core_models_service_url,
        api_key=ROBOFLOW_API_KEY,
    ).select_api_v0()

    # when
    result = client.ocr_image(IMAGE_URL)

    # then
    assert isinstance(result, dict), "Expected dict as response"
    assert set(result.keys()) == {
        "result",
        "time",
        "parent_id",
    }, "Expected all fields to be present in output"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_ocr_model_when_non_https_input_url_given(
    core_models_service_url: str,
) -> None:
    # when
    response = requests.post(
        f"{core_models_service_url}/doctr/ocr",
        params={
            "api_key": ROBOFLOW_API_KEY,
        },
        json={
            "image": {"type": "url", "value": "http://some.com/image.jpg"},
        },
    )

    # then
    assert response.status_code == 400, "Expected bad request"
    error_message = response.json()["message"]
    assert (
        "non https:// URL" in error_message
    ), "Expected bad request be caused by http protocol"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_easy_ocr_model_when_non_https_input_url_given(
    core_models_service_url: str,
) -> None:
    # when
    response = requests.post(
        f"{core_models_service_url}/easy_ocr/ocr",
        params={
            "api_key": ROBOFLOW_API_KEY,
        },
        json={
            "image": {"type": "url", "value": "http://some.com/image.jpg"},
        },
    )

    # then
    assert response.status_code == 400, "Expected bad request"
    error_message = response.json()["message"]
    assert (
        "non https:// URL" in error_message
    ), "Expected bad request be caused by http protocol"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_ocr_model_when_ip_based_input_url_given(
    core_models_service_url: str,
) -> None:
    # when
    response = requests.post(
        f"{core_models_service_url}/doctr/ocr",
        params={
            "api_key": ROBOFLOW_API_KEY,
        },
        json={
            "image": {"type": "url", "value": "https://127.0.0.1/image.jpg"},
        },
    )

    # then
    assert response.status_code == 400, "Expected bad request"
    error_message = response.json()["message"]
    assert (
        "URL without FQDN" in error_message
    ), "Expected bad request be caused by lack of FQDN"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_easy_ocr_model_when_ip_based_input_url_given(
    core_models_service_url: str,
) -> None:
    # when
    response = requests.post(
        f"{core_models_service_url}/easy_ocr/ocr",
        params={
            "api_key": ROBOFLOW_API_KEY,
        },
        json={
            "image": {"type": "url", "value": "https://127.0.0.1/image.jpg"},
        },
    )

    # then
    assert response.status_code == 400, "Expected bad request"
    error_message = response.json()["message"]
    assert (
        "URL without FQDN" in error_message
    ), "Expected bad request be caused by lack of FQDN"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_clip_model_when_valid_input_given(
    core_models_service_url: str,
) -> None:
    # given
    client = InferenceHTTPClient(
        api_url=core_models_service_url,
        api_key=ROBOFLOW_API_KEY,
    ).select_api_v0()

    # when
    result = client.clip_compare(subject=IMAGE_URL, prompt=["cat", "dog"])

    # then
    assert isinstance(result, dict), "Expected dict as response"
    assert set(result.keys()) == {
        "similarity",
        "time",
        "parent_id",
        "inference_id",
        "frame_id",
    }, "Expected all fields to be present in output"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_clip_model_when_non_https_input_url_given(
    core_models_service_url: str,
) -> None:
    # when
    response = requests.post(
        f"{core_models_service_url}/clip/compare",
        params={
            "api_key": ROBOFLOW_API_KEY,
        },
        json={
            "subject": {"type": "url", "value": "http://some.com/image.jpg"},
            "prompt": ["cat", "dog"],
        },
    )

    # then
    assert response.status_code == 400, "Expected bad request"
    error_message = response.json()["message"]
    assert (
        "non https:// URL" in error_message
    ), "Expected bad request be caused by http protocol"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_clip_model_when_ip_based_input_url_given(
    core_models_service_url: str,
) -> None:
    # when
    response = requests.post(
        f"{core_models_service_url}/clip/compare",
        params={
            "api_key": ROBOFLOW_API_KEY,
        },
        json={
            "subject": {"type": "url", "value": "https://127.0.0.1/image.jpg"},
            "prompt": ["cat", "dog"],
        },
    )

    # then
    assert response.status_code == 400, "Expected bad request"
    error_message = response.json()["message"]
    assert (
        "URL without FQDN" in error_message
    ), "Expected bad request be caused by lack of FQDN"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_yolo_world_model_when_valid_input_given(
    core_models_service_url: str,
) -> None:
    # given
    client = InferenceHTTPClient(
        api_url=core_models_service_url,
        api_key=ROBOFLOW_API_KEY,
    ).select_api_v0()

    # when
    result = client.infer_from_yolo_world(IMAGE_URL, class_names=["cat", "dog"])

    # then
    assert isinstance(result, list), "Expected list as response"
    assert len(result), "One image provided - one output expected"
    assert set(result[0].keys()) == {
        "predictions",
        "image",
        "time",
    }, "Expected all fields to be present in output"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_yolo_world_model_when_non_https_input_url_given(
    core_models_service_url: str,
) -> None:
    # when
    response = requests.post(
        f"{core_models_service_url}/yolo_world/infer",
        params={
            "api_key": ROBOFLOW_API_KEY,
        },
        json={
            "image": {"type": "url", "value": "http://some.com/image.jpg"},
            "text": ["cat", "dog"],
        },
    )

    # then
    assert response.status_code == 400, "Expected bad request"
    error_message = response.json()["message"]
    assert (
        "non https:// URL" in error_message
    ), "Expected bad request be caused by http protocol"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_yolo_world_model_when_ip_based_input_url_given(
    core_models_service_url: str,
) -> None:
    # when
    response = requests.post(
        f"{core_models_service_url}/yolo_world/infer",
        params={
            "api_key": ROBOFLOW_API_KEY,
        },
        json={
            "image": {"type": "url", "value": "https://127.0.0.1/image.jpg"},
            "text": ["cat", "dog"],
        },
    )

    # then
    assert response.status_code == 400, "Expected bad request"
    error_message = response.json()["message"]
    assert (
        "URL without FQDN" in error_message
    ), "Expected bad request be caused by lack of FQDN"
