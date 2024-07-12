import pytest

from inference_sdk import InferenceHTTPClient
from inference_sdk.http.errors import HTTPCallErrorError
from tests.inference.hosted_platform_tests.conftest import IMAGE_URL


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
        api_url=core_models_service_url, api_key="invalid"
    ).select_api_v0()

    # when
    with pytest.raises(HTTPCallErrorError) as error:
        _ = client.ocr_image(IMAGE_URL)

    # then
    assert error.value.status_code == 403, "Expected to see unauthorised error"
