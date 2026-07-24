import pytest

from inference_sdk import InferenceHTTPClient
from tests.inference.hosted_platform_tests.conftest import IMAGE_URL, ROBOFLOW_API_KEY


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_rfdetr_object_detection_model_when_valid_response_expected(
    object_detection_service_url: str,
    rfdetr_od_model_id: str,
) -> None:
    # given
    client = InferenceHTTPClient(
        api_url=object_detection_service_url,
        api_key=ROBOFLOW_API_KEY,
    ).select_api_v0()

    # when
    response = client.infer(IMAGE_URL, model_id=rfdetr_od_model_id)

    # then
    assert isinstance(response, dict), "Expected dict as response"
    assert "predictions" in response, "Expected 'predictions' key in response"
    assert (
        len(response["predictions"]) > 0
    ), f"Expected at least one instance detected in {IMAGE_URL}"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_rfdetr_instance_segmentation_model_when_valid_response_expected(
    instance_segmentation_service_url: str,
    rfdetr_is_model_id: str,
) -> None:
    # given
    client = InferenceHTTPClient(
        api_url=instance_segmentation_service_url,
        api_key=ROBOFLOW_API_KEY,
    ).select_api_v0()

    # when
    response = client.infer(IMAGE_URL, model_id=rfdetr_is_model_id)

    # then
    assert isinstance(response, dict), "Expected dict as response"
    assert "predictions" in response, "Expected 'predictions' key in response"
    assert (
        len(response["predictions"]) > 0
    ), f"Expected at least one instance detected in {IMAGE_URL}"
