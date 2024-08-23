import pickle

import numpy as np
import pytest
import requests

from inference_sdk import (
    InferenceConfiguration,
    InferenceHTTPClient,
    VisualisationResponseFormat,
)
from tests.inference.hosted_platform_tests.conftest import IMAGE_URL, ROBOFLOW_API_KEY


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_object_detection_model_without_api_key(
    object_detection_service_url: str,
    detection_model_id: str,
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/{detection_model_id}",
        params={
            "image": IMAGE_URL,
        },
    )

    # then
    assert response.status_code == 401, "Expected to see unauthorised error"


@pytest.mark.flaky(retries=4, delay=1)
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
        },
    )

    # then
    assert response.status_code == 403, "Expected to see unauthorised error"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_object_detection_model_with_invalid_model_id(
    object_detection_service_url: str,
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/some/38",
        params={
            "image": IMAGE_URL,
            "api_key": ROBOFLOW_API_KEY,
        },
    )

    # then
    assert (
        response.status_code == 403
    ), "Expected to see unauthorised error, as there is no such model in workspace"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_object_detection_model_when_non_https_image_url_given(
    object_detection_service_url: str,
    detection_model_id: str,
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/{detection_model_id}",
        params={
            "image": f"http://some.com/image.jpg",
            "api_key": ROBOFLOW_API_KEY,
        },
    )

    # then
    assert response.status_code == 400, "Expected to see bad request"
    error_message = response.json()["message"]
    assert (
        "non https:// URL" in error_message
    ), "Expected bad request be caused by http protocol"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_object_detection_model_when_ip_address_as_url_given(
    object_detection_service_url: str,
    detection_model_id: str,
) -> None:
    # when
    response = requests.post(
        f"{object_detection_service_url}/{detection_model_id}",
        params={
            "image": f"https://127.0.0.1/image.jpg",
            "api_key": ROBOFLOW_API_KEY,
        },
    )

    # then
    assert response.status_code == 400, "Expected to see bad request"
    error_message = response.json()["message"]
    assert (
        "URL without FQDN" in error_message
    ), "Expected bad request be caused by lack of FQDN"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_object_detection_model_when_numpy_array_given(
    object_detection_service_url: str,
    detection_model_id: str,
) -> None:
    # given
    data = np.zeros((192, 168, 3), dtype=np.uint8)
    data_bytes = pickle.dumps(data)

    # when
    response = requests.post(
        f"{object_detection_service_url}/{detection_model_id}",
        params={
            "image_type": "numpy",
            "api_key": ROBOFLOW_API_KEY,
        },
        headers={"Content-Type": "application/json"},
        data=data_bytes,
    )

    # then
    assert response.status_code == 400, "Expected to see bad request"
    error_message = response.json()["message"]
    assert (
        "NumPy image type is not supported" in error_message
    ), "Expected bad request be caused by Numpy input"


@pytest.mark.flaky(retries=4, delay=1)
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
    assert set(response.keys()) == {
        "image",
        "predictions",
        "time",
        "inference_id",
    }, "Expected all required keys to be provided in response"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_object_detection_model_when_valid_response_expected_with_visualisation(
    object_detection_service_url: str,
    detection_model_id: str,
) -> None:
    # given
    configuration = InferenceConfiguration(
        format="image",
        output_visualisation_format=VisualisationResponseFormat.NUMPY,
    )
    client = (
        InferenceHTTPClient(
            api_url=object_detection_service_url,
            api_key=ROBOFLOW_API_KEY,
        )
        .configure(configuration)
        .select_api_v0()
    )

    # when
    response = client.infer(IMAGE_URL, model_id=detection_model_id)

    # then
    assert isinstance(response, dict), "Expected dict as response"
    assert set(response.keys()) == {
        "visualization"
    }, "Expected all required keys to be provided in response"
    assert isinstance(
        response["visualization"], np.ndarray
    ), "Expected np array with visualisation as response"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_object_detection_model_when_valid_response_expected_with_visualisation_and_payload(
    object_detection_service_url: str,
    detection_model_id: str,
) -> None:
    # given
    configuration = InferenceConfiguration(
        format="image_and_json",
        output_visualisation_format=VisualisationResponseFormat.NUMPY,
    )
    client = (
        InferenceHTTPClient(
            api_url=object_detection_service_url,
            api_key=ROBOFLOW_API_KEY,
        )
        .configure(configuration)
        .select_api_v0()
    )

    # when
    response = client.infer(IMAGE_URL, model_id=detection_model_id)

    # then
    assert isinstance(response, dict), "Expected dict as response"
    assert set(response.keys()) == {
        "visualization",
        "image",
        "predictions",
        "time",
        "inference_id",
    }, "Expected all required keys to be provided in response"
    assert isinstance(
        response["visualization"], np.ndarray
    ), "Expected np array with visualisation as response"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_instance_segmentation_model_without_api_key(
    instance_segmentation_service_url: str,
    segmentation_model_id: str,
) -> None:
    # when
    response = requests.post(
        f"{instance_segmentation_service_url}/{segmentation_model_id}",
        params={
            "image": IMAGE_URL,
        },
    )

    # then
    assert response.status_code == 401, "Expected to see unauthorised error"


@pytest.mark.flaky(retries=4, delay=1)
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
        },
    )

    # then
    assert response.status_code == 403, "Expected to see unauthorised error"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_instance_segmentation_model_with_invalid_model_id(
    instance_segmentation_service_url: str,
) -> None:
    # when
    response = requests.post(
        f"{instance_segmentation_service_url}/some/38",
        params={
            "image": IMAGE_URL,
            "api_key": ROBOFLOW_API_KEY,
        },
    )

    # then
    assert (
        response.status_code == 403
    ), "Expected to see unauthorised error, as there is no such model in workspace"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_instance_segmentation_model_when_non_https_image_url_given(
    instance_segmentation_service_url: str,
    segmentation_model_id: str,
) -> None:
    # when
    response = requests.post(
        f"{instance_segmentation_service_url}/{segmentation_model_id}",
        params={
            "image": f"http://some.com/image.jpg",
            "api_key": ROBOFLOW_API_KEY,
        },
    )

    # then
    assert response.status_code == 400, "Expected to see bad request"
    error_message = response.json()["message"]
    assert (
        "non https:// URL" in error_message
    ), "Expected bad request be caused by http protocol"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_instance_segmentation_model_when_ip_address_as_url_given(
    instance_segmentation_service_url: str,
    segmentation_model_id: str,
) -> None:
    # when
    response = requests.post(
        f"{instance_segmentation_service_url}/{segmentation_model_id}",
        params={
            "image": f"https://127.0.0.1/image.jpg",
            "api_key": ROBOFLOW_API_KEY,
        },
    )

    # then
    assert response.status_code == 400, "Expected to see bad request"
    error_message = response.json()["message"]
    assert (
        "URL without FQDN" in error_message
    ), "Expected bad request be caused by lack of FQDN"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_instance_segmentation_model_when_numpy_array_given(
    instance_segmentation_service_url: str,
    segmentation_model_id: str,
) -> None:
    # given
    data = np.zeros((192, 168, 3), dtype=np.uint8)
    data_bytes = pickle.dumps(data)

    # when
    response = requests.post(
        f"{instance_segmentation_service_url}/{segmentation_model_id}",
        params={
            "image_type": "numpy",
            "api_key": ROBOFLOW_API_KEY,
        },
        headers={"Content-Type": "application/json"},
        data=data_bytes,
    )

    # then
    assert response.status_code == 400, "Expected to see bad request"
    error_message = response.json()["message"]
    assert (
        "NumPy image type is not supported" in error_message
    ), "Expected bad request be caused by Numpy input"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_instance_segmentation_model_when_valid_response_expected(
    instance_segmentation_service_url: str,
    segmentation_model_id: str,
) -> None:
    # given
    client = InferenceHTTPClient(
        api_url=instance_segmentation_service_url,
        api_key=ROBOFLOW_API_KEY,
    ).select_api_v0()

    # when
    response = client.infer(IMAGE_URL, model_id=segmentation_model_id)

    # then
    assert isinstance(response, dict), "Expected dict as response"
    assert set(response.keys()) == {
        "image",
        "predictions",
        "time",
        "inference_id",
    }, "Expected all required keys to be provided in response"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_instance_segmentation_model_when_valid_response_expected_with_visualisation(
    instance_segmentation_service_url: str,
    segmentation_model_id: str,
) -> None:
    # given
    configuration = InferenceConfiguration(
        format="image",
        output_visualisation_format=VisualisationResponseFormat.NUMPY,
    )
    client = (
        InferenceHTTPClient(
            api_url=instance_segmentation_service_url,
            api_key=ROBOFLOW_API_KEY,
        )
        .configure(configuration)
        .select_api_v0()
    )

    # when
    response = client.infer(IMAGE_URL, model_id=segmentation_model_id)

    # then
    assert isinstance(response, dict), "Expected dict as response"
    assert set(response.keys()) == {
        "visualization"
    }, "Expected all required keys to be provided in response"
    assert isinstance(
        response["visualization"], np.ndarray
    ), "Expected np array with visualisation as response"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_instance_segmentation_model_when_valid_response_expected_with_visualisation_and_payload(
    instance_segmentation_service_url: str,
    segmentation_model_id: str,
) -> None:
    # given
    configuration = InferenceConfiguration(
        format="image_and_json",
        output_visualisation_format=VisualisationResponseFormat.NUMPY,
    )
    client = (
        InferenceHTTPClient(
            api_url=instance_segmentation_service_url,
            api_key=ROBOFLOW_API_KEY,
        )
        .configure(configuration)
        .select_api_v0()
    )

    # when
    response = client.infer(IMAGE_URL, model_id=segmentation_model_id)

    # then
    assert isinstance(response, dict), "Expected dict as response"
    assert set(response.keys()) == {
        "visualization",
        "image",
        "predictions",
        "time",
        "inference_id",
    }, "Expected all required keys to be provided in response"
    assert isinstance(
        response["visualization"], np.ndarray
    ), "Expected np array with visualisation as response"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_classification_model_without_api_key(
    classification_service_url: str,
    classification_model_id: str,
) -> None:
    # when
    response = requests.post(
        f"{classification_service_url}/{classification_model_id}",
        params={
            "image": IMAGE_URL,
        },
    )

    # then
    assert response.status_code == 401, "Expected to see unauthorised error"


@pytest.mark.flaky(retries=4, delay=1)
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
        },
    )

    # then
    assert response.status_code == 403, "Expected to see unauthorised error"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_classification_model_with_invalid_model_id(
    classification_service_url: str,
) -> None:
    # when
    response = requests.post(
        f"{classification_service_url}/some/38",
        params={
            "image": IMAGE_URL,
            "api_key": ROBOFLOW_API_KEY,
        },
    )

    # then
    assert (
        response.status_code == 403
    ), "Expected to see unauthorised error, as there is no such model in workspace"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_classification_model_when_non_https_image_url_given(
    classification_service_url: str,
    classification_model_id: str,
) -> None:
    # when
    response = requests.post(
        f"{classification_service_url}/{classification_model_id}",
        params={
            "image": f"http://some.com/image.jpg",
            "api_key": ROBOFLOW_API_KEY,
        },
    )

    # then
    assert response.status_code == 400, "Expected to see bad request"
    error_message = response.json()["message"]
    assert (
        "non https:// URL" in error_message
    ), "Expected bad request be caused by http protocol"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_classification_model_when_ip_address_as_url_given(
    classification_service_url: str,
    classification_model_id: str,
) -> None:
    # when
    response = requests.post(
        f"{classification_service_url}/{classification_model_id}",
        params={
            "image": f"https://127.0.0.1/image.jpg",
            "api_key": ROBOFLOW_API_KEY,
        },
    )

    # then
    assert response.status_code == 400, "Expected to see bad request"
    error_message = response.json()["message"]
    assert (
        "URL without FQDN" in error_message
    ), "Expected bad request be caused by lack of FQDN"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_classification_model_when_numpy_array_given(
    classification_service_url: str,
    classification_model_id: str,
) -> None:
    # given
    data = np.zeros((192, 168, 3), dtype=np.uint8)
    data_bytes = pickle.dumps(data)

    # when
    response = requests.post(
        f"{classification_service_url}/{classification_model_id}",
        params={
            "image_type": "numpy",
            "api_key": ROBOFLOW_API_KEY,
        },
        headers={"Content-Type": "application/json"},
        data=data_bytes,
    )

    # then
    assert response.status_code == 400, "Expected to see bad request"
    error_message = response.json()["message"]
    assert (
        "NumPy image type is not supported" in error_message
    ), "Expected bad request be caused by Numpy input"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_classification_model_when_valid_response_expected(
    classification_service_url: str,
    classification_model_id: str,
) -> None:
    # given
    client = InferenceHTTPClient(
        api_url=classification_service_url,
        api_key=ROBOFLOW_API_KEY,
    ).select_api_v0()

    # when
    response = client.infer(IMAGE_URL, model_id=classification_model_id)

    # then
    assert isinstance(response, dict), "Expected dict as response"
    assert set(response.keys()) == {
        "image",
        "predictions",
        "inference_id",
        "predicted_classes",
        "time",
    }, "Expected all required keys to be provided in response"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_classification_model_when_valid_response_expected_with_visualisation(
    classification_service_url: str,
    classification_model_id: str,
) -> None:
    # given
    configuration = InferenceConfiguration(
        format="image",
        output_visualisation_format=VisualisationResponseFormat.NUMPY,
    )
    client = (
        InferenceHTTPClient(
            api_url=classification_service_url,
            api_key=ROBOFLOW_API_KEY,
        )
        .configure(configuration)
        .select_api_v0()
    )

    # when
    response = client.infer(IMAGE_URL, model_id=classification_model_id)

    # then
    assert isinstance(response, dict), "Expected dict as response"
    assert set(response.keys()) == {
        "visualization"
    }, "Expected all required keys to be provided in response"
    assert isinstance(
        response["visualization"], np.ndarray
    ), "Expected np array with visualisation as response"


@pytest.mark.flaky(retries=4, delay=1)
def test_infer_from_classification_model_when_valid_response_expected_with_visualisation_and_payload(
    classification_service_url: str,
    classification_model_id: str,
) -> None:
    # given
    configuration = InferenceConfiguration(
        format="image_and_json",
        output_visualisation_format=VisualisationResponseFormat.NUMPY,
    )
    client = (
        InferenceHTTPClient(
            api_url=classification_service_url,
            api_key=ROBOFLOW_API_KEY,
        )
        .configure(configuration)
        .select_api_v0()
    )

    # when
    response = client.infer(IMAGE_URL, model_id=classification_model_id)

    # then
    assert isinstance(response, dict), "Expected dict as response"
    assert set(response.keys()) == {
        "visualization",
        "image",
        "inference_id",
        "predictions",
        "predicted_classes",
        "time",
    }, "Expected all required keys to be provided in response"
    assert isinstance(
        response["visualization"], np.ndarray
    ), "Expected np array with visualisation as response"
