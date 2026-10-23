import math
import os

import cv2
import numpy as np
import pytest

from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from tests.inference.hosted_platform_tests.conftest import (
    ROBOFLOW_API_KEY,
    apply_auth_mode,
)

# Images of the anomaly reference project, far from the threshold on every model.
ASSETS_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "workflows",
        "integration_tests",
        "execution",
        "assets",
    )
)
ANOMALOUS_IMAGE = cv2.imread(os.path.join(ASSETS_DIR, "anomaly_anomalous.jpg"))
NORMAL_IMAGE = cv2.imread(os.path.join(ASSETS_DIR, "anomaly_normal.jpg"))


def _client(service_url: str, auth_mode: str, api_v0: bool, **configuration):
    client = InferenceHTTPClient(api_url=service_url, api_key=ROBOFLOW_API_KEY)
    if api_v0:
        client.select_api_v0()
    client.configure(InferenceConfiguration(**configuration))
    return apply_auth_mode(client, auth_mode)


def _anomalous_confidence(response: dict) -> float:
    margin = (response["anomaly_score"] - response["anomaly_threshold"]) / response[
        "anomaly_scale"
    ]
    return 1 / (1 + math.exp(-margin))


@pytest.mark.flaky(retries=4, delay=1)
@pytest.mark.parametrize("api_v0", [True, False], ids=["v0", "v1"])
def test_anomaly_model_decides_at_the_calibrated_threshold_by_default(
    classification_service_url: str,
    anomaly_detection_model_id: str,
    auth_mode: str,
    api_v0: bool,
) -> None:
    # given
    client = _client(classification_service_url, auth_mode, api_v0)

    # when
    anomalous, normal = (
        client.infer(image, model_id=anomaly_detection_model_id)
        for image in (ANOMALOUS_IMAGE, NORMAL_IMAGE)
    )

    # then
    assert anomalous["is_anomalous"] is True and anomalous["top"] == "anomalous"
    assert normal["is_anomalous"] is False and normal["top"] == "normal"
    for response in (anomalous, normal):
        assert response["anomaly_scale"] > 0
        assert "anomaly_map" not in response or response["anomaly_map"] is None
        anomalous_confidence = _anomalous_confidence(response)
        served = {p["class"]: p["confidence"] for p in response["predictions"]}
        assert served["anomalous"] == pytest.approx(anomalous_confidence, abs=1e-4)
        assert response["is_anomalous"] is (anomalous_confidence >= 0.5)


@pytest.mark.flaky(retries=4, delay=1)
@pytest.mark.parametrize("api_v0", [True, False], ids=["v0", "v1"])
def test_anomaly_model_verdict_follows_the_request_confidence(
    classification_service_url: str,
    anomaly_detection_model_id: str,
    auth_mode: str,
    api_v0: bool,
) -> None:
    # given
    default_response = _client(classification_service_url, auth_mode, api_v0).infer(
        NORMAL_IMAGE, model_id=anomaly_detection_model_id
    )
    anomalous_confidence = _anomalous_confidence(default_response)
    below = _client(
        classification_service_url,
        auth_mode,
        api_v0,
        confidence_threshold=round(anomalous_confidence / 2, 4),
    )

    # when
    response = below.infer(NORMAL_IMAGE, model_id=anomaly_detection_model_id)

    # then: the threshold in effect never moves, only the verdict does
    assert default_response["is_anomalous"] is False
    assert response["is_anomalous"] is True and response["top"] == "anomalous"
    assert response["anomaly_threshold"] == default_response["anomaly_threshold"]
    assert response["anomaly_score"] == default_response["anomaly_score"]


@pytest.mark.flaky(retries=4, delay=1)
def test_anomaly_model_serves_the_map_at_network_resolution_on_request(
    classification_service_url: str,
    anomaly_detection_model_id: str,
    auth_mode: str,
) -> None:
    # given
    client = _client(
        classification_service_url, auth_mode, False, include_anomaly_map=True
    )

    # when
    response = client.infer(ANOMALOUS_IMAGE, model_id=anomaly_detection_model_id)

    # then
    anomaly_map = response["anomaly_map"]
    assert isinstance(anomaly_map, np.ndarray)
    assert anomaly_map.dtype == np.float32
    assert anomaly_map.ndim == 2 and anomaly_map.shape[0] == anomaly_map.shape[1]
    assert float(anomaly_map.max()) > 0
