import os

import pytest
import requests

from tests.inference.integration_tests.regression_test import bool_env

api_key = os.environ.get("API_KEY")




@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_DEPTH_ESTIMATION_TEST", False)),
    reason="Skipping Depth Estimation test",
)
def test_depth_estimation_inference(
    server_url: str, clean_loaded_models_every_test_fixture
) -> None:
    # given
    payload = {
        "api_key": api_key,
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "model_id": "depth-anything-v2/small",
    }

    # when
    response = requests.post(
        f"{server_url}/infer/depth-estimation",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert "normalized_depth" in data, "Expected normalized_depth in response"
    assert "image" in data, "Expected image in response"
    assert len(data["normalized_depth"]) > 0, "Expected non-empty depth map"

