import os

import pytest
import requests

from tests.inference.integration_tests.regression_test import bool_env

api_key = os.environ.get("STAGING_API_KEY_FOR_HACKATON_PROJECT")


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_FLORENCE_TEST", False)),
    reason="Skipping Florence test",
)
def test_florence_inference(server_url: str, clean_loaded_models_fixture) -> None:
    # given
    payload = {
        "api_key": api_key,
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "prompt": "<CAPTION>",
        "model_id": "beer-can-hackathon/127"
    }

    # when
    response = requests.post(
        f"{server_url}/infer/lmm",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert len(data["response"]) > 0, "Expected non empty generatiom"
