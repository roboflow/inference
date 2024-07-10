import os

import pytest
import requests

from tests.inference.integration_tests.regression_test import bool_env

api_key = os.environ.get("API_KEY")



@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_PALIGEMMA_TEST", False)) or bool_env(os.getenv("SKIP_LMM_TEST", False)),
    reason="Skipping Paligemma test",
)
def test_paligemma_inference(server_url: str, clean_loaded_models_fixture) -> None:
    # given
    payload = {
        "api_key": api_key,
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "prompt": "Describe the image",
        "model_id": "paligemma-3b-mix-224"
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
