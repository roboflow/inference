import os

import pytest
import requests

from tests.inference.integration_tests.regression_test import bool_env

doc_api_key = os.environ.get("API_KEY")


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_QWEN25_TEST", False)) or bool_env(os.getenv("SKIP_LMM_TEST", False)),
    reason="Skipping Qwen2.5 test",
)
def test_qwen25_inference(server_url: str, clean_loaded_models_fixture) -> None:
    # given
    payload = {
        "api_key": doc_api_key,
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "prompt": "Tell me something about this dog!<system_prompt>You are a helpful assistant.",
        "model_id": "pallet-load-manifest-json-2/13",
    }

    # when
    response = requests.post(
        f"{server_url}/infer/lmm",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert len(data.get("response")) > 0, "Expected a response from Qwen2.5 model"


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_QWEN3_TEST", False)) or bool_env(os.getenv("SKIP_LMM_TEST", False)),
    reason="Skipping Qwen3 test",
)
def test_qwen3_inference(server_url: str, clean_loaded_models_fixture) -> None:
    # given
    payload = {
        "api_key": doc_api_key,
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "prompt": "Tell me something about this dog!<system_prompt>You are a helpful assistant.",
        "model_id": "qwen3vl-2b-instruct",
    }

    # when
    response = requests.post(
        f"{server_url}/infer/lmm",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert len(data.get("response")) > 0, "Expected a response from Qwen3 model"