import os

import pytest
import requests

from tests.inference.integration_tests.regression_test import bool_env

# Keep up to date with inference.models.aliases.FLORENCE_ALIASES
# Can't import because adds a lot of requirements to testing environment
FLORENCE_ALIASES = {
    "florence-2-base": "florence-pretrains/3",  # since transformers 0.53.3 need newer version of florence2 weights
    "florence-2-large": "florence-pretrains/4",
}

api_key = os.environ.get("melee_API_KEY")


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_FLORENCE_TEST", False))
    or bool_env(os.getenv("SKIP_LMM_TEST", False)),
    reason="Skipping Florence test",
)
def test_florence_lora_inference(server_url: str, clean_loaded_models_fixture) -> None:
    # given
    payload = {
        "api_key": api_key,
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "prompt": "<CAPTION>",
        "model_id": "qwen_playground/80",
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


@pytest.mark.parametrize("model_id", FLORENCE_ALIASES.keys())
@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_FLORENCE_TEST", False))
    or bool_env(os.getenv("SKIP_LMM_TEST", False)),
    reason="Skipping Florence test",
)
def test_florence_inference(
    model_id: str, server_url: str, clean_loaded_models_every_test_fixture
) -> None:
    # given
    payload = {
        "api_key": api_key,
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "prompt": "<CAPTION>",
        "model_id": model_id,
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
