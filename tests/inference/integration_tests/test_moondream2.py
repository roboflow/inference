import os

import pytest
import requests

from tests.inference.integration_tests.regression_test import bool_env

# Keep up to date with inference.models.aliases.MOONDREAM2_ALIASES
# Can't import because adds a lot of requirements to testing environment
MOONDREAM2_ALIASES = {
    "moondream2": "moondream2",
}

api_key = os.environ.get("API_KEY")


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_MOONDREAM2_TEST", False))
    or bool_env(os.getenv("SKIP_LMM_TEST", False)),
    reason="Skipping Moondream2 test",
)
@pytest.mark.parametrize("model_id", MOONDREAM2_ALIASES.keys())
def test_moondream2_inference_object_detection(
    model_id: str, server_url: str, clean_loaded_models_every_test_fixture
) -> None:
    # given
    payload = {
        "api_key": api_key,
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "task_type": "phrase-grounded-object-detection",
        "prompt": "dog",
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
    assert len(data["response"]) > 0, "Expected non empty generation"



@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_MOONDREAM2_TEST", False))
    or bool_env(os.getenv("SKIP_LMM_TEST", False)),
    reason="Skipping SmolVLM test",
)
@pytest.mark.parametrize("model_id", MOONDREAM2_ALIASES.keys())
def test_moondream2_inference_caption(
    model_id: str, server_url: str, clean_loaded_models_every_test_fixture
) -> None:
    # given
    payload = {
        "api_key": api_key,
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "task_type": "caption",
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
    assert len(data["response"]) > 0, "Expected non empty generation"


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_MOONDREAM2_TEST", False))
    or bool_env(os.getenv("SKIP_LMM_TEST", False)),
    reason="Skipping SmolVLM test",
)
@pytest.mark.parametrize("model_id", MOONDREAM2_ALIASES.keys())
def test_moondream2_inference_vqa(
    model_id: str, server_url: str, clean_loaded_models_every_test_fixture
) -> None:
    # given
    payload = {
        "api_key": api_key,
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "task_type": "query",
        "prompt": "describe the image",
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
    assert len(data["response"]) > 0, "Expected non empty generation"

