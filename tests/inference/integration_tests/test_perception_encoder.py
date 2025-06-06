import os

import pytest
import requests

from tests.inference.integration_tests.regression_test import bool_env


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_PE_TEST", False)),
    reason="Skipping SmolVLM test",
)
def test_text_embedding(server_url: str, clean_loaded_models_fixture) -> None:
    # given
    payload = {"text": "Come with me if you want to live"}

    # when
    response = requests.post(
        f"{server_url}/perception_encoder/embed_text",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert len(data["embeddings"]) > 0, "Expected embeddings"


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_PE_TEST", False)),
    reason="Skipping SmolVLM test",
)
def test_image_embedding(server_url: str, clean_loaded_models_fixture) -> None:
    # given
    payload = {
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        }
    }

    # when
    response = requests.post(
        f"{server_url}/perception_encoder/embed_image",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert len(data["embeddings"]) > 0, "Expected embeddings"


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_PE_TEST", False)),
    reason="Skipping SmolVLM test",
)
def test_comparison(server_url: str, clean_loaded_models_fixture) -> None:
    # given
    payload = {
        "subject": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "prompt": "Image with dog and a men.",
    }

    # when
    response = requests.post(
        f"{server_url}/perception_encoder/compare",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert len(data["similarity"]) == 1, "Expected similarity"
