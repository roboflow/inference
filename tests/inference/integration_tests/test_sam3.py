import os
import time

import pytest
import requests

from tests.inference.integration_tests.regression_test import bool_env

API_KEY = os.environ.get("API_KEY")
PORT = os.environ.get("PORT", 9001)
BASE_URL = os.environ.get("BASE_URL", "http://localhost")


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_SAM3_TESTS", True)),
    reason="Skipping SAM test",
)
def test_image_embedding(
    clean_loaded_models_every_test_fixture
) -> None:
    # given
    payload = {
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
    }

    # when
    response = requests.post(
        f"{BASE_URL}:{PORT}/sam3/embed_image",
        json=payload,
        params={"api_key": API_KEY}
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert "image_id" in data
    assert "time" in data


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_SAM3_TESTS", True)),
    reason="Skipping SAM test",
)
def test_visual_segmentation(
    clean_loaded_models_every_test_fixture
) -> None:
    payload = {
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        }
    }

    # when
    response = requests.post(
        f"{BASE_URL}/sam3/visual_segment",
        json=payload,
        params={"api_key": API_KEY}
    )

    # then
    response.raise_for_status()


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_SAM3_TESTS", True)),
    reason="Skipping SAM test",
)
def test_concept_segmentation(
    clean_loaded_models_every_test_fixture
) -> None:
    payload = {
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "prompts": [
            {
                "type": "text",
                "text": "dog",
            }
        ]
    }

    # when
    response = requests.post(
        f"{BASE_URL}/sam3/concept_segment",
        json=payload,
        params={"api_key": API_KEY}
    )

    # then
    response.raise_for_status()


@pytest.fixture(scope="session", autouse=True)
def setup():
    try:
        res = requests.get(f"{BASE_URL}:{PORT}")
        res.raise_for_status()
        success = True
    except:
        success = False
    MAX_WAIT = int(os.getenv("MAX_WAIT", 30))
    waited = 0
    while not success:
        print("Waiting for server to start...")
        time.sleep(5)
        waited += 5
        try:
            res = requests.get(f"{BASE_URL}:{PORT}")
            res.raise_for_status()
            success = True
        except:
            success = False
        if waited > MAX_WAIT:
            raise Exception("Test server failed to start")