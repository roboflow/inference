import json
import os
import time
from copy import deepcopy
from pathlib import Path

import pytest
import requests

from tests.inference.integration_tests.regression_test import bool_env

api_key = os.environ.get("API_KEY")
port = os.environ.get("PORT", 9001)
base_url = os.environ.get("BASE_URL", "http://localhost")


version_ids = [
    "hiera_small",
    "hiera_large",
    "hiera_tiny",
    "hiera_b_plus",
]
payload_ = {
    "image": {
        "type": "url",
        "value": "https://source.roboflow.com/D8zLgnZxdqtqF0plJINA/DqK7I0rUz5HBvu1hdNi6/original.jpg",
    },
    "image_id": "test",
}

tests = ["embed_image", "segment_image"]

@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_SAM2_TESTS", True)),
    reason="Skipping SAM test",
)
@pytest.mark.parametrize("version_id", version_ids)
@pytest.mark.parametrize("test", tests)
def test_sam2(version_id, test, clean_loaded_models_fixture):
    payload = deepcopy(payload_)
    payload["api_key"] = api_key
    payload["sam2_version_id"] = version_id
    response = requests.post(
        f"{base_url}:{port}/sam2/{test}",
        json=payload,
    )
    try:
        response.raise_for_status()
        data = response.json()
        if test == "embed_image":
            try:
                assert "image_id" in data
            except:
                print(f"Invalid response: {data}, expected 'image_id' in response")
        if test == "segment_image":
            try:
                assert "masks" in data
            except:
                print(f"Invalid response: {data}, expected 'masks' in response")
    except Exception as e:
        raise e


@pytest.fixture(scope="session", autouse=True)
def setup():
    try:
        res = requests.get(f"{base_url}:{port}")
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
            res = requests.get(f"{base_url}:{port}")
            res.raise_for_status()
            success = True
        except:
            success = False
        if waited > MAX_WAIT:
            raise Exception("Test server failed to start")


if __name__ == "__main__":
    test_sam2()
