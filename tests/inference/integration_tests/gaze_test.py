import json
import os
import time
from copy import deepcopy
from pathlib import Path

import pytest
import requests

api_key = os.environ.get("API_KEY")
port = os.environ.get("PORT", 9001)
base_url = os.environ.get("BASE_URL", "http://localhost")

TESTS = [
    {
        "description": "Gaze Detection",
        "type": "gaze_detection",
        "payload": {
            "image": {
                "type": "url",
                "value": "https://media.roboflow.com/inference/man.jpg",
            }
        },
    }
]


def bool_env(val):
    if isinstance(val, bool):
        return val
    return val.lower() in ["true", "1", "t", "y", "yes"]


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_GAZE_TEST", False)), reason="Skipping gaze test"
)
@pytest.mark.parametrize("test", TESTS)
def test_gaze(test, clean_loaded_models_fixture):
    payload = deepcopy(test["payload"])
    payload["api_key"] = api_key
    response = requests.post(
        f"{base_url}:{port}/gaze/gaze_detection",
        json=payload,
    )
    try:
        response.raise_for_status()
        data = response.json()[0]
        try:
            assert "predictions" in data
        except:
            print(f"Invalid response: {data}, expected 'predictions' in data")

        try:
            assert (
                isinstance(data["predictions"], list) and len(data["predictions"]) > 0
            )
        except:
            print(
                f"Invalid response: {data['predictions']}, expected at least one face"
            )
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
    from conftest import on_demand_clean_loaded_models
    test_gaze(TESTS[0], on_demand_clean_loaded_models())
