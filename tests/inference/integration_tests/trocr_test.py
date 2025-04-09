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
        "description": "TrOCR",
        "type": "ocr",
        "payload": {
            "image": {
                "type": "url",
                "value": "https://media.roboflow.com/serial_number.png",
            }
        },
        "expected_response": {
            "result": "3702692432",
            "time": 3,
        },
    }
]


def bool_env(val):
    if isinstance(val, bool):
        return val
    return val.lower() in ["true", "1", "t", "y", "yes"]


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_TROCR_TEST", False)), reason="Skipping TrOCR test"
)
@pytest.mark.parametrize("test", TESTS)
def test_trocr(test):
    # given
    payload = deepcopy(test["payload"])
    payload["api_key"] = api_key

    # when
    response = requests.post(
        f"{base_url}:{port}/ocr/trocr",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert "result" in data
    assert isinstance(data["result"], str) and len(data["result"]) > 0
    assert isinstance(data["time"], float) and data["time"] > 0
    assert data["result"] == test["expected_response"]["result"]


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
    test_trocr(TESTS[0])
