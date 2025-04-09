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
        "description": "DocTR",
        "type": "ocr",
        "payload": {
            "image": {
                "type": "url",
                "value": "https://media.roboflow.com/swift.png",
            }
        },
        "expected_response": {
            "result": "- was thinking earlier today that I have gone through, to use the lingo, eras of listening to each of Swift's Eras. Meta indeed. I started listening to Ms. Swift's music after hearing the Midnights album. A few weeks after hearing the album for the first time, - found myself playing various songs on repeat. I listened to the album in order multiple times.",
            "time": 2.61976716702338,
        },
    }
]


def bool_env(val):
    if isinstance(val, bool):
        return val
    return val.lower() in ["true", "1", "t", "y", "yes"]


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_DOCTR_TEST", False)), reason="Skipping DocTR test"
)
@pytest.mark.parametrize("test", TESTS)
def test_doctr(test, clean_loaded_models_fixture):
    payload = deepcopy(test["payload"])
    payload["api_key"] = api_key
    response = requests.post(
        f"{base_url}:{port}/doctr/ocr",
        json=payload,
    )
    try:
        response.raise_for_status()
        data = response.json()
        try:
            assert "result" in data
        except:
            print(f"Invalid response: {data}, expected 'result' in data")

        try:
            assert isinstance(data["result"], str) and len(data["result"]) > 0
        except:
            print(f"Invalid response: {data['result']}, expected a non-empty string")

        try:
            assert data["result"] == test["expected_response"]["result"]
        except:
            print(
                f"Invalid response: {data['result']}, expected {test['expected_response']['result']}"
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
    test_doctr()
