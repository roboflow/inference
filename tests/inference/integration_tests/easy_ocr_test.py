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
        "description": "EasyOCR",
        "type": "ocr",
        "payload": {
            "image": [
                {
                    "type": "url",
                    "value": "https://media.roboflow.com/swift.png",
                },
                {
                    "type": "url",
                    "value": "https://media.roboflow.com/swift.png",
                },
            ]
        },
        "expected_response": {
            "result": [
                "was thinking earlier today that have gone through, to use the lingo, eras Of listening to each of Swift's Eras. Meta indeed: started listening to Ms. Swift's music after hearing the Midnights album: few weeks after hearing the album for the first time, found myself playing various songs on repeat. listened to the album in order multiple times:, expected was thinking earlier today that have gone through, to use the lingo, eras of listening to each of Swift's Eras: Meta indeed: started listening to Ms. Swift's music after hearing the Midnights album. A few weeks after hearing the album for the first time, found myself playing various songs on repeat. listened to the album in order multiple times:",
                "was thinking earlier today that have gone through, to use the lingo, eras Of listening to each of Swift's Eras. Meta indeed: started listening to Ms. Swift's music after hearing the Midnights album: few weeks after hearing the album for the first time, found myself playing various songs on repeat. listened to the album in order multiple times:, expected was thinking earlier today that have gone through, to use the lingo, eras of listening to each of Swift's Eras: Meta indeed: started listening to Ms. Swift's music after hearing the Midnights album. A few weeks after hearing the album for the first time, found myself playing various songs on repeat. listened to the album in order multiple times:",
            ],
            "time": 2.61976716702338,
        },
    },
    {
        "description": "EasyOCR",
        "type": "ocr",
        "payload": {
            "image": {
                "type": "url",
                "value": "https://media.roboflow.com/swift.png",
            }
        },
        "expected_response": {
            "result": "was thinking earlier today that have gone through, to use the lingo, eras Of listening to each of Swift's Eras. Meta indeed: started listening to Ms. Swift's music after hearing the Midnights album: few weeks after hearing the album for the first time, found myself playing various songs on repeat. listened to the album in order multiple times:, expected was thinking earlier today that have gone through, to use the lingo, eras of listening to each of Swift's Eras: Meta indeed: started listening to Ms. Swift's music after hearing the Midnights album. A few weeks after hearing the album for the first time, found myself playing various songs on repeat. listened to the album in order multiple times:",
            "time": 2.61976716702338,
        },
    },
]


def bool_env(val):
    if isinstance(val, bool):
        return val
    return val.lower() in ["true", "1", "t", "y", "yes"]


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_EASY_OCR_TEST", False)), reason="Skipping EasyOCR test"
)
@pytest.mark.parametrize("test", TESTS)
def test_easy_ocr(test, clean_loaded_models_fixture):
    payload = deepcopy(test["payload"])
    payload["api_key"] = api_key
    response = requests.post(
        f"{base_url}:{port}/easy_ocr/ocr",
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
            assert "predictions" in data
        except:
            print(f"Invalid response: {data}, expected 'predictions' in data")

        if type(test["payload"]["image"]) is not list:
            try:
                assert isinstance(data["result"], str) and len(data["result"]) > 0
            except:
                print(
                    f"Invalid response: {data['result']}, expected a non-empty string"
                )

        if type(test["payload"]["image"]) is not list:
            try:
                assert data["result"] == test["expected_response"]["result"]
            except:
                print(
                    f"Invalid response: {data['result']}, expected {test['expected_response']['result']}"
                )
        else:
            result = [d["result"] for d in data]
            try:
                assert result == test["expected_response"]["result"]
            except:
                print(
                    f"Invalid response: {result}, expected {test['expected_response']['result']}"
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
    test_easy_ocr()
