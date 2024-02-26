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
        "description": "YOLO-World",
        "type": "zero-shot-object-detection",
        "payload": {
            "api_key": api_key,
            "image": {
                "type": "url",
                "value": "https://media.roboflow.com/dog.jpeg",
            },
            "text": ["person", "backpack", "dog", "eye", "nose", "ear", "tongue"],
        },
        "expected_response": {
            "time": 3.0543342079909053,
            "image": {"width": 720, "height": 1280},
            "predictions": [
                {
                    "x": 324.58551025390625,
                    "y": 816.8829345703125,
                    "width": 645.8734130859375,
                    "height": 924.895751953125,
                    "confidence": 0.8848897814750671,
                    "class": "person",
                    "class_id": 0,
                    "detection_id": "b52aaea3-2972-49dc-ad9c-21faf5338539",
                },
                {
                    "x": 356.41864013671875,
                    "y": 587.4201049804688,
                    "width": 575.7637939453125,
                    "height": 679.5279541015625,
                    "confidence": 0.6294476985931396,
                    "class": "dog",
                    "class_id": 2,
                    "detection_id": "2498ffa2-a82b-4568-a475-249800ebcaf5",
                },
                {
                    "x": 222.44143676757812,
                    "y": 975.375244140625,
                    "width": 442.60906982421875,
                    "height": 609.14892578125,
                    "confidence": 0.13416942954063416,
                    "class": "backpack",
                    "class_id": 1,
                    "detection_id": "63a17397-3343-49ae-946e-a394e07dde6a",
                },
                {
                    "x": 403.83270263671875,
                    "y": 512.683837890625,
                    "width": 131.404541015625,
                    "height": 100.52346801757812,
                    "confidence": 0.0321989580988884,
                    "class": "tongue",
                    "class_id": 6,
                    "detection_id": "f843ddde-9f3b-448b-b205-90d2446fd61d",
                },
            ],
        },
    }
]


def bool_env(val):
    if isinstance(val, bool):
        return val
    return val.lower() in ["true", "1", "t", "y", "yes"]


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_YOLO_WORLD_TEST", False)),
    reason="Skipping YOLO-World test",
)
@pytest.mark.parametrize("test", TESTS)
def test_yolo_world(test):
    payload = deepcopy(test["payload"])
    payload["api_key"] = api_key
    response = requests.post(
        f"{base_url}:{port}/yolo_world/infer",
        json=payload,
    )
    try:
        response.raise_for_status()
        data = response.json()
        try:
            assert "predictions" in data
        except:
            print(f"Invalid response: {data}, expected 'predictions' in data")
        try:
            assert data == test
        except:
            print(f"Invalid response: {data}, expected {test['expected_response']}")
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
    test_yolo_world()
