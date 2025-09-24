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
        "description": "Grounding DINO",
        "type": "ocr",
        "payload": {
            "api_key": api_key,
            "image": {
                "type": "url",
                "value": "https://media.roboflow.com/fruit.png",
            },
            "text": ["apple"],
        },
        "expected_response": {
            "time": 4.5613939999602735,
            "image": {"width": 950, "height": 796},
            "predictions": [
                {
                    "x": 425.0,
                    "y": 358.0,
                    "width": 282.0,
                    "height": 250.0,
                    "confidence": 0.7121592164039612,
                    "class_name": "apple",
                    "class_confidence": None,
                    "class_id": 0,
                    "tracker_id": None,
                },
                {
                    "x": 612.0,
                    "y": 113.0,
                    "width": 253.0,
                    "height": 207.0,
                    "confidence": 0.7059137225151062,
                    "class_name": "apple",
                    "class_confidence": None,
                    "class_id": 0,
                    "tracker_id": None,
                },
                {
                    "x": 298.0,
                    "y": 154.0,
                    "width": 313.0,
                    "height": 202.0,
                    "confidence": 0.6812718510627747,
                    "class_name": "apple",
                    "class_confidence": None,
                    "class_id": 0,
                    "tracker_id": None,
                },
                {
                    "x": 397.0,
                    "y": 186.0,
                    "width": 179.0,
                    "height": 142.0,
                    "confidence": 0.6505898833274841,
                    "class_name": "apple",
                    "class_confidence": None,
                    "class_id": 0,
                    "tracker_id": None,
                },
                {
                    "x": 587.0,
                    "y": 207.0,
                    "width": 215.0,
                    "height": 249.0,
                    "confidence": 0.6408542394638062,
                    "class_name": "apple",
                    "class_confidence": None,
                    "class_id": 0,
                    "tracker_id": None,
                },
                {
                    "x": 221.0,
                    "y": 333.0,
                    "width": 232.0,
                    "height": 242.0,
                    "confidence": 0.675274670124054,
                    "class_name": "apple",
                    "class_confidence": None,
                    "class_id": 0,
                    "tracker_id": None,
                },
                {
                    "x": 663.0,
                    "y": 390.0,
                    "width": 193.0,
                    "height": 182.0,
                    "confidence": 0.6492464542388916,
                    "class_name": "apple",
                    "class_confidence": None,
                    "class_id": 0,
                    "tracker_id": None,
                },
                {
                    "x": 798.0,
                    "y": 317.0,
                    "width": 237.0,
                    "height": 265.0,
                    "confidence": 0.6517468690872192,
                    "class_name": "apple",
                    "class_confidence": None,
                    "class_id": 0,
                    "tracker_id": None,
                },
                {
                    "x": 234.0,
                    "y": 637.0,
                    "width": 216.0,
                    "height": 286.0,
                    "confidence": 0.5249432325363159,
                    "class_name": "apple",
                    "class_confidence": None,
                    "class_id": 0,
                    "tracker_id": None,
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
    bool_env(os.getenv("SKIP_GROUNDING_DINO_TEST", False)),
    reason="Skipping grounding dino test",
)
@pytest.mark.parametrize("test", TESTS)
def test_grounding_dino(test, clean_loaded_models_fixture):
    payload = deepcopy(test["payload"])
    payload["api_key"] = api_key
    response = requests.post(
        f"{base_url}:{port}/grounding_dino/infer",
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
    test_grounding_dino()
