import json
import os
import requests
from copy import deepcopy
from pathlib import Path
import pytest
import time

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
                "value": "https://raw.githubusercontent.com/serengil/deepface/master/tests/dataset/img12.jpg",
            }
        },
        "expected_response": [
            {
                "predictions": [
                    {
                        "face": {
                            "x": 298.0,
                            "y": 175.0,
                            "width": 160.0,
                            "height": 160.0,
                            "confidence": 0.9520819187164307,
                            "class": "face",
                            "landmarks": [
                                {"x": 257.0, "y": 141.0},
                                {"x": 323.0, "y": 145.0},
                                {"x": 282.0, "y": 185.0},
                                {"x": 283.0, "y": 213.0},
                                {"x": 230.0, "y": 147.0},
                                {"x": 370.0, "y": 157.0},
                            ],
                        },
                        "pitch": -0.14007748663425446,
                        "yaw": 0.1661173403263092,
                    }
                ],
                "time_total": 0.7869974579953123,
                "time_load_img": 0.6480592500010971,
                "time_face_det": 0.005114250001497567,
                "time_gaze_det": 0.13382395799271762,
            }
        ],
    }
]


@pytest.mark.parametrize("test", TESTS)
def test_gaze(test):
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
        if waited > 30:
            raise Exception("Test server failed to start")


if __name__ == "__main__":
    test_gaze()
