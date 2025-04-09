import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path

import pytest
import requests
from PIL import Image

from tests.inference.integration_tests.regression_test import (
    compare_prediction_response,
)

PIXEL_TOLERANCE = 2
CONFIDENCE_TOLERANCE = 0.005
TIME_TOLERANCE = 0.75
api_key = os.environ.get("API_KEY")
port = os.environ.get("PORT", 9001)
base_url = os.environ.get("BASE_URL", "http://localhost")


def bool_env(val):
    if isinstance(val, bool):
        return val
    return val.lower() in ["true", "1", "t", "y", "yes"]


def infer_request_with_image_url(
    test, port=9001, api_key="", base_url="http://localhost", batch_size=1
):
    payload = {
        "model_id": f"{test['project']}/{test['version']}",
        "image": [
            {
                "type": "url",
                "value": test["image_url"],
            }
        ]
        * batch_size,
        "confidence": test["confidence"],
        "iou_threshold": test["iou_threshold"],
        "api_key": api_key,
    }
    return (
        requests.post(
            f"{base_url}:{port}/infer/{test['type']}",
            json=payload,
        ),
        "url",
    )


def infer_request_with_base64_image_dif_size(
    test, port=9001, api_key="", base_url="http://localhost", batch_size=1
):
    sizes = [
        (155, 73),
        (125, 43),
        (165, 82),
        (115, 36),
        (134, 58),
        (115, 40),
        (164, 91),
        (200, 88),
        (137, 60),
        (155, 77),
        (144, 65),
        (132, 49),
        (140, 52),
        (178, 75),
        (155, 61),
        (147, 56),
    ]
    images = []
    for i in range(batch_size):
        buffered = BytesIO()
        im = test["pil_image"]
        im = im.resize(sizes[i])
        im.save(buffered, quality=100, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        img_str = img_str.decode("ascii")
        images.append(img_str)
    payload = {
        "model_id": f"{test['project']}/{test['version']}",
        "image": [
            {
                "type": "base64",
                "value": img_str,
            }
            for img_str in images
        ],
        "confidence": test["confidence"],
        "iou_threshold": test["iou_threshold"],
        "api_key": api_key,
    }
    return (
        requests.post(
            f"{base_url}:{port}/infer/{test['type']}",
            json=payload,
        ),
        "base64_diffsize",
    )


def infer_request_with_base64_image_dif_size_fixed(
    test, port=9001, api_key="", base_url="http://localhost", batch_size=1
):
    sizes = [
        (155, 73),
        (125, 43),
        (165, 82),
        (115, 36),
        (134, 58),
        (115, 40),
        (164, 91),
        (200, 88),
        (137, 60),
        (155, 77),
        (144, 65),
        (132, 49),
        (140, 52),
        (178, 75),
        (155, 61),
        (147, 56),
    ]
    images = []
    for i in range(batch_size):
        buffered = BytesIO()
        im = test["pil_image"]
        im = im.resize(sizes[i])
        im.save(buffered, quality=100, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        img_str = img_str.decode("ascii")
        images.append(img_str)

    payload = {
        "model_id": f"{test['project']}/{test['version']}",
        "image": [
            {
                "type": "base64",
                "value": img_str,
            }
            for img_str in images
        ],
        "confidence": test["confidence"],
        "iou_threshold": test["iou_threshold"],
        "api_key": api_key,
        "fix_batch_size": True,
    }
    return (
        requests.post(
            f"{base_url}:{port}/infer/{test['type']}",
            json=payload,
        ),
        "base64_diffsize",
    )


def infer_request_with_base64_image(
    test, port=9001, api_key="", base_url="http://localhost", batch_size=1
):
    buffered = BytesIO()
    test["pil_image"].save(buffered, quality=100, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    img_str = img_str.decode("ascii")
    payload = {
        "model_id": f"{test['project']}/{test['version']}",
        "image": [
            {
                "type": "base64",
                "value": img_str,
            }
        ]
        * batch_size,
        "confidence": test["confidence"],
        "iou_threshold": test["iou_threshold"],
        "api_key": api_key,
    }
    return (
        requests.post(
            f"{base_url}:{port}/infer/{test['type']}",
            json=payload,
        ),
        "base64",
    )


with open(os.path.join(Path(__file__).resolve().parent, "batch_tests.json"), "r") as f:
    TESTS = json.load(f)


INFER_RESPONSE_FUNCTIONS = [
    infer_request_with_image_url,
    infer_request_with_base64_image,
    infer_request_with_base64_image_dif_size,
    infer_request_with_base64_image_dif_size_fixed,
]

DETECTION_TEST_PARAMS = []
is_parallel_server = bool_env(os.getenv("IS_PARALLEL_SERVER", False))
for test in TESTS:
    if test["description"] == "YOLACT Instance Segmentation" and is_parallel_server:
        continue # Skip YOLACT tests for parallel server
    if "expected_response" in test:
        for res_func in INFER_RESPONSE_FUNCTIONS:
            DETECTION_TEST_PARAMS.append((test, res_func))


@pytest.mark.parametrize("test,res_function", DETECTION_TEST_PARAMS)
def test_detection(test, res_function, clean_loaded_models_fixture):
    try:
        try:
            pil_image = Image.open(
                requests.get(test["image_url"], stream=True).raw
            ).convert("RGB")
            test["pil_image"] = pil_image
        except Exception as e:
            raise ValueError(f"Unable to load image from URL: {test['image_url']}")

        response, image_type = res_function(
            test, port, api_key=os.getenv(f"{test['project'].replace('-','_')}_API_KEY")
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise ValueError(f"Failed to make request to {res_function.__name__}: {e}")
        try:
            data = response.json()
        except ValueError:
            raise ValueError(f"Invalid JSON response: {response.text}")
        try:
            assert "expected_response" in test
        except AssertionError:
            raise ValueError(
                f"Invalid test: {test}, Missing 'expected_response' field in test."
            )
        try:
            assert image_type in test["expected_response"]
        except AssertionError:
            raise ValueError(
                f"Invalid test: {test}, Missing 'expected_response' field for image type {image_type}."
            )
        if not bool_env(os.getenv("FUNCTIONAL", False)):
            for d, test_data in zip(data, test["expected_response"][image_type]):
                compare_prediction_response(
                    d,
                    test_data,
                    prediction_type=test["type"],
                )
        print(
            "\u2713"
            + f" Test {test['project']}/{test['version']} passed with {res_function.__name__}."
        )
    except Exception as e:
        raise Exception(f"Error in test {test['description']}: {e}")


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
    # This helps figure out which test is failing
    for i, param in enumerate(DETECTION_TEST_PARAMS):
        print(i, param[0]["project"], param[0]["version"], param[1].__name__)
