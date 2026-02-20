import base64
import json
import os
import time
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import pytest
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

from tests.common import (
    assert_classification_predictions_match,
    assert_localized_predictions_match,
)

PIXEL_TOLERANCE = 2
CONFIDENCE_TOLERANCE = 0.02
TIME_TOLERANCE = 0.75
api_key = os.environ.get("API_KEY")
port = os.environ.get("PORT", 9001)
base_url = os.environ.get("BASE_URL", "http://localhost")


def bool_env(val):
    if isinstance(val, bool):
        return val
    return val.lower() in ["true", "1", "t", "y", "yes"]


def model_add(test, port=9001, api_key="", base_url="http://localhost"):
    return requests.post(
        f"{base_url}:{port}/{test['project']}/{test['version']}?"
        + "&".join(
            [
                f"api_key={api_key}",
                f"confidence={test['confidence']}",
                f"overlap={test['iou_threshold']}",
                f"image={test['image_url']}",
            ]
        )
    )


def legacy_infer_with_image_url(
    test, port=9001, api_key="", base_url="http://localhost"
):
    return (
        requests.post(
            f"{base_url}:{port}/{test['project']}/{test['version']}?"
            + "&".join(
                [
                    f"api_key={api_key}",
                    f"confidence={test['confidence']}",
                    f"overlap={test['iou_threshold']}",
                    f"image={test['image_url']}",
                    f'format={test.get("format", "json")}',
                ]
            )
        ),
        "url",
    )


def legacy_infer_with_base64_image(
    test, port=9001, api_key="", base_url="http://localhost"
):
    buffered = BytesIO()
    test["pil_image"].save(buffered, quality=100, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    img_str = img_str.decode("ascii")
    if test.get("simulate_rfwidget_upload", False):
        img_str = f"data:image/jpeg;base64,{img_str}"
    return (
        requests.post(
            f"{base_url}:{port}/{test['project']}/{test['version']}?"
            + "&".join(
                [
                    f"api_key={api_key}",
                    f"confidence={test['confidence']}",
                    f"overlap={test['iou_threshold']}",
                    f'format={test.get("format", "json")}',
                ]
            ),
            data=img_str,
            headers={"Content-Type": "application/json"},
        ),
        "base64",
    )


def legacy_infer_with_multipart_form_image(
    test, port=9001, api_key="", base_url="http://localhost"
):
    buffered = BytesIO()
    test["pil_image"].save(buffered, quality=100, format="JPEG")
    m = MultipartEncoder(
        fields={"file": ("original.jpeg", buffered.getvalue(), "image/jpeg")}
    )
    return (
        requests.post(
            f"{base_url}:{port}/{test['project']}/{test['version']}?"
            + "&".join(
                [
                    f"api_key={api_key}",
                    f"confidence={test['confidence']}",
                    f"overlap={test['iou_threshold']}",
                    f"format={test.get('format', 'json')}",
                ]
            ),
            data=m,
            headers={"Content-Type": m.content_type},
        ),
        "multipart_form",
    )


def infer_request_with_image_url(
    test, port=9001, api_key="", base_url="http://localhost"
):
    payload = {
        "model_id": f"{test['project']}/{test['version']}",
        "image": {
            "type": "url",
            "value": test["image_url"],
        },
        "confidence": test["confidence"],
        "iou_threshold": test["iou_threshold"],
        "api_key": api_key,
        "visualize_predictions": test.get("format") is not None,
    }
    return (
        requests.post(
            f"{base_url}:{port}/infer/{test['type']}",
            json=payload,
        ),
        "url",
    )


def infer_request_with_base64_image(
    test, port=9001, api_key="", base_url="http://localhost"
):
    buffered = BytesIO()
    test["pil_image"].save(buffered, quality=100, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    img_str = img_str.decode("ascii")
    payload = {
        "model_id": f"{test['project']}/{test['version']}",
        "image": {
            "type": "base64",
            "value": img_str,
        },
        "confidence": test["confidence"],
        "iou_threshold": test["iou_threshold"],
        "api_key": api_key,
        "visualize_predictions": test.get("format") is not None,
    }
    return (
        requests.post(
            f"{base_url}:{port}/infer/{test['type']}",
            json=payload,
        ),
        "base64",
    )


def compare_prediction_response(
    response: dict,
    expected_response: dict,
    prediction_type: Literal["object_detection", "instance_segmentation", "classification"] = "object_detection",
):
    # note that these casts do type checking internally via pydantic
    if prediction_type == "object_detection":
        assert_localized_predictions_match(
            result_prediction=response,
            reference_prediction=expected_response,
            box_pixel_tolerance=PIXEL_TOLERANCE,
            box_confidence_tolerance=CONFIDENCE_TOLERANCE,
        )
    elif prediction_type == "instance_segmentation":
        # this test for YOLACT used to totally fail on GPU, setting a threshold of .95 passes but seems low
        # TODO: look into why YOLACT seems to be so impacted by GPU vs CPU deployment
        assert_localized_predictions_match(
            result_prediction=response,
            reference_prediction=expected_response,
            mask_iou_threshold=0.95,
            box_pixel_tolerance=PIXEL_TOLERANCE,
            box_confidence_tolerance=CONFIDENCE_TOLERANCE,
        )
    elif prediction_type == "classification":
        assert_classification_predictions_match(
            result_prediction=response,
            reference_prediction=expected_response,
            confidence_tolerance=CONFIDENCE_TOLERANCE,
        )


TESTS_FILE = "tests.json" if os.getenv("USE_INFERENCE_MODELS", "false").lower() != "true" else "tests_inference_models.json"
with open(os.path.join(Path(__file__).resolve().parent, TESTS_FILE), "r") as f:
    TESTS = json.load(f)

INFER_RESPONSE_FUNCTIONS = [
    infer_request_with_image_url,
    infer_request_with_base64_image,
    legacy_infer_with_image_url,
    legacy_infer_with_base64_image,
    legacy_infer_with_multipart_form_image,
]

SKIP_YOLOV8_TEST = bool_env(os.getenv("SKIP_YOLOV8_TEST", False))
is_parallel_server = bool_env(os.getenv("IS_PARALLEL_SERVER", False))
DETECTION_TEST_PARAMS = []
for test in TESTS:
    if test["description"] == "YOLACT Instance Segmentation" and is_parallel_server:
        continue # Skip YOLACT tests for parallel server
    if "expected_response" in test:
        if not SKIP_YOLOV8_TEST or "YOLOv8" not in test["description"]:
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
            compare_prediction_response(
                data,
                test["expected_response"][image_type],
                prediction_type=test["type"],
            )
        print(
            "\u2713"
            + f" Test {test['project']}/{test['version']} passed with {res_function.__name__}."
        )
    except Exception as e:
        raise Exception(f"Error in test {test['description']}: {e}")


VISUALIZATION_TEST_PARAMS = [
    p for p in DETECTION_TEST_PARAMS if p[0]["type"] != "classification"
]


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_VISUALISATION_TESTS", False)),
    reason="Skipping visualisation test",
)
@pytest.mark.parametrize("test,res_function", VISUALIZATION_TEST_PARAMS)
def test_visualization(test, res_function, clean_loaded_models_fixture):
    test = deepcopy(test)
    try:
        try:
            pil_image = Image.open(
                requests.get(test["image_url"], stream=True).raw
            ).convert("RGB")
            test["pil_image"] = pil_image
        except Exception as e:
            raise ValueError(f"Unable to load image from URL: {test['image_url']}")

        test["format"] = "image"
        response, _image_type = res_function(
            test, port, api_key=os.getenv(f"{test['project'].replace('-','_')}_API_KEY")
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise ValueError(f"Failed to make request to {res_function.__name__}: {e}")
        try:
            data = base64.b64decode(response.json()["visualization"])
        except KeyError:
            raise ValueError("Response json lacks visualization key")
        except json.JSONDecodeError:
            data = response.content
        try:
            im = BytesIO(data)
            Image.open(im).convert("RGB")
        except Exception as error:
            raise ValueError("Invalid image response") from error
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
