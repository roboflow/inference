import base64
import json
import os
import requests
import time

import pytest

from io import BytesIO
from pathlib import Path
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

PIXEL_TOLERANCE = 2
CONFIDENCE_TOLERANCE = 0.005
TIME_TOLERANCE = 0.75

api_key = os.environ.get("API_KEY")
port = os.environ.get("PORT", 9001)
base_url = os.environ.get("BASE_URL", "http://localhost")


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
    return (
        requests.post(
            f"{base_url}:{port}/{test['project']}/{test['version']}?"
            + "&".join(
                [
                    f"api_key={api_key}",
                    f"confidence={test['confidence']}",
                    f"overlap={test['iou_threshold']}",
                ]
            ),
            data=img_str,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
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
    }
    return (
        requests.post(
            f"{base_url}:{port}/infer/{test['type']}",
            json=payload,
        ),
        "base64",
    )


def compare_detection_response(
    response, expected_response, type="object_detection", multilabel=False
):
    try:
        assert "time" in response
    except AssertionError:
        raise ValueError(f"Invalid response: {response}, Missing 'time' field.")
    # try:
    #     assert response["time"] == pytest.approx(
    #         expected_response["time"], rel=None, abs=TIME_TOLERANCE
    #     )
    # except AssertionError:
    #     raise ValueError(
    #         f"Invalid response: {response}, 'time' field does not match expected value. Expected {expected_response['time']}, got {response['time']}."
    #     )
    try:
        assert "image" in response
    except AssertionError:
        raise ValueError(f"Invalid response: {response}, Missing 'image' field.")
    try:
        assert response["image"]["width"] == expected_response["image"]["width"]
    except AssertionError:
        raise ValueError(
            f"Invalid response: {response}, 'image' field does not match expected value. Expected {expected_response['image']['width']}, got {response['image']['width']}."
        )
    try:
        assert response["image"]["height"] == expected_response["image"]["height"]
    except AssertionError:
        raise ValueError(
            f"Invalid response: {response}, 'image' field does not match expected value. Expected {expected_response['image']['height']}, got {response['image']['height']}."
        )
    try:
        assert "predictions" in response
    except AssertionError:
        raise ValueError(f"Invalid response: {response}, Missing 'predictions' field.")
    try:
        assert len(response["predictions"]) == len(expected_response["predictions"])
    except AssertionError:
        raise ValueError(
            f"Invalid response: {response}, number of predictions does not match expected value. Expected {len(expected_response['predictions'])} predictions, got {len(response['predictions'])}."
        )
    if type in ["object_detection", "instance_segmentation"]:
        for i, prediction in enumerate(response["predictions"]):
            try:
                assert prediction["x"] == pytest.approx(
                    expected_response["predictions"][i]["x"],
                    rel=None,
                    abs=PIXEL_TOLERANCE,
                )
            except AssertionError:
                raise ValueError(
                    f"Invalid response: {response}, 'x' field does not match expected value for prediction {i}. Expected {expected_response['predictions'][i]['x']}, got {prediction['x']}."
                )
            try:
                assert prediction["y"] == pytest.approx(
                    expected_response["predictions"][i]["y"],
                    rel=None,
                    abs=PIXEL_TOLERANCE,
                )
            except AssertionError:
                raise ValueError(
                    f"Invalid response: {response}, 'y' field does not match expected value for prediction {i}. Expected {expected_response['predictions'][i]['y']}, got {prediction['y']}."
                )
            try:
                assert prediction["width"] == pytest.approx(
                    expected_response["predictions"][i]["width"],
                    rel=None,
                    abs=PIXEL_TOLERANCE,
                )
            except AssertionError:
                raise ValueError(
                    f"Invalid response: {response}, 'width' field does not match expected value for prediction {i}. Expected {expected_response['predictions'][i]['width']}, got {prediction['width']}."
                )
            try:
                assert prediction["height"] == pytest.approx(
                    expected_response["predictions"][i]["height"],
                    rel=None,
                    abs=PIXEL_TOLERANCE,
                )
            except AssertionError:
                raise ValueError(
                    f"Invalid response: {response}, 'height' field does not match expected value for prediction {i}. Expected {expected_response['predictions'][i]['height']}, got {prediction['height']}."
                )
            try:
                assert prediction["confidence"] == pytest.approx(
                    expected_response["predictions"][i]["confidence"],
                    rel=None,
                    abs=CONFIDENCE_TOLERANCE,
                )
            except AssertionError:
                raise ValueError(
                    f"Invalid response: {response}, 'confidence' field does not match expected value for prediction {i}. Expected {expected_response['predictions'][i]['confidence']}, got {prediction['confidence']}."
                )
            try:
                assert (
                    prediction["class"] == expected_response["predictions"][i]["class"]
                )
            except AssertionError:
                raise ValueError(
                    f"Invalid response: {response}, 'class' field does not match expected value for prediction {i}. Expected {expected_response['predictions'][i]['class']}, got {prediction['class']}."
                )
            if type == "instance_segmentation":
                try:
                    assert "points" in prediction
                except AssertionError:
                    raise ValueError(
                        f"Invalid response: {response}, Missing 'points' field for prediction {i}."
                    )
                for j, point in enumerate(prediction["points"]):
                    try:
                        assert point["x"] == pytest.approx(
                            expected_response["predictions"][i]["points"][j]["x"],
                            rel=None,
                            abs=PIXEL_TOLERANCE,
                        )
                    except AssertionError:
                        raise ValueError(
                            f"Invalid response: {response}, 'x' field does not match expected value for prediction {i}, point {j}. Expected {expected_response['predictions'][i]['points'][j]['x']}, got {point['x']}."
                        )
                    try:
                        assert point["y"] == pytest.approx(
                            expected_response["predictions"][i]["points"][j]["y"],
                            rel=None,
                            abs=PIXEL_TOLERANCE,
                        )
                    except AssertionError:
                        raise ValueError(
                            f"Invalid response: {response}, 'y' field does not match expected value for prediction {i}, point {j}. Expected {expected_response['predictions'][i]['points'][j]['y']}, got {point['y']}."
                        )
    elif type == "classification":
        if multilabel:
            for class_name, confidence in response["predictions"].items():
                try:
                    assert class_name in expected_response["predictions"]
                except AssertionError:
                    raise ValueError(
                        f"Invalid response: {response}, Unexpected class {class_name}. Expected classes: {expected_response['predictions'].keys()}."
                    )
                try:
                    assert "confidence" in confidence
                except AssertionError:
                    raise ValueError(
                        f"Invalid response: {response}, Missing 'confidence' field for class {class_name}."
                    )
                try:
                    assert confidence["confidence"] == pytest.approx(
                        expected_response["predictions"][class_name]["confidence"],
                        rel=None,
                        abs=CONFIDENCE_TOLERANCE,
                    )
                except AssertionError:
                    raise ValueError(
                        f"Invalid response: {response}, 'confidence' field does not match expected value for class {class_name}. Expected {expected_response['predictions'][class_name]['confidence']}, got {confidence['confidence']}."
                    )
            try:
                assert "predicted_classes" in response
            except AssertionError:
                raise ValueError(
                    f"Invalid response: {response}, Missing 'predicted_classes' field."
                )
            for class_name in response["predicted_classes"]:
                try:
                    assert class_name in expected_response["predictions"]
                except AssertionError:
                    raise ValueError(
                        f"Invalid response: {response}, Unexpected class {class_name}. Expected classes: {expected_response['predicted_classes']}."
                    )
        else:
            try:
                assert "top" in response
            except AssertionError:
                raise ValueError(f"Invalid response: {response}, Missing 'top' field.")
            try:
                assert response["top"] == expected_response["top"]
            except AssertionError:
                raise ValueError(
                    f"Invalid response: {response}, 'top' field does not match expected value. Expected {expected_response['top']}, got {response['top']}."
                )
            try:
                assert "confidence" in response
            except AssertionError:
                raise ValueError(
                    f"Invalid response: {response}, Missing 'confidence' field."
                )
            try:
                assert response["confidence"] == pytest.approx(
                    expected_response["confidence"],
                    rel=None,
                    abs=CONFIDENCE_TOLERANCE,
                )
            except AssertionError:
                raise ValueError(
                    f"Invalid response: {response}, 'confidence' field does not match expected value. Expected {expected_response['confidence']}, got {response['confidence']}."
                )
            for i, prediction in enumerate(response["predictions"]):
                try:
                    assert "class" in prediction
                except AssertionError:
                    raise ValueError(
                        f"Invalid response: {response}, Missing 'class' field for prediction {i}."
                    )
                try:
                    assert "confidence" in prediction
                except AssertionError:
                    raise ValueError(
                        f"Invalid response: {response}, Missing 'confidence' field for prediction {i}."
                    )
                try:
                    assert prediction["confidence"] == pytest.approx(
                        expected_response["predictions"][i]["confidence"],
                        rel=None,
                        abs=CONFIDENCE_TOLERANCE,
                    )
                except AssertionError:
                    raise ValueError(
                        f"Invalid response: {response}, 'confidence' field does not match expected value for prediction {i}. Expected {expected_response['predictions'][i]['confidence']}, got {prediction['confidence']}."
                    )
                try:
                    assert (
                        prediction["class"]
                        == expected_response["predictions"][i]["class"]
                    )
                except AssertionError:
                    raise ValueError(
                        f"Invalid response: {response}, 'class' field does not match expected value for prediction {i}. Expected {expected_response['predictions'][i]['class']}, got {prediction['class']}."
                    )


with open(os.path.join(Path(__file__).resolve().parent, "tests.json"), "r") as f:
    TESTS = json.load(f)

INFER_RESPONSE_FUNCTIONS = [
    infer_request_with_image_url,
    infer_request_with_base64_image,
    legacy_infer_with_image_url,
    legacy_infer_with_base64_image,
    legacy_infer_with_multipart_form_image,
]

DETECTION_TEST_PARAMS = []
for test in TESTS:
    if "expected_response" in test:
        for res_func in INFER_RESPONSE_FUNCTIONS:
            DETECTION_TEST_PARAMS.append((test, res_func))


@pytest.mark.parametrize("test,res_function", DETECTION_TEST_PARAMS)
def test_detection(test, res_function):
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
        compare_detection_response(
            data,
            test["expected_response"][image_type],
            type=test["type"],
            multilabel=test.get("multi_label", False),
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
    # This helps figure out which test is failing
    for i, param in enumerate(DETECTION_TEST_PARAMS):
        print(i, param[0]["project"], param[0]["version"], param[1].__name__)
