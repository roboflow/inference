import os
import time

import numpy as np
import pytest
import requests

from tests.inference.integration_tests.conftest import on_demand_clean_loaded_models

api_key = os.environ.get("API_KEY")
port = os.environ.get("PORT", 9001)
base_url = os.environ.get("BASE_URL", "http://localhost")


def bool_env(val):
    if isinstance(val, bool):
        return val
    return val.lower() in ["true", "1", "t", "y", "yes"]


@pytest.fixture(scope="session", autouse=True)
def ensure_server_runs():
    try:
        res = requests.get(f"{base_url}:{port}")
        res.raise_for_status()
        success = True
    except:
        success = False
    max_wait = int(os.getenv("MAX_WAIT", 30))
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
        if waited > max_wait:
            raise Exception("Test server failed to start")


def get_bounding_boxes(response: dict) -> np.ndarray:
    result = []
    for prediction in response["predictions"]:
        result.append(
            [
                prediction["x"],
                prediction["y"],
                prediction["width"],
                prediction["height"],
            ]
        )
    return np.array(result)


def get_class_ids(response: dict) -> np.ndarray:
    result = []
    for prediction in response["predictions"]:
        result.append(prediction["class_id"])
    return np.array(result)


def get_classes_confidence(response: dict) -> np.ndarray:
    result = []
    for prediction in response["predictions"]:
        result.append(prediction["confidence"])
    return np.array(result)


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_YOLO_WORLD_TEST", False)),
    reason="Skipping YOLO-World test",
)
def test_yolo_world_v1_s() -> None:
    # given
    on_demand_clean_loaded_models()
    payload = {
        "api_key": api_key,
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "text": ["person", "backpack", "dog", "eye", "nose", "ear", "tongue"],
        "yolo_world_version_id": "s",
        "confidence": 0.2,
    }
    expected_boxes = np.array(
        [[322.516, 779.195, 642.808, 998.81], [348.708, 764.75, 595.524, 1030.5]]
    )
    expected_confidences = np.array([0.60646, 0.54239])
    expected_classes = np.array([0, 2])

    # when
    response = requests.post(
        f"{base_url}:{port}/yolo_world/infer",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert "predictions" in data, "'predictions' key not present in response"
    assert np.allclose(
        get_bounding_boxes(data), expected_boxes, atol=1e-1
    ), "Predicted boxes missmatch"
    assert np.allclose(
        get_classes_confidence(data), expected_confidences, atol=1e-3
    ), "Predicted confidence missmatch"
    assert (get_class_ids(data) == expected_classes).all(), "Predicted classes"


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_YOLO_WORLD_TEST", False)),
    reason="Skipping YOLO-World test",
)
def test_yolo_world_v1_m() -> None:
    # given
    on_demand_clean_loaded_models()
    payload = {
        "api_key": api_key,
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "text": ["person", "backpack", "dog", "eye", "nose", "ear", "tongue"],
        "yolo_world_version_id": "m",
        "confidence": 0.3,
    }
    expected_boxes = np.array(
        [
            [353.8045, 585.415, 572.311, 674.67],
            [323.7132, 817.05, 645.3735, 923.9],
            [322.3683, 817.35, 642.9434, 924.7],
        ]
    )
    expected_confidences = np.array([0.89549, 0.8669, 0.52278])
    expected_classes = np.array([2, 0, 1])

    # when
    response = requests.post(
        f"{base_url}:{port}/yolo_world/infer",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert "predictions" in data, "'predictions' key not present in response"
    assert np.allclose(
        get_bounding_boxes(data), expected_boxes, atol=1e-1
    ), "Predicted boxes missmatch"
    assert np.allclose(
        get_classes_confidence(data), expected_confidences, atol=1e-3
    ), "Predicted confidence missmatch"
    assert (get_class_ids(data) == expected_classes).all(), "Predicted classes"


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_YOLO_WORLD_TEST", False)),
    reason="Skipping YOLO-World test",
)
def test_yolo_world_v1_l() -> None:
    # given
    on_demand_clean_loaded_models()
    payload = {
        "api_key": api_key,
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "text": ["person", "backpack", "dog", "eye", "nose", "ear", "tongue"],
        "yolo_world_version_id": "l",
        "confidence": 0.3,
    }
    expected_boxes = np.array(
        [
            [325.0455, 818.295, 649.4891, 923.01],
            [355.225, 586.075, 575.85, 676.77],
            [222.5447, 973.93, 442.6706, 612.14],
        ]
    )
    expected_confidences = np.array([0.94843, 0.89687, 0.36721])
    expected_classes = np.array([0, 2, 1])

    # when
    response = requests.post(
        f"{base_url}:{port}/yolo_world/infer",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert "predictions" in data, "'predictions' key not present in response"
    assert np.allclose(
        get_bounding_boxes(data), expected_boxes, atol=1e-1
    ), "Predicted boxes missmatch"
    assert np.allclose(
        get_classes_confidence(data), expected_confidences, atol=1e-3
    ), "Predicted confidence missmatch"
    assert (get_class_ids(data) == expected_classes).all(), "Predicted classes"


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_YOLO_WORLD_TEST", False)),
    reason="Skipping YOLO-World test",
)
def test_yolo_world_v1_x() -> None:
    # given
    on_demand_clean_loaded_models()
    payload = {
        "api_key": api_key,
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "text": ["person", "backpack", "dog", "eye", "nose", "ear", "tongue"],
        "yolo_world_version_id": "x",
        "confidence": 0.3,
    }
    expected_boxes = np.array(
        [
            [323.09, 818.045, 646.18, 923.91],
            [358.277, 591.49, 578.206, 686.16],
            [224.7407, 972.395, 446.5987, 615.21],
        ]
    )
    expected_confidences = np.array([0.80944, 0.77479, 0.37293])
    expected_classes = np.array([0, 2, 1])

    # when
    response = requests.post(
        f"{base_url}:{port}/yolo_world/infer",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert "predictions" in data, "'predictions' key not present in response"
    assert np.allclose(
        get_bounding_boxes(data), expected_boxes, atol=1e-1
    ), "Predicted boxes missmatch"
    assert np.allclose(
        get_classes_confidence(data), expected_confidences, atol=1e-3
    ), "Predicted confidence missmatch"
    assert (get_class_ids(data) == expected_classes).all(), "Predicted classes"


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_YOLO_WORLD_TEST", False)),
    reason="Skipping YOLO-World test",
)
def test_yolo_world_v2_s() -> None:
    on_demand_clean_loaded_models()
    # given
    payload = {
        "api_key": api_key,
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "text": ["person", "backpack", "dog", "eye", "nose", "ear", "tongue"],
        "yolo_world_version_id": "v2-s",
        "confidence": 0.2,
    }
    expected_boxes = np.array(
        [[355.9675, 583.0, 576.785, 668.44], [303.955, 818.075, 607.91, 923.85]]
    )
    expected_confidences = np.array([0.66347, 0.58448])
    expected_classes = np.array([2, 0])

    # when
    response = requests.post(
        f"{base_url}:{port}/yolo_world/infer",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert "predictions" in data, "'predictions' key not present in response"
    assert np.allclose(
        get_bounding_boxes(data), expected_boxes, atol=1e-1
    ), "Predicted boxes missmatch"
    assert np.allclose(
        get_classes_confidence(data), expected_confidences, atol=1e-3
    ), "Predicted confidence missmatch"
    assert (get_class_ids(data) == expected_classes).all(), "Predicted classes"


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_YOLO_WORLD_TEST", False)),
    reason="Skipping YOLO-World test",
)
def test_yolo_world_v2_m() -> None:
    # given
    on_demand_clean_loaded_models()
    payload = {
        "api_key": api_key,
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "text": ["person", "backpack", "dog", "eye", "nose", "ear", "tongue"],
        "yolo_world_version_id": "v2-m",
        "confidence": 0.3,
    }
    expected_boxes = np.array(
        [
            [356.8265, 589.485, 578.387, 680.77],
            [320.4491, 817.1, 639.1018, 924.4],
            [237.6314, 821.12, 474.4972, 917.16],
        ]
    )
    expected_confidences = np.array([0.78977, 0.70636, 0.33532])
    expected_classes = np.array([2, 0, 1])

    # when
    response = requests.post(
        f"{base_url}:{port}/yolo_world/infer",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert "predictions" in data, "'predictions' key not present in response"
    assert np.allclose(
        get_bounding_boxes(data), expected_boxes, atol=1e-1
    ), "Predicted boxes missmatch"
    assert np.allclose(
        get_classes_confidence(data), expected_confidences, atol=1e-3
    ), "Predicted confidence missmatch"
    assert (get_class_ids(data) == expected_classes).all(), "Predicted classes"


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_YOLO_WORLD_TEST", False)),
    reason="Skipping YOLO-World test",
)
def test_yolo_world_v2_l() -> None:
    # given
    on_demand_clean_loaded_models()
    payload = {
        "api_key": api_key,
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "text": ["person", "backpack", "dog", "eye", "nose", "ear", "tongue"],
        "yolo_world_version_id": "v2-l",
        "confidence": 0.3,
    }
    expected_boxes = np.array(
        [
            [324.5763, 818.29, 647.4673, 922.82],
            [356.6655, 584.255, 571.969, 678.41],
            [249.1565, 968.995, 497.9069, 621.41],
            [404.58, 511.675, 130.68, 102.77],
        ]
    )
    expected_confidences = np.array([0.89018, 0.85904, 0.52798, 0.33998])
    expected_classes = np.array([0, 2, 1, 6])

    # when
    response = requests.post(
        f"{base_url}:{port}/yolo_world/infer",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert "predictions" in data, "'predictions' key not present in response"
    assert np.allclose(
        get_bounding_boxes(data), expected_boxes, atol=1e-1
    ), "Predicted boxes missmatch"
    assert np.allclose(
        get_classes_confidence(data), expected_confidences, atol=1e-3
    ), "Predicted confidence missmatch"
    assert (get_class_ids(data) == expected_classes).all(), "Predicted classes"


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_YOLO_WORLD_TEST", False)),
    reason="Skipping YOLO-World test",
)
def test_yolo_world_v2_x() -> None:
    # given
    on_demand_clean_loaded_models()
    payload = {
        "api_key": api_key,
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "text": ["person", "backpack", "dog", "eye", "nose", "ear", "tongue"],
        "yolo_world_version_id": "v2-x",
        "confidence": 0.3,
    }
    expected_boxes = np.array(
        [
            [356.991, 591.07, 573.198, 680.68],
            [324.252, 817.545, 647.5961, 924.71],
            [403.78, 512.505, 130.56, 99.47],
        ]
    )
    expected_confidences = np.array([0.94871, 0.93121, 0.63789])
    expected_classes = np.array([2, 0, 6])

    # when
    response = requests.post(
        f"{base_url}:{port}/yolo_world/infer",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert "predictions" in data, "'predictions' key not present in response"
    assert np.allclose(
        get_bounding_boxes(data), expected_boxes, atol=1e-1
    ), "Predicted boxes missmatch"
    assert np.allclose(
        get_classes_confidence(data), expected_confidences, atol=1e-3
    ), "Predicted confidence missmatch"
    assert (get_class_ids(data) == expected_classes).all(), "Predicted classes"
