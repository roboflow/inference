import os
import time

import numpy as np
import supervision as sv
import pytest
import requests

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


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_YOLO_WORLD_TEST", False)),
    reason="Skipping YOLO-World test",
)
def test_yolo_world_v1_s() -> None:
    # given
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
    expected_response = sv.Detections(
        xyxy=np.array(
            [[1.1122, 279.79, 643.92, 1278.6], [50.946, 249.5, 646.47, 1280]]
        ),
        confidence=np.array([0.60646, 0.54239]),
        class_id=np.array([0, 2]),
    )

    # when
    response = requests.post(
        f"{base_url}:{port}/yolo_world/infer",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert "predictions" in data, "'predictions' key not present in response"
    result_detections = sv.Detections.from_roboflow(data)
    assert np.allclose(
        result_detections.xyxy, expected_response.xyxy, atol=0.05
    ), "Predicted boxes missmatch"
    assert np.allclose(
        result_detections.confidence, expected_response.confidence, atol=1e-5
    ), "Predicted confidence missmatch"
    assert (
        result_detections.class_id == expected_response.class_id
    ).all(), "Predicted classes"


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_YOLO_WORLD_TEST", False)),
    reason="Skipping YOLO-World test",
)
def test_yolo_world_v1_m() -> None:
    # given
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
    expected_response = sv.Detections(
        xyxy=np.array(
            [
                [67.649, 248.08, 639.96, 922.75],
                [1.0265, 355.1, 646.4, 1279],
                [0.89661, 355, 643.84, 1279.7],
            ]
        ),
        confidence=np.array([0.89549, 0.8669, 0.52278]),
        class_id=np.array([2, 0, 1]),
    )

    # when
    response = requests.post(
        f"{base_url}:{port}/yolo_world/infer",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert "predictions" in data, "'predictions' key not present in response"
    result_detections = sv.Detections.from_roboflow(data)
    assert np.allclose(
        result_detections.xyxy, expected_response.xyxy, atol=0.01
    ), "Predicted boxes missmatch"
    assert np.allclose(
        result_detections.confidence, expected_response.confidence, atol=1e-5
    ), "Predicted confidence missmatch"
    assert (
        result_detections.class_id == expected_response.class_id
    ).all(), "Predicted classes"


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_YOLO_WORLD_TEST", False)),
    reason="Skipping YOLO-World test",
)
def test_yolo_world_v1_l() -> None:
    # given
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
    expected_response = sv.Detections(
        xyxy=np.array(
            [
                [0.3009, 356.79, 649.79, 1279.8],
                [67.3, 247.69, 643.15, 924.46],
                [1.2094, 667.86, 443.88, 1280],
            ]
        ),
        confidence=np.array([0.94843, 0.89687, 0.36721]),
        class_id=np.array([0, 2, 1]),
    )

    # when
    response = requests.post(
        f"{base_url}:{port}/yolo_world/infer",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert "predictions" in data, "'predictions' key not present in response"
    result_detections = sv.Detections.from_roboflow(data)
    assert np.allclose(
        result_detections.xyxy, expected_response.xyxy, atol=0.05
    ), "Predicted boxes missmatch"
    assert np.allclose(
        result_detections.confidence, expected_response.confidence, atol=1e-5
    ), "Predicted confidence missmatch"
    assert (
        result_detections.class_id == expected_response.class_id
    ).all(), "Predicted classes"


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_YOLO_WORLD_TEST", False)),
    reason="Skipping YOLO-World test",
)
def test_yolo_world_v1_x() -> None:
    # given
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
    expected_response = sv.Detections(
        xyxy=np.array(
            [
                [0, 356.09, 646.18, 1280],
                [69.174, 248.41, 647.38, 934.57],
                [1.4413, 664.79, 448.04, 1280],
            ]
        ),
        confidence=np.array([0.80944, 0.77479, 0.37293]),
        class_id=np.array([0, 2, 1]),
    )

    # when
    response = requests.post(
        f"{base_url}:{port}/yolo_world/infer",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert "predictions" in data, "'predictions' key not present in response"
    result_detections = sv.Detections.from_roboflow(data)
    assert np.allclose(
        result_detections.xyxy, expected_response.xyxy, atol=0.05
    ), "Predicted boxes missmatch"
    assert np.allclose(
        result_detections.confidence, expected_response.confidence, atol=1e-4
    ), "Predicted confidence missmatch"
    assert (
        result_detections.class_id == expected_response.class_id
    ).all(), "Predicted classes"


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_YOLO_WORLD_TEST", False)),
    reason="Skipping YOLO-World test",
)
def test_yolo_world_v2_s() -> None:
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
    expected_response = sv.Detections(
        xyxy=np.array([[67.575, 248.78, 644.36, 917.22], [0, 356.15, 607.91, 1280]]),
        confidence=np.array([0.66347, 0.58448]),
        class_id=np.array([2, 0]),
    )

    # when
    response = requests.post(
        f"{base_url}:{port}/yolo_world/infer",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert "predictions" in data, "'predictions' key not present in response"
    result_detections = sv.Detections.from_roboflow(data)
    assert np.allclose(
        result_detections.xyxy, expected_response.xyxy, atol=0.05
    ), "Predicted boxes missmatch"
    assert np.allclose(
        result_detections.confidence, expected_response.confidence, atol=1e-5
    ), "Predicted confidence missmatch"
    assert (
        result_detections.class_id == expected_response.class_id
    ).all(), "Predicted classes"


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_YOLO_WORLD_TEST", False)),
    reason="Skipping YOLO-World test",
)
def test_yolo_world_v2_m() -> None:
    # given
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
    expected_response = sv.Detections(
        xyxy=np.array(
            [
                [67.633, 249.1, 646.02, 929.87],
                [0.89825, 354.9, 640, 1279.3],
                [0.38277, 362.54, 474.88, 1279.7],
            ]
        ),
        confidence=np.array([0.78977, 0.70636, 0.33532]),
        class_id=np.array([2, 0, 1]),
    )

    # when
    response = requests.post(
        f"{base_url}:{port}/yolo_world/infer",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert "predictions" in data, "'predictions' key not present in response"
    result_detections = sv.Detections.from_roboflow(data)
    assert np.allclose(
        result_detections.xyxy, expected_response.xyxy, atol=0.05
    ), "Predicted boxes missmatch"
    assert np.allclose(
        result_detections.confidence, expected_response.confidence, atol=1e-4
    ), "Predicted confidence missmatch"
    assert (
        result_detections.class_id == expected_response.class_id
    ).all(), "Predicted classes"


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_YOLO_WORLD_TEST", False)),
    reason="Skipping YOLO-World test",
)
def test_yolo_world_v2_l() -> None:
    # given
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
    expected_response = sv.Detections(
        xyxy=np.array(
            [
                [0.84265, 356.88, 648.31, 1279.7],
                [70.681, 245.05, 642.65, 923.46],
                [0.20306, 658.29, 498.11, 1279.7],
                [339.24, 460.29, 469.92, 563.06],
            ]
        ),
        confidence=np.array([0.89018, 0.85904, 0.52798, 0.33998]),
        class_id=np.array([0, 2, 1, 6]),
    )

    # when
    response = requests.post(
        f"{base_url}:{port}/yolo_world/infer",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert "predictions" in data, "'predictions' key not present in response"
    result_detections = sv.Detections.from_roboflow(data)
    assert np.allclose(
        result_detections.xyxy, expected_response.xyxy, atol=0.05
    ), "Predicted boxes missmatch"
    assert np.allclose(
        result_detections.confidence, expected_response.confidence, atol=1e-4
    ), "Predicted confidence missmatch"
    assert (
        result_detections.class_id == expected_response.class_id
    ).all(), "Predicted classes"


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_YOLO_WORLD_TEST", False)),
    reason="Skipping YOLO-World test",
)
def test_yolo_world_v2_x() -> None:
    # given
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
    expected_response = sv.Detections(
        xyxy=np.array(
            [
                [70.392, 250.73, 643.59, 931.41],
                [0.45392, 355.19, 648.05, 1279.9],
                [338.5, 462.77, 469.06, 562.24],
            ]
        ),
        confidence=np.array([0.94871, 0.93121, 0.63789]),
        class_id=np.array([2, 0, 6]),
    )

    # when
    response = requests.post(
        f"{base_url}:{port}/yolo_world/infer",
        json=payload,
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert "predictions" in data, "'predictions' key not present in response"
    result_detections = sv.Detections.from_roboflow(data)
    assert np.allclose(
        result_detections.xyxy, expected_response.xyxy, atol=0.05
    ), "Predicted boxes missmatch"
    assert np.allclose(
        result_detections.confidence, expected_response.confidence, atol=1e-4
    ), "Predicted confidence missmatch"
    assert (
        result_detections.class_id == expected_response.class_id
    ).all(), "Predicted classes"
