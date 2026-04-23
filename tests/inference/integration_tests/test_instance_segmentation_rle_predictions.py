import os

import requests
import supervision as sv
from pycocotools import mask as mask_utils

USE_INFERENCE_MODELS = os.getenv("USE_INFERENCE_MODELS", "false").lower() == "true"
API_KEY = os.environ.get("API_KEY")
PORT = os.environ.get("PORT", 9001)
BASE_URL = os.environ.get("BASE_URL", "http://localhost")



def test_v1_endpoint_with_valid_payload() -> None:
    payload = {
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "response_mask_format": "rle",
        "api_key": API_KEY,
        "model_id": "yolov8n-seg-640"
    }
    response = requests.post(
        f"{BASE_URL}:{PORT}/infer/instance_segmentation",
        json=payload,
    )
    response.raise_for_status()
    data = response.json()[0]
    if not USE_INFERENCE_MODELS:
        for detection in data["predictions"]:
            assert detection["mask_format"] == "polygon"
        detections = sv.Detections.from_inference(data)
        assert isinstance(detections, sv.Detections)
    else:
        rles = []
        for detection in data["predictions"]:
            assert detection["mask_format"] == "rle"
            detection["mask_format"] = "polygon"
            rles.append(detection["rle"])
        masks = mask_utils.decode(rles).transpose(2, 0, 1).astype(bool)
        assert masks.shape[1:] == (10, 20)


def test_v1_endpoint_with_invalid_payload() -> None:
    payload = {
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "response_mask_format": "dummy",
        "api_key": API_KEY,
        "model_id": "yolov8n-seg-640"
    }

    response = requests.post(
        f"{BASE_URL}:{PORT}/infer/instance_segmentation",
        json=payload,
    )

    assert response.status_code == 400



def test_legacy_endpoint_valid_payload() -> None:
    response = requests.post(
        f"{BASE_URL}:{PORT}/coco-dataset-vdnr1/2",
        params={
            "image": "https://source.roboflow.com/D8zLgnZxdqtqF0plJINA/DqK7I0rUz5HBvu1hdNi6/original.jpg",
            "response_mask_format": "rle",
            "api_key": API_KEY,
        },
    )
    response.raise_for_status()
    data = response.json()[0]
    if not USE_INFERENCE_MODELS:
        for detection in data["predictions"]:
            assert detection["mask_format"] == "polygon"
        detections = sv.Detections.from_inference(data)
        assert isinstance(detections, sv.Detections)
    else:
        rles = []
        for detection in data["predictions"]:
            assert detection["mask_format"] == "rle"
            detection["mask_format"] = "polygon"
            rles.append(detection["rle"])
        masks = mask_utils.decode(rles).transpose(2, 0, 1).astype(bool)
        assert masks.shape[1:] == (10, 20)


def test_legacy_endpoint_invalid_payload() -> None:
    response = requests.post(
        f"{BASE_URL}:{PORT}/coco-dataset-vdnr1/2",
        params={
            "image": "https://source.roboflow.com/D8zLgnZxdqtqF0plJINA/DqK7I0rUz5HBvu1hdNi6/original.jpg",
            "response_mask_format": "dummy",
            "api_key": API_KEY,
        },
    )

    assert response.status_code == 400
