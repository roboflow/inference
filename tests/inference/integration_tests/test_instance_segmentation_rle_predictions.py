import os

import numpy as np
import requests
import supervision as sv
from numpy import ndarray
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
    data = response.json()
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
            detection["rle"]["counts"] = detection["rle"]["counts"].encode("ascii")
            print(detection["rle"])
            rles.append(detection["rle"])
        masks = mask_utils.decode(rles).transpose(2, 0, 1).astype(bool)
        assert masks.shape[1:] == (1280, 720)


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

    assert response.status_code == 422



def test_legacy_endpoint_valid_payload() -> None:
    response = requests.post(
        f"{BASE_URL}:{PORT}/coco-dataset-vdnr1/2",
        params={
            "image": "https://media.roboflow.com/dog.jpeg",
            "response_mask_format": "rle",
            "api_key": API_KEY,
        },
    )
    response.raise_for_status()
    data = response.json()
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
        assert masks.shape[1:] == (1280, 720)


def test_legacy_endpoint_invalid_payload() -> None:
    response = requests.post(
        f"{BASE_URL}:{PORT}/coco-dataset-vdnr1/2",
        params={
            "image": "https://media.roboflow.com/dog.jpeg",
            "response_mask_format": "dummy",
            "api_key": API_KEY,
        },
    )

    assert response.status_code == 422


def test_legacy_endpoint_both_masks_variants_comparison() -> None:
    response_rle = requests.post(
        f"{BASE_URL}:{PORT}/coco-dataset-vdnr1/2",
        params={
            "image": "https://media.roboflow.com/dog.jpeg",
            "response_mask_format": "rle",
            "api_key": API_KEY,
        },
    )
    response_rle.raise_for_status()
    rle_data = response_rle.json()
    response_polygon = requests.post(
        f"{BASE_URL}:{PORT}/coco-dataset-vdnr1/2",
        params={
            "image": "https://media.roboflow.com/dog.jpeg",
            "api_key": API_KEY,
        },
    )
    response_polygon.raise_for_status()
    polygon_data = response_polygon.json()

    if not USE_INFERENCE_MODELS:
        for detection in rle_data["predictions"]:
            assert detection["mask_format"] == "polygon"
        detections = sv.Detections.from_inference(rle_data)
        assert isinstance(detections, sv.Detections)
        rle_data_mask = detections.mask
    else:
        rles = []
        for detection in rle_data["predictions"]:
            assert detection["mask_format"] == "rle"
            detection["mask_format"] = "polygon"
            rles.append(detection["rle"])
        rle_data_mask = mask_utils.decode(rles).transpose(2, 0, 1).astype(bool)

    detections_polygon = sv.Detections.from_inference(polygon_data)
    assert isinstance(detections_polygon, sv.Detections)
    polygon_data_mask = detections_polygon.mask

    if not USE_INFERENCE_MODELS:
        assert np.allclose(polygon_data_mask, rle_data_mask)
