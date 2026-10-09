import os

import cv2
import numpy as np
import pytest
import requests
import supervision as sv
from numpy import ndarray
from pycocotools import mask as mask_utils

from tests.inference.integration_tests.conftest import (
    api_key_auth_headers,
    without_api_key_in_header_mode,
)

USE_INFERENCE_MODELS = os.getenv("USE_INFERENCE_MODELS", "false").lower() == "true"
API_KEY = os.environ.get("API_KEY")
PORT = os.environ.get("PORT", 9001)
BASE_URL = os.environ.get("BASE_URL", "http://localhost")


@pytest.mark.skipif(
    not USE_INFERENCE_MODELS, reason="Resolution control uses inference-models"
)
@pytest.mark.parametrize("opt_in", [None, False, True])
@pytest.mark.parametrize("response_format", ["polygon", "rle"])
@pytest.mark.parametrize(
    "mode,factor", [("accurate", 1.0), ("tradeoff", 0.5), ("fast", 0.0)]
)
def test_mask_resolution_round_trip_through_server(
    auth_mode: str, response_format: str, mode: str, factor: float, opt_in
) -> None:
    payload = {
        "image": {"type": "url", "value": "https://media.roboflow.com/dog.jpeg"},
        "api_key": API_KEY,
        "model_id": "yolov8n-seg-640",
        "response_mask_format": response_format,
        "mask_decode_mode": mode,
        "tradeoff_factor": factor,
    }
    if opt_in is not None:
        payload["allow_reduced_mask_resolution"] = opt_in

    response = requests.post(
        f"{BASE_URL}:{PORT}/infer/instance_segmentation",
        json=without_api_key_in_header_mode(auth_mode, payload),
        headers=api_key_auth_headers(auth_mode, API_KEY),
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()
    metadata = result.get("mask_metadata")
    if opt_in:
        assert metadata["coordinate_system"] == "mask_grid"
        assert result["image"] == {
            "width": metadata["width"],
            "height": metadata["height"],
        }
        assert result["original_image"] == {"width": 720, "height": 1280}
        assert metadata["scale_x"] == pytest.approx(720 / metadata["width"])
        assert metadata["scale_y"] == pytest.approx(1280 / metadata["height"])
        if mode != "accurate":
            assert metadata["width"] < 720 and metadata["height"] < 1280
    else:
        assert metadata is None
        assert result.get("original_image") is None
    detections = sv.Detections.from_inference(result)
    height, width = result["image"]["height"], result["image"]["width"]
    assert len(detections) > 0
    assert all(
        prediction["mask_format"] == response_format
        for prediction in result["predictions"]
    )
    assert detections.mask.shape == (len(detections), height, width)
    assert detections.mask.any()
    sv.MaskAnnotator().annotate(
        np.zeros((height, width, 3), dtype=np.uint8), detections
    )

    if response_format == "rle":
        sizes = {
            tuple(prediction["rle"]["size"]) for prediction in result["predictions"]
        }
        assert len(sizes) == 1
        mask_height, mask_width = sizes.pop()
        if metadata:
            assert (mask_height, mask_width) == (metadata["height"], metadata["width"])
        assert (mask_height, mask_width) == (height, width)

    reference_response = requests.post(
        f"{BASE_URL}:{PORT}/infer/instance_segmentation",
        json=without_api_key_in_header_mode(
            auth_mode, {**payload, "mask_decode_mode": "accurate"}
        ),
        headers=api_key_auth_headers(auth_mode, API_KEY),
        timeout=120,
    )
    reference_response.raise_for_status()
    reference = sv.Detections.from_inference(reference_response.json())
    scale_x = metadata["scale_x"] if metadata else 1.0
    scale_y = metadata["scale_y"] if metadata else 1.0
    np.testing.assert_allclose(
        detections.xyxy * [scale_x, scale_y, scale_x, scale_y], reference.xyxy, atol=1
    )
    np.testing.assert_array_equal(detections.class_id, reference.class_id)
    restored_masks = np.stack(
        [
            cv2.resize(
                mask.astype(np.uint8), (720, 1280), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            for mask in detections.mask
        ]
    )
    intersection = np.logical_and(restored_masks, reference.mask).sum(axis=(1, 2))
    union = np.logical_or(restored_masks, reference.mask).sum(axis=(1, 2))
    assert np.all(intersection / np.maximum(union, 1) > 0.8)


def test_v1_endpoint_with_valid_payload(auth_mode: str) -> None:
    payload = {
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "response_mask_format": "rle",
        "api_key": API_KEY,
        "model_id": "yolov8n-seg-640",
    }
    response = requests.post(
        f"{BASE_URL}:{PORT}/infer/instance_segmentation",
        json=without_api_key_in_header_mode(auth_mode, payload),
        headers=api_key_auth_headers(auth_mode, API_KEY),
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


def test_v1_endpoint_with_invalid_payload(auth_mode: str) -> None:
    payload = {
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "response_mask_format": "dummy",
        "api_key": API_KEY,
        "model_id": "yolov8n-seg-640",
    }

    response = requests.post(
        f"{BASE_URL}:{PORT}/infer/instance_segmentation",
        json=without_api_key_in_header_mode(auth_mode, payload),
        headers=api_key_auth_headers(auth_mode, API_KEY),
    )

    assert response.status_code == 422


def test_legacy_endpoint_valid_payload(auth_mode: str) -> None:
    response = requests.post(
        f"{BASE_URL}:{PORT}/coco-dataset-vdnr1/2",
        params=without_api_key_in_header_mode(
            auth_mode,
            {
                "image": "https://media.roboflow.com/dog.jpeg",
                "response_mask_format": "rle",
                "api_key": API_KEY,
            },
        ),
        headers=api_key_auth_headers(auth_mode, API_KEY),
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


def test_legacy_endpoint_invalid_payload(auth_mode: str) -> None:
    response = requests.post(
        f"{BASE_URL}:{PORT}/coco-dataset-vdnr1/2",
        params=without_api_key_in_header_mode(
            auth_mode,
            {
                "image": "https://media.roboflow.com/dog.jpeg",
                "response_mask_format": "dummy",
                "api_key": API_KEY,
            },
        ),
        headers=api_key_auth_headers(auth_mode, API_KEY),
    )

    assert response.status_code == 422


def test_legacy_endpoint_both_masks_variants_comparison(auth_mode: str) -> None:
    response_rle = requests.post(
        f"{BASE_URL}:{PORT}/coco-dataset-vdnr1/2",
        params=without_api_key_in_header_mode(
            auth_mode,
            {
                "image": "https://media.roboflow.com/dog.jpeg",
                "response_mask_format": "rle",
                "api_key": API_KEY,
            },
        ),
        headers=api_key_auth_headers(auth_mode, API_KEY),
    )
    response_rle.raise_for_status()
    rle_data = response_rle.json()
    response_polygon = requests.post(
        f"{BASE_URL}:{PORT}/coco-dataset-vdnr1/2",
        params=without_api_key_in_header_mode(
            auth_mode,
            {
                "image": "https://media.roboflow.com/dog.jpeg",
                "api_key": API_KEY,
            },
        ),
        headers=api_key_auth_headers(auth_mode, API_KEY),
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
