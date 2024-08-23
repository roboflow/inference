import json
import os
import requests
from copy import deepcopy
from pathlib import Path
import pytest
import time
import numpy as np
import supervision as sv
from PIL import Image
import requests
from io import BytesIO
from typing import Dict

from copy import deepcopy

from tests.inference.integration_tests.regression_test import bool_env
from inference.core.workflows.core_steps.models.foundation.segment_anything2.v1 import (
    convert_sam2_segmentation_response_to_inference_instances_seg_response,
)
from inference.core.workflows.core_steps.common.utils import (
    convert_inference_detections_batch_to_sv_detections,
)
from inference.core.entities.responses.sam2 import Sam2SegmentationPrediction


api_key = os.environ.get("API_KEY")
port = os.environ.get("PORT", 9001)
base_url = os.environ.get("BASE_URL", "http://localhost")


version_ids = [
    "hiera_small",
    "hiera_large",
    "hiera_tiny",
    "hiera_b_plus",
]
payload_ = {
    "image": {
        "type": "url",
        "value": "https://source.roboflow.com/D8zLgnZxdqtqF0plJINA/DqK7I0rUz5HBvu1hdNi6/original.jpg",
    },
    "image_id": "test",
}

tests = ["embed_image", "segment_image"]


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_SAM2_TESTS", True)),
    reason="Skipping SAM test",
)
@pytest.mark.parametrize("version_id", version_ids)
@pytest.mark.parametrize("test", tests)
def test_sam2(version_id, test, clean_loaded_models_fixture):
    payload = deepcopy(payload_)
    payload["api_key"] = api_key
    payload["sam2_version_id"] = version_id
    response = requests.post(
        f"{base_url}:{port}/sam2/{test}",
        json=payload,
    )
    try:
        response.raise_for_status()
        data = response.json()
        if test == "embed_image":
            try:
                assert "image_id" in data
            except:
                print(f"Invalid response: {data}, expected 'image_id' in response")
        if test == "segment_image":
            try:
                assert "masks" in data
            except:
                print(f"Invalid response: {data}, expected 'masks' in response")
    except Exception as e:
        raise e


def convert_response_dict_to_sv_detections(image: Image, response_dict: Dict):
    class DummyImage:
        def __init__(self, image_array):
            self.numpy_image = image_array

    image_object = DummyImage(np.asarray(image))
    preds = convert_sam2_segmentation_response_to_inference_instances_seg_response(
        [Sam2SegmentationPrediction(**p) for p in response_dict["predictions"]],
        image_object,
        [],
        [],
        0,
    )
    preds = preds.model_dump(by_alias=True, exclude_none=True)
    return convert_inference_detections_batch_to_sv_detections([preds])[0]


def test_sam2_multi_poly(clean_loaded_models_fixture, sam2_multipolygon_response):
    version_id = "hiera_tiny"
    payload = deepcopy(payload_)
    payload["api_key"] = api_key
    payload["sam2_version_id"] = version_id
    image_url = "https://media.roboflow.com/inference/seawithdock.jpeg"
    payload["image"]["value"] = image_url
    payload["prompts"] = {
        "prompts": [{"points": [{"x": 58, "y": 379, "positive": True}]}]
    }
    response = requests.post(
        f"{base_url}:{port}/sam2/segment_image",
        json=payload,
    )
    try:
        sam2_multipolygon_response = deepcopy(sam2_multipolygon_response)
        response.raise_for_status()
        data = response.json()
        with open("test_multi.json", "w") as f:
            json.dump(data, f)
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        preds = convert_response_dict_to_sv_detections(image, data)
        ground_truth = convert_response_dict_to_sv_detections(
            image, sam2_multipolygon_response
        )
        print(preds.mask.min(), preds.mask.max(), preds.mask.shape)
        print(ground_truth.mask.min(), ground_truth.mask.max(), ground_truth.mask.shape)
        preds_bool_mask = np.logical_or.reduce(preds.mask, axis=0)
        ground_truth_bool_mask = np.logical_or.reduce(ground_truth.mask, axis=0)
        iou = (
            np.logical_and(preds_bool_mask, ground_truth_bool_mask).sum()
            / np.logical_or(preds_bool_mask, ground_truth_bool_mask).sum()
        )
        assert iou > 0.99
        try:
            assert "predictions" in data
        except:
            print(f"Invalid response: {data}, expected 'predictions' in response")
            raise
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
    test_sam2()
