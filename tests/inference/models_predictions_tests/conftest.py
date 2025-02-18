import json
import os.path
import shutil
import zipfile
from typing import Dict, Generator

import cv2
import numpy as np
import pytest
import requests

from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    InstanceSegmentationInferenceResponse,
    KeypointsDetectionInferenceResponse,
    MultiLabelClassificationInferenceResponse,
    ObjectDetectionInferenceResponse,
)
from inference.core.env import MODEL_CACHE_DIR

ASSETS_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "assets",
    )
)
EXAMPLE_IMAGE_PATH = os.path.join(ASSETS_DIR, "example_image.jpg")
PERSON_IMAGE_PATH = os.path.join(ASSETS_DIR, "person_image.jpg")
BEER_IMAGE_PATH = os.path.join(ASSETS_DIR, "beer.jpg")
TRUCK_IMAGE_PATH = os.path.join(ASSETS_DIR, "truck.jpg")
SAM2_TRUCK_LOGITS = os.path.join(ASSETS_DIR, "low_res_logits.npy")
SAM2_TRUCK_MASK_FROM_CACHE = os.path.join(ASSETS_DIR, "mask_from_cached_logits.npy")
SAM2_MULTI_POLY_RESPONSE_PATH = os.path.join(
    ASSETS_DIR, "sam2_multipolygon_response.json"
)


@pytest.fixture(scope="function")
def sam2_multipolygon_response() -> Dict:
    with open(SAM2_MULTI_POLY_RESPONSE_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="function")
def example_image() -> np.ndarray:
    return cv2.imread(EXAMPLE_IMAGE_PATH)


@pytest.fixture(scope="function")
def person_image() -> np.ndarray:
    return cv2.imread(PERSON_IMAGE_PATH)


@pytest.fixture(scope="function")
def beer_image() -> np.ndarray:
    return cv2.imread(BEER_IMAGE_PATH)


@pytest.fixture(scope="function")
def truck_image() -> np.ndarray:
    return cv2.imread(TRUCK_IMAGE_PATH)


@pytest.fixture(scope="function")
def vit_multi_class_model() -> Generator[str, None, None]:
    model_id = "vit_multi_class/1"
    model_cache_dir = fetch_and_place_model_in_cache(
        model_id=model_id,
        model_package_url="https://storage.googleapis.com/roboflow-tests-assets/vit_multi_class.zip",
    )
    yield model_id
    shutil.rmtree(model_cache_dir)


@pytest.fixture(scope="function")
def vit_multi_class_reference_prediction() -> ClassificationInferenceResponse:
    with open(
        os.path.join(ASSETS_DIR, "vit_multi_class_reference_prediction.json"), "r"
    ) as f:
        return ClassificationInferenceResponse.model_validate(json.load(f))


@pytest.fixture(scope="function")
def vit_multi_label_model() -> Generator[str, None, None]:
    model_id = "vit_multi_label/1"
    model_cache_dir = fetch_and_place_model_in_cache(
        model_id=model_id,
        model_package_url="https://storage.googleapis.com/roboflow-tests-assets/vit_multi_label.zip",
    )
    yield model_id
    shutil.rmtree(model_cache_dir)


@pytest.fixture(scope="function")
def vit_multi_label_reference_prediction() -> MultiLabelClassificationInferenceResponse:
    with open(
        os.path.join(ASSETS_DIR, "vit_multi_label_reference_prediction.json"), "r"
    ) as f:
        return MultiLabelClassificationInferenceResponse.model_validate(json.load(f))


@pytest.fixture(scope="function")
def yolov5_det_model() -> Generator[str, None, None]:
    model_id = "yolov5_det/1"
    model_cache_dir = fetch_and_place_model_in_cache(
        model_id=model_id,
        model_package_url="https://storage.googleapis.com/roboflow-tests-assets/yolov5_det.zip",
    )
    yield model_id
    shutil.rmtree(model_cache_dir)


@pytest.fixture(scope="function")
def yolov5_det_reference_prediction() -> ObjectDetectionInferenceResponse:
    with open(
        os.path.join(ASSETS_DIR, "yolov5_det_reference_prediction.json"), "r"
    ) as f:
        return ObjectDetectionInferenceResponse.model_validate(json.load(f))


@pytest.fixture(scope="function")
def yolov5_seg_model() -> Generator[str, None, None]:
    model_id = "yolov5_seg/1"
    model_cache_dir = fetch_and_place_model_in_cache(
        model_id=model_id,
        model_package_url="https://storage.googleapis.com/roboflow-tests-assets/yolov5_seg.zip",
    )
    yield model_id
    shutil.rmtree(model_cache_dir)


@pytest.fixture(scope="function")
def yolov5_seg_reference_prediction() -> InstanceSegmentationInferenceResponse:
    with open(
        os.path.join(ASSETS_DIR, "yolov5_seg_reference_prediction.json"), "r"
    ) as f:
        return InstanceSegmentationInferenceResponse.model_validate(json.load(f))


@pytest.fixture(scope="function")
def yolov7_seg_model() -> Generator[str, None, None]:
    model_id = "yolov7_seg/1"
    model_cache_dir = fetch_and_place_model_in_cache(
        model_id=model_id,
        model_package_url="https://storage.googleapis.com/roboflow-tests-assets/yolov7_seg.zip",
    )
    yield model_id
    shutil.rmtree(model_cache_dir)


@pytest.fixture(scope="function")
def yolov7_seg_reference_prediction() -> InstanceSegmentationInferenceResponse:
    with open(
        os.path.join(ASSETS_DIR, "yolov7_seg_reference_prediction.json"), "r"
    ) as f:
        return InstanceSegmentationInferenceResponse.model_validate(json.load(f))


@pytest.fixture(scope="function")
def yolov8_cls_model() -> Generator[str, None, None]:
    model_id = "yolov8_cls/1"
    model_cache_dir = fetch_and_place_model_in_cache(
        model_id=model_id,
        model_package_url="https://storage.googleapis.com/roboflow-tests-assets/yolov8_cls.zip",
    )
    yield model_id
    shutil.rmtree(model_cache_dir)


@pytest.fixture(scope="function")
def yolov8_cls_reference_prediction() -> ClassificationInferenceResponse:
    with open(
        os.path.join(ASSETS_DIR, "yolov8_cls_reference_prediction.json"), "r"
    ) as f:
        return ClassificationInferenceResponse.model_validate(json.load(f))


@pytest.fixture(scope="function")
def yolov8_det_model() -> Generator[str, None, None]:
    model_id = "yolov8_det/1"
    model_cache_dir = fetch_and_place_model_in_cache(
        model_id=model_id,
        model_package_url="https://storage.googleapis.com/roboflow-tests-assets/yolov8_det.zip",
    )
    yield model_id
    shutil.rmtree(model_cache_dir)


@pytest.fixture(scope="function")
def yolov8_det_reference_prediction() -> ObjectDetectionInferenceResponse:
    with open(
        os.path.join(ASSETS_DIR, "yolov8_det_reference_prediction.json"), "r"
    ) as f:
        return ObjectDetectionInferenceResponse.model_validate(json.load(f))


@pytest.fixture(scope="function")
def yolov8_pose_model() -> Generator[str, None, None]:
    model_id = "yolov8_pose/1"
    model_cache_dir = fetch_and_place_model_in_cache(
        model_id=model_id,
        model_package_url="https://storage.googleapis.com/roboflow-tests-assets/yolov8_pose.zip",
    )
    yield model_id
    shutil.rmtree(model_cache_dir)


@pytest.fixture(scope="function")
def yolov8_pose_reference_prediction() -> KeypointsDetectionInferenceResponse:
    with open(
        os.path.join(ASSETS_DIR, "yolov8_pose_reference_prediction.json"), "r"
    ) as f:
        return KeypointsDetectionInferenceResponse.model_validate(json.load(f))


@pytest.fixture(scope="function")
def yolov8_seg_model() -> Generator[str, None, None]:
    model_id = "yolov8_seg/1"
    model_cache_dir = fetch_and_place_model_in_cache(
        model_id=model_id,
        model_package_url="https://storage.googleapis.com/roboflow-tests-assets/yolov8_seg.zip",
    )
    yield model_id
    shutil.rmtree(model_cache_dir)


@pytest.fixture(scope="function")
def yolov8_seg_reference_prediction() -> InstanceSegmentationInferenceResponse:
    with open(
        os.path.join(ASSETS_DIR, "yolov8_seg_reference_prediction.json"), "r"
    ) as f:
        return InstanceSegmentationInferenceResponse.model_validate(json.load(f))


@pytest.fixture(scope="function")
def yolonas_det_model() -> Generator[str, None, None]:
    model_id = "yolonas/1"
    model_cache_dir = fetch_and_place_model_in_cache(
        model_id=model_id,
        model_package_url="https://storage.googleapis.com/roboflow-tests-assets/yolonas.zip",
    )
    yield model_id
    shutil.rmtree(model_cache_dir)


@pytest.fixture(scope="function")
def yolonas_det_reference_prediction() -> ObjectDetectionInferenceResponse:
    with open(
        os.path.join(ASSETS_DIR, "yolonas_det_reference_prediction.json"), "r"
    ) as f:
        return ObjectDetectionInferenceResponse.model_validate(json.load(f))


@pytest.fixture(scope="function")
def yolov10_det_model() -> Generator[str, None, None]:
    model_id = "yolov10_det/1"
    model_cache_dir = fetch_and_place_model_in_cache(
        model_id=model_id,
        model_package_url="https://storage.googleapis.com/roboflow-tests-assets/yolov10_det.zip",
    )
    yield model_id
    shutil.rmtree(model_cache_dir)


@pytest.fixture(scope="function")
def yolov10_det_reference_prediction() -> ObjectDetectionInferenceResponse:
    with open(
        os.path.join(ASSETS_DIR, "yolov10_det_reference_prediction.json"), "r"
    ) as f:
        return ObjectDetectionInferenceResponse.model_validate(json.load(f))


@pytest.fixture(scope="function")
def sam2_tiny_model() -> Generator[str, None, None]:
    model_id = "sam2/hiera_tiny"
    model_cache_dir = fetch_and_place_model_in_cache(
        model_id=model_id,
        model_package_url="https://storage.googleapis.com/roboflow-tests-assets/sam2_tiny.zip",
    )
    yield model_id
    shutil.rmtree(model_cache_dir)


@pytest.fixture(scope="function")
def sam2_small_model() -> Generator[str, None, None]:
    model_id = "sam2/hiera_small"
    model_cache_dir = fetch_and_place_model_in_cache(
        model_id=model_id,
        model_package_url="https://storage.googleapis.com/roboflow-tests-assets/sam2_small.zip",
    )
    yield model_id
    shutil.rmtree(model_cache_dir)


@pytest.fixture(scope="function")
def sam2_tiny_model() -> Generator[str, None, None]:
    model_id = "sam2/hiera_tiny"
    model_cache_dir = fetch_and_place_model_in_cache(
        model_id=model_id,
        model_package_url="https://storage.googleapis.com/roboflow-tests-assets/sam2_tiny.zip",
    )
    yield model_id
    shutil.rmtree(model_cache_dir)


@pytest.fixture(scope="function")
def sam2_small_truck_logits() -> Generator[np.ndarray, None, None]:
    yield np.load(SAM2_TRUCK_LOGITS)


@pytest.fixture(scope="function")
def sam2_small_truck_mask_from_cached_logits() -> Generator[np.ndarray, None, None]:
    yield np.load(SAM2_TRUCK_MASK_FROM_CACHE)


def fetch_and_place_model_in_cache(
    model_id: str,
    model_package_url: str,
) -> str:
    target_model_directory = os.path.join(MODEL_CACHE_DIR, model_id)
    if os.path.isdir(target_model_directory):
        shutil.rmtree(target_model_directory)
    download_location = os.path.join(ASSETS_DIR, os.path.basename(model_package_url))
    if not os.path.exists(download_location):
        download_file(file_url=model_package_url, target_path=download_location)
    extract_zip_package(zip_path=download_location, target_dir=target_model_directory)
    return target_model_directory


def download_file(
    file_url: str,
    target_path: str,
    chunk_size: int = 8192,
) -> None:
    with requests.get(file_url, stream=True) as response:
        response.raise_for_status()
        with open(target_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)


def extract_zip_package(zip_path: str, target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
