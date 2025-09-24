import os.path

import cv2
import numpy as np
import pytest
import requests
import torch
import torchvision.io
from filelock import FileLock
from PIL import Image

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
MODELS_DIR = os.path.join(ASSETS_DIR, "models")
DOG_IMAGE_PATH = os.path.join(ASSETS_DIR, "dog.jpeg")
DOG_IMAGE_URL = "https://media.roboflow.com/dog.jpeg"
CLIP_RN50_TORCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/clip_packages/RN50/torch/model.pt"
CLIP_RN50_ONNX_VISUAL = "https://storage.googleapis.com/roboflow-tests-assets/clip_packages/RN50/onnx/visual.onnx"
CLIP_RN50_ONNX_TEXTUAL = "https://storage.googleapis.com/roboflow-tests-assets/clip_packages/RN50/onnx/textual.onnx"
PE_MODEL_URL = "https://storage.googleapis.com/roboflow-tests-assets/perception-encoder/pe-core-b16-224/model.pt"
PE_CONFIG_URL = "https://storage.googleapis.com/roboflow-tests-assets/perception-encoder/pe-core-b16-224/config.json"


@pytest.fixture(scope="module")
def original_clip_download_dir() -> str:
    clip_dir = os.path.join(MODELS_DIR, "clip_original")
    os.makedirs(clip_dir, exist_ok=True)
    return clip_dir


@pytest.fixture(scope="module")
def clip_rn50_pytorch_path() -> str:
    package_path = os.path.join(MODELS_DIR, "clip_rn50", "torch")
    os.makedirs(package_path, exist_ok=True)
    model_path = os.path.join(package_path, "model.pt")
    _download_if_not_exists(file_path=model_path, url=CLIP_RN50_TORCH_URL)
    return package_path


@pytest.fixture(scope="module")
def clip_rn50_onnx_path() -> str:
    package_path = os.path.join(MODELS_DIR, "clip_rn50", "onnx")
    os.makedirs(package_path, exist_ok=True)
    visual_path = os.path.join(package_path, "visual.onnx")
    textual_path = os.path.join(package_path, "textual.onnx")
    _download_if_not_exists(file_path=visual_path, url=CLIP_RN50_ONNX_VISUAL)
    _download_if_not_exists(file_path=textual_path, url=CLIP_RN50_ONNX_TEXTUAL)
    return package_path


@pytest.fixture(scope="module")
def perception_encoder_path() -> str:
    package_path = os.path.join(MODELS_DIR, "perception_encoder")
    os.makedirs(package_path, exist_ok=True)
    model_path = os.path.join(package_path, "model.pt")
    config_path = os.path.join(package_path, "config.json")
    _download_if_not_exists(file_path=model_path, url=PE_MODEL_URL)
    _download_if_not_exists(file_path=config_path, url=PE_CONFIG_URL)
    return package_path


@pytest.fixture(scope="function")
def dog_image_numpy() -> np.ndarray:
    _download_if_not_exists(file_path=DOG_IMAGE_PATH, url=DOG_IMAGE_URL)
    image = cv2.imread(DOG_IMAGE_PATH)
    assert image is not None, "Could not load test image"
    return image


@pytest.fixture(scope="function")
def dog_image_torch() -> torch.Tensor:
    _download_if_not_exists(file_path=DOG_IMAGE_PATH, url=DOG_IMAGE_URL)
    return torchvision.io.read_image(DOG_IMAGE_PATH)


@pytest.fixture(scope="function")
def dog_image_pil() -> Image.Image:
    _download_if_not_exists(file_path=DOG_IMAGE_PATH, url=DOG_IMAGE_URL)
    return Image.open(DOG_IMAGE_PATH)


def _download_if_not_exists(file_path: str, url: str, lock_timeout: int = 120) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    lock_path = f"{file_path}.lock"
    with FileLock(lock_file=lock_path, timeout=lock_timeout):
        if os.path.exists(file_path):
            return None
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
