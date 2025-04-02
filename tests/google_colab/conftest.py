import os

import cv2
import numpy as np
import pytest
import gdown

if os.getenv("ENFORCE_GPU_EXECUTION"):
    os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"
else:
    os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"


REFERENCE_IMAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets", "dog.jpeg"))
REFERENCE_VIDEO_URL = "https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu"
REFERENCE_VIDEO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets", "video.mp4"))
PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"
PLAYER_CLASS_ID = 2
FOOTBALL_FIELD_DETECTOR_MODEL_ID = "football-field-detection-f07vi/14"


@pytest.fixture(scope="function")
def reference_image() -> np.ndarray:
    return cv2.imread(REFERENCE_IMAGE_PATH)


@pytest.fixture()
def roboflow_api_key() -> str:
    return os.environ["ROBOFLOW_API_KEY"]


@pytest.fixture(scope="function")
def reference_video() -> str:
    if os.path.isfile(REFERENCE_VIDEO_PATH):
        return REFERENCE_VIDEO_PATH
    gdown.download(REFERENCE_VIDEO_URL, REFERENCE_VIDEO_PATH)
    return REFERENCE_VIDEO_PATH


