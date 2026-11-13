import os
import zipfile

import pytest
import requests
from filelock import FileLock

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

YOLOV8N_TS_PACKAGE_URL = (
    "https://storage.googleapis.com/roboflow-tests-assets/"
    "rf-platform-models/yolov8n-torchscript-static-bs-letterbox.zip"
)
YOLOV8N_TS_PACKAGE_NAME = "yolov8n-torchscript-static-bs-letterbox"


def _download_if_not_exists(file_path: str, url: str, lock_timeout: int = 180) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    lock_path = f"{file_path}.lock"
    with FileLock(lock_file=lock_path, timeout=lock_timeout):
        if os.path.exists(file_path):
            return
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)


@pytest.fixture(scope="module")
def yolov8n_model_path() -> str:
    """Download and extract a small YOLOv8n model package. Returns local path."""
    package_dir = os.path.join(ASSETS_DIR, "backends_test")
    zip_path = os.path.join(package_dir, f"{YOLOV8N_TS_PACKAGE_NAME}.zip")
    os.makedirs(package_dir, exist_ok=True)
    _download_if_not_exists(file_path=zip_path, url=YOLOV8N_TS_PACKAGE_URL)
    # Zip extracts files directly into package_dir (no subdirectory)
    marker = os.path.join(package_dir, "weights.torchscript")
    if not os.path.exists(marker):
        lock_path = f"{zip_path}.extract.lock"
        with FileLock(lock_path, timeout=180):
            if not os.path.exists(marker):
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(package_dir)
    return package_dir
