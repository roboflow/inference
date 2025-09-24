import os.path
import tempfile
from typing import Generator

import pytest
import requests

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
IMAGES_URLS = [
    "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/aFq7tthQAK6d4pvtupX7/original.jpg",
    "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/KmFskd2RQMfcnDNjzeeA/original.jpg",
    "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/3FBCYL5SX7VPrg0OVkdN/original.jpg",
]
VIDEO_URL = "https://media.roboflow.com/inference/people-walking.mp4"

INFERENCE_CLI_TESTS_API_KEY = os.getenv("INFERENCE_CLI_TESTS_API_KEY")
RUN_TESTS_WITH_INFERENCE_PACKAGE = (
    os.getenv("RUN_TESTS_WITH_INFERENCE_PACKAGE", "False").lower() == "true"
)
RUN_TESTS_EXPECTING_ERROR_WHEN_INFERENCE_NOT_INSTALLED = (
    os.getenv("RUN_TESTS_EXPECTING_ERROR_WHEN_INFERENCE_NOT_INSTALLED", "False").lower()
    == "true"
)


@pytest.fixture(scope="function")
def empty_directory() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(scope="function")
def example_env_file_path() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "assets", "example.env")
    )


@pytest.fixture
def dataset_directory() -> str:
    dataset_directory = os.path.join(ASSETS_DIR, "test_images")
    os.makedirs(dataset_directory, exist_ok=True)
    expected_video_name = "video.mp4"
    current_content = set(os.listdir(dataset_directory))
    all_images_present = all(
        f"{i}.jpg" in current_content for i in range(len(IMAGES_URLS))
    )
    if all_images_present and expected_video_name in current_content:
        return dataset_directory
    for i, image_url in enumerate(IMAGES_URLS):
        response = requests.get(image_url)
        response.raise_for_status()
        image_bytes = response.content
        with open(os.path.join(dataset_directory, f"{i}.jpg"), "wb") as f:
            f.write(image_bytes)
    response = requests.get(VIDEO_URL)
    response.raise_for_status()
    video_bytes = response.content
    with open(os.path.join(dataset_directory, "video.mp4"), "wb") as f:
        f.write(video_bytes)
    return dataset_directory


@pytest.fixture
def video_to_be_processed(dataset_directory: str) -> str:
    return os.path.join(dataset_directory, "video.mp4")


@pytest.fixture
def image_to_be_processed(dataset_directory: str) -> str:
    return os.path.join(dataset_directory, "0.jpg")
