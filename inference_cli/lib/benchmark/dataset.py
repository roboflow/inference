import os.path
from glob import glob
from itertools import chain
from typing import List, Optional

import cv2
import numpy as np
import requests
from tqdm import tqdm

from inference_cli.lib.exceptions import DatasetLoadingError
from inference_cli.lib.logger import CLI_LOGGER
from inference_sdk.http.utils.encoding import bytes_to_opencv_image

MAX_IMAGES_TO_LOAD = 8
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
PREDEFINED_DATASETS = {
    "coco": [
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/aFq7tthQAK6d4pvtupX7/original.jpg",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/KmFskd2RQMfcnDNjzeeA/original.jpg",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/3FBCYL5SX7VPrg0OVkdN/original.jpg",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/K2KrTzjxYu0kJCScGcoH/original.jpg",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/XzDB9zVrIxJm17iVKleP/original.jpg",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/0fsReHjmHk3hBadXdNk4/original.jpg",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/t23lZ0inksJwRRLd3J1b/original.jpg",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/3iCH40NuJxcf8l2tXgQn/original.jpg",
    ]
}


def load_dataset_images(dataset_reference: str) -> List[np.ndarray]:
    if os.path.isdir(dataset_reference):
        return load_images(directory=dataset_reference)
    if dataset_reference not in PREDEFINED_DATASETS:
        raise DatasetLoadingError(f"Could not find dataset: {dataset_reference}")
    return download_images(urls=PREDEFINED_DATASETS[dataset_reference])


def load_images(
    directory: str, max_images_to_load: int = MAX_IMAGES_TO_LOAD
) -> List[np.ndarray]:
    file_paths = sorted(
        list(
            chain.from_iterable(
                glob(os.path.join(directory, e)) for e in IMAGE_EXTENSIONS
            )
        )
    )
    results = []
    progress_bar = tqdm(desc="Loading images...", total=max_images_to_load)
    for file_path in file_paths:
        image = load_image(path=file_path)
        if image is None:
            continue
        results.append(image)
        progress_bar.update()
    progress_bar.close()
    if len(results) < 1:
        raise DatasetLoadingError(f"Could not load images from {directory}")
    return results


def load_image(path: str) -> Optional[np.ndarray]:
    try:
        return cv2.imread(path)
    except Exception as error:
        CLI_LOGGER.warning(f"Could not load image: {path}. Cause: {error}")
        return None


def download_images(urls: List[str]) -> List[np.ndarray]:
    results = [download_image(url=url) for url in tqdm(urls, "Loading images...")]
    non_empty_results = [r for r in results if r is not None]
    if len(non_empty_results) < 1:
        raise DatasetLoadingError(f"Could not load images")
    return non_empty_results


def download_image(url: str) -> Optional[np.ndarray]:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return bytes_to_opencv_image(payload=response.content)
    except Exception as error:
        CLI_LOGGER.warning(f"Could not load image: {url}. Cause: {error}")
        return None
