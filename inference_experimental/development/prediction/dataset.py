from typing import List, Tuple

import backoff
import cv2
import numpy as np
import requests
from tqdm import tqdm

REFERENCE_IMAGES_URL = [
    (
        "tennis",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/3iCH40NuJxcf8l2tXgQn/original.jpg",
    ),
    (
        "beach",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/TAnwdBgfDCoPH2jT1ghx/original.jpg",
    ),
    (
        "giraffe",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/T4nrLKwEA0vHp8aJFPTt/original.jpg",
    ),
    (
        "car",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/aFq7tthQAK6d4pvtupX7/original.jpg",
    ),
    (
        "crowd",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/3FBCYL5SX7VPrg0OVkdN/original.jpg",
    ),
    (
        "food",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/XzDB9zVrIxJm17iVKleP/original.jpg",
    ),
    (
        "animals",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/0fsReHjmHk3hBadXdNk4/original.jpg",
    ),
    (
        "elephants",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/t23lZ0inksJwRRLd3J1b/original.jpg",
    ),
]


def download_dataset() -> List[Tuple[str, np.ndarray]]:
    results = []
    for image_id, url in tqdm(REFERENCE_IMAGES_URL, desc="Downloading dataset..."):
        image = load_image_from_url(url=url)
        results.append((image_id, image))
    return results


@backoff.on_exception(
    backoff.constant,
    exception=requests.RequestException,
    max_tries=3,
    interval=1,
)
def load_image_from_url(url: str) -> np.ndarray:
    response = requests.get(url)
    response.raise_for_status()
    payload = response.content
    bytes_array = np.frombuffer(payload, dtype=np.uint8)
    decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)
    if decoding_result is None:
        raise ValueError("Could not encode bytes to OpenCV image.")
    return decoding_result
