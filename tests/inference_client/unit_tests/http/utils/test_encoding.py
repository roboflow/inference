import base64
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
import pytest
from PIL import Image, ImageChops

from inference_client.http.errors import EncodingError
from inference_client.http.utils.encoding import (
    numpy_array_to_base64_jpeg,
    pillow_image_to_base64_jpeg,
    encode_base_64,
    bytes_to_opencv_image,
    bytes_to_pillow_image,
)


def test_numpy_array_to_base64_jpeg() -> None:
    # given
    image = np.zeros((128, 128, 3), dtype=np.uint8)

    # when
    encoded_image = numpy_array_to_base64_jpeg(image=image)
    decoded_image = base64.b64decode(encoded_image)
    bytes_array = np.frombuffer(decoded_image, dtype=np.uint8)
    recovered_image = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)

    # then
    assert image.shape == recovered_image.shape
    assert (image == recovered_image).all()


def test_pillow_image_to_base64_jpeg() -> None:
    # given
    image = Image.new(mode="RGB", size=(128, 128), color=(0, 0, 0))

    # when
    encoded_image = pillow_image_to_base64_jpeg(image=image)
    decoded_image = base64.b64decode(encoded_image)
    recovered_image = Image.open(BytesIO(decoded_image))

    # then
    assert image.size == recovered_image.size
    difference = ImageChops.difference(image, recovered_image)
    assert difference.getbbox() is None


def test_encode_base_64() -> None:
    # given
    payload = b"""
    My name is Maximus Decimus Meridius, commander of the Armies of the North, General of the Felix Legions
    and loyal servant to the true emperor, Marcus Aurelius. Father to a murdered son. Husband to a 
    murdered wife. And I will have my vengeance, in this life or the next.
    """

    # when
    encoded_payload = encode_base_64(payload=payload)
    decoded_payload = base64.b64decode(encoded_payload)

    # then
    assert decoded_payload == payload


@pytest.mark.parametrize("encoding", [".jpg", ".png"])
def test_bytes_to_opencv_image_when_bytes_represent_image(encoding: str) -> None:
    # given
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    _, encoded_image = cv2.imencode(encoding, image)

    # when
    recovered_image = bytes_to_opencv_image(payload=np.array(encoded_image).tobytes())

    # then
    assert image.shape == recovered_image.shape
    assert (image == recovered_image).all()


def test_bytes_to_opencv_image_when_bytes_do_not_represent_image() -> None:
    # given
    payload = b"For sure not an image :)"

    # when
    with pytest.raises(EncodingError):
        _ = bytes_to_opencv_image(payload=payload)


@pytest.mark.parametrize("encoding", ["PNG", "JPEG"])
def test_bytes_to_pillow_image_when_bytes_represent_image(
    encoding: Optional[str],
) -> None:
    # given
    image = Image.new(mode="RGB", size=(128, 128), color=(0, 0, 0))
    with BytesIO() as buffer:
        image.save(buffer, format=encoding)
        payload = buffer.getvalue()

    # when
    recovered_image = bytes_to_pillow_image(payload=payload)

    # then
    assert image.size == recovered_image.size
    difference = ImageChops.difference(image, recovered_image)
    assert difference.getbbox() is None


def test_bytes_to_pillow_image_when_bytes_do_not_represent_image() -> None:
    # given
    payload = b"For sure not an image :)"

    # when
    with pytest.raises(EncodingError):
        _ = bytes_to_pillow_image(payload=payload)
