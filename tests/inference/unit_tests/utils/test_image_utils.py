import io
import pickle

import cv2
import numpy as np
import pytest
from requests_mock import Mocker

from inference.core.exceptions import InputImageLoadError, InvalidNumpyInput
from inference.core.utils.image_utils import load_image_from_url, load_image_from_numpy_str, load_image_from_buffer, \
    load_image_base64


@pytest.mark.parametrize(
    "response_status_code",
    [400, 401, 403, 404, 500, 501, 502, 503, 504]
)
def test_load_image_from_url_when_request_not_succeeded(
    requests_mock: Mocker,
    response_status_code: int,
) -> None:
    # given
    resource_url = "https://some.com/image.jpg"
    requests_mock.get(
        resource_url,
        status_code=response_status_code,
    )

    # when

    with pytest.raises(InputImageLoadError):
        _ = load_image_from_url(value=resource_url)


def test_load_image_from_url_when_payload_does_not_contain_image(
    requests_mock: Mocker,
) -> None:
    # given
    resource_url = "https://some.com/image.jpg"
    requests_mock.get(
        resource_url,
        content=b"FOR SURE NOT AN IMAGE :)",
    )

    # when

    with pytest.raises(InputImageLoadError):
        _ = load_image_from_url(value=resource_url)


def test_load_image_from_url_when_jpeg_image_should_be_successfully_decoded(
    requests_mock: Mocker,
    image_as_numpy: np.ndarray,
    image_as_jpeg_bytes: bytes,
) -> None:
    # given
    resource_url = "https://some.com/image.jpg"
    requests_mock.get(
        resource_url,
        content=image_as_jpeg_bytes,
    )

    # when
    result = load_image_from_url(value=resource_url)

    # then
    assert image_as_numpy.shape == result.shape
    assert np.allclose(image_as_numpy, result)


def test_load_image_from_url_when_png_image_should_be_successfully_decoded(
    requests_mock: Mocker,
    image_as_numpy: np.ndarray,
    image_as_png_bytes: bytes,
) -> None:
    resource_url = "https://some.com/image.png"
    requests_mock.get(
        resource_url,
        content=image_as_png_bytes,
    )

    # when
    result = load_image_from_url(value=resource_url)

    # then
    assert image_as_numpy.shape == result.shape
    assert np.allclose(image_as_numpy, result)


def test_load_image_from_numpy_str_when_empty_bytes_given() -> None:
    # when
    with pytest.raises(InvalidNumpyInput):
        _ = load_image_from_numpy_str(value=b"")


def test_load_image_from_numpy_str_when_string_given() -> None:
    # when
    with pytest.raises(InvalidNumpyInput):
        _ = load_image_from_numpy_str(value="some")


def test_load_image_from_numpy_str_when_non_object_bytes_given() -> None:
    # when
    with pytest.raises(InvalidNumpyInput):
        _ = load_image_from_numpy_str(value=b"some")


def test_load_image_from_numpy_str_when_non_array_bytes_given() -> None:
    # given
    payload = pickle.dumps([1, 2, 3])

    # when
    with pytest.raises(InvalidNumpyInput):
        _ = load_image_from_numpy_str(value=payload)


def test_load_image_from_numpy_str_when_invalid_shape_array_bytes_given() -> None:
    # given
    payload = pickle.dumps(np.array([1, 2, 3]))

    # when
    with pytest.raises(InvalidNumpyInput):
        _ = load_image_from_numpy_str(value=payload)


def test_load_image_from_numpy_str_when_array_with_non_standard_channels_given() -> None:
    # given
    payload = pickle.dumps(np.ones((128, 128, 4), dtype=np.uint8))

    # when
    with pytest.raises(InvalidNumpyInput):
        _ = load_image_from_numpy_str(value=payload)


def test_load_image_from_numpy_str_when_array_with_invalid_values_given() -> None:
    # given
    payload = pickle.dumps(1024 * np.ones((128, 128, 3), dtype=np.uint8))

    # when
    with pytest.raises(InvalidNumpyInput):
        _ = load_image_from_numpy_str(value=payload)


def test_load_image_from_numpy_str_when_valid_image_given(
    image_as_pickled_bytes: bytes,
    image_as_numpy: np.ndarray,
) -> None:
    # when
    result = load_image_from_numpy_str(value=image_as_pickled_bytes)

    # then
    assert image_as_numpy.shape == result.shape
    assert np.allclose(image_as_numpy, result)


def test_load_image_from_buffer_when_valid_input_provided(
    image_as_jpeg_bytes: bytes,
    image_as_numpy,
) -> None:
    # given
    with io.BytesIO() as buffer:
        buffer.write(image_as_jpeg_bytes)

        # when
        result = load_image_from_buffer(value=buffer)

        # then
        assert image_as_numpy.shape == result.shape
        assert np.allclose(image_as_numpy, result)


def test_load_image_from_buffer_when_non_image_input_provided() -> None:
    # given
    with io.BytesIO() as buffer:
        buffer.write(b"Non-image")

        # when
        with pytest.raises(InputImageLoadError):
            _ = load_image_from_buffer(value=buffer)


def test_load_image_base64_when_valid_string_given(
    image_as_jpeg_base64_bytes: bytes,
    image_as_numpy: np.ndarray,
) -> None:
    # given
    payload = image_as_jpeg_base64_bytes.decode("utf-8")

    # when
    result = load_image_base64(value=payload)

    # then
    assert image_as_numpy.shape == result.shape
    assert np.allclose(image_as_numpy, result)


def test_load_image_base64_when_valid_bytes_given(
    image_as_jpeg_base64_bytes: bytes,
    image_as_numpy: np.ndarray,
) -> None:
    # when
    result = load_image_base64(value=image_as_jpeg_base64_bytes)

    # then
    assert image_as_numpy.shape == result.shape
    assert np.allclose(image_as_numpy, result)


def test_load_image_base64_when_valid_bytes_given_with_type_preamble(
    image_as_jpeg_base64_bytes: bytes,
    image_as_numpy: np.ndarray,
) -> None:
    # given
    payload = image_as_jpeg_base64_bytes.decode("utf-8")
    payload = f"data:image/jpeg;base64, {payload}"

    # when
    result = load_image_base64(value=payload)

    # then
    assert image_as_numpy.shape == result.shape
    assert np.allclose(image_as_numpy, result)


def test_load_image_base64_when_invalid_bytes_given() -> None:
    # when
    with pytest.raises(InputImageLoadError):
        _ = load_image_base64(value=b"some")

