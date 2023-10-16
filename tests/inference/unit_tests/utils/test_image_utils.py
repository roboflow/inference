import io
import pickle
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from PIL import Image
from _pytest.fixtures import FixtureRequest
from requests_mock import Mocker

from inference.core.exceptions import InputImageLoadError, InvalidNumpyInput
from inference.core.utils.image_utils import load_image_from_url, load_image_from_numpy_str, load_image_from_buffer, \
    load_image_base64, load_image_inferred, attempt_loading_image_from_string, load_image_from_encoded_bytes
from inference.core.utils import image_utils


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


def test_load_image_from_numpy_str_when_arbitrary_object_given() -> None:
    # when
    with pytest.raises(InvalidNumpyInput):
        _ = load_image_from_numpy_str(value=[1, 2, 3])


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
    image_as_buffer: io.BytesIO,
    image_as_numpy,
) -> None:
    # when
    result = load_image_from_buffer(value=image_as_buffer)

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


def test_load_image_inferred_when_value_is_np_array(image_as_numpy: np.ndarray) -> None:
    # when
    result = load_image_inferred(value=image_as_numpy)

    # then
    assert result == (image_as_numpy, True)


def test_load_image_inferred_when_value_is_pillow_image(
    image_as_pillow: Image.Image,
    image_as_numpy: np.ndarray,
) -> None:
    # when
    result = load_image_inferred(value=image_as_pillow)

    # then
    assert result[1] is False
    assert image_as_numpy.shape == result[0].shape
    assert np.allclose(image_as_numpy, result[0])


@mock.patch.object(image_utils, "load_image_from_url")
@pytest.mark.parametrize("url", ["http://some/image.jpg", "https://some/image.jpg"])
def test_load_image_inferred_when_value_is_url(
    load_image_from_url_mock: MagicMock,
    url: str,
) -> None:
    # when
    result = load_image_inferred(value=url, cv_imread_flags=cv2.IMREAD_COLOR)

    # then
    assert result[0] is load_image_from_url_mock.return_value
    assert result[1] is True
    load_image_from_url_mock.assert_called_once_with(value=url, cv_imread_flags=cv2.IMREAD_COLOR)


def test_load_image_inferred_when_value_is_local_image_path(
    image_as_local_path: str,
    image_as_numpy: np.ndarray,
) -> None:
    # when
    result = load_image_inferred(value=image_as_local_path)

    # then
    assert result[1] is True
    assert image_as_numpy.shape == result[0].shape
    assert np.allclose(image_as_numpy, result[0])


@mock.patch.object(image_utils, "attempt_loading_image_from_string")
@pytest.mark.parametrize("value", ["aaa", b"some", io.BytesIO(), [1, 2, 3]])
def test_load_image_inferred_when_value_is_unknown_and_should_be_tried_against_set_of_methods(
    attempt_loading_image_from_string_mock: MagicMock,
    value: Any,
) -> None:
    # when
    result = load_image_inferred(value=value, cv_imread_flags=cv2.IMREAD_COLOR)

    # then
    assert result is attempt_loading_image_from_string_mock.return_value
    attempt_loading_image_from_string_mock.assert_called_once_with(
        value=value, cv_imread_flags=cv2.IMREAD_COLOR,
    )


@pytest.mark.parametrize(
    "fixture_name",
    [
        "image_as_png_bytes",
        "image_as_jpeg_bytes",
    ],
)
def test_load_image_from_encoded_bytes_when_decoding_should_succeed(
    fixture_name: str,
    image_as_numpy: np.ndarray,
    request: FixtureRequest,
) -> None:
    # given
    value = request.getfixturevalue(fixture_name)

    # when
    result = load_image_from_encoded_bytes(value=value)

    # then
    assert image_as_numpy.shape == result.shape
    assert np.allclose(image_as_numpy, result)


def test_load_image_from_encoded_bytes_when_decoding_should_fail() -> None:
    # when
    with pytest.raises(InputImageLoadError):
        _ = load_image_from_encoded_bytes(value=b"FOR SURE NOT AN IMAGE :)")


@pytest.mark.parametrize(
    "fixture_name",
    [
        "image_as_jpeg_base64_bytes",
        "image_as_jpeg_base64_string",
        "image_as_png_bytes",
        "image_as_jpeg_bytes",
        "image_as_pickled_bytes",
    ],
)
def test_attempt_loading_image_from_string_when_parsing_should_be_successful(
    fixture_name: str,
    image_as_numpy: np.ndarray,
    request: FixtureRequest,
) -> None:
    # given
    value = request.getfixturevalue(fixture_name)

    # when
    result = attempt_loading_image_from_string(value=value)

    # then
    assert result[1] is True
    assert image_as_numpy.shape == result[0].shape
    assert np.allclose(image_as_numpy, result[0])


@pytest.mark.parametrize(
    "value",
    [
        b"some",
        "some",
        1,
        [1, 2, 3],
        {4, 5, 6}
    ],
)
def test_attempt_loading_image_from_string_when_parsing_should_fail(
    value: Any,
    image_as_numpy: np.ndarray,
) -> None:
    # when
    with pytest.raises(InputImageLoadError):
        _ = attempt_loading_image_from_string(value=value)
