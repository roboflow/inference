import io
import pickle
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest
from PIL import Image
from requests_mock import Mocker

from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.exceptions import (
    InputFormatInferenceFailed,
    InputImageLoadError,
    InvalidImageTypeDeclared,
    InvalidNumpyInput,
)
from inference.core.utils import image_utils
from inference.core.utils.image_utils import (
    ImageType,
    attempt_loading_image_from_string,
    choose_image_decoding_flags,
    convert_gray_image_to_bgr,
    extract_image_payload_and_type,
    load_image,
    load_image_base64,
    load_image_from_buffer,
    load_image_from_encoded_bytes,
    load_image_from_numpy_str,
    load_image_from_url,
    load_image_rgb,
    load_image_with_inferred_type,
    load_image_with_known_type,
)


@pytest.mark.parametrize(
    "response_status_code", [400, 401, 403, 404, 500, 501, 502, 503, 504]
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


@mock.patch.object(image_utils, "ALLOW_URL_INPUT", False)
def test_load_image_from_url_when_url_loading_not_allowed() -> None:
    with pytest.raises(InvalidImageTypeDeclared):
        _ = load_image_from_url(value="https://google.com/image.jpg")


@mock.patch.object(image_utils, "ALLOW_URL_INPUT", True)
@mock.patch.object(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", False)
@pytest.mark.parametrize(
    "url",
    [
        "http://google.com/image.jpg",
        "http://127.0.0.1:90/image.jpg",
        "http://[fe80::1ff:fe23:4567:890a%25eth0]:90/image.jpg",
        "http://[2001:0db8:85a3:0000:0000:8a2e:0370:7334]/image.jpg",
    ],
)
def test_load_image_from_url_when_https_is_enforced_and_provided_urls_with_http_schema(
    url: str,
) -> None:
    with pytest.raises(InputImageLoadError):
        _ = load_image_from_url(value=url)


@mock.patch.object(image_utils, "ALLOW_URL_INPUT", True)
@mock.patch.object(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT_WITHOUT_FQDN", False)
@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/image.jpg",
        "https://127.0.0.1:90/image.jpg",
        "https://[fe80::1ff:fe23:4567:890a%25eth0]/image.jpg",
        "https://[fe80::1ff:fe23:4567:890a%25eth0]:90/image.jpg",
        "https://[2001:0db8:85a3:0000:0000:8a2e:0370:7334]/image.jpg",
    ],
)
def test_load_image_from_url_when_fqdns_are_enforced_and_urls_based_on_ips_provided(
    url: str,
) -> None:
    with pytest.raises(InputImageLoadError):
        _ = load_image_from_url(value=url)


@mock.patch.object(image_utils, "ALLOW_URL_INPUT", True)
@mock.patch.object(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT_WITHOUT_FQDN", True)
@mock.patch.object(
    image_utils, "WHITELISTED_DESTINATIONS_FOR_URL_INPUT", {"not_existing"}
)
@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/image.jpg",
        "https://127.0.0.1:90/image.jpg",
        "https://[fe80::1ff:fe23:4567:890a%25eth0]/image.jpg",
        "https://[fe80::1ff:fe23:4567:890a%25eth0]:90/image.jpg",
        "https://[2001:0db8:85a3:0000:0000:8a2e:0370:7334]/image.jpg",
    ],
)
def test_load_image_from_url_when_locations_not_whitelisted(url: str) -> None:
    # when
    with pytest.raises(InputImageLoadError) as error:
        _ = load_image_from_url(value=url)

    # then
    assert "whitelisted" in str(error.value)


@mock.patch.object(image_utils, "ALLOW_URL_INPUT", True)
@mock.patch.object(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT_WITHOUT_FQDN", True)
@mock.patch.object(
    image_utils,
    "WHITELISTED_DESTINATIONS_FOR_URL_INPUT",
    {
        "127.0.0.1",
        "fe80::1ff:fe23:4567:890a%25eth0",
        "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
        "google.com",
        "subdomain.google.com",
    },
)
@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/image.jpg",
        "https://127.0.0.1:90/image.jpg",
        "https://[fe80::1ff:fe23:4567:890a%25eth0]/image.jpg",
        "https://[fe80::1ff:fe23:4567:890a%25eth0]:90/image.jpg",
        "https://[2001:0db8:85a3:0000:0000:8a2e:0370:7334]/image.jpg",
        "https://subdomain.google.com/image.jpg?param=some",
        "https://google.com/image.jpg?param=some",
    ],
)
def test_load_image_from_url_when_locations_whitelisted(
    url: str,
    requests_mock: Mocker,
    image_as_numpy: np.ndarray,
    image_as_png_bytes: bytes,
) -> None:
    requests_mock.get(
        url,
        content=image_as_png_bytes,
    )

    # when
    result = load_image_from_url(value=url)

    # then
    assert image_as_numpy.shape == result.shape
    assert np.allclose(image_as_numpy, result)


@mock.patch.object(image_utils, "ALLOW_URL_INPUT", True)
@mock.patch.object(image_utils, "ALLOW_NON_HTTPS_URL_INPUT", False)
@mock.patch.object(image_utils, "ALLOW_URL_INPUT_WITHOUT_FQDN", True)
@mock.patch.object(
    image_utils,
    "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT",
    {
        "127.0.0.1",
        "fe80::1ff:fe23:4567:890a%25eth0",
        "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
        "google.com",
        "subdomain.google.com",
    },
)
@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/image.jpg",
        "https://127.0.0.1:90/image.jpg",
        "https://[fe80::1ff:fe23:4567:890a%25eth0]/image.jpg",
        "https://[fe80::1ff:fe23:4567:890a%25eth0]:90/image.jpg",
        "https://[2001:0db8:85a3:0000:0000:8a2e:0370:7334]/image.jpg",
        "https://subdomain.google.com/image.jpg?param=some",
        "https://google.com/image.jpg?param=some",
    ],
)
def test_load_image_from_url_when_locations_blacklisted(
    url: str,
) -> None:
    # when
    with pytest.raises(InputImageLoadError) as error:
        _ = load_image_from_url(value=url)

    # then
    assert "blacklisted" in str(error.value)


@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", True)
def test_load_image_from_numpy_str_when_empty_bytes_given() -> None:
    # when
    with pytest.raises(InvalidNumpyInput):
        _ = load_image_from_numpy_str(value=b"")


@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", True)
def test_load_image_from_numpy_str_when_arbitrary_object_given() -> None:
    # when
    with pytest.raises(InvalidNumpyInput):
        _ = load_image_from_numpy_str(value=[1, 2, 3])


@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", True)
def test_load_image_from_numpy_str_when_string_given() -> None:
    # when
    with pytest.raises(InvalidNumpyInput):
        _ = load_image_from_numpy_str(value="some")


@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", True)
def test_load_image_from_numpy_str_when_non_object_bytes_given() -> None:
    # when
    with pytest.raises(InvalidNumpyInput):
        _ = load_image_from_numpy_str(value=b"some")


@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", True)
def test_load_image_from_numpy_str_when_non_array_bytes_given() -> None:
    # given
    payload = pickle.dumps([1, 2, 3])

    # when
    with pytest.raises(InvalidNumpyInput):
        _ = load_image_from_numpy_str(value=payload)


@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", True)
def test_load_image_from_numpy_str_when_invalid_shape_array_bytes_given() -> None:
    # given
    payload = pickle.dumps(np.array([1, 2, 3]))

    # when
    with pytest.raises(InvalidNumpyInput):
        _ = load_image_from_numpy_str(value=payload)


@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", True)
def test_load_image_from_numpy_str_when_array_with_non_standard_channels_given() -> (
    None
):
    # given
    payload = pickle.dumps(np.ones((128, 128, 4), dtype=np.uint8))

    # when
    with pytest.raises(InvalidNumpyInput):
        _ = load_image_from_numpy_str(value=payload)


@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", True)
def test_load_image_from_numpy_str_when_valid_image_given(
    image_as_pickled_bytes: bytes,
    image_as_numpy: np.ndarray,
) -> None:
    # when
    result = load_image_from_numpy_str(value=image_as_pickled_bytes)

    # then
    assert image_as_numpy.shape == result.shape
    assert np.allclose(image_as_numpy, result)


@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", False)
def test_load_image_from_numpy_str_when_valid_image_given_but_not_allowed_to_unpickle(
    image_as_pickled_bytes: bytes,
) -> None:
    # when
    with pytest.raises(InvalidImageTypeDeclared):
        _ = load_image_from_numpy_str(value=image_as_pickled_bytes)


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


def test_load_image_with_inferred_type_when_value_is_np_array(
    image_as_numpy: np.ndarray,
) -> None:
    # when
    result = load_image_with_inferred_type(value=image_as_numpy)

    # then
    assert result == (image_as_numpy, True)


def test_load_image_with_inferred_type_when_value_is_pillow_image(
    image_as_pillow: Image.Image,
    image_as_numpy: np.ndarray,
) -> None:
    # when
    result = load_image_with_inferred_type(value=image_as_pillow)

    # then
    assert result[1] is False
    assert image_as_numpy.shape == result[0].shape
    assert np.allclose(image_as_numpy, result[0])


@mock.patch.object(image_utils, "load_image_from_url")
@pytest.mark.parametrize("url", ["http://some/image.jpg", "https://some/image.jpg"])
def test_load_image_with_inferred_type_when_value_is_url(
    load_image_from_url_mock: MagicMock,
    url: str,
) -> None:
    # when
    result = load_image_with_inferred_type(value=url, cv_imread_flags=cv2.IMREAD_COLOR)

    # then
    assert result[0] is load_image_from_url_mock.return_value
    assert result[1] is True
    load_image_from_url_mock.assert_called_once_with(
        value=url, cv_imread_flags=cv2.IMREAD_COLOR
    )


def test_load_image_with_inferred_type_when_value_is_local_image_path(
    image_as_local_path: str,
    image_as_numpy: np.ndarray,
) -> None:
    # when
    result = load_image_with_inferred_type(value=image_as_local_path)

    # then
    assert result[1] is True
    assert image_as_numpy.shape == result[0].shape
    assert np.allclose(image_as_numpy, result[0])


@mock.patch.object(image_utils, "attempt_loading_image_from_string")
@pytest.mark.parametrize("value", ["aaa", b"some", io.BytesIO(), [1, 2, 3]])
def test_load_image_with_inferred_type_when_value_is_unknown_and_should_be_tried_against_set_of_methods(
    attempt_loading_image_from_string_mock: MagicMock,
    value: Any,
) -> None:
    # when
    result = load_image_with_inferred_type(
        value=value, cv_imread_flags=cv2.IMREAD_COLOR
    )

    # then
    assert result is attempt_loading_image_from_string_mock.return_value
    attempt_loading_image_from_string_mock.assert_called_once_with(
        value=value,
        cv_imread_flags=cv2.IMREAD_COLOR,
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


@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", True)
@pytest.mark.parametrize(
    "fixture_name",
    [
        "image_as_jpeg_base64_bytes",
        "image_as_jpeg_base64_string",
        "image_as_png_bytes",
        "image_as_jpeg_bytes",
        "image_as_pickled_bytes",
        "image_as_base64_encoded_pickled_bytes",
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


@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", False)
@pytest.mark.parametrize(
    "fixture_name",
    [
        "image_as_pickled_bytes",
        "image_as_base64_encoded_pickled_bytes",
    ],
)
def test_attempt_loading_image_from_string_when_parsing_should_be_fail_due_to_unpickling_being_prohibited(
    fixture_name: str,
    request: FixtureRequest,
) -> None:
    # given
    value = request.getfixturevalue(fixture_name)

    # when
    with pytest.raises(InvalidImageTypeDeclared):
        _ = attempt_loading_image_from_string(value=value)


@pytest.mark.parametrize(
    "value",
    [b"some", "some", 1, [1, 2, 3], {4, 5, 6}],
)
def test_attempt_loading_image_from_string_when_parsing_should_fail(
    value: Any,
    image_as_numpy: np.ndarray,
) -> None:
    # when
    with pytest.raises(InputImageLoadError):
        _ = attempt_loading_image_from_string(value=value)


def test_choose_image_decoding_flags_when_disabled_auto_orient() -> None:
    # when
    result = choose_image_decoding_flags(disable_preproc_auto_orient=True)

    # then
    assert result == cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION


def test_choose_image_decoding_flags_when_enabled_auto_orient() -> None:
    # when
    result = choose_image_decoding_flags(disable_preproc_auto_orient=False)

    # then
    assert result == cv2.IMREAD_COLOR


def test_extract_image_payload_and_type_when_type_cannot_be_inferred(
    image_as_jpeg_base64_bytes: bytes,
) -> None:
    # when
    result = extract_image_payload_and_type(value=image_as_jpeg_base64_bytes)

    # then
    assert result == (image_as_jpeg_base64_bytes, None)


@pytest.mark.parametrize(
    "type_name, expected_type_enum",
    [
        ("base64", ImageType.BASE64),
        ("file", ImageType.FILE),
        ("multipart", ImageType.MULTIPART),
        ("numpy", ImageType.NUMPY),
        ("pil", ImageType.PILLOW),
        ("url", ImageType.URL),
        ("BASE64", ImageType.BASE64),
        ("FILE", ImageType.FILE),
        ("MULTIPART", ImageType.MULTIPART),
        ("NUMPY", ImageType.NUMPY),
        ("PIL", ImageType.PILLOW),
        ("URL", ImageType.URL),
    ],
)
def test_extract_image_payload_and_type_when_value_is_dict_and_type_is_recognised(
    type_name: str,
    expected_type_enum: ImageType,
) -> None:
    # given
    value = {"value": "some", "type": type_name}

    # when
    result = extract_image_payload_and_type(value=value)

    # then
    assert result == ("some", expected_type_enum)


@pytest.mark.parametrize(
    "type_name, expected_type_enum",
    [
        ("base64", ImageType.BASE64),
        ("file", ImageType.FILE),
        ("multipart", ImageType.MULTIPART),
        ("numpy", ImageType.NUMPY),
        ("pil", ImageType.PILLOW),
        ("url", ImageType.URL),
        ("BASE64", ImageType.BASE64),
        ("FILE", ImageType.FILE),
        ("MULTIPART", ImageType.MULTIPART),
        ("NUMPY", ImageType.NUMPY),
        ("PIL", ImageType.PILLOW),
        ("URL", ImageType.URL),
    ],
)
def test_extract_image_payload_and_type_when_value_is_request_and_type_is_recognised(
    type_name: str,
    expected_type_enum: ImageType,
) -> None:
    # given
    value = InferenceRequestImage(value="some", type=type_name)

    # when
    result = extract_image_payload_and_type(value=value)

    # then
    assert result == ("some", expected_type_enum)


def test_extract_image_payload_and_type_when_value_is_dict_and_type_is_not_recognised() -> (
    None
):
    # when
    with pytest.raises(InvalidImageTypeDeclared):
        _ = extract_image_payload_and_type(value={"value": "some", "type": "unknown"})


def test_extract_image_payload_and_type_when_value_is_request_and_type_is_not_recognised() -> (
    None
):
    # given
    value = InferenceRequestImage(value="some", type="unknown")

    # when
    with pytest.raises(InvalidImageTypeDeclared):
        _ = extract_image_payload_and_type(value=value)


@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", True)
@pytest.mark.parametrize(
    "fixture_name, image_type, is_bgr",
    [
        ("image_as_jpeg_base64_bytes", ImageType.BASE64, True),
        ("image_as_jpeg_base64_string", ImageType.BASE64, True),
        ("image_as_local_path", ImageType.FILE, True),
        ("image_as_buffer", ImageType.MULTIPART, True),
        ("image_as_pickled_bytes", ImageType.NUMPY, True),
        ("image_as_base64_encoded_pickled_bytes", ImageType.NUMPY, True),
        ("image_as_pillow", ImageType.PILLOW, False),
    ],
)
def test_load_image_with_known_type_when_load_should_succeed(
    fixture_name: str,
    image_type: ImageType,
    is_bgr: bool,
    image_as_numpy: np.ndarray,
    request: FixtureRequest,
) -> None:
    # given
    value = request.getfixturevalue(fixture_name)

    # when
    result = load_image_with_known_type(value=value, image_type=image_type)

    # then
    assert result[1] is is_bgr
    assert image_as_numpy.shape == result[0].shape
    assert np.allclose(image_as_numpy, result[0])


@mock.patch.object(image_utils, "IMAGE_LOADERS")
def test_load_image_with_known_type_when_load_from_url_succeeds(
    image_loaders_mock: MagicMock,
) -> None:
    # given
    url_loader_mock = MagicMock()
    image_loaders_mock.__getitem__.return_value = url_loader_mock

    # when
    result = load_image_with_known_type(
        value="http://some/image.jpg",
        image_type=ImageType.URL,
        cv_imread_flags=cv2.IMREAD_COLOR,
    )

    # then
    assert result[1] is True
    assert result[0] == url_loader_mock.return_value
    url_loader_mock.assert_called_once_with("http://some/image.jpg", cv2.IMREAD_COLOR)


@mock.patch.object(image_utils, "IMAGE_LOADERS")
def test_load_image_with_known_type_when_load_from_url_fails(
    image_loaders_mock: MagicMock,
) -> None:
    # given
    url_loader_mock = MagicMock()
    url_loader_mock.side_effect = InputImageLoadError(message="", public_message="")
    image_loaders_mock.__getitem__.return_value = url_loader_mock

    # when
    with pytest.raises(InputImageLoadError):
        _ = load_image_with_known_type(
            value="http://some/image.jpg",
            image_type=ImageType.URL,
            cv_imread_flags=cv2.IMREAD_COLOR,
        )


@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", False)
def test_load_image_with_known_type_when_numpy_load_disabled_and_numpy_value_given() -> (
    None
):
    # when
    with pytest.raises(InvalidImageTypeDeclared):
        _ = load_image_with_known_type(
            value=np.array([1, 2, 3]),
            image_type=ImageType.NUMPY,
        )


def test_convert_gray_image_to_bgr_when_three_chanel_input_submitted(
    image_as_numpy: np.ndarray,
) -> None:
    # when
    result = convert_gray_image_to_bgr(image=image_as_numpy)

    # then
    assert result is image_as_numpy


def test_convert_gray_image_to_bgr_when_single_chanel_input_submitted(
    image_as_numpy: np.ndarray,
) -> None:
    # given
    image = np.zeros((128, 128, 1), dtype=np.uint8)

    # when
    result = convert_gray_image_to_bgr(image=image)

    # then
    assert image_as_numpy.shape == result.shape
    assert np.allclose(image_as_numpy, result)


def test_convert_gray_image_to_bgr_when_2d_input_submitted(
    image_as_numpy: np.ndarray,
) -> None:
    # given
    image = np.zeros((128, 128), dtype=np.uint8)

    # when
    result = convert_gray_image_to_bgr(image=image)

    # then
    assert image_as_numpy.shape == result.shape
    assert np.allclose(image_as_numpy, result)


@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", True)
@pytest.mark.parametrize(
    "fixture_name, is_bgr",
    [
        ("image_as_jpeg_base64_bytes", True),
        ("image_as_jpeg_base64_string", True),
        ("image_as_local_path", True),
        ("image_as_buffer", True),
        ("image_as_rgba_buffer", True),  # works due to cv load flags
        ("image_as_gray_buffer", True),
        ("image_as_pickled_bytes", True),
        ("image_as_base64_encoded_pickled_bytes", True),
        ("image_as_pickled_bytes_gray", True),
        ("image_as_pillow", False),
    ],
)
def test_load_image_when_load_should_succeed_from_inferred_type(
    fixture_name: str,
    is_bgr: bool,
    image_as_numpy: np.ndarray,
    request: FixtureRequest,
) -> None:
    # given
    value = request.getfixturevalue(fixture_name)

    # when
    result = load_image(value=value)

    # then
    assert result[1] is is_bgr
    assert image_as_numpy.shape == result[0].shape
    assert np.allclose(image_as_numpy, result[0])


@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", True)
def test_load_image_when_load_should_fail_on_rgba_numpy_input(
    image_as_pickled_bytes_rgba: bytes,
) -> None:
    # when
    with pytest.raises(InputFormatInferenceFailed):
        _ = load_image(value=image_as_pickled_bytes_rgba)


@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", False)
def test_load_image_when_load_should_fail_on_not_allowed_input(
    image_as_pickled_bytes_rgba: bytes,
) -> None:
    # when
    with pytest.raises(InvalidImageTypeDeclared):
        _ = load_image(value=image_as_pickled_bytes_rgba)


@pytest.mark.parametrize(
    "value", ["", b"", "NOT AN IMAGE", [1, 2, 3], np.zeros((2, 3, 4))]
)
def test_load_image_when_load_should_fail_on_invalid_input(value: Any) -> None:
    # when
    with pytest.raises(InputImageLoadError):
        _ = load_image(value=value)


@pytest.mark.parametrize(
    "fixture_name, image_type, is_bgr",
    [
        ("image_as_jpeg_base64_bytes", ImageType.BASE64, True),
        ("image_as_jpeg_base64_string", ImageType.BASE64, True),
        ("image_as_local_path", ImageType.FILE, True),
        ("image_as_buffer", ImageType.MULTIPART, True),
        (
            "image_as_rgba_buffer",
            ImageType.MULTIPART,
            True,
        ),  # works due to cv load flags
        ("image_as_gray_buffer", ImageType.MULTIPART, True),
        ("image_as_pickled_bytes", ImageType.NUMPY, True),
        ("image_as_pickled_bytes_gray", ImageType.NUMPY, True),
        ("image_as_pillow", ImageType.PILLOW, False),
    ],
)
@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", True)
def test_load_image_when_load_should_succeed_from_known_type(
    fixture_name: str,
    image_type: ImageType,
    is_bgr: bool,
    image_as_numpy: np.ndarray,
    request: FixtureRequest,
) -> None:
    # given
    value = request.getfixturevalue(fixture_name)
    request = InferenceRequestImage(value=value, type=image_type.value)

    # when
    result = load_image(value=request)

    # then
    assert result[1] is is_bgr
    assert image_as_numpy.shape == result[0].shape
    assert np.allclose(image_as_numpy, result[0])


@mock.patch.object(image_utils, "ALLOW_NUMPY_INPUT", False)
def test_load_image_when_load_should_fail_from_known_type_due_to_numpy_unpickling_forbidden(
    image_as_pickled_bytes: bytes,
) -> None:
    # given
    request = InferenceRequestImage(value=image_as_pickled_bytes, type=ImageType.NUMPY)

    # when
    with pytest.raises(InvalidImageTypeDeclared):
        _ = load_image(value=request)


@mock.patch.object(image_utils, "load_image")
def test_load_image_rgb_on_bgr_image(load_image_mock: MagicMock) -> None:
    # given
    image = np.ones((128, 128, 3), dtype=np.uint8)
    image[:, :, -1] = 255
    load_image_mock.return_value = (image, True)

    # when
    result = load_image_rgb(value="some")

    # then
    load_image_mock.assert_called_once_with(
        value="some", disable_preproc_auto_orient=False
    )
    assert result.shape == (128, 128, 3)
    assert np.all(result[:, :, -1] == 1)
    assert np.all(result[:, :, 0] == 255)


@mock.patch.object(image_utils, "load_image")
def test_load_image_rgb_on_rgb_image(load_image_mock: MagicMock) -> None:
    # given
    image = np.ones((128, 128, 3), dtype=np.uint8)
    image[:, :, -1] = 255
    load_image_mock.return_value = (image, False)

    # when
    result = load_image_rgb(value="some")

    # then
    load_image_mock.assert_called_once_with(
        value="some", disable_preproc_auto_orient=False
    )
    assert result.shape == (128, 128, 3)
    assert np.all(result[:, :, 0] == 1)
    assert np.all(result[:, :, -1] == 255)
