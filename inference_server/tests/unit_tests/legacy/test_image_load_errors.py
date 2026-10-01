import base64
import json

import numpy as np
import pytest

from inference_server.errors import error_response
from inference_server.framework.input_parsers.url_fetch import fetch_images_from_urls
from inference_server.legacy.common import (
    ImagePayload,
    decode_inline_image,
    load_request_images,
)
from inference_server.legacy.errors import LegacyHTTPError
from inference_server.legacy.visualization import payload_to_rgb
from tests.unit_tests.legacy.conftest import FakeGateway

PREFIX = "Could not load input image. Cause: "
MALFORMED_BASE64 = "Malformed base64 input image."
EMPTY_PAYLOAD = "Empty image payload."
RAW_BYTES = (
    "Invalid base64 input: the image payload contains raw bytes instead of a "
    "base64-encoded string. Please base64-encode the image before sending."
)
NUMPY_UNSUPPORTED = (
    "NumPy image type is not supported in this configuration of `inference`."
)
UNKNOWN_TYPE = "Image declaration contains not recognised image type."
LOCAL_FILE_DISABLED = "Loading images from local filesystem is disabled."
NOT_NDARRAY = "Data provided as input could not be decoded into np.ndarray object."
NDARRAY_DIMENSIONS = "For image given as np.ndarray expected 2 or 3 dimensions."
URL_OFFLINE = "Cannot load an image from URL while OFFLINE_MODE is enabled."
URL_INPUT_DISABLED = (
    "Providing images via URL is not supported in this configuration of `inference`."
)
URL_INVALID = "Provided image URL is invalid"
URL_NON_HTTPS = (
    "Providing images via non https:// URL is not supported in this configuration "
    "of `inference`."
)
URL_WHITELIST = (
    "It is not allowed to reach image URL - prohibited by whitelisted destinations."
)
URL_BLACKLIST = (
    "It is not allowed to reach image URL - prohibited by blacklisted destinations."
)
URL_DESTINATION = "URL points to a network destination that is not allowed."
URL_FETCH = "Data pointed by URL could not be decoded into image."
URL_NOT_IMAGE = "Data is not image."

NOT_AN_IMAGE_B64 = base64.b64encode(b"hello").decode()


def _client(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = FakeGateway(model_info={"ds/1": {"class_names": ["cat"]}})

    return legacy_client(gateway)


def _post_image(client, image):
    response = client.post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "api_key": "k", "image": image},
    )

    return response


@pytest.mark.parametrize(
    "image,public_message",
    [
        ({"type": "base64", "value": "abc"}, MALFORMED_BASE64),
        ({"type": "base64", "value": NOT_AN_IMAGE_B64}, MALFORMED_BASE64),
        ({"type": "base64", "value": ""}, EMPTY_PAYLOAD),
        ({"type": "base64", "value": "!!!"}, EMPTY_PAYLOAD),
        ({"type": "numpy", "value": "gASVAAAAAAAAAAAu"}, NUMPY_UNSUPPORTED),
        ({"type": "bogus", "value": "x"}, UNKNOWN_TYPE),
        ({"type": "numpy_object", "value": [1, 2, 3]}, NOT_NDARRAY),
    ],
)
def test_inline_image_failure_answers_like_legacy(
    legacy_client, fake_stat, image, public_message
):
    response = _post_image(_client(legacy_client, fake_stat), image)

    assert response.status_code == 400
    assert response.json() == {"message": f"{PREFIX}{public_message}"}


def test_local_file_image_is_refused_like_legacy(legacy_client, fake_stat, monkeypatch):
    monkeypatch.setattr(
        "inference_server.legacy.common.ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM",
        False,
    )

    response = _post_image(
        _client(legacy_client, fake_stat), {"type": "file", "value": "/tmp/a.jpg"}
    )

    assert response.status_code == 400
    assert response.json() == {"message": f"{PREFIX}{LOCAL_FILE_DISABLED}"}


@pytest.mark.parametrize(
    "value,public_message",
    [
        (b"\xff\xd8\xff\xe0", RAW_BYTES),
        (bytearray(b"\xff\xd8\xff\xe0"), RAW_BYTES),
        (b"abc", MALFORMED_BASE64),
    ],
)
def test_base64_bytes_failure_answers_like_legacy(value, public_message):
    with pytest.raises(LegacyHTTPError) as exc:
        decode_inline_image({"type": "base64", "value": value}, ndarray_ok=False)

    assert exc.value.status_code == 400
    assert exc.value.message == f"{PREFIX}{public_message}"


def test_numpy_object_with_one_dimension_answers_like_legacy():
    with pytest.raises(LegacyHTTPError) as exc:
        decode_inline_image(
            {"type": "numpy_object", "value": np.zeros((4,), dtype=np.uint8)},
            ndarray_ok=True,
        )

    assert exc.value.status_code == 400
    assert exc.value.message == f"{PREFIX}{NDARRAY_DIMENSIONS}"


def test_undecodable_visualization_payload_answers_like_legacy():
    with pytest.raises(LegacyHTTPError) as exc:
        payload_to_rgb(ImagePayload(b"not an image", 1, 1))

    assert exc.value.status_code == 400
    assert exc.value.message == f"{PREFIX}{MALFORMED_BASE64}"


@pytest.mark.parametrize(
    "setting,public_message",
    [("OFFLINE_MODE", URL_OFFLINE), ("ALLOW_URL_INPUT", URL_INPUT_DISABLED)],
)
def test_refused_url_input_answers_like_legacy(
    legacy_client, fake_stat, monkeypatch, setting, public_message
):
    monkeypatch.setattr(
        f"inference_server.legacy.common.{setting}", setting == "OFFLINE_MODE"
    )

    response = _post_image(
        _client(legacy_client, fake_stat),
        {"type": "url", "value": "https://example.com/a.jpg"},
    )

    assert response.status_code == 400
    assert response.json() == {"message": f"{PREFIX}{public_message}"}


def _configure_url_fetch(monkeypatch, *, blocked=None, allowed=None, addresses=()):
    async def _resolve(host):
        return list(addresses)

    monkeypatch.setattr(
        "inference_server.configuration.BLACKLISTED_DESTINATIONS_FOR_URL_INPUT",
        blocked,
    )
    monkeypatch.setattr(
        "inference_server.configuration.WHITELISTED_DESTINATIONS_FOR_URL_INPUT",
        allowed,
    )
    monkeypatch.setattr(
        "inference_server.configuration.ALLOW_URL_TO_NON_GLOBAL_ADDRESSES", False
    )
    monkeypatch.setattr("inference_server.configuration.OFFLINE_MODE", False)
    monkeypatch.setattr("inference_server.configuration.ALLOW_URL_INPUT", True)
    monkeypatch.setattr(
        "inference_server.framework.input_parsers.url_fetch.resolve_host", _resolve
    )


URL_FAILURES = [
    pytest.param(
        "https://example.com/a.jpg",
        {"blocked": frozenset({"example.com"})},
        URL_BLACKLIST,
        403,
        {
            "error_code": "URL_DESTINATION_FORBIDDEN",
            "description": "image URL destination",
        },
        id="blacklisted",
    ),
    pytest.param(
        "https://example.com/a.jpg",
        {"allowed": frozenset({"images.internal"})},
        URL_WHITELIST,
        403,
        {
            "error_code": "URL_DESTINATION_FORBIDDEN",
            "description": "image URL destination",
        },
        id="not-whitelisted",
    ),
    pytest.param(
        "https://internal.example/a.jpg",
        {"addresses": ("127.0.0.1",)},
        URL_DESTINATION,
        403,
        {
            "error_code": "URL_DESTINATION_FORBIDDEN",
            "description": "image URL destination",
        },
        id="non-global-address",
    ),
    pytest.param(
        "https://missing.example/a.jpg",
        {"addresses": ()},
        URL_FETCH,
        502,
        {"error_code": "URL_FETCH_FAILED", "description": "fetching image URL failed"},
        id="fetch-failed",
    ),
    pytest.param(
        "ftp://example.com/a.jpg",
        {},
        URL_NON_HTTPS,
        400,
        {
            "error_code": "INVALID_URL",
            "description": "image URL must start with http:// or https://",
        },
        id="other-scheme",
    ),
    pytest.param(
        "example.com/a.jpg",
        {},
        URL_INVALID,
        400,
        {
            "error_code": "INVALID_URL",
            "description": "image URL must start with http:// or https://",
        },
        id="no-scheme",
    ),
    pytest.param(
        "https:///a.jpg",
        {},
        URL_INVALID,
        400,
        {"error_code": "INVALID_URL", "description": "image URL has no host"},
        id="no-host",
    ),
]


@pytest.mark.parametrize("url,settings,public_message,v2_status,v2_body", URL_FAILURES)
def test_url_image_failure_answers_like_legacy(
    legacy_client,
    fake_stat,
    monkeypatch,
    url,
    settings,
    public_message,
    v2_status,
    v2_body,
):
    client = _client(legacy_client, fake_stat)
    _configure_url_fetch(monkeypatch, **settings)

    response = _post_image(client, {"type": "url", "value": url})

    assert response.status_code == 400
    assert response.json() == {"message": f"{PREFIX}{public_message}"}


@pytest.mark.parametrize("url,settings,public_message,v2_status,v2_body", URL_FAILURES)
@pytest.mark.asyncio
async def test_url_image_failure_keeps_the_v2_answer(
    monkeypatch, url, settings, public_message, v2_status, v2_body
):
    _configure_url_fetch(monkeypatch, **settings)

    images, error = await fetch_images_from_urls([url])

    assert images is None
    assert error.status_code == v2_status
    assert json.loads(error.body) == v2_body


@pytest.mark.parametrize(
    "fetch_error,status,message",
    [
        (
            error_response(
                504, "URL_FETCH_TIMEOUT", "fetching image URL timed out after 10s"
            ),
            400,
            f"{PREFIX}{URL_FETCH}",
        ),
        (
            error_response(
                502, "URL_FETCH_FAILED", "fetching image URL returned status 404"
            ),
            400,
            f"{PREFIX}{URL_FETCH}",
        ),
        (
            error_response(
                502, "URL_FETCH_FAILED", "too many redirects fetching image URL"
            ),
            400,
            f"{PREFIX}{URL_FETCH}",
        ),
        (
            error_response(
                413, "URL_IMAGE_TOO_LARGE", "image at URL exceeds 50MB limit"
            ),
            413,
            "image at URL exceeds 50MB limit",
        ),
        (
            error_response(
                413,
                "PAYLOAD_TOO_LARGE",
                "combined size of images at URLs exceeds 104857600 byte limit",
            ),
            413,
            "combined size of images at URLs exceeds 104857600 byte limit",
        ),
    ],
)
@pytest.mark.asyncio
async def test_url_fetch_response_is_translated_for_legacy_routes(
    monkeypatch, fetch_error, status, message
):
    async def _fetch(urls):
        return None, fetch_error

    monkeypatch.setattr("inference_server.legacy.common.fetch_images_from_urls", _fetch)

    with pytest.raises(LegacyHTTPError) as exc:
        await load_request_images(
            [{"type": "url", "value": "https://example.com/a.jpg"}], ndarray_ok=False
        )

    assert exc.value.status_code == status
    assert exc.value.message == message


@pytest.mark.asyncio
async def test_url_content_that_is_not_an_image_answers_like_legacy(monkeypatch):
    async def _fetch(urls):
        return [b"hello"], None

    monkeypatch.setattr("inference_server.legacy.common.fetch_images_from_urls", _fetch)

    with pytest.raises(LegacyHTTPError) as exc:
        await load_request_images(
            [{"type": "url", "value": "https://example.com/a.jpg"}], ndarray_ok=False
        )

    assert exc.value.status_code == 400
    assert exc.value.message == f"{PREFIX}{URL_NOT_IMAGE}"
