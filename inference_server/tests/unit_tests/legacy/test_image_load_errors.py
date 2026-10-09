import base64
import io
import json
import os
import pickle
import subprocess
import sys
import threading
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageFile

from inference_server.errors import error_response
from inference_server.framework.input_parsers.json_base64 import extract_json_base64
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
NDARRAY_CHANNELS = "For image given as np.ndarray expected 1 or 3 channels."
LOCAL_FILE_NOT_LOADED = "Could not load image from the local file."
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


def _png(width=7, height=5) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="PNG")

    return buffer.getvalue()


def _detection_gateway():
    detections = SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 5]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    gateway = FakeGateway(
        predictions={("ds/1", "infer"): detections},
        model_info={"ds/1": {"class_names": ["cat"], "actions": {"infer": {}}}},
    )

    return gateway


def _inferred_images(gateway):
    return [call[4] for call in gateway.calls if call[0] == "infer"]


def _fetch_png_from_urls(monkeypatch):
    seen = []

    async def _fetch(urls, destination_policy=None):
        seen.append(urls)
        return [_png() for _ in urls], None

    monkeypatch.setattr("inference_server.legacy.common.fetch_images_from_urls", _fetch)

    return seen


def _set_local_file_loading(monkeypatch, allowed):
    monkeypatch.setattr(
        "inference_server.legacy.common.ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM",
        allowed,
    )


def _clip_gateway():
    gateway = FakeGateway(
        predictions={
            ("clip/ViT-B-16", "embed_images"): np.array([[1.0, 0.0]]),
            ("clip/ViT-B-16", "embed_text"): lambda image, params: np.array(
                [[1.0, 0.0]] * len(params["texts"])
            ),
        },
        model_info={
            "clip/ViT-B-16": {
                "actions": {"embed_text": {}, "embed_images": {}, "compare": {}}
            }
        },
    )

    return gateway


def _compare_image_subject(client, subject):
    response = client.post(
        "/clip/compare",
        json={
            "subject": subject,
            "subject_type": "image",
            "prompt": ["a cat"],
            "prompt_type": "text",
            "api_key": "k",
        },
    )

    return response


@pytest.mark.parametrize("declared_type", ["BASE64", "Base64"])
def test_base64_type_is_accepted_in_any_letter_case(
    legacy_client, fake_stat, declared_type
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _detection_gateway()

    response = _post_image(
        legacy_client(gateway),
        {"type": declared_type, "value": base64.b64encode(_png()).decode()},
    )

    assert response.status_code == 200, response.text
    assert response.json()["image"] == {"width": 7, "height": 5}
    assert _inferred_images(gateway) == [_png()]


@pytest.mark.parametrize("declared_type", ["Url", "URL"])
def test_url_type_is_accepted_in_any_letter_case(
    legacy_client, fake_stat, monkeypatch, declared_type
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _detection_gateway()
    client = legacy_client(gateway)
    seen = _fetch_png_from_urls(monkeypatch)

    response = _post_image(
        client, {"type": declared_type, "value": "https://example.com/a.png"}
    )

    assert response.status_code == 200, response.text
    assert seen == [["https://example.com/a.png"]]
    assert _inferred_images(gateway) == [_png()]


def test_app_loads_local_file_image_when_the_setting_is_left_unset(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png())
    code = (
        "import sys; import inference_server.app; "
        "from inference_server.legacy import common; "
        "payload = common.decode_inline_image("
        "{'type': 'file', 'value': sys.argv[1]}, ndarray_ok=False); "
        "print(common.ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM, "
        "payload.width, payload.height)"
    )
    env = {
        name: value
        for name, value in os.environ.items()
        if name != "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM"
    }

    result = subprocess.run(
        [sys.executable, "-c", code, str(image_path)],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "True 7 5"


@pytest.mark.parametrize("declared_type", ["file", "FILE"])
def test_local_file_image_is_loaded_when_allowed(
    legacy_client, fake_stat, tmp_path, monkeypatch, declared_type
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _detection_gateway()
    client = legacy_client(gateway)
    _set_local_file_loading(monkeypatch, True)
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png())

    response = _post_image(client, {"type": declared_type, "value": str(image_path)})

    assert response.status_code == 200, response.text
    assert response.json()["image"] == {"width": 7, "height": 5}
    assert _inferred_images(gateway) == [_png()]


def test_local_file_image_is_read_through_a_symlink_and_a_relative_path(
    legacy_client, fake_stat, tmp_path, monkeypatch
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _detection_gateway()
    client = legacy_client(gateway)
    _set_local_file_loading(monkeypatch, True)
    (tmp_path / "image.png").write_bytes(_png())
    (tmp_path / "link.png").symlink_to(tmp_path / "image.png")
    monkeypatch.chdir(tmp_path)

    response = _post_image(client, {"type": "file", "value": "./link.png"})

    assert response.status_code == 200, response.text
    assert _inferred_images(gateway) == [_png()]


@pytest.mark.parametrize("target", ["missing.png", "notes.txt", "."])
def test_unreadable_local_file_is_refused_with_one_answer(
    legacy_client, fake_stat, tmp_path, monkeypatch, target
):
    client = _client(legacy_client, fake_stat)
    _set_local_file_loading(monkeypatch, True)
    (tmp_path / "notes.txt").write_text("hello")

    response = _post_image(client, {"type": "file", "value": str(tmp_path / target)})

    assert response.status_code == 400
    assert response.json() == {"message": f"{PREFIX}{LOCAL_FILE_NOT_LOADED}"}


@pytest.mark.parametrize("value", [None, 3, ["a.png"], {"path": "a.png"}])
def test_local_file_value_that_is_not_a_path_is_refused_with_the_same_answer(
    legacy_client, fake_stat, monkeypatch, value
):
    client = _client(legacy_client, fake_stat)
    _set_local_file_loading(monkeypatch, True)

    response = _post_image(client, {"type": "file", "value": value})

    assert response.status_code == 400
    assert response.json() == {"message": f"{PREFIX}{LOCAL_FILE_NOT_LOADED}"}


def test_existing_local_file_is_refused_when_loading_is_disabled(
    legacy_client, fake_stat, tmp_path, monkeypatch
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _detection_gateway()
    client = legacy_client(gateway)
    _set_local_file_loading(monkeypatch, False)
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png())

    response = _post_image(client, {"type": "file", "value": str(image_path)})

    assert response.status_code == 400
    assert response.json() == {"message": f"{PREFIX}{LOCAL_FILE_DISABLED}"}
    assert _inferred_images(gateway) == []


@pytest.mark.parametrize("allowed", [True, False])
@pytest.mark.parametrize("declared_type", ["numpy", "NUMPY"])
def test_pickled_numpy_image_is_refused_in_every_configuration(
    legacy_client, fake_stat, monkeypatch, allowed, declared_type
):
    client = _client(legacy_client, fake_stat)
    _set_local_file_loading(monkeypatch, allowed)
    pickled = base64.b64encode(pickle.dumps(np.zeros((4, 6, 3), np.uint8))).decode()

    response = _post_image(client, {"type": declared_type, "value": pickled})

    assert response.status_code == 400
    assert response.json() == {"message": f"{PREFIX}{NUMPY_UNSUPPORTED}"}


def test_image_without_declared_type_given_as_url_is_fetched(
    legacy_client, fake_stat, monkeypatch
):
    gateway = _clip_gateway()
    client = legacy_client(gateway)
    seen = _fetch_png_from_urls(monkeypatch)

    response = _compare_image_subject(client, "https://example.com/a.png")

    assert response.status_code == 200, response.text
    assert seen == [["https://example.com/a.png"]]
    assert _inferred_images(gateway)[0] == _png()


def test_image_without_declared_type_given_as_local_path_is_loaded(
    legacy_client, fake_stat, tmp_path, monkeypatch
):
    gateway = _clip_gateway()
    client = legacy_client(gateway)
    _set_local_file_loading(monkeypatch, True)
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png())

    response = _compare_image_subject(client, str(image_path))

    assert response.status_code == 200, response.text
    assert _inferred_images(gateway)[0] == _png()


def test_image_without_declared_type_given_as_base64_is_decoded(
    legacy_client, fake_stat
):
    gateway = _clip_gateway()

    response = _compare_image_subject(
        legacy_client(gateway), base64.b64encode(_png()).decode()
    )

    assert response.status_code == 200, response.text
    assert _inferred_images(gateway)[0] == _png()


def test_image_without_declared_type_given_as_non_image_file_is_refused(
    legacy_client, fake_stat, tmp_path, monkeypatch
):
    client = legacy_client(_clip_gateway())
    _set_local_file_loading(monkeypatch, True)
    notes_path = tmp_path / "notes.txt"
    notes_path.write_text("hello")

    response = _compare_image_subject(client, str(notes_path))

    assert response.status_code == 400
    assert response.json() == {"message": f"{PREFIX}{LOCAL_FILE_NOT_LOADED}"}


def test_image_without_declared_type_is_not_read_from_disk_when_disabled(
    legacy_client, fake_stat, tmp_path, monkeypatch
):
    client = legacy_client(_clip_gateway())
    _set_local_file_loading(monkeypatch, False)
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png())

    response = _compare_image_subject(client, str(image_path))

    assert response.status_code == 400
    assert response.json() == {"message": f"{PREFIX}{NUMPY_UNSUPPORTED}"}


@pytest.mark.parametrize(
    "subject",
    [
        "hello world",
        "",
        "/no/such/file.png",
        NOT_AN_IMAGE_B64,
        base64.b64encode(pickle.dumps(np.zeros((4, 6, 3), np.uint8))).decode(),
    ],
)
def test_image_without_declared_type_that_cannot_be_inferred_answers_like_legacy(
    legacy_client, fake_stat, subject
):
    response = _compare_image_subject(legacy_client(_clip_gateway()), subject)

    assert response.status_code == 400
    assert response.json() == {"message": f"{PREFIX}{NUMPY_UNSUPPORTED}"}


def test_raw_image_bytes_without_declared_type_are_accepted():
    payload = decode_inline_image(_png(), ndarray_ok=False)

    assert payload.data == _png()
    assert (payload.width, payload.height) == (7, 5)


def test_image_buffer_without_declared_type_is_accepted():
    payload = decode_inline_image(io.BytesIO(_png()), ndarray_ok=False)

    assert payload.data == _png()
    assert (payload.width, payload.height) == (7, 5)


def test_ndarray_without_declared_type_is_accepted():
    array = np.zeros((5, 7, 3), dtype=np.uint8)

    payload = decode_inline_image(array, ndarray_ok=True)

    assert payload.data is array
    assert (payload.width, payload.height) == (7, 5)


def test_nested_list_numpy_object_is_refused_like_legacy(legacy_client, fake_stat):
    response = _post_image(
        _client(legacy_client, fake_stat),
        {"type": "numpy_object", "value": [[[0, 0, 0], [0, 0, 0]]]},
    )

    assert response.status_code == 400
    assert response.json() == {"message": f"{PREFIX}{NOT_NDARRAY}"}


@pytest.mark.parametrize(
    "shape,public_message",
    [
        ((1, 4, 6, 3), NDARRAY_DIMENSIONS),
        ((4, 6, 4), NDARRAY_CHANNELS),
        ((4, 6, 2), NDARRAY_CHANNELS),
    ],
)
def test_numpy_object_shape_is_checked_like_legacy(shape, public_message):
    with pytest.raises(LegacyHTTPError) as exc:
        decode_inline_image(
            {"type": "numpy_object", "value": np.zeros(shape, dtype=np.uint8)},
            ndarray_ok=True,
        )

    assert exc.value.status_code == 400
    assert exc.value.message == f"{PREFIX}{public_message}"


@pytest.mark.parametrize("shape", [(4, 6), (4, 6, 1), (4, 6, 3)])
def test_numpy_object_with_image_shape_is_accepted(shape):
    array = np.zeros(shape, dtype=np.uint8)

    payload = decode_inline_image(
        {"type": "numpy_object", "value": array}, ndarray_ok=True
    )

    assert payload.data is array
    assert (payload.width, payload.height) == (6, 4)


class _JsonRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


@pytest.mark.parametrize("declared_type", ["file", "BASE64"])
@pytest.mark.asyncio
async def test_v2_json_image_keeps_refusing_other_declared_types(
    tmp_path, declared_type
):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png())
    value = str(image_path)
    if declared_type == "BASE64":
        value = base64.b64encode(_png()).decode()

    images, params, error = await extract_json_base64(
        _JsonRequest({"inputs": {"image": {"type": declared_type, "value": value}}})
    )

    assert images == []
    assert error.status_code == 400
    assert json.loads(error.body) == {
        "error_code": "INVALID_IMAGE",
        "description": 'inputs.image[0] must be {"type": "base64", "value": "..."}',
    }


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
    target = "LEGACY_OFFLINE_MODE" if setting == "OFFLINE_MODE" else setting
    monkeypatch.setattr(
        f"inference_server.legacy.common.{target}", setting == "OFFLINE_MODE"
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
    monkeypatch.setattr(
        "inference_server.configuration.ALLOW_URL_INPUT_WITHOUT_FQDN", True
    )

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
    async def _fetch(urls, destination_policy=None):
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
    async def _fetch(urls, destination_policy=None):
        return [b"hello"], None

    monkeypatch.setattr("inference_server.legacy.common.fetch_images_from_urls", _fetch)

    with pytest.raises(LegacyHTTPError) as exc:
        await load_request_images(
            [{"type": "url", "value": "https://example.com/a.jpg"}], ndarray_ok=False
        )

    assert exc.value.status_code == 400
    assert exc.value.message == f"{PREFIX}{URL_NOT_IMAGE}"


def _noise_png(side=64) -> bytes:
    pixels = np.random.default_rng(0).integers(0, 255, (side, side, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")

    return buffer.getvalue()


def _count_file_reads(monkeypatch):
    read_sizes = []
    original_read = os.read

    def _read(descriptor, size):
        chunk = original_read(descriptor, size)
        read_sizes.append(len(chunk))
        return chunk

    monkeypatch.setattr("inference_server.legacy.common.os.read", _read)

    return read_sizes


def test_local_file_larger_than_the_image_cap_is_refused_without_reading_past_it(
    legacy_client, fake_stat, tmp_path, monkeypatch
):
    client = _client(legacy_client, fake_stat)
    _set_local_file_loading(monkeypatch, True)
    cap = len(_png()) + 16
    monkeypatch.setattr("inference_server.legacy.common.URL_FETCH_MAX_BYTES", cap)
    monkeypatch.setattr("inference_server.legacy.common._FILE_CHUNK_BYTES", 32)
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png() + b"\0" * 100_000)
    read_sizes = _count_file_reads(monkeypatch)

    response = _post_image(client, {"type": "file", "value": str(image_path)})

    assert response.status_code == 413
    assert response.json() == {"message": "image file exceeds 0MB limit"}
    assert sum(read_sizes) == cap + 1


@pytest.mark.asyncio
async def test_local_files_of_one_request_share_the_aggregate_budget(
    tmp_path, monkeypatch
):
    _set_local_file_loading(monkeypatch, True)
    budget = len(_png()) * 2 - 1
    monkeypatch.setattr("inference_server.configuration.MAX_BODY_BYTES", budget)
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png())
    read_sizes = _count_file_reads(monkeypatch)
    image = {"type": "file", "value": str(image_path)}

    with pytest.raises(LegacyHTTPError) as exc:
        await load_request_images([image, image], ndarray_ok=False)

    assert exc.value.status_code == 413
    assert exc.value.message == (
        f"combined size of image files exceeds {budget} byte limit"
    )
    assert sum(read_sizes) == budget + 1


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="platform has no named pipes")
@pytest.mark.timeout(20)
def test_named_pipe_given_as_local_file_is_refused_without_blocking(
    legacy_client, fake_stat, tmp_path, monkeypatch
):
    client = _client(legacy_client, fake_stat)
    _set_local_file_loading(monkeypatch, True)
    pipe_path = tmp_path / "pipe.png"
    os.mkfifo(pipe_path)

    response = _post_image(client, {"type": "file", "value": str(pipe_path)})

    assert response.status_code == 400
    assert response.json() == {"message": f"{PREFIX}{LOCAL_FILE_NOT_LOADED}"}


@pytest.mark.skipif(not os.path.exists("/dev/zero"), reason="platform has no /dev/zero")
@pytest.mark.timeout(20)
def test_device_given_as_local_file_is_refused_without_being_read(
    legacy_client, fake_stat, monkeypatch
):
    client = _client(legacy_client, fake_stat)
    _set_local_file_loading(monkeypatch, True)
    read_sizes = _count_file_reads(monkeypatch)

    response = _post_image(client, {"type": "file", "value": "/dev/zero"})

    assert response.status_code == 400
    assert response.json() == {"message": f"{PREFIX}{LOCAL_FILE_NOT_LOADED}"}
    assert read_sizes == []


def test_local_file_is_opened_exactly_once(
    legacy_client, fake_stat, tmp_path, monkeypatch
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _detection_gateway()
    client = legacy_client(gateway)
    _set_local_file_loading(monkeypatch, True)
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png())
    opened = []
    original_open = os.open

    def _open(path, *args, **kwargs):
        opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("inference_server.legacy.common.os.open", _open)
    monkeypatch.setattr(
        "builtins.open",
        lambda *args, **kwargs: pytest.fail(f"opened again: {args[0]}"),
    )
    monkeypatch.setattr(
        "inference_server.legacy.common.os.path.isfile",
        lambda path: pytest.fail("checked by path"),
    )

    response = _post_image(client, {"type": "file", "value": str(image_path)})

    monkeypatch.undo()
    assert response.status_code == 200, response.text
    assert opened.count(str(image_path)) == 1
    assert _inferred_images(gateway) == [_png()]


def test_local_file_with_truncated_pixel_data_is_refused(
    legacy_client, fake_stat, tmp_path, monkeypatch
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _detection_gateway()
    client = legacy_client(gateway)
    _set_local_file_loading(monkeypatch, True)
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_noise_png()[: len(_noise_png()) // 2])

    response = _post_image(client, {"type": "file", "value": str(image_path)})

    assert response.status_code == 400
    assert response.json() == {"message": f"{PREFIX}{LOCAL_FILE_NOT_LOADED}"}
    assert _inferred_images(gateway) == []


def test_npy_container_without_declared_type_is_not_an_image(legacy_client, fake_stat):
    gateway = _clip_gateway()
    buffer = io.BytesIO()
    np.save(buffer, np.zeros((5, 7, 3), dtype=np.uint8), allow_pickle=False)

    response = _compare_image_subject(
        legacy_client(gateway), base64.b64encode(buffer.getvalue()).decode()
    )

    assert response.status_code == 400
    assert response.json() == {"message": f"{PREFIX}{NUMPY_UNSUPPORTED}"}
    assert _inferred_images(gateway) == []


def test_declared_base64_image_keeps_the_header_check_only():
    truncated = _noise_png()[: len(_noise_png()) // 2]

    payload = decode_inline_image(
        {"type": "base64", "value": base64.b64encode(truncated).decode()},
        ndarray_ok=False,
    )

    assert payload.data == truncated
    assert (payload.width, payload.height) == (64, 64)


def _lower_pixel_ceiling(monkeypatch, ceiling):
    monkeypatch.setattr(
        "inference_server.legacy.common.max_decoded_pixels", lambda: ceiling
    )


def _spy_on_pixel_loading(monkeypatch):
    loads = []
    original_load = ImageFile.ImageFile.load

    def _load(image, *args, **kwargs):
        loads.append(image.size)
        return original_load(image, *args, **kwargs)

    monkeypatch.setattr(ImageFile.ImageFile, "load", _load)

    return loads


def test_local_file_above_the_pixel_ceiling_is_refused_before_pixels_are_loaded(
    legacy_client, fake_stat, tmp_path, monkeypatch
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _detection_gateway()
    client = legacy_client(gateway)
    _set_local_file_loading(monkeypatch, True)
    _lower_pixel_ceiling(monkeypatch, 7 * 5 - 1)
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png())
    loads = _spy_on_pixel_loading(monkeypatch)

    response = _post_image(client, {"type": "file", "value": str(image_path)})

    assert response.status_code == 400
    assert response.json() == {"message": f"{PREFIX}{LOCAL_FILE_NOT_LOADED}"}
    assert loads == []
    assert _inferred_images(gateway) == []


def test_local_file_at_the_pixel_ceiling_is_loaded(
    legacy_client, fake_stat, tmp_path, monkeypatch
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _detection_gateway()
    client = legacy_client(gateway)
    _set_local_file_loading(monkeypatch, True)
    _lower_pixel_ceiling(monkeypatch, 7 * 5)
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png())
    loads = _spy_on_pixel_loading(monkeypatch)

    response = _post_image(client, {"type": "file", "value": str(image_path)})

    assert response.status_code == 200, response.text
    assert loads == [(7, 5)]


def test_local_file_above_the_pillow_pixel_ceiling_is_refused(
    legacy_client, fake_stat, tmp_path, monkeypatch
):
    client = _client(legacy_client, fake_stat)
    _set_local_file_loading(monkeypatch, True)
    _lower_pixel_ceiling(monkeypatch, 0)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 7 * 5 - 1)
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png())
    loads = _spy_on_pixel_loading(monkeypatch)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        response = _post_image(client, {"type": "file", "value": str(image_path)})

    assert response.status_code == 400
    assert response.json() == {"message": f"{PREFIX}{LOCAL_FILE_NOT_LOADED}"}
    assert loads == []


def test_local_file_that_pillow_calls_a_decompression_bomb_is_refused(
    legacy_client, fake_stat, tmp_path, monkeypatch
):
    client = _client(legacy_client, fake_stat)
    _set_local_file_loading(monkeypatch, True)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 7 * 5 // 2 - 1)
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png())

    response = _post_image(client, {"type": "file", "value": str(image_path)})

    assert response.status_code == 400
    assert response.json() == {"message": f"{PREFIX}{LOCAL_FILE_NOT_LOADED}"}


def test_image_without_declared_type_above_the_pixel_ceiling_is_not_accepted(
    legacy_client, fake_stat, monkeypatch
):
    gateway = _clip_gateway()
    client = legacy_client(gateway)
    _lower_pixel_ceiling(monkeypatch, 7 * 5 - 1)
    subject = base64.b64encode(_png()).decode()
    loads = _spy_on_pixel_loading(monkeypatch)

    response = _compare_image_subject(client, subject)

    assert response.status_code == 400
    assert response.json() == {"message": f"{PREFIX}{NUMPY_UNSUPPORTED}"}
    assert loads == []
    assert _inferred_images(gateway) == []


def test_decoding_leaves_the_warning_filters_untouched():
    filters_before = list(warnings.filters)

    payload = decode_inline_image(_png(), ndarray_ok=False)

    assert payload.data == _png()
    assert list(warnings.filters) == filters_before


@pytest.mark.timeout(30)
def test_overlapping_decodes_leave_the_warning_filters_untouched(monkeypatch):
    first_is_loading = threading.Event()
    second_is_loading = threading.Event()
    first_is_done = threading.Event()
    original_load = ImageFile.ImageFile.load
    encoded = _png()
    results = {}

    def _load(image, *args, **kwargs):
        if threading.current_thread().name == "first":
            first_is_loading.set()
            assert second_is_loading.wait(10)
        else:
            second_is_loading.set()
            assert first_is_done.wait(10)
        return original_load(image, *args, **kwargs)

    def _decode():
        name = threading.current_thread().name
        try:
            results[name] = decode_inline_image(encoded, ndarray_ok=False).data
        finally:
            if name == "first":
                first_is_done.set()

    monkeypatch.setattr(ImageFile.ImageFile, "load", _load)
    filters_before = list(warnings.filters)
    first = threading.Thread(target=_decode, name="first")
    second = threading.Thread(target=_decode, name="second")

    first.start()
    assert first_is_loading.wait(10)
    second.start()
    first.join(20)
    second.join(20)
    monkeypatch.undo()

    assert results == {"first": encoded, "second": encoded}
    assert list(warnings.filters) == filters_before
    accepted = decode_inline_image(
        {"type": "base64", "value": base64.b64encode(_png()).decode()},
        ndarray_ok=False,
    )
    assert accepted.data == _png()
