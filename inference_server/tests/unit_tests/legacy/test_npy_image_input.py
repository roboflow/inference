import base64
import io
import pickle
from unittest.mock import Mock

import numpy as np
import pytest

from inference_server.legacy import common
from inference_server.legacy.common import decode_inline_image
from inference_server.legacy.errors import InvalidNumpyInput
from inference_server.legacy.visualization import payload_to_rgb
from tests.unit_tests.legacy.test_image_load_errors import (
    _detection_gateway,
    _inferred_images,
    _post_image,
)


def _npy(image, *, version=(1, 0), allow_pickle=False):
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, image, version=version, allow_pickle=allow_pickle)
    return buffer.getvalue()


@pytest.mark.parametrize("ndarray_ok", [False, True])
@pytest.mark.parametrize("version", [(1, 0), (2, 0)])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided"])
@pytest.mark.parametrize("encoded", [False, True])
def test_explicit_npy_preserves_bgr_pixels(ndarray_ok, version, layout, encoded):
    image = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
    if layout == "fortran":
        image = np.asfortranarray(image)
    elif layout == "strided":
        image = image[::2, ::2, ::-1]
    data = _npy(image, version=version)
    value = base64.b64encode(data).decode("ascii") if encoded else data

    payload = decode_inline_image(
        {"type": "NPY", "value": value}, ndarray_ok=ndarray_ok
    )

    assert (payload.width, payload.height) == (image.shape[1], image.shape[0])
    if ndarray_ok:
        np.testing.assert_array_equal(payload.data, image)
    else:
        assert payload.data == data
        np.testing.assert_array_equal(
            np.load(io.BytesIO(payload.data), allow_pickle=False), image
        )
    np.testing.assert_array_equal(payload_to_rgb(payload), image[..., ::-1])


@pytest.mark.parametrize("ndarray_ok", [False, True])
@pytest.mark.parametrize("channels_axis", [False, True])
def test_npy_grayscale_preserves_pixels(ndarray_ok, channels_axis):
    gray = np.arange(35, dtype=np.uint8).reshape(5, 7)
    image = gray[..., None] if channels_axis else gray

    payload = decode_inline_image(
        {"type": "npy", "value": _npy(image)}, ndarray_ok=ndarray_ok
    )

    assert (payload.width, payload.height) == (7, 5)
    np.testing.assert_array_equal(
        payload_to_rgb(payload), np.repeat(gray[..., None], 3, axis=2)
    )


@pytest.mark.parametrize("ndarray_ok", [False, True])
@pytest.mark.parametrize(
    "image",
    [
        np.zeros((5, 7, 3), dtype=np.float32),
        np.zeros((5, 7, 3), dtype=np.uint16),
        np.zeros((5, 7, 3), dtype=[("pixel", "u1")]),
        np.zeros((5, 7, 4), dtype=np.uint8),
        np.zeros((5,), dtype=np.uint8),
        np.zeros((1, 5, 7, 3), dtype=np.uint8),
        np.zeros((0, 7, 3), dtype=np.uint8),
    ],
    ids=["float", "uint16", "structured", "rgba", "1d", "batch", "empty"],
)
def test_npy_rejects_non_image_layouts(monkeypatch, ndarray_ok, image):
    decode = Mock(side_effect=AssertionError("must reject before allocating"))
    monkeypatch.setattr(common.np, "load", decode)

    with pytest.raises(InvalidNumpyInput) as error:
        decode_inline_image(
            {"type": "npy", "value": _npy(image)}, ndarray_ok=ndarray_ok
        )

    assert error.value.status_code == 400
    decode.assert_not_called()


@pytest.mark.parametrize(
    "case", ["truncated", "trailing", "npz", "huge-shape", "huge-header"]
)
def test_npy_rejects_invalid_containers_before_loading(monkeypatch, case):
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    data = _npy(image)
    if case == "truncated":
        data = data[:-1]
    elif case == "trailing":
        data += data
    elif case == "npz":
        buffer = io.BytesIO()
        np.savez(buffer, image)
        data = buffer.getvalue()
    elif case == "huge-shape":
        buffer = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            buffer, {"shape": (2**40, 2**40, 3), "fortran_order": False, "descr": "|u1"}
        )
        data = buffer.getvalue()
    elif case == "huge-header":
        data = b"\x93NUMPY\x02\x00" + (2**32 - 1).to_bytes(4, "little") + b"{}"
    decode = Mock(side_effect=AssertionError("must reject before allocating"))
    monkeypatch.setattr(common.np, "load", decode)

    with pytest.raises(InvalidNumpyInput):
        decode_inline_image({"type": "npy", "value": data}, ndarray_ok=True)

    decode.assert_not_called()


@pytest.mark.parametrize("value", ["not base64!", "", b"", None, {}, "☃"])
def test_npy_rejects_malformed_input(value):
    with pytest.raises(InvalidNumpyInput):
        decode_inline_image({"type": "npy", "value": value}, ndarray_ok=False)


_UNPICKLED = []


def _record_unpickle():
    _UNPICKLED.append(True)


class _PickleSentinel:
    def __reduce__(self):
        return _record_unpickle, ()


@pytest.mark.parametrize("container", ["pickle", "object-npy"])
@pytest.mark.parametrize("ndarray_ok", [False, True])
def test_npy_never_executes_pickle(container, ndarray_ok):
    _UNPICKLED.clear()
    image = np.full((2, 3, 1), _PickleSentinel(), dtype=object)
    data = (
        pickle.dumps(image) if container == "pickle" else _npy(image, allow_pickle=True)
    )

    with pytest.raises(InvalidNumpyInput):
        decode_inline_image({"type": "npy", "value": data}, ndarray_ok=ndarray_ok)

    assert not _UNPICKLED


@pytest.mark.parametrize("ndarray_ok", [False, True])
def test_npy_applies_existing_pixel_ceiling_before_loading(monkeypatch, ndarray_ok):
    monkeypatch.setattr(common, "max_decoded_pixels", lambda: 34)
    decode = Mock(side_effect=AssertionError("must reject before allocating"))
    monkeypatch.setattr(common.np, "load", decode)
    data = _npy(np.zeros((5, 7, 3), dtype=np.uint8))

    with pytest.raises(InvalidNumpyInput):
        decode_inline_image({"type": "npy", "value": data}, ndarray_ok=ndarray_ok)

    decode.assert_not_called()


def test_http_npy_reaches_gateway_with_original_pixels(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _detection_gateway()
    image = np.arange(105, dtype=np.uint8).reshape(5, 7, 3)
    data = _npy(image)

    response = _post_image(
        legacy_client(gateway),
        {"type": "npy", "value": base64.b64encode(data).decode("ascii")},
    )

    assert response.status_code == 200, response.text
    forwarded = _inferred_images(gateway)
    assert len(forwarded) == 1
    np.testing.assert_array_equal(
        np.load(io.BytesIO(forwarded[0]), allow_pickle=False), image
    )


@pytest.mark.parametrize("kind", ["npy", "numpy"])
def test_http_pickle_remains_rejected(legacy_client, fake_stat, kind):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _detection_gateway()
    data = pickle.dumps(np.zeros((5, 7, 3), dtype=np.uint8))

    response = _post_image(
        legacy_client(gateway),
        {"type": kind, "value": base64.b64encode(data).decode("ascii")},
    )

    assert response.status_code == 400, response.text
    assert not _inferred_images(gateway)


_MALFORMED_HEADERS = {
    "unterminated-brace": b"{",
    "truncated-dict": b"{'descr': '|u1', 'fortran_order': False, 'shape': (2,2),",
    "deep-nesting": b"[" * 3000,
    "binary": b"\x00\xff",
    "bad-syntax": b"{'descr': '|u1' 'x'}",
    "not-a-dict": b"(1, 2)",
}


def _malformed_npy(version, header: bytes) -> bytes:
    size = len(header).to_bytes(2 if version == (1, 0) else 4, "little")
    return b"\x93NUMPY" + bytes(version) + size + header


@pytest.mark.parametrize("version", [(1, 0), (2, 0)], ids=["v1", "v2"])
@pytest.mark.parametrize("name", list(_MALFORMED_HEADERS))
def test_npy_malformed_header_is_a_client_error(monkeypatch, version, name):
    decode = Mock(side_effect=AssertionError("must reject before allocating"))
    monkeypatch.setattr(common.np, "load", decode)

    with pytest.raises(InvalidNumpyInput):
        decode_inline_image(
            {"type": "npy", "value": _malformed_npy(version, _MALFORMED_HEADERS[name])},
            ndarray_ok=True,
        )

    decode.assert_not_called()


@pytest.mark.parametrize("version", [(1, 0), (2, 0)], ids=["v1", "v2"])
@pytest.mark.parametrize("name", ["unterminated-brace", "deep-nesting", "binary"])
def test_http_npy_malformed_header_returns_400(legacy_client, fake_stat, version, name):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _detection_gateway()
    data = _malformed_npy(version, _MALFORMED_HEADERS[name])

    response = _post_image(
        legacy_client(gateway),
        {"type": "npy", "value": base64.b64encode(data).decode("ascii")},
    )

    assert response.status_code == 400, response.text
    assert not _inferred_images(gateway)
