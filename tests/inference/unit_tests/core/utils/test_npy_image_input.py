import base64
import io
import pickle
import warnings
from unittest.mock import Mock

import numpy as np
import pytest

from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.exceptions import (
    InputFormatInferenceFailed,
    InvalidImageTypeDeclared,
    InvalidNumpyInput,
)
from inference.core.utils import image_utils
from inference.core.warnings import InferenceDeprecationWarning


def _npy(image, *, version=(1, 0), allow_pickle=False):
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, image, version=version, allow_pickle=allow_pickle)
    return buffer.getvalue()


def _value(data, *, encoded):
    return base64.b64encode(data).decode("ascii") if encoded else data


@pytest.mark.parametrize("version", [(1, 0), (2, 0)])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided"])
@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("allow_pickle_input", [False, True])
def test_explicit_npy_preserves_bgr_pixels(
    monkeypatch, version, layout, encoded, allow_pickle_input
):
    monkeypatch.setattr(image_utils, "ALLOW_NUMPY_INPUT", allow_pickle_input)
    image = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
    if layout == "fortran":
        image = np.asfortranarray(image)
    elif layout == "strided":
        image = image[::2, ::2, ::-1]
    request = InferenceRequestImage(
        type="NPY", value=_value(_npy(image, version=version), encoded=encoded)
    )

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always", InferenceDeprecationWarning)
        loaded, is_bgr = image_utils.load_image(request)

    assert is_bgr
    assert loaded.dtype == image.dtype
    np.testing.assert_array_equal(loaded, image)
    np.testing.assert_array_equal(image_utils.load_image_rgb(request), image[..., ::-1])
    assert not emitted


@pytest.mark.parametrize("channels_axis", [False, True])
def test_npy_grayscale_uses_existing_bgr_conversion(channels_axis):
    gray = np.arange(35, dtype=np.uint8).reshape(5, 7)
    image = gray[..., None] if channels_axis else gray

    loaded, is_bgr = image_utils.load_image({"type": "npy", "value": _npy(image)})

    assert is_bgr
    np.testing.assert_array_equal(loaded, np.repeat(gray[..., None], 3, axis=2))


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
def test_npy_rejects_non_image_layouts_before_loading(monkeypatch, image):
    decode = Mock(side_effect=AssertionError("must reject before allocating"))
    monkeypatch.setattr(image_utils.np, "load", decode)

    with pytest.raises(InvalidNumpyInput):
        image_utils.load_image_from_npy(_npy(image))

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
    monkeypatch.setattr(image_utils.np, "load", decode)

    with pytest.raises(InvalidNumpyInput):
        image_utils.load_image_from_npy(data)

    decode.assert_not_called()


@pytest.mark.parametrize("value", ["not base64!", "", b"", None, {}, "☃"])
def test_npy_rejects_malformed_input(value):
    with pytest.raises(InvalidNumpyInput):
        image_utils.load_image_from_npy(value)


_UNPICKLED = []


def _record_unpickle():
    _UNPICKLED.append(True)


class _PickleSentinel:
    def __reduce__(self):
        return _record_unpickle, ()


@pytest.mark.parametrize("container", ["pickle", "object-npy"])
@pytest.mark.parametrize("enabled", [False, True])
def test_npy_never_executes_pickle(monkeypatch, container, enabled):
    monkeypatch.setattr(image_utils, "ALLOW_NUMPY_INPUT", enabled)
    _UNPICKLED.clear()
    image = np.full((2, 3, 1), _PickleSentinel(), dtype=object)
    data = (
        pickle.dumps(image) if container == "pickle" else _npy(image, allow_pickle=True)
    )

    with pytest.raises(InvalidNumpyInput):
        image_utils.load_image({"type": "npy", "value": _value(data, encoded=True)})

    assert not _UNPICKLED


@pytest.mark.parametrize("allow_numpy_input", [False, True])
@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("protocol", [4, 5])
@pytest.mark.parametrize("declared", [False, True])
def test_pickled_numpy_is_never_unpickled(
    monkeypatch, allow_numpy_input, encoded, protocol, declared
):
    monkeypatch.setattr(image_utils, "ALLOW_NUMPY_INPUT", allow_numpy_input)
    image = np.arange(105, dtype=np.uint8).reshape(5, 7, 3)
    value = _value(pickle.dumps(image, protocol=protocol), encoded=encoded)
    request = {"type": "numpy", "value": value} if declared else value
    unpickle = Mock(side_effect=AssertionError("request data must never be unpickled"))
    monkeypatch.setattr(pickle, "loads", unpickle)

    expected = (
        InvalidImageTypeDeclared
        if declared or not allow_numpy_input
        else InputFormatInferenceFailed
    )
    with pytest.raises(expected):
        image_utils.load_image(request)

    unpickle.assert_not_called()
    assert not _UNPICKLED


def test_numpy_object_has_no_pickle_deprecation_warning():
    image = np.zeros((5, 7, 3), dtype=np.uint8)

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always", InferenceDeprecationWarning)
        loaded, _ = image_utils.load_image({"type": "numpy_object", "value": image})

    assert loaded is image
    assert not emitted


def test_untyped_npy_still_requires_an_explicit_type(monkeypatch):
    monkeypatch.setattr(image_utils, "ALLOW_NUMPY_INPUT", False)
    data = _value(_npy(np.zeros((5, 7, 3), dtype=np.uint8)), encoded=True)

    with pytest.raises(InvalidImageTypeDeclared):
        image_utils.load_image(data)


@pytest.mark.parametrize("tensor_mode", [False, True])
def test_workflow_npy_retains_type_and_metadata(monkeypatch, tensor_mode):
    from roboflow_workflows.core_steps.common import deserializers, deserializers_tensor

    from inference.core.interfaces.workflows_image_codec import ServerImageCodec

    monkeypatch.setattr(image_utils, "ALLOW_NUMPY_INPUT", False)
    image = np.arange(105, dtype=np.uint8).reshape(5, 7, 3)
    deserializer = deserializers_tensor if tensor_mode else deserializers

    result = deserializer.deserialize_image_kind(
        parameter="image",
        image={
            "type": "npy",
            "value": _value(_npy(image), encoded=True),
            "parent_id": "source-frame",
        },
        image_codec=ServerImageCodec(),
    )

    np.testing.assert_array_equal(result.numpy_image, image)
    assert result.parent_metadata.parent_id == "source-frame"
    assert base64.b64decode(result.base64_image).startswith(b"\xff\xd8")


_MALFORMED_HEADERS = {
    "unterminated-brace": b"{",
    "truncated-dict": b"{'descr': '|u1', 'fortran_order': False, 'shape': (2,2),",
    "deep-nesting": b"[" * 3000,
    "binary": b"\x00\xff",
    "bad-syntax": b"{'descr': '|u1' 'x'}",
    "not-a-dict": b"(1, 2)",
}


@pytest.mark.parametrize("version", [(1, 0), (2, 0)], ids=["v1", "v2"])
@pytest.mark.parametrize("name", list(_MALFORMED_HEADERS))
def test_malformed_npy_header_raises_invalid_numpy_input(monkeypatch, version, name):
    header = _MALFORMED_HEADERS[name]
    size = len(header).to_bytes(2 if version == (1, 0) else 4, "little")
    data = b"\x93NUMPY" + bytes(version) + size + header
    decode = Mock(side_effect=AssertionError("must reject before allocating"))
    monkeypatch.setattr(image_utils.np, "load", decode)

    with pytest.raises(InvalidNumpyInput):
        image_utils.load_image({"type": "npy", "value": data})

    decode.assert_not_called()


_MALFORMED_DESCRIPTORS = {
    "empty-tuple": b"{'descr': (), 'fortran_order': False, 'shape': (1, 1)}",
    "incomplete-tuple": b"{'descr': ('|u1',), 'fortran_order': False, 'shape': (1, 1)}",
    "list-of-garbage": b"{'descr': [()], 'fortran_order': False, 'shape': (1, 1)}",
    "missing-keys": b"{'shape': (1, 1)}",
}


@pytest.mark.parametrize("version", [(1, 0), (2, 0)], ids=["v1", "v2"])
@pytest.mark.parametrize("name", list(_MALFORMED_DESCRIPTORS))
def test_malformed_npy_descriptor_raises_invalid_numpy_input(monkeypatch, version, name):
    header = _MALFORMED_DESCRIPTORS[name]
    size = len(header).to_bytes(2 if version == (1, 0) else 4, "little")
    data = b"\x93NUMPY" + bytes(version) + size + header
    decode = Mock(side_effect=AssertionError("must reject before allocating"))
    monkeypatch.setattr(image_utils.np, "load", decode)

    with pytest.raises(InvalidNumpyInput):
        image_utils.load_image({"type": "npy", "value": data})

    decode.assert_not_called()
