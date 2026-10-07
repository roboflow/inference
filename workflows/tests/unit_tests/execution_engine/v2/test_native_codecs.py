"""Tests of the faithful native wire format in ``v2/blocks/payload_codec.py``.

Round trips go through ``json.dumps``/``json.loads`` and are compared with
``_assert_same``, a structural comparison that does not use the codec: types
(tuple vs list, int vs str keys, NumPy scalar vs float), dtypes, shapes and
exact element bits must all survive.
"""

import base64
import dataclasses
import json
import math
from datetime import datetime
from typing import Any

import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.v2.blocks.payload_codec import (
    NATIVE_ENVELOPE_KEY,
    SUPPORTED_TENSOR_DTYPES,
    decode_native,
    encode_native,
    is_native_envelope,
    tensor_dtype_problem,
)
from roboflow_workflows.execution_engine.v2.errors import ContractError

from inference_models.models.base.object_detection import Detections


def _round_trip(value: Any) -> Any:
    wire = json.loads(json.dumps(encode_native(value), allow_nan=False))

    return decode_native(wire)


def _assert_same(actual: Any, expected: Any, path: str = "value") -> None:
    assert type(actual) is type(expected), f"{path}: {type(actual)} != {type(expected)}"
    if isinstance(expected, torch.Tensor):
        assert (actual.dtype, actual.shape) == (expected.dtype, expected.shape), path
        assert actual.device.type == "cpu", path
        assert _bits(actual) == _bits(expected), path
    elif isinstance(expected, (np.ndarray, np.generic)):
        assert (actual.dtype, actual.shape) == (expected.dtype, expected.shape), path
        assert actual.tobytes() == expected.tobytes(), path
    elif dataclasses.is_dataclass(expected):
        for field in dataclasses.fields(expected):
            _assert_same(
                getattr(actual, field.name),
                getattr(expected, field.name),
                f"{path}.{field.name}",
            )
    elif isinstance(expected, dict):
        assert list(actual) == list(expected), path
        assert [type(key) for key in actual] == [type(key) for key in expected], path
        for key in expected:
            _assert_same(actual[key], expected[key], f"{path}[{key!r}]")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_same(left, right, f"{path}[{index}]")
    elif isinstance(expected, float) and math.isnan(expected):
        assert math.isnan(actual), path
    else:
        assert actual == expected, path


def _bits(tensor: torch.Tensor) -> bytes:
    host = tensor.detach().cpu().contiguous().reshape(-1)

    return host.view(torch.uint8).numpy().tobytes()


def _tensor_wire(dtype: str, shape: list, data: bytes) -> dict:
    return {
        NATIVE_ENVELOPE_KEY: 1,
        "value": {
            "tensor": {
                "dtype": dtype,
                "shape": shape,
                "data": base64.b64encode(data).decode("ascii"),
            }
        },
    }


TENSORS = {
    "float16 (0, 8)": torch.zeros(0, 8, dtype=torch.float16),
    "float64 0-d": torch.tensor(2.5, dtype=torch.float64),
    "int64 0-d": torch.tensor(-7),
    "bfloat16": torch.tensor([1.5, -2.0], dtype=torch.bfloat16),
    "bool": torch.tensor([[True, False], [False, True]]),
    "uint8 (2, 0, 3)": torch.zeros(2, 0, 3, dtype=torch.uint8),
    "int32 transposed view": torch.arange(6, dtype=torch.int32).reshape(2, 3).t(),
    "float32 strided slice": torch.arange(10.0)[::3],
    "float32 non-finite": torch.tensor([math.nan, math.inf, -math.inf, -0.0]),
    "int16 extremes": torch.tensor([-32768, 32767], dtype=torch.int16),
    "int8": torch.tensor([-128, 127], dtype=torch.int8),
}


@pytest.mark.parametrize("tensor", TENSORS.values(), ids=list(TENSORS))
def test_tensors_round_trip_dtype_shape_and_bits(tensor: torch.Tensor) -> None:
    # when
    decoded = _round_trip(tensor)

    # then
    _assert_same(decoded, tensor)
    assert decoded.is_contiguous()


def test_every_supported_tensor_dtype_round_trips() -> None:
    for name, dtype in SUPPORTED_TENSOR_DTYPES.items():
        tensor = torch.ones(3, dtype=dtype)
        _assert_same(_round_trip(tensor), tensor, name)


@pytest.mark.parametrize("dtype", list(SUPPORTED_TENSOR_DTYPES.values()))
@pytest.mark.parametrize(
    "shape,strides",
    [((1,), (38,)), ((0,), (38,)), ((1, 1), (76, 38)), ((1,), (0,))],
    ids=["singleton-column", "empty-column", "singleton-matrix", "zero-stride"],
)
def test_contiguous_views_with_nonunit_strides_round_trip(dtype, shape, strides):
    storage = torch.arange(76).to(dtype)
    tensor = storage.as_strided(shape, strides, storage_offset=3)
    original = storage.clone()
    assert tensor.is_contiguous()
    assert tensor.stride(-1) != 1

    decoded = _round_trip(tensor)

    assert decoded.dtype == dtype
    assert decoded.shape == tensor.shape
    assert torch.equal(decoded, tensor)
    assert torch.equal(storage, original)


def test_tensor_wire_is_compact_base64_not_number_lists() -> None:
    # when
    wire = encode_native(torch.arange(4, dtype=torch.int16).reshape(2, 2))

    # then
    assert wire == _tensor_wire("int16", [2, 2], bytes([0, 0, 1, 0, 2, 0, 3, 0]))


def test_metadata_grammar_keeps_types_and_cannot_collide_with_tags() -> None:
    # given: ordinary keys spelled like tags or like the envelope key
    metadata = {
        "class_names": {0: "car", 1: "truck"},
        "mixed": {1: "one", "two": 2},
        "scaling_relative_to_root_parent": [0.5, 0.25],
        "offset": (3.0, 4.5),
        "parent_frame_id": None,
        "flag": True,
        "bytes": b"\x00\xffcounts",
        "polygon": np.array([[1, 2], [3, 4]], dtype=np.int16),
        "score": np.float32(0.5),
        "tracker": np.int64(9),
        "velocity": torch.tensor([1.0, -2.0]),
        "nan": math.nan,
        "inf": -math.inf,
        "tensor": {"dtype": "int8", "shape": [1], "data": "AA=="},
        "dict": {"dict": 1},
        NATIVE_ENVELOPE_KEY: 1,
        "nested": [{"deep": ({"x": [None, 1.25]},)}],
        "big": 2**70,
    }

    # when
    decoded = _round_trip(metadata)

    # then
    _assert_same(decoded, metadata)
    assert decoded["polygon"].flags.writeable


@pytest.mark.parametrize("value", [0.5, -0.0, math.nan, math.inf, -math.inf])
def test_numpy_float64_metadata_keeps_its_scalar_type_and_bits(value: float) -> None:
    metadata = {"score": np.float64(value)}

    _assert_same(_round_trip(metadata), metadata)


@pytest.mark.parametrize("version", [1.0, True, np.int64(1)])
def test_envelope_version_requires_a_python_integer(version: Any) -> None:
    with pytest.raises(ContractError, match="Unsupported native envelope version"):
        decode_native({NATIVE_ENVELOPE_KEY: version, "value": None})


def test_native_carriers_round_trip_with_their_fields() -> None:
    # given
    detections = Detections(
        xyxy=torch.zeros(0, 4),
        class_id=torch.zeros(0, dtype=torch.long),
        confidence=torch.zeros(0),
        image_metadata={"parent_id": "crop", "class_names": {}},
        bboxes_metadata=[],
    )

    # when
    decoded = _round_trip(detections)

    # then
    _assert_same(decoded, detections)
    assert encode_native(detections)["value"]["carrier"]["type"] == "Detections"


class _Secret:
    def __repr__(self) -> str:
        return "SECRET-REPR"


UNSUPPORTED_VALUES = [
    ({"time": datetime(2026, 9, 30)}, r"value\['time'\]: unsupported type datetime"),
    ({"tags": {1, 2}}, r"value\['tags'\]: unsupported type set"),
    ({"obj": [_Secret()]}, r"value\['obj'\]\[0\]: unsupported type _Secret"),
    ({1.5: "float key"}, "dict keys must be str or int, got float"),
    ({True: "bool key"}, "dict keys must be str or int, got bool"),
    ({"a": np.array(["x"])}, "numpy dtype <U1 is not supported"),
    ({"a": np.array([1 + 2j])}, "numpy dtype complex128 is not supported"),
    (torch.zeros(2, dtype=torch.complex64), "tensor dtype torch.complex64 is not"),
    (torch.zeros(2, dtype=torch.uint16), "tensor dtype torch.uint16 is not"),
]


@pytest.mark.parametrize(("value", "message"), UNSUPPORTED_VALUES)
def test_unsupported_values_are_rejected_by_path_without_repr(
    value: Any, message: str
) -> None:
    # when
    with pytest.raises(ContractError, match=message) as error:
        encode_native(value)

    # then
    assert "SECRET-REPR" not in str(error.value)


def test_validators_and_codec_share_one_dtype_table() -> None:
    assert tensor_dtype_problem(torch.float16) is None
    assert "torch.uint16 is not supported" in tensor_dtype_problem(torch.uint16)


MALFORMED_ENVELOPES = [
    ({"value": 1}, "exactly the keys"),
    ({NATIVE_ENVELOPE_KEY: 1, "value": 1, "kind": "tensor"}, "exactly the keys"),
    ({NATIVE_ENVELOPE_KEY: 2, "value": 1}, "Unsupported native envelope version 2"),
    ({NATIVE_ENVELOPE_KEY: True, "value": 1}, "Unsupported native envelope version"),
    ({NATIVE_ENVELOPE_KEY: 1, "value": {"a": 1}}, "unknown tag 'a'"),
    (
        {NATIVE_ENVELOPE_KEY: 1, "value": {"dict": {}, "tuple": []}},
        "a node with one tag, got object",
    ),
    ({NATIVE_ENVELOPE_KEY: 1, "value": {"float": "1.5"}}, "'nan', 'inf' or '-inf'"),
    ({NATIVE_ENVELOPE_KEY: 1, "value": {"bytes": "***"}}, "invalid base64"),
    ({NATIVE_ENVELOPE_KEY: 1, "value": {"map": [[1.5, 2]]}}, "keys are str or int"),
    ({NATIVE_ENVELOPE_KEY: 1, "value": {"map": [[1]]}}, r"\[key, value\] pairs"),
    (_tensor_wire("float32", [2], b"\x00" * 7), "needs 8 bytes, got 7"),
    (_tensor_wire("float32", [-1], b""), "non-negative integers"),
    (_tensor_wire("float32", [True], b"\x00" * 4), "non-negative integers"),
    (_tensor_wire("complex64", [1], b"\x00" * 8), "complex64 is not supported"),
    (_tensor_wire("bool", [2], b"\x00\x02"), "only bytes 0 and 1"),
    (
        {
            NATIVE_ENVELOPE_KEY: 1,
            "value": {
                "tensor": {"dtype": "int8", "shape": [1], "data": "AA==", "x": 1}
            },
        },
        "needs exactly the keys",
    ),
    (
        {
            NATIVE_ENVELOPE_KEY: 1,
            "value": {"numpy_scalar": {"dtype": "int8", "shape": [1], "data": "AA=="}},
        },
        r"numpy_scalar needs shape \[\]",
    ),
    (
        {NATIVE_ENVELOPE_KEY: 1, "value": {"carrier": {"type": "Image", "fields": {}}}},
        "unknown type 'Image'",
    ),
    (
        {
            NATIVE_ENVELOPE_KEY: 1,
            "value": {"carrier": {"type": "Detections", "fields": {"xyxy": None}}},
        },
        r"missing \['bboxes_metadata', 'class_id', 'confidence', 'image_metadata'\]",
    ),
]


@pytest.mark.parametrize(("envelope", "message"), MALFORMED_ENVELOPES)
def test_malformed_envelopes_are_rejected_before_values_are_built(
    envelope: Any, message: str
) -> None:
    # when / then
    with pytest.raises(ContractError, match=message):
        decode_native(envelope)


def test_only_dicts_with_the_discriminator_claim_to_be_envelopes() -> None:
    assert is_native_envelope({NATIVE_ENVELOPE_KEY: 7})
    assert not is_native_envelope({"type": "tensor", "dtype": "float32"})
    assert not is_native_envelope([NATIVE_ENVELOPE_KEY])
