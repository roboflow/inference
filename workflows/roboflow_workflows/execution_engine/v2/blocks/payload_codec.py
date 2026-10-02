"""Faithful JSON wire format of native V2 payloads.

One versioned envelope wraps every native payload::

    {"__workflows_v2_native__": 1, "value": <value>}

``<value>`` follows a small grammar. JSON scalars and arrays stand for Python
scalars and lists. Every JSON object is a tagged node with exactly one key, so
ordinary metadata keys can never be mistaken for a tag::

    {"float": "nan" | "inf" | "-inf"}        non-finite float
    {"tuple": [value, ...]}                  tuple
    {"dict": {"<str key>": value, ...}}      dict with str keys
    {"map": [[key, value], ...]}             dict with int (or mixed) keys
    {"bytes": "<base64>"}                    bytes
    {"tensor": <array>}                      torch.Tensor
    {"ndarray": <array>}                     numpy.ndarray
    {"numpy_scalar": <array>}                NumPy scalar, shape []
    {"carrier": {"type": ..., "fields": {...}}}   one of NATIVE_CARRIERS

    <array> = {"dtype": "float16", "shape": [0, 8], "data": "<base64>"}

Array data are the raw little-endian C-order element bytes, so dtype, shape
(including zero-sized axes) and every value round-trip exactly; a
non-contiguous tensor is written in logical order. Encoding copies tensors to
the host: this module is the explicit CPU boundary. Decoding builds CPU
tensors; placing them on a device is the receiver's choice. Autograd state is
not carried.

Anything outside the grammar is rejected with its path and type name, never
with its ``repr``.
"""

import base64
import binascii
import dataclasses
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from roboflow_workflows.execution_engine.v2.errors import ContractError

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

NATIVE_ENVELOPE_KEY = "__workflows_v2_native__"
NATIVE_ENVELOPE_VERSION = 1

SUPPORTED_TENSOR_DTYPES: Dict[str, torch.dtype] = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}
SUPPORTED_NUMPY_DTYPES = (
    "bool",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
)
NATIVE_CARRIERS: Dict[str, type] = {
    carrier.__name__: carrier
    for carrier in (
        Detections,
        InstanceDetections,
        InstancesRLEMasks,
        KeyPoints,
        ClassificationPrediction,
        MultiLabelClassificationPrediction,
    )
}

_TENSOR_DTYPE_NAMES = {dtype: name for name, dtype in SUPPORTED_TENSOR_DTYPES.items()}
_CARRIER_NAMES = {carrier: name for name, carrier in NATIVE_CARRIERS.items()}
_NON_FINITE_FLOATS = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}
_ARRAY_KEYS = {"dtype", "shape", "data"}


def tensor_dtype_problem(dtype: torch.dtype) -> Optional[str]:
    """Explain why a tensor dtype cannot be carried by the wire format.

    Args:
        dtype: Tensor dtype to check.

    Returns:
        ``None`` for a supported dtype, otherwise a message naming the
        supported dtypes. Kind validators use it so that every payload they
        accept can also be serialized.
    """
    if dtype in _TENSOR_DTYPE_NAMES:
        return None

    problem = _unsupported_tensor_dtype(dtype)

    return problem


def _unsupported_tensor_dtype(dtype: Any) -> str:
    message = (
        f"tensor dtype {dtype} is not supported; supported dtypes: "
        f"{', '.join(SUPPORTED_TENSOR_DTYPES)}"
    )

    return message


def is_native_envelope(value: Any) -> bool:
    """Return whether ``value`` claims to be a native envelope.

    Args:
        value: Wire value.

    Returns:
        ``True`` for a dict holding the envelope discriminator key, even when
        the rest of it is malformed; ``decode_native`` reports the problem.
    """
    claims_envelope = isinstance(value, dict) and NATIVE_ENVELOPE_KEY in value

    return claims_envelope


def encode_native(payload: Any) -> Dict[str, Any]:
    """Encode a native payload into the versioned JSON envelope.

    Args:
        payload: A native carrier, tuple of carriers, tensor or any value of
            the grammar in the module docstring.

    Returns:
        ``{"__workflows_v2_native__": 1, "value": ...}``, JSON-serializable.

    Raises:
        ContractError: When the payload holds a value outside the grammar.
    """
    envelope = {
        NATIVE_ENVELOPE_KEY: NATIVE_ENVELOPE_VERSION,
        "value": _encode(payload, path="value"),
    }

    return envelope


def decode_native(envelope: Any) -> Any:
    """Decode a versioned JSON envelope into native values on the CPU.

    Args:
        envelope: Output of ``encode_native``, possibly after a JSON round
            trip.

    Returns:
        The payload. Tensors are new CPU tensors.

    Raises:
        ContractError: On an unknown version, extra or missing keys, unknown
            tags, carriers or dtypes, or array data that do not match their
            dtype and shape.
    """
    if not isinstance(envelope, dict) or set(envelope) != {
        NATIVE_ENVELOPE_KEY,
        "value",
    }:
        raise ContractError(
            f"A native envelope has exactly the keys '{NATIVE_ENVELOPE_KEY}' and "
            "'value'"
        )
    version = envelope[NATIVE_ENVELOPE_KEY]
    if type(version) is not int or version != NATIVE_ENVELOPE_VERSION:
        raise ContractError(
            f"Unsupported native envelope version {version!r}; this reader "
            f"supports {NATIVE_ENVELOPE_VERSION}"
        )

    payload = _decode(envelope["value"], path="value")

    return payload


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _encode(value: Any, *, path: str) -> Any:
    if isinstance(value, np.generic):
        return {"numpy_scalar": _encode_array(np.asarray(value), path=path)}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"float": "nan" if math.isnan(value) else str(value)}
    if isinstance(value, list):
        return [
            _encode(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        items = [
            _encode(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        ]
        return {"tuple": items}
    if isinstance(value, dict):
        return _encode_dict(value, path=path)
    if isinstance(value, bytes):
        return {"bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, torch.Tensor):
        return {"tensor": _encode_tensor(value, path=path)}
    if isinstance(value, np.ndarray):
        return {"ndarray": _encode_array(value, path=path)}
    if type(value) in _CARRIER_NAMES:
        return {"carrier": _encode_carrier(value, path=path)}

    raise ContractError(f"{path}: unsupported type {type(value).__name__}")


def _encode_dict(value: dict, *, path: str) -> Dict[str, Any]:
    for key in value:
        if isinstance(key, bool) or not isinstance(key, (str, int)):
            raise ContractError(
                f"{path}: dict keys must be str or int, got {type(key).__name__}"
            )

    items = {key: _encode(item, path=f"{path}[{key!r}]") for key, item in value.items()}
    if all(isinstance(key, str) for key in items):
        return {"dict": items}

    encoded = {"map": [[key, item] for key, item in items.items()]}

    return encoded


def _encode_tensor(tensor: torch.Tensor, *, path: str) -> Dict[str, Any]:
    problem = tensor_dtype_problem(tensor.dtype)
    if problem is not None:
        raise ContractError(f"{path}: {problem}")

    host = tensor.detach().cpu().contiguous()
    raw = host.reshape(-1).view(torch.uint8).numpy().tobytes()
    encoded = _array_node(
        dtype=_TENSOR_DTYPE_NAMES[tensor.dtype], shape=host.shape, raw=raw
    )

    return encoded


def _encode_array(array: np.ndarray, *, path: str) -> Dict[str, Any]:
    if array.dtype.name not in SUPPORTED_NUMPY_DTYPES:
        raise ContractError(
            f"{path}: numpy dtype {array.dtype} is not supported; supported dtypes: "
            f"{', '.join(SUPPORTED_NUMPY_DTYPES)}"
        )

    little_endian = array.astype(array.dtype.newbyteorder("<"), copy=False)
    raw = np.ascontiguousarray(little_endian).tobytes()
    encoded = _array_node(dtype=array.dtype.name, shape=array.shape, raw=raw)

    return encoded


def _array_node(*, dtype: str, shape: Tuple[int, ...], raw: bytes) -> Dict[str, Any]:
    _require_little_endian_host()
    node = {
        "dtype": dtype,
        "shape": [int(size) for size in shape],
        "data": base64.b64encode(raw).decode("ascii"),
    }

    return node


def _encode_carrier(carrier: Any, *, path: str) -> Dict[str, Any]:
    carrier_type = _CARRIER_NAMES[type(carrier)]
    fields = {
        field.name: _encode(getattr(carrier, field.name), path=f"{path}.{field.name}")
        for field in dataclasses.fields(carrier)
    }
    encoded = {"type": carrier_type, "fields": fields}

    return encoded


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _decode(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [
            _decode(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        ]
    if not isinstance(value, dict) or len(value) != 1:
        raise ContractError(
            f"{path}: expected a JSON scalar, array or a node with one tag, got "
            f"{_describe(value)}"
        )

    ((tag, body),) = value.items()
    decoder = _NODE_DECODERS.get(tag)
    if decoder is None:
        raise ContractError(
            f"{path}: unknown tag '{tag}'; expected one of {sorted(_NODE_DECODERS)}"
        )

    decoded = decoder(body, path=path)

    return decoded


def _decode_float(body: Any, *, path: str) -> float:
    if body not in _NON_FINITE_FLOATS:
        raise ContractError(f"{path}: float tag holds 'nan', 'inf' or '-inf'")

    return _NON_FINITE_FLOATS[body]


def _decode_tuple(body: Any, *, path: str) -> tuple:
    items = _require_type(body, list, path=f"{path}.tuple")
    decoded = tuple(
        _decode(item, path=f"{path}[{index}]") for index, item in enumerate(items)
    )

    return decoded


def _decode_dict(body: Any, *, path: str) -> dict:
    items = _require_type(body, dict, path=f"{path}.dict")
    decoded = {
        key: _decode(item, path=f"{path}[{key!r}]") for key, item in items.items()
    }

    return decoded


def _decode_map(body: Any, *, path: str) -> dict:
    pairs = _require_type(body, list, path=f"{path}.map")
    decoded = {}
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ContractError(f"{path}.map: entries are [key, value] pairs")
        key, item = pair
        if isinstance(key, bool) or not isinstance(key, (str, int)):
            raise ContractError(
                f"{path}.map: keys are str or int, got {_describe(key)}"
            )
        decoded[key] = _decode(item, path=f"{path}[{key!r}]")

    return decoded


def _decode_bytes(body: Any, *, path: str) -> bytes:
    decoded = _base64(body, path=f"{path}.bytes")

    return decoded


def _decode_tensor(body: Any, *, path: str) -> torch.Tensor:
    dtype_name, shape, raw = _read_array_node(body, path=f"{path}.tensor")
    dtype = SUPPORTED_TENSOR_DTYPES.get(dtype_name)
    if dtype is None:
        raise ContractError(f"{path}.tensor: {_unsupported_tensor_dtype(dtype_name)}")

    item_size = torch.empty((), dtype=dtype).element_size()
    _check_array_bytes(
        raw, shape=shape, item_size=item_size, dtype_name=dtype_name, path=path
    )
    if not raw:
        return torch.empty(shape, dtype=dtype)

    tensor = (
        torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dtype).reshape(shape)
    )

    return tensor


def _decode_ndarray(body: Any, *, path: str) -> np.ndarray:
    dtype_name, shape, raw = _read_array_node(body, path=f"{path}.ndarray")
    if dtype_name not in SUPPORTED_NUMPY_DTYPES:
        raise ContractError(
            f"{path}.ndarray: numpy dtype {dtype_name!r} is not supported; supported "
            f"dtypes: {', '.join(SUPPORTED_NUMPY_DTYPES)}"
        )

    dtype = np.dtype(dtype_name)
    _check_array_bytes(
        raw, shape=shape, item_size=dtype.itemsize, dtype_name=dtype_name, path=path
    )
    wire_dtype = dtype.newbyteorder("<")
    array = np.frombuffer(raw, dtype=wire_dtype).reshape(shape).astype(dtype)

    return array


def _decode_numpy_scalar(body: Any, *, path: str) -> np.generic:
    array = _decode_ndarray(body, path=path)
    if array.shape != ():
        raise ContractError(
            f"{path}: numpy_scalar needs shape [], got {list(array.shape)}"
        )

    scalar = array[()]

    return scalar


def _decode_carrier(body: Any, *, path: str) -> Any:
    body = _require_type(body, dict, path=f"{path}.carrier")
    if set(body) != {"type", "fields"}:
        raise ContractError(
            f"{path}.carrier: needs exactly the keys 'type' and 'fields'"
        )

    carrier_type = body["type"]
    carrier_class = (
        NATIVE_CARRIERS.get(carrier_type) if isinstance(carrier_type, str) else None
    )
    if carrier_class is None:
        shown = repr(carrier_type) if isinstance(carrier_type, str) else "non-string"
        raise ContractError(
            f"{path}.carrier: unknown type {shown}; expected one of "
            f"{sorted(NATIVE_CARRIERS)}"
        )
    fields = _require_type(body["fields"], dict, path=f"{path}.carrier.fields")
    expected = {field.name for field in dataclasses.fields(carrier_class)}
    if set(fields) != expected:
        raise ContractError(
            f"{path}: {body['type']} fields must be {sorted(expected)}; missing "
            f"{sorted(expected - set(fields))}, unexpected {sorted(set(fields) - expected)}"
        )

    decoded_fields = {
        name: _decode(item, path=f"{path}.{name}") for name, item in fields.items()
    }
    carrier = carrier_class(**decoded_fields)

    return carrier


_NODE_DECODERS = {
    "float": _decode_float,
    "tuple": _decode_tuple,
    "dict": _decode_dict,
    "map": _decode_map,
    "bytes": _decode_bytes,
    "tensor": _decode_tensor,
    "ndarray": _decode_ndarray,
    "numpy_scalar": _decode_numpy_scalar,
    "carrier": _decode_carrier,
}


def _read_array_node(body: Any, *, path: str) -> Tuple[str, List[int], bytes]:
    body = _require_type(body, dict, path=path)
    if set(body) != _ARRAY_KEYS:
        raise ContractError(f"{path}: needs exactly the keys {sorted(_ARRAY_KEYS)}")

    shape = body["shape"]
    shape_is_valid = isinstance(shape, list) and all(
        isinstance(size, int) and not isinstance(size, bool) and size >= 0
        for size in shape
    )
    if not shape_is_valid:
        raise ContractError(f"{path}: shape must be a list of non-negative integers")
    if not isinstance(body["dtype"], str):
        raise ContractError(f"{path}: dtype must be a string")

    raw = _base64(body["data"], path=f"{path}.data")

    return body["dtype"], shape, raw


def _check_array_bytes(
    raw: bytes, *, shape: List[int], item_size: int, dtype_name: str, path: str
) -> None:
    _require_little_endian_host()
    expected_bytes = math.prod(shape) * item_size
    if len(raw) != expected_bytes:
        raise ContractError(
            f"{path}: {dtype_name} data of shape {shape} needs {expected_bytes} bytes, "
            f"got {len(raw)}"
        )
    if dtype_name == "bool" and raw.translate(None, b"\x00\x01"):
        raise ContractError(f"{path}: bool data may hold only bytes 0 and 1")


def _base64(value: Any, *, path: str) -> bytes:
    if not isinstance(value, str):
        raise ContractError(f"{path}: expected a base64 string, got {_describe(value)}")

    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ContractError(f"{path}: invalid base64 ({error})") from error

    return decoded


def _require_type(value: Any, expected: type, *, path: str) -> Any:
    if not isinstance(value, expected):
        raise ContractError(
            f"{path}: expected a JSON {'object' if expected is dict else 'array'}, got "
            f"{_describe(value)}"
        )

    return value


def _require_little_endian_host() -> None:
    # Array data are little-endian on the wire and read through native views.
    if sys.byteorder != "little":
        raise ContractError("The native wire format needs a little-endian host")


def _describe(value: Any) -> str:
    """Name a wire value by its JSON type, never by its content."""
    json_types = {dict: "object", list: "array", str: "string", bool: "boolean"}
    if value is None:
        return "null"

    description = json_types.get(type(value), type(value).__name__)

    return description
