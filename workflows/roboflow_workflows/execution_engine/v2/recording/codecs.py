"""Type-keyed payload codecs of the recording store.

A recorded value is a JSON node plus raw bytes in a blob file. Every JSON
object is a tagged node with exactly one key, so payload keys can never be
mistaken for a tag::

    null | bool | int | finite float | str      the same Python scalar
    [node, ...]                                  list
    {"float": "nan" | "inf" | "-inf"}            non-finite float
    {"tuple": [node, ...]}                       tuple
    {"set": [node, ...]}  {"frozenset": [...]}   set / frozenset
    {"dict": {"<str>": node, ...}}               dict with str keys
    {"map": [[key, node], ...]}                  dict with int (or mixed) keys
    {"batch": {"parent": [..], "indices": [[..], ...], "items": [node, ...]}}
    {"codec": {"name": "<codec name>", "value": <codec JSON>}}

Anything else goes through a ``PayloadCodec`` chosen by the value's exact type
name; subclasses need their own codec. Codecs write bulk data with
``BlobSink.put`` and keep only the returned opaque reference in JSON, so
tensors and images are stored as raw bytes, never as base64 text.

The generic built-in codecs handle ``bytes``, ``numpy.ndarray`` and NumPy
scalars. NumPy is imported only when one of them runs. Domain codecs, such as
the native image and prediction codecs, are registered on a ``Catalogue``.
"""

import math
import sys
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)

from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.recording.errors import (
    RecordingCodecError,
    RecordingCorruptError,
    RecordingError,
)

JsonValue = Any
"""``None``, bool, int, float, str, list or str-keyed dict of JSON values."""

BlobRef = JsonValue
"""Opaque JSON reference to bytes stored by a ``BlobSink``."""

BYTES_CODEC_NAME = "v2/bytes@1"
NDARRAY_CODEC_NAME = "v2/ndarray@1"
NUMPY_SCALAR_CODEC_NAME = "v2/numpy_scalar@1"
NUMPY_DTYPES = (
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

_NON_FINITE_FLOATS = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


class BlobSink(Protocol):
    """Destination of raw bytes written while encoding one record."""

    def put(self, data: Union[bytes, bytearray, memoryview]) -> BlobRef:
        """Store bytes and return a JSON reference to them."""


class BlobSource(Protocol):
    """Read side of a ``BlobSink``."""

    def get(self, ref: BlobRef) -> bytes:
        """Return the bytes of a reference, verified for bounds and integrity.

        A returned ``bytearray`` is fresh and owned by the caller, so decoders
        may wrap it without copying. Immutable ``bytes`` must be copied first.
        """


@dataclass(frozen=True)
class PayloadCodec:
    """Encoder and decoder of one exact payload type.

    Args:
        name: Stable identity written to disk, e.g. ``"v2/image_data@1"``.
            Change the version suffix when the JSON form changes.
        type_name: Exact ``f"{cls.__module__}.{cls.__qualname__}"`` of the
            encoded type; see ``type_name_of``.
        encode: ``encode(value, sink, encoder) -> JSON``. Store bulk data with
            ``sink.put`` and nested values with ``encoder.encode``.
        decode: ``decode(node, source, decoder) -> value``. Read bulk data with
            ``source.get`` and nested values with ``decoder.decode``. Decoded
            values must be independently owned and on the CPU.

    Raises:
        RecordingCodecError: On an empty name or type name, or a non-callable
            encoder or decoder.
    """

    name: str
    type_name: str
    encode: Callable[[Any, BlobSink, "PayloadEncoder"], JsonValue]
    decode: Callable[[JsonValue, BlobSource, "PayloadDecoder"], Any]

    def __post_init__(self) -> None:
        for label in ("name", "type_name"):
            if not isinstance(getattr(self, label), str) or not getattr(self, label):
                raise RecordingCodecError(
                    f"PayloadCodec {label} must be a non-empty string, "
                    f"got {getattr(self, label)!r}"
                )
        for label in ("encode", "decode"):
            if not callable(getattr(self, label)):
                raise RecordingCodecError(
                    f"PayloadCodec {self.name!r} {label} must be callable"
                )


def type_name_of(value_type: type) -> str:
    """Return the exact type name codecs are looked up by.

    Args:
        value_type: Payload class.

    Returns:
        ``f"{value_type.__module__}.{value_type.__qualname__}"``.
    """
    type_name = f"{value_type.__module__}.{value_type.__qualname__}"

    return type_name


class CodecRegistry:
    """Immutable set of payload codecs, including the generic built-ins.

    Args:
        codecs: Additional codecs, usually ``catalogue.codecs.values()``.

    Raises:
        RecordingCodecError: When two codecs share a name or a type name, or a
            codec replaces a built-in.
    """

    def __init__(self, codecs: Iterable[PayloadCodec] = ()):
        self._by_name: Dict[str, PayloadCodec] = {}
        self._by_type: Dict[str, PayloadCodec] = {}
        for codec in (*BUILTIN_CODECS, *codecs):
            problem = codec_conflict(
                codec, by_name=self._by_name, by_type=self._by_type
            )
            if problem is not None:
                raise RecordingCodecError(problem)
            self._by_name[codec.name] = codec
            self._by_type[codec.type_name] = codec

    @property
    def names(self) -> Tuple[str, ...]:
        """Codec names in registration order."""
        return tuple(self._by_name)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def require(self, names: Iterable[str], *, directory: Any = None) -> None:
        """Check that every named codec is available.

        Args:
            names: Codec names a recording uses.
            directory: Recording directory, for the error location.

        Raises:
            RecordingCodecError: Naming every missing codec.
        """
        missing = [name for name in names if name not in self._by_name]
        if missing:
            raise RecordingCodecError(
                f"the recording uses codecs {missing} that are not registered; "
                "register them on the catalogue",
                directory=directory,
            )

    def encoder(self, sink: BlobSink, *, location: str = "") -> "PayloadEncoder":
        """Create an encoder writing bulk data to ``sink``.

        Args:
            sink: Blob destination of the record being encoded.
            location: Path prefix of error messages, e.g. ``"group.field"``.

        Returns:
            A new encoder; it records which codecs it used.
        """
        encoder = PayloadEncoder(self, sink=sink, location=location, used=set())

        return encoder

    def decoder(self, source: BlobSource, *, location: str = "") -> "PayloadDecoder":
        """Create a decoder reading bulk data from ``source``.

        Args:
            source: Blob source of the record being decoded.
            location: Path prefix of error messages.

        Returns:
            A new decoder.
        """
        decoder = PayloadDecoder(self, source=source, location=location)

        return decoder

    def _for_type(self, value_type: type) -> Optional[PayloadCodec]:
        codec = self._by_type.get(type_name_of(value_type))
        if codec is None and value_type.__module__ == "numpy":
            codec = _numpy_scalar_codec_for(value_type)

        return codec


class PayloadEncoder:
    """Encodes values into the tagged JSON grammar and blob data.

    Codecs receive an encoder positioned at their value; calling ``encode``
    with a relative ``path`` such as ``".xyxy"`` extends error locations.
    """

    def __init__(
        self,
        registry: CodecRegistry,
        *,
        sink: BlobSink,
        location: str,
        used: Set[str],
    ):
        self._registry = registry
        self._sink = sink
        self._location = location
        self._used = used

    @property
    def used_codecs(self) -> Set[str]:
        """Names of the codecs this encoder (and its nested encoders) used."""
        return self._used

    def encode(self, value: Any, *, path: str = "") -> JsonValue:
        """Encode one value.

        Args:
            value: Value to encode.
            path: Location of ``value`` relative to this encoder.

        Returns:
            The JSON node.

        Raises:
            RecordingCodecError: When no codec handles a value or a codec
                fails; the message names the value's location and type.
        """
        node = self._encode(value, path=self._location + path)

        return node

    def _encode(self, value: Any, *, path: str) -> JsonValue:
        value_type = type(value)
        if value is None or value_type in (bool, int, str):
            return value
        if value_type is float:
            if math.isfinite(value):
                return value
            return {"float": "nan" if math.isnan(value) else str(value)}
        if value_type is list:
            return [
                self._encode(item, path=f"{path}[{position}]")
                for position, item in enumerate(value)
            ]
        if value_type is tuple:
            return {"tuple": self._encode_items(value, path=path)}
        if value_type in (set, frozenset):
            return {value_type.__name__: self._encode_items(value, path=path)}
        if value_type is dict:
            return self._encode_dict(value, path=path)
        if value_type is Batch:
            return self._encode_batch(value, path=path)

        codec = self._registry._for_type(value_type)
        if codec is None:
            raise _located(
                RecordingCodecError,
                f"{path}: no recording codec for {type_name_of(value_type)}; "
                "register a PayloadCodec on the catalogue",
            )

        try:
            codec_value = codec.encode(value, self._sink, self._at(path))
        except Exception as error:
            if getattr(error, "located", False):
                raise
            raise _located(
                RecordingCodecError,
                f"{path}: codec {codec.name!r} cannot encode "
                f"{type_name_of(value_type)}: {error}",
            ) from error

        self._used.add(codec.name)
        node = {"codec": {"name": codec.name, "value": codec_value}}

        return node

    def _encode_items(self, items: Iterable[Any], *, path: str) -> List[JsonValue]:
        encoded = [
            self._encode(item, path=f"{path}[{position}]")
            for position, item in enumerate(items)
        ]

        return encoded

    def _encode_dict(self, value: dict, *, path: str) -> JsonValue:
        for key in value:
            if type(key) not in (str, int):
                raise _located(
                    RecordingCodecError,
                    f"{path}: dict keys must be str or int, got {type(key).__name__}",
                )

        items = {
            key: self._encode(item, path=f"{path}[{key!r}]")
            for key, item in value.items()
        }
        if all(type(key) is str for key in items):
            return {"dict": items}

        encoded = {"map": [[key, item] for key, item in items.items()]}

        return encoded

    def _encode_batch(self, batch: Batch, *, path: str) -> JsonValue:
        items = [
            self._encode(item, path=f"{path}{list(index)}")
            for index, item in batch.iter_with_indices()
        ]
        encoded = {
            "batch": {
                "parent": list(batch.parent_index),
                "indices": [list(index) for index in batch.indices],
                "items": items,
            }
        }

        return encoded

    def _at(self, path: str) -> "PayloadEncoder":
        encoder = PayloadEncoder(
            self._registry, sink=self._sink, location=path, used=self._used
        )

        return encoder


class PayloadDecoder:
    """Decodes the tagged JSON grammar written by ``PayloadEncoder``.

    Decoded ``Batch`` trees carry no layout or metadata view; the store
    attaches the recorded entry's views.
    """

    def __init__(self, registry: CodecRegistry, *, source: BlobSource, location: str):
        self._registry = registry
        self._source = source
        self._location = location

    def decode(self, node: JsonValue, *, path: str = "") -> Any:
        """Decode one node.

        Args:
            node: JSON node.
            path: Location of ``node`` relative to this decoder.

        Returns:
            The value; tensors and arrays are new CPU objects.

        Raises:
            RecordingCodecError: On an unknown codec name or a codec failure.
            RecordingCorruptError: On a node outside the grammar.
        """
        value = self._decode(node, path=self._location + path)

        return value

    def _decode(self, node: JsonValue, *, path: str) -> Any:
        node_type = type(node)
        if node is None or node_type in (bool, int, float, str):
            return node
        if node_type is list:
            return [
                self._decode(item, path=f"{path}[{position}]")
                for position, item in enumerate(node)
            ]
        if node_type is not dict or len(node) != 1:
            raise _located(
                RecordingCorruptError, f"{path}: expected a JSON value or a tagged node"
            )

        ((tag, body),) = node.items()
        if tag == "float":
            if body not in _NON_FINITE_FLOATS:
                raise _located(
                    RecordingCorruptError, f"{path}: invalid non-finite float {body!r}"
                )
            return _NON_FINITE_FLOATS[body]
        if tag in ("tuple", "set", "frozenset"):
            items = self._decode_items(_require(body, list, path=path), path=path)
            return {"tuple": tuple, "set": set, "frozenset": frozenset}[tag](items)
        if tag == "dict":
            items = _require(body, dict, path=path)
            return {
                key: self._decode(item, path=f"{path}[{key!r}]")
                for key, item in items.items()
            }
        if tag == "map":
            return self._decode_map(body, path=path)
        if tag == "batch":
            return self._decode_batch(body, path=path)
        if tag == "codec":
            return self._decode_codec(body, path=path)

        raise _located(RecordingCorruptError, f"{path}: unknown tag {tag!r}")

    def _decode_items(self, items: List[JsonValue], *, path: str) -> List[Any]:
        decoded = [
            self._decode(item, path=f"{path}[{position}]")
            for position, item in enumerate(items)
        ]

        return decoded

    def _decode_map(self, body: JsonValue, *, path: str) -> Dict[Any, Any]:
        decoded = {}
        for pair in _require(body, list, path=path):
            if (
                type(pair) is not list
                or len(pair) != 2
                or type(pair[0])
                not in (
                    str,
                    int,
                )
            ):
                raise _located(
                    RecordingCorruptError,
                    f"{path}: map entries must be [str|int key, value] pairs",
                )
            key, item = pair
            decoded[key] = self._decode(item, path=f"{path}[{key!r}]")

        return decoded

    def _decode_batch(self, body: JsonValue, *, path: str) -> Batch:
        body = _require(body, dict, path=path)
        if set(body) != {"parent", "indices", "items"}:
            raise _located(
                RecordingCorruptError,
                f"{path}: batch needs exactly 'parent', 'indices' and 'items'",
            )

        parent = _index(body["parent"], path=path)
        indices = [
            _index(index, path=path)
            for index in _require(body["indices"], list, path=path)
        ]
        encoded_items = _require(body["items"], list, path=path)
        if len(encoded_items) != len(indices):
            raise _located(
                RecordingCorruptError,
                f"{path}: batch has {len(indices)} indices and "
                f"{len(encoded_items)} items",
            )

        items = [
            self._decode(item, path=f"{path}{list(index)}")
            for index, item in zip(indices, encoded_items)
        ]
        try:
            batch = Batch(items, indices=indices, parent_index=parent)
        except ContractError as error:
            raise _located(
                RecordingCorruptError, f"{path}: invalid batch: {error}"
            ) from error

        return batch

    def _decode_codec(self, body: JsonValue, *, path: str) -> Any:
        body = _require(body, dict, path=path)
        if set(body) != {"name", "value"} or type(body["name"]) is not str:
            raise _located(
                RecordingCorruptError,
                f"{path}: codec node needs exactly a string 'name' and a 'value'",
            )

        codec = self._registry._by_name.get(body["name"])
        if codec is None:
            raise _located(
                RecordingCodecError,
                f"{path}: unknown recording codec {body['name']!r}; register it "
                "on the catalogue",
            )

        try:
            value = codec.decode(body["value"], self._source, self._at(path))
        except Exception as error:
            if getattr(error, "located", False):
                raise
            error_class = (
                RecordingCorruptError
                if isinstance(error, RecordingCorruptError)
                else RecordingCodecError
            )
            raise _located(
                error_class, f"{path}: codec {codec.name!r} cannot decode: {error}"
            ) from error

        return value

    def _at(self, path: str) -> "PayloadDecoder":
        decoder = PayloadDecoder(self._registry, source=self._source, location=path)

        return decoder


def codec_conflict(
    codec: Any,
    *,
    by_name: Mapping[str, PayloadCodec],
    by_type: Mapping[str, PayloadCodec],
) -> Optional[str]:
    """Explain why ``codec`` cannot join a set of codecs.

    Args:
        codec: Candidate codec.
        by_name: Registered codecs by name.
        by_type: Registered codecs by type name.

    Returns:
        ``None`` when ``codec`` is new or already registered, otherwise a
        message. Built-in codec names and types are always taken.
    """
    if not isinstance(codec, PayloadCodec):
        return f"Recording codecs must be PayloadCodec objects, got {codec!r}"

    known = by_name.get(codec.name)
    if known is not None:
        if known == codec:
            return None
        return f"Two different recording codecs are named {codec.name!r}"

    owner = by_type.get(codec.type_name)
    if owner is not None:
        return (
            f"Recording codecs {owner.name!r} and {codec.name!r} both encode "
            f"{codec.type_name}"
        )

    builtin = _BUILTIN_BY_NAME.get(codec.name) or _BUILTIN_BY_TYPE.get(codec.type_name)
    if builtin is not None and builtin != codec:
        return (
            f"Recording codec {codec.name!r} for {codec.type_name} conflicts with "
            f"the built-in codec {builtin.name!r}"
        )

    return None


def _located(error_class: type, message: str) -> RecordingError:
    """Create an error whose message already starts with its value path."""
    error = error_class(message)
    error.located = True

    return error


def _require(value: Any, expected: type, *, path: str) -> Any:
    if type(value) is not expected:
        raise _located(
            RecordingCorruptError,
            f"{path}: expected a JSON {'object' if expected is dict else 'array'}",
        )

    return value


def _index(value: Any, *, path: str) -> Tuple[int, ...]:
    if type(value) is not list or not all(
        type(component) is int and component >= 0 for component in value
    ):
        raise _located(
            RecordingCorruptError,
            f"{path}: an index must be a list of non-negative integers",
        )

    return tuple(value)


# ---------------------------------------------------------------------------
# Generic built-in codecs
# ---------------------------------------------------------------------------


def _encode_bytes(value: bytes, sink: BlobSink, encoder: PayloadEncoder) -> JsonValue:
    ref = sink.put(value)

    return ref


def _decode_bytes(
    node: JsonValue, source: BlobSource, decoder: PayloadDecoder
) -> bytes:
    data = bytes(source.get(node))

    return data


def encode_array_bytes(
    array: Any, sink: BlobSink, *, dtype_names: Tuple[str, ...] = NUMPY_DTYPES
) -> JsonValue:
    """Store a NumPy array's elements as raw little-endian C-order bytes.

    Args:
        array: NumPy array of any memory layout.
        sink: Blob destination.
        dtype_names: Accepted dtype names.

    Returns:
        ``{"dtype", "shape", "data": <blob ref>}``.

    Raises:
        RecordingCodecError: On an unsupported dtype or a big-endian host.
    """
    import numpy as np

    _require_little_endian_host()
    if array.dtype.name not in dtype_names:
        raise RecordingCodecError(
            f"numpy dtype {array.dtype} is not supported; supported dtypes: "
            f"{', '.join(dtype_names)}"
        )

    contiguous = np.ascontiguousarray(
        array.astype(array.dtype.newbyteorder("<"), copy=False)
    )
    node = {
        "dtype": array.dtype.name,
        "shape": [int(size) for size in array.shape],
        "data": sink.put(memoryview(contiguous.reshape(-1)).cast("B")),
    }

    return node


def decode_array_bytes(
    node: JsonValue, source: BlobSource, *, dtype_names: Tuple[str, ...] = NUMPY_DTYPES
) -> Any:
    """Rebuild a writable NumPy array written by ``encode_array_bytes``.

    Args:
        node: ``{"dtype", "shape", "data"}`` node.
        source: Blob source.
        dtype_names: Accepted dtype names.

    Returns:
        A new, independently owned array.

    Raises:
        RecordingCorruptError: When the node is malformed or the byte count
            does not match dtype and shape.
    """
    import numpy as np

    dtype_name, shape = read_array_header(node, dtype_names=dtype_names)
    dtype = np.dtype(dtype_name)
    raw = source.get(node["data"])
    check_array_bytes(raw, shape=shape, item_size=dtype.itemsize, dtype_name=dtype_name)

    owned = raw if isinstance(raw, bytearray) else bytearray(raw)
    array = np.frombuffer(owned, dtype=dtype.newbyteorder("<")).reshape(shape)
    array = array.astype(dtype, copy=False)

    return array


def read_array_header(
    node: JsonValue, *, dtype_names: Tuple[str, ...]
) -> Tuple[str, List[int]]:
    """Validate the ``dtype`` and ``shape`` of a raw array node.

    Args:
        node: ``{"dtype", "shape", "data"}`` node.
        dtype_names: Accepted dtype names.

    Returns:
        The dtype name and shape.

    Raises:
        RecordingCorruptError: On a malformed node or an unknown dtype.
    """
    if type(node) is not dict or set(node) != {"dtype", "shape", "data"}:
        raise RecordingCorruptError(
            "an array node needs exactly 'dtype', 'shape' and 'data'"
        )

    shape = node["shape"]
    if type(shape) is not list or not all(
        type(size) is int and size >= 0 for size in shape
    ):
        raise RecordingCorruptError("an array shape must be non-negative integers")
    if node["dtype"] not in dtype_names:
        raise RecordingCorruptError(f"unsupported recorded dtype {node['dtype']!r}")

    return node["dtype"], shape


def check_array_bytes(
    raw: bytes, *, shape: List[int], item_size: int, dtype_name: str
) -> None:
    """Check that raw bytes fit an array's dtype and shape.

    Args:
        raw: Element bytes.
        shape: Array shape.
        item_size: Bytes per element.
        dtype_name: Dtype name; ``bool`` data may hold only bytes 0 and 1.

    Raises:
        RecordingCorruptError: On a size mismatch or invalid bool bytes.
    """
    _require_little_endian_host()
    expected = math.prod(shape) * item_size
    if len(raw) != expected:
        raise RecordingCorruptError(
            f"{dtype_name} data of shape {shape} needs {expected} bytes, got {len(raw)}"
        )
    if dtype_name == "bool" and raw:
        import numpy as np

        # A reduction over a byte view of the buffer; no full-size temporary.
        if np.frombuffer(raw, dtype=np.uint8).max() > 1:
            raise RecordingCorruptError("bool data may hold only bytes 0 and 1")


def _encode_ndarray(value: Any, sink: BlobSink, encoder: PayloadEncoder) -> JsonValue:
    node = encode_array_bytes(value, sink)

    return node


def _decode_ndarray(
    node: JsonValue, source: BlobSource, decoder: PayloadDecoder
) -> Any:
    array = decode_array_bytes(node, source)

    return array


def _encode_numpy_scalar(
    value: Any, sink: BlobSink, encoder: PayloadEncoder
) -> JsonValue:
    import numpy as np

    node = encode_array_bytes(np.asarray(value), sink)

    return node


def _decode_numpy_scalar(
    node: JsonValue, source: BlobSource, decoder: PayloadDecoder
) -> Any:
    array = decode_array_bytes(node, source)
    if array.shape != ():
        raise RecordingCorruptError(
            f"a numpy scalar needs shape [], got {list(array.shape)}"
        )

    scalar = array[()]

    return scalar


def _numpy_scalar_codec_for(value_type: type) -> Optional[PayloadCodec]:
    import numpy as np

    if issubclass(value_type, np.generic) and np.dtype(value_type).name in NUMPY_DTYPES:
        return NUMPY_SCALAR_CODEC

    return None


def _require_little_endian_host() -> None:
    if sys.byteorder != "little":
        raise RecordingCodecError("Recordings need a little-endian host")


BYTES_CODEC = PayloadCodec(
    name=BYTES_CODEC_NAME,
    type_name=type_name_of(bytes),
    encode=_encode_bytes,
    decode=_decode_bytes,
)
NDARRAY_CODEC = PayloadCodec(
    name=NDARRAY_CODEC_NAME,
    type_name="numpy.ndarray",
    encode=_encode_ndarray,
    decode=_decode_ndarray,
)
NUMPY_SCALAR_CODEC = PayloadCodec(
    name=NUMPY_SCALAR_CODEC_NAME,
    type_name="numpy.generic",
    encode=_encode_numpy_scalar,
    decode=_decode_numpy_scalar,
)
BUILTIN_CODECS: Tuple[PayloadCodec, ...] = (
    BYTES_CODEC,
    NDARRAY_CODEC,
    NUMPY_SCALAR_CODEC,
)

_BUILTIN_BY_NAME = {codec.name: codec for codec in BUILTIN_CODECS}
_BUILTIN_BY_TYPE = {codec.type_name: codec for codec in BUILTIN_CODECS}
