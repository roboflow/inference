"""Generic recording codecs: the tagged grammar, binary built-ins, registries."""

import enum
import json
import math
import zlib
from collections import OrderedDict
from typing import Any, List

import numpy as np
import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.errors import CatalogueError
from roboflow_workflows.execution_engine.v2.recording import (
    BUILTIN_CODECS,
    CodecRegistry,
    PayloadCodec,
    RecordingCodecError,
    RecordingCorruptError,
    type_name_of,
)


class MemoryBlobs:
    """In-memory blob sink and source with the store's reference shape."""

    def __init__(self) -> None:
        self.data = bytearray()
        self.puts: List[bytes] = []

    def put(self, data) -> list:
        raw = bytes(data)
        self.puts.append(raw)
        ref = [len(self.data), len(raw), zlib.crc32(raw)]
        self.data += raw
        return ref

    def get(self, ref) -> bytearray:
        offset, length, _ = ref
        return bytearray(self.data[offset : offset + length])


def round_trip(value: Any, registry: CodecRegistry = None) -> Any:
    registry = registry or CodecRegistry()
    blobs = MemoryBlobs()
    node = registry.encoder(blobs, location="g.f").encode(value)
    text = json.dumps(node, allow_nan=False)
    decoded = registry.decoder(blobs, location="g.f").decode(json.loads(text))
    return decoded


class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x, self.y = x, y


def encode_point(point: Point, sink, encoder) -> Any:
    return {"xy": encoder.encode((point.x, point.y), path=".xy")}


def decode_point(node: Any, source, decoder) -> Point:
    x, y = decoder.decode(node["xy"], path=".xy")
    return Point(x, y)


POINT_CODEC = PayloadCodec(
    name="test/point@1",
    type_name=type_name_of(Point),
    encode=encode_point,
    decode=decode_point,
)


# Grammar -------------------------------------------------------------------


def test_plain_values_keep_their_exact_python_types() -> None:
    value = {
        "none": None,
        "flags": [True, False, 0, 1],
        "floats": (0.1, -0.0, 1e300, math.inf, -math.inf),
        "text": "tab\tand\nnewline ünïcode",
        "nested": [(1, [2, (3,)]), {"inner": ()}],
        "sets": [{1, 2}, frozenset({"a"})],
    }

    decoded = round_trip(value)

    assert decoded == value
    assert [type(item) for item in decoded["flags"]] == [bool, bool, int, int]
    assert type(decoded["floats"]) is tuple
    assert math.copysign(1.0, decoded["floats"][1]) == -1.0
    assert type(decoded["sets"][1]) is frozenset
    assert type(decoded["nested"][0][1][1]) is tuple


def test_nan_round_trips_as_nan() -> None:
    decoded = round_trip([math.nan])

    assert math.isnan(decoded[0])


def test_int_and_mixed_key_dicts_keep_keys_and_order() -> None:
    value = {2: "b", 1: "a", "x": {0: None}}

    decoded = round_trip(value)

    assert decoded == value
    assert list(decoded) == [2, 1, "x"]


def test_nested_batches_keep_indices_gaps_and_empty_parents() -> None:
    inner_with_gap = Batch(["a", "c"], indices=[(0, 0), (0, 2)], parent_index=(0,))
    empty_inner = Batch.empty(parent_index=(2,))
    outer = Batch([inner_with_gap, empty_inner], indices=[(0,), (2,)])

    decoded = round_trip(outer)

    assert decoded == outer
    assert decoded.indices == ((0,), (2,))
    assert decoded.content[0].indices == ((0, 0), (0, 2))
    assert decoded.content[1].parent_index == (2,)
    assert len(decoded.content[1]) == 0


def test_bytes_go_to_the_blob_not_to_json() -> None:
    payload = bytes(range(256)) * 4
    registry = CodecRegistry()
    blobs = MemoryBlobs()

    node = registry.encoder(blobs).encode({"raw": payload})

    assert blobs.puts == [payload]
    assert len(json.dumps(node)) < 200
    assert registry.decoder(blobs).decode(node) == {"raw": payload}


@pytest.mark.parametrize(
    "array",
    [
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        np.arange(24, dtype=">i4").reshape(4, 6),
        np.arange(60, dtype=np.uint16).reshape(6, 10)[::2, 1::3],
        np.zeros((0, 8), dtype=np.float16),
        np.array([True, False, True]),
        np.array(7, dtype=np.int64),
    ],
    ids=["float32", "big-endian", "non-contiguous", "zero-size", "bool", "0-d"],
)
def test_numpy_arrays_round_trip_exactly_as_raw_bytes(array: np.ndarray) -> None:
    registry = CodecRegistry()
    blobs = MemoryBlobs()

    node = registry.encoder(blobs).encode(array)
    decoded = registry.decoder(blobs).decode(json.loads(json.dumps(node)))

    assert decoded.dtype.name == array.dtype.name
    assert decoded.shape == array.shape
    np.testing.assert_array_equal(decoded, array)
    assert blobs.puts == [
        np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<")).tobytes()
    ]
    assert decoded.flags.writeable


def test_numpy_scalars_keep_their_dtype() -> None:
    value = [np.float32(1.5), np.int16(-3), np.bool_(True)]

    decoded = round_trip(value)

    assert [type(item) for item in decoded] == [type(item) for item in value]
    assert decoded == value


def test_array_decode_wraps_an_owned_read_buffer_and_copies_immutable_bytes() -> None:
    registry = CodecRegistry()
    blobs = MemoryBlobs()
    node = registry.encoder(blobs).encode(np.arange(4, dtype=np.uint8))
    returned = []

    class OwnedBuffers:
        def get(self, ref) -> bytearray:
            returned.append(blobs.get(ref))
            return returned[-1]

    class ImmutableBytes:
        def get(self, ref) -> bytes:
            return bytes(returned[0])

    wrapped = registry.decoder(OwnedBuffers()).decode(node)
    returned[0][0] = 9
    copied = registry.decoder(ImmutableBytes()).decode(node)
    copied[1] = 7

    assert wrapped.tolist() == [9, 1, 2, 3]
    assert copied.tolist() == [9, 7, 2, 3]
    assert returned[0] == bytearray([9, 1, 2, 3])


def test_bool_arrays_with_bytes_other_than_0_and_1_are_corruption() -> None:
    registry = CodecRegistry()
    blobs = MemoryBlobs()
    node = registry.encoder(blobs).encode(np.array([True, False, True]))
    empty = registry.encoder(blobs).encode(np.zeros((0, 3), dtype=bool))
    blobs.data[1] = 2

    with pytest.raises(RecordingCorruptError, match="^g\\.f: .*only bytes 0 and 1"):
        registry.decoder(blobs, location="g.f").decode(node)
    assert registry.decoder(blobs).decode(empty).shape == (0, 3)


def test_decoded_arrays_are_independent_of_each_other() -> None:
    registry = CodecRegistry()
    blobs = MemoryBlobs()
    node = registry.encoder(blobs).encode(np.zeros(4))

    first = registry.decoder(blobs).decode(node)
    first[0] = 5.0
    second = registry.decoder(blobs).decode(node)

    assert second[0] == 0.0


# Explicit failures -----------------------------------------------------------


class Color(enum.IntEnum):
    RED = 1


@pytest.mark.parametrize(
    "value, type_name",
    [
        (object(), "builtins.object"),
        (Color.RED, f"{__name__}.Color"),
        (OrderedDict(a=1), "collections.OrderedDict"),
        (np.array([1 + 2j]), None),
    ],
    ids=["object", "int-subclass", "dict-subclass", "complex-array"],
)
def test_unsupported_payloads_fail_with_their_location(
    value: Any, type_name: str
) -> None:
    encoder = CodecRegistry().encoder(MemoryBlobs(), location="analysis.frame")

    with pytest.raises(RecordingCodecError) as caught:
        encoder.encode({"items": [1, value]})

    message = str(caught.value)
    assert "analysis.frame['items'][1]" in message
    if type_name is not None:
        assert f"no recording codec for {type_name}" in message
        assert "register a PayloadCodec on the catalogue" in message


def test_non_string_dict_keys_are_rejected() -> None:
    with pytest.raises(RecordingCodecError, match="dict keys must be str or int"):
        CodecRegistry().encoder(MemoryBlobs()).encode({(1, 2): "x"})


def test_codec_failures_are_wrapped_with_codec_name_and_path() -> None:
    def broken(value, sink, encoder):
        raise ValueError("boom")

    registry = CodecRegistry(
        [PayloadCodec("test/broken@1", type_name_of(Point), broken, decode_point)]
    )

    with pytest.raises(RecordingCodecError) as caught:
        registry.encoder(MemoryBlobs(), location="g.f").encode([Point(1, 2)])

    assert "g.f[0]" in str(caught.value)
    assert "test/broken@1" in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)


def test_nested_codec_values_report_the_full_path() -> None:
    registry = CodecRegistry([POINT_CODEC])
    point = Point(1.0, object())

    with pytest.raises(RecordingCodecError, match=r"g\.f\[0\]\.xy\[1\]"):
        registry.encoder(MemoryBlobs(), location="g.f").encode([point])


def test_unknown_codec_name_on_read_is_a_codec_error() -> None:
    blobs = MemoryBlobs()
    node = CodecRegistry([POINT_CODEC]).encoder(blobs).encode(Point(1.0, 2.0))

    with pytest.raises(
        RecordingCodecError, match="unknown recording codec 'test/point@1'"
    ):
        CodecRegistry().decoder(blobs, location="g[3].f").decode(node)


@pytest.mark.parametrize(
    "node",
    [
        {"unknown": 1},
        {"a": 1, "b": 2},
        {"float": "huge"},
        {"batch": {"parent": [], "indices": [[0], [0]], "items": [1, 2]}},
        {"batch": {"parent": [], "indices": [[-1]], "items": [1]}},
        {"codec": {"name": 3, "value": None}},
        {"map": [[1.5, 2]]},
    ],
)
def test_nodes_outside_the_grammar_are_corruption(node: Any) -> None:
    with pytest.raises(RecordingCorruptError, match=r"^g\[3\]\.f"):
        CodecRegistry().decoder(MemoryBlobs(), location="g[3].f").decode(node)


@pytest.mark.parametrize(
    "indices, items, counts",
    [
        ([], ["discarded"], "0 indices and 1 items"),
        ([[0]], [1, 2], "1 indices and 2 items"),
        ([[0], [1]], [1], "2 indices and 1 items"),
    ],
)
def test_batch_indices_and_items_must_have_equal_counts(
    indices: list, items: list, counts: str
) -> None:
    node = {"batch": {"parent": [], "indices": indices, "items": items}}

    with pytest.raises(RecordingCorruptError, match=rf"^g\[3\]\.f: batch has {counts}"):
        CodecRegistry().decoder(MemoryBlobs(), location="g[3].f").decode(node)


# Custom codecs and registries -----------------------------------------------------


def test_custom_codecs_recurse_and_record_their_use() -> None:
    registry = CodecRegistry([POINT_CODEC])
    blobs = MemoryBlobs()
    encoder = registry.encoder(blobs)

    node = encoder.encode({"p": Point(1.5, math.inf)})
    decoded = registry.decoder(blobs).decode(node)

    assert (decoded["p"].x, decoded["p"].y) == (1.5, math.inf)
    assert encoder.used_codecs == {"test/point@1"}


def test_registry_rejects_conflicting_codecs() -> None:
    other = PayloadCodec("test/point@1", "x.Other", encode_point, decode_point)
    second_for_type = PayloadCodec(
        "test/point@2", type_name_of(Point), encode_point, decode_point
    )
    bytes_override = PayloadCodec(
        "test/bytes@1", "builtins.bytes", encode_point, decode_point
    )

    with pytest.raises(RecordingCodecError, match="Two different recording codecs"):
        CodecRegistry([POINT_CODEC, other])
    with pytest.raises(RecordingCodecError, match="both encode"):
        CodecRegistry([POINT_CODEC, second_for_type])
    with pytest.raises(RecordingCodecError, match="both encode builtins.bytes"):
        CodecRegistry([bytes_override])
    CodecRegistry([POINT_CODEC, POINT_CODEC])


def test_require_names_every_missing_codec() -> None:
    registry = CodecRegistry([POINT_CODEC])

    registry.require(["test/point@1", "v2/bytes@1"])
    with pytest.raises(RecordingCodecError, match=r"\['a@1', 'b@1'\]"):
        registry.require(["a@1", "test/point@1", "b@1"], directory="/tmp/rec")


def test_codec_declarations_are_validated() -> None:
    with pytest.raises(RecordingCodecError, match="name must be a non-empty string"):
        PayloadCodec("", "x.Y", encode_point, decode_point)
    with pytest.raises(RecordingCodecError, match="encode must be callable"):
        PayloadCodec("a@1", "x.Y", None, decode_point)


def test_catalogue_collects_merges_and_validates_codecs() -> None:
    catalogue = Catalogue(codecs=[POINT_CODEC], namespace="demo")
    merged = Catalogue.merge(catalogue, Catalogue(codecs=[POINT_CODEC]), Catalogue())

    assert dict(merged.codecs) == {"test/point@1": POINT_CODEC}
    assert dict(Catalogue().codecs) == {}
    assert dict(catalogue.with_blocks([]).codecs) == {"test/point@1": POINT_CODEC}

    rival = PayloadCodec(
        "test/point@2", type_name_of(Point), encode_point, decode_point
    )
    with pytest.raises(CatalogueError, match="both encode"):
        Catalogue.merge(catalogue, Catalogue(codecs=[rival]))
    with pytest.raises(CatalogueError, match="built-in codec"):
        Catalogue(
            codecs=[
                PayloadCodec(BUILTIN_CODECS[0].name, "x.Y", encode_point, decode_point)
            ]
        )
    with pytest.raises(CatalogueError, match="must be PayloadCodec"):
        Catalogue(codecs=["not a codec"])
