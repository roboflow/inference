import enum
import itertools
import math

import pytest
from roboflow_workflows.execution_engine.v2.state.codec import (
    INT64_MAX,
    INT64_MIN,
    decode_value,
    encode_value,
    global_storage_key,
    machine_storage_key,
    source_storage_key,
    validate_amount,
    validate_name,
)
from roboflow_workflows.execution_engine.v2.state.errors import StateValueError


class _Level(enum.IntEnum):
    LOW = 1


class _Text(str):
    pass


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        -7,
        INT64_MIN,
        INT64_MAX,
        1.5,
        -0.0,
        1e300,
        "",
        "żółw \ud800 \u0000",
        [],
        {},
        {"b": [1, {"c": None}], "a": "x"},
    ],
)
def test_portable_values_round_trip(value):
    encoded = encode_value(value)

    decoded = decode_value(encoded)

    assert decoded == value
    assert type(decoded) is type(value)
    assert encoded.isascii()


@pytest.mark.parametrize(
    "value",
    [
        (1, 2),
        {1, 2},
        b"bytes",
        INT64_MAX + 1,
        INT64_MIN - 1,
        math.nan,
        math.inf,
        -math.inf,
        _Level.LOW,
        _Text("x"),
        {1: "non-str key"},
        {"nested": [object()]},
        [1, [math.nan]],
    ],
)
def test_non_portable_values_are_rejected(value):
    with pytest.raises(StateValueError):
        encode_value(value)


def test_cyclic_and_too_deep_values_are_rejected():
    cyclic = []
    cyclic.append(cyclic)
    deep = []
    for _ in range(100):
        deep = [deep]

    with pytest.raises(StateValueError, match="contains itself"):
        encode_value(cyclic)
    with pytest.raises(StateValueError, match="deeper"):
        encode_value(deep)


def test_shared_subvalue_is_not_a_cycle():
    shared = [1]

    encoded = encode_value({"a": shared, "b": shared})

    assert decode_value(encoded) == {"a": [1], "b": [1]}


def test_encoding_is_canonical_and_type_sensitive():
    assert encode_value({"b": 1, "a": 2}) == encode_value({"a": 2, "b": 1})
    assert encode_value(1) != encode_value(1.0)
    assert encode_value(1) != encode_value(True)
    assert encode_value(0.0) != encode_value(-0.0)
    assert encode_value(12) == "12"


def test_numpy_scalars_are_rejected():
    numpy = pytest.importorskip("numpy")

    with pytest.raises(StateValueError, match="int64"):
        encode_value(numpy.int64(3))


@pytest.mark.parametrize("amount", [True, 1.0, "1", INT64_MAX + 1, None])
def test_invalid_amounts_are_rejected(amount):
    with pytest.raises(StateValueError):
        validate_amount(amount)


@pytest.mark.parametrize("name", ["", None, 3, "\ud800"])
def test_invalid_names_are_rejected(name):
    with pytest.raises(StateValueError):
        validate_name(name, what="key")


def test_storage_keys_never_collide():
    components = [
        "a",
        "a:b",
        "b",
        "a%3Ab",
        "%",
        ":",
        "g",
        "s",
        "a:g:b",
        "s:a",
        "mg",
        "ms:a",
    ]

    keys = set()
    triples = 0
    for namespace, source_id, key in itertools.product(components, repeat=3):
        keys.add(source_storage_key(namespace, source_id, key))
        keys.add(machine_storage_key(namespace, source_id, key))
        triples += 2
    for namespace, key in itertools.product(components, repeat=2):
        keys.add(global_storage_key(namespace, key))
        keys.add(machine_storage_key(namespace, None, key))
        triples += 2

    assert len(keys) == triples
