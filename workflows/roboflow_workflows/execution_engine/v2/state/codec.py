"""Portable value codec and storage keys of managed state.

Every backend stores the same strings, so memory and Redis agree on values,
equality and integer behaviour::

    value                         stored text
    None                          null
    3                             3          (also Redis's integer string)
    1.0                           1.0
    {"b": [1, "x"], "a": True}    {"a":true,"b":[1,"x"]}

Accepted values are exact ``None``, ``bool``, ``int`` (signed 64-bit),
finite ``float``, ``str``, ``list`` and ``dict`` with ``str`` keys, nested.
Anything else (tuples, numpy scalars, tensors, ``IntEnum``, NaN, cycles) is
rejected instead of being converted. Two values are equal when their
encodings are equal: ``1`` differs from ``1.0`` and ``True`` from ``1``.

Storage keys escape ``%`` and ``:`` in each component, so different
(namespace, source, key) triples never map to one key::

    wf2state:<namespace>:g:<key>
    wf2state:<namespace>:s:<source>:<key>
    wf2state:<namespace>:mg:<machine>            (state-machine records)
    wf2state:<namespace>:ms:<source>:<machine>
"""

import json
import math
from typing import Any, Optional

from roboflow_workflows.execution_engine.v2.state.errors import StateValueError

__all__ = [
    "INT64_MAX",
    "INT64_MIN",
    "decode_value",
    "encode_value",
    "global_storage_key",
    "machine_storage_key",
    "source_storage_key",
    "validate_amount",
    "validate_name",
]

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1

_KEY_PREFIX = "wf2state"
_MAX_DEPTH = 64


def encode_value(value: Any) -> str:
    """Encode a portable JSON value into its canonical stored text.

    Args:
        value: Value to store.

    Returns:
        Canonical JSON text: sorted keys, no spaces, ASCII only (other
        characters as ``\\u`` escapes, so any ``str`` round-trips).

    Raises:
        StateValueError: When the value, or anything nested in it, is not a
            portable JSON value.
    """
    _validate_value(value, path="value", depth=0, parents=set())
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )

    return encoded


def decode_value(encoded: str) -> Any:
    """Decode stored text into a fresh value.

    Args:
        encoded: Text produced by ``encode_value``.

    Returns:
        A new object; mutating it does not change the stored state.
    """
    value = json.loads(encoded)

    return value


def validate_amount(amount: Any) -> int:
    """Check an ``incr`` amount.

    Args:
        amount: Requested increment.

    Returns:
        ``amount`` unchanged.

    Raises:
        StateValueError: When ``amount`` is not an exact ``int`` in the signed
            64-bit range.
    """
    if type(amount) is not int:
        raise StateValueError(
            f"incr amount must be an int, got {type(amount).__name__}"
        )
    if not INT64_MIN <= amount <= INT64_MAX:
        raise StateValueError(f"incr amount {amount} is outside signed 64-bit range")

    return amount


def validate_name(name: Any, *, what: str) -> str:
    """Check a key, namespace or source id.

    Args:
        name: Name to check.
        what: Role of the name, used in the error message.

    Returns:
        ``name`` unchanged.

    Raises:
        StateValueError: When ``name`` is not a non-empty ``str`` or holds
            characters UTF-8 cannot encode (lone surrogates).
    """
    if type(name) is not str or not name:
        raise StateValueError(f"state {what} must be a non-empty str, got {name!r}")
    if not name.isascii():
        try:
            name.encode("utf-8")
        except UnicodeEncodeError as error:
            raise StateValueError(
                f"state {what} {name!r} is not valid UTF-8"
            ) from error

    return name


def global_storage_key(namespace: str, key: str) -> str:
    """Return the storage key of an execution-global entry."""
    storage_key = f"{_KEY_PREFIX}:{_escape(namespace)}:g:{_escape(key)}"

    return storage_key


def source_storage_key(namespace: str, source_id: str, key: str) -> str:
    """Return the storage key of a source-scoped entry."""
    storage_key = (
        f"{_KEY_PREFIX}:{_escape(namespace)}:s:{_escape(source_id)}:{_escape(key)}"
    )

    return storage_key


def machine_storage_key(namespace: str, source_id: Optional[str], machine: str) -> str:
    """Return the storage key of a state-machine record (engine-internal).

    Markers ``mg`` / ``ms`` differ from the ``g`` / ``s`` markers of user
    keys, and every component is escaped, so no user key can alias a record.

    Args:
        namespace: Namespace of the ``ManagedState``.
        source_id: Source of a per-source machine; ``None`` for a global one.
        machine: Scoped machine name, e.g. ``child/door``.

    Returns:
        The storage key.
    """
    if source_id is None:
        storage_key = f"{_KEY_PREFIX}:{_escape(namespace)}:mg:{_escape(machine)}"
        return storage_key

    storage_key = (
        f"{_KEY_PREFIX}:{_escape(namespace)}:ms:{_escape(source_id)}:"
        f"{_escape(machine)}"
    )

    return storage_key


def _escape(component: str) -> str:
    escaped = component.replace("%", "%25").replace(":", "%3A")

    return escaped


def _validate_value(value: Any, *, path: str, depth: int, parents: set) -> None:
    value_type = type(value)
    if value is None or value_type in (bool, str):
        return

    if value_type is int:
        if not INT64_MIN <= value <= INT64_MAX:
            raise StateValueError(f"{path}: int {value} is outside signed 64-bit range")
        return

    if value_type is float:
        if not math.isfinite(value):
            raise StateValueError(f"{path}: float {value} is not finite")
        return

    if value_type not in (list, dict):
        raise StateValueError(
            f"{path}: {value_type.__name__} is not a portable JSON value; use "
            "None, bool, int, float, str, list or dict with str keys"
        )

    if depth >= _MAX_DEPTH:
        raise StateValueError(f"{path}: nesting deeper than {_MAX_DEPTH} levels")
    if id(value) in parents:
        raise StateValueError(f"{path}: value contains itself")

    parents.add(id(value))
    if value_type is list:
        for position, item in enumerate(value):
            _validate_value(
                item, path=f"{path}[{position}]", depth=depth + 1, parents=parents
            )
    else:
        for item_key, item in value.items():
            if type(item_key) is not str:
                raise StateValueError(f"{path}: dict key {item_key!r} is not a str")
            _validate_value(
                item, path=f"{path}[{item_key!r}]", depth=depth + 1, parents=parents
            )
    parents.discard(id(value))
