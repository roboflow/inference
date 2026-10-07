"""Owned snapshots of event fields handed to asynchronous handlers.

An asynchronous handler reads its fields later, on another thread, while the
emitting block and downstream steps keep using theirs. ``snapshot_fields``
gives the handler values nobody else can change::

    value                         snapshot
    ----------------------------  ----------------------------------------------
    None, bool, int, float, str,  the same object (immutable)
    bytes, Enum, engine metadata
    numpy array                   array.copy()        dtype kept, host memory
    numpy object array            same shape and dtype, members snapshotted;
                                  structured dtype with object fields raises
    torch tensor                  tensor.clone()      same device, no CPU trip
    list / dict / set             new container of snapshotted members
    tuple / frozenset / mapping-  the same object when no member changed,
      proxy / frozen dataclass    else rebuilt (dataclasses.replace)
    mutable dataclass             rebuilt with snapshotted fields
    Batch                         same indices, layout and metadata; members
                                  snapshotted
    pydantic model                model_copy(deep=True)
    __workflows_snapshot__()      what the method returns (custom owner hook)
    register_snapshot(type, fn)   what fn returns
    anything else                 EventEmissionError: never silently shared

Values shared inside one emission stay shared in its snapshot (one copy per
original object), so aliasing between fields is preserved. ``ImageData`` is
a frozen dataclass: its tensor is cloned and its identity, provenance and
video metadata are kept.

Device completion contract: a tensor field must be complete as seen from the
emitting thread's current device stream when ``emit`` is called. The clone
is queued on that stream, so it follows work the producer queued there.
Work on another stream must be synchronized, or returned as a ``Future``
that resolves when complete, before ``emit``; the engine does not probe
side streams. Futures are resolved before snapshotting.

Synchronous handlers never get snapshots: they run before ``emit`` returns,
with ordinary aliasing rules.
"""

import dataclasses
import enum
import sys
import threading
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional

from roboflow_workflows.execution_engine.v2.data import (
    Axis,
    Batch,
    EntryLayout,
    EntryMetadata,
    SampleContext,
    TemporalContext,
    TimeSpan,
    Timestamp,
)
from roboflow_workflows.execution_engine.v2.errors import EventEmissionError

__all__ = ["SNAPSHOT_HOOK", "Snapshot", "register_snapshot", "snapshot_fields"]

SNAPSHOT_HOOK = "__workflows_snapshot__"
"""Method name a payload class defines to return an owned copy of itself."""

_ATOMS = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    Fraction,
    Decimal,
    range,
    enum.Enum,
    Axis,
    EntryLayout,
    EntryMetadata,
    SampleContext,
    TemporalContext,
    TimeSpan,
    Timestamp,
)

_REGISTERED: Dict[type, Callable[[Any], Any]] = {}
_REGISTERED_LOCK = threading.Lock()


def register_snapshot(payload_type: type, snapshot: Callable[[Any], Any]) -> None:
    """Teach the engine how to take an owned copy of a payload type.

    Use this for a plugin payload the generic rules cannot copy safely, for
    example one holding a native handle. Subclasses use the registration of
    their nearest registered base.

    Args:
        payload_type: Class whose instances ``snapshot`` copies.
        snapshot: Returns a copy that shares no mutable state with its
            argument.

    Raises:
        TypeError: When ``payload_type`` is not a class or ``snapshot`` is
            not callable.
    """
    if not isinstance(payload_type, type):
        raise TypeError(f"payload_type must be a class, got {payload_type!r}")
    if not callable(snapshot):
        raise TypeError(f"snapshot must be callable, got {snapshot!r}")

    with _REGISTERED_LOCK:
        _REGISTERED[payload_type] = snapshot


@dataclass(frozen=True)
class Snapshot:
    """Owned field values of one event for one asynchronous handler.

    Args:
        fields: Field name to snapshotted value.
        known_bytes: Bytes copied into arrays, tensors and byte arrays.
        unknown_sizes: Copies whose size is unknown (custom hooks,
            registrations, pydantic models); ``known_bytes`` excludes them.
    """

    fields: Mapping[str, Any]
    known_bytes: int = 0
    unknown_sizes: int = 0


class _Copies:
    """One snapshot's copies by original id (aliasing) and their sizes."""

    def __init__(self) -> None:
        self.memo: Dict[int, Any] = {}
        self.known_bytes = 0
        self.unknown_sizes = 0


def snapshot_fields(fields: Mapping[str, Any], *, where: str) -> Snapshot:
    """Return owned copies of event field values for an asynchronous handler.

    Args:
        fields: Ready field values (futures already resolved).
        where: Emission description used in error messages.

    Returns:
        The snapshotted fields with best-effort size accounting.

    Raises:
        EventEmissionError: When a value has no supported snapshot.
    """
    copies = _Copies()
    snapshotted = {
        name: _snapshot(value, copies=copies, where=f"{where} field '{name}'")
        for name, value in fields.items()
    }
    snapshot = Snapshot(
        fields=snapshotted,
        known_bytes=copies.known_bytes,
        unknown_sizes=copies.unknown_sizes,
    )

    return snapshot


def _snapshot(value: Any, *, copies: "_Copies", where: str) -> Any:
    if isinstance(value, _ATOMS):
        return value
    if id(value) in copies.memo:
        return copies.memo[id(value)]

    copied = _copy(value, copies=copies, where=where)
    copies.memo[id(value)] = copied

    return copied


def _copy(value: Any, *, copies: "_Copies", where: str) -> Any:
    registered = _registered_for(type(value))
    if registered is not None:
        copies.unknown_sizes += 1
        return registered(value)
    hook = getattr(type(value), SNAPSHOT_HOOK, None)
    if hook is not None:
        copies.unknown_sizes += 1
        return hook(value)

    numpy = sys.modules.get("numpy")
    if numpy is not None and isinstance(value, numpy.ndarray):
        return _array(value, numpy=numpy, copies=copies, where=where)
    if numpy is not None and isinstance(value, numpy.generic):
        return value
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        copies.known_bytes += value.numel() * value.element_size()
        return value.clone()

    if isinstance(value, Batch):
        content = [
            _snapshot(item, copies=copies, where=where) for item in value.content
        ]
        copied = Batch(
            content,
            indices=value.indices,
            layout=value.layout,
            metadata=value.metadata,
            parent_index=value.parent_index,
        )
        return copied
    if isinstance(value, list):
        return [_snapshot(item, copies=copies, where=where) for item in value]
    if isinstance(value, dict):
        return {
            key: _snapshot(item, copies=copies, where=where)
            for key, item in value.items()
        }
    if isinstance(value, bytearray):
        copies.known_bytes += len(value)
        return bytearray(value)
    if isinstance(value, set):
        return {_snapshot(item, copies=copies, where=where) for item in value}
    if isinstance(value, (tuple, frozenset, MappingProxyType)):
        return _immutable_container(value, copies=copies, where=where)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _dataclass(value, copies=copies, where=where)
    if hasattr(value, "model_copy") and hasattr(type(value), "model_fields"):
        copies.unknown_sizes += 1
        return value.model_copy(deep=True)

    raise EventEmissionError(
        f"{where}: {type(value).__module__}.{type(value).__qualname__} has no "
        "supported ownership snapshot, so an asynchronous handler cannot "
        f"receive it safely. Define {SNAPSHOT_HOOK}(self) on the class, call "
        "reactions.snapshots.register_snapshot, bind a different field, or "
        "use a synchronous handler"
    )


def _array(value: Any, *, numpy: Any, copies: "_Copies", where: str) -> Any:
    """Copy a numeric array; snapshot each member of an object array."""
    if not value.dtype.hasobject:
        copies.known_bytes += value.nbytes
        return value.copy()
    if value.dtype.fields is not None:
        raise EventEmissionError(
            f"{where}: a structured array with object fields has no supported "
            f"ownership snapshot; define {SNAPSHOT_HOOK}(self) on a wrapper "
            "class, call reactions.snapshots.register_snapshot, or use a "
            "synchronous handler"
        )

    copied = numpy.empty(value.shape, dtype=value.dtype)
    for index in numpy.ndindex(value.shape):
        copied[index] = _snapshot(value[index], copies=copies, where=where)

    return copied


def _registered_for(payload_type: type) -> Optional[Callable[[Any], Any]]:
    if not _REGISTERED:
        return None

    for base in payload_type.__mro__:
        if base in _REGISTERED:
            return _REGISTERED[base]

    return None


def _immutable_container(value: Any, *, copies: "_Copies", where: str) -> Any:
    """Rebuild an immutable container only when a member needed a copy."""
    if isinstance(value, MappingProxyType):
        items = {
            key: _snapshot(item, copies=copies, where=where)
            for key, item in value.items()
        }
        changed = any(items[key] is not value[key] for key in items)
        return MappingProxyType(items) if changed else value

    members = [_snapshot(item, copies=copies, where=where) for item in value]
    if all(new is old for new, old in zip(members, value)):
        return value
    if isinstance(value, tuple) and hasattr(value, "_make"):
        return value._make(members)

    rebuilt = type(value)(members)

    return rebuilt


def _dataclass(value: Any, *, copies: "_Copies", where: str) -> Any:
    """Copy changed fields; a frozen instance without changes is kept."""
    frozen = type(value).__dataclass_params__.frozen
    changes = {}
    for item in dataclasses.fields(value):
        current = getattr(value, item.name)
        copied = _snapshot(current, copies=copies, where=f"{where}.{item.name}")
        if copied is current:
            continue
        if not item.init:
            raise EventEmissionError(
                f"{where}: field '{item.name}' of {type(value).__qualname__} "
                f"needs a copy but cannot be passed to its constructor; define "
                f"{SNAPSHOT_HOOK}(self) on the class"
            )
        changes[item.name] = copied
    if frozen and not changes:
        return value

    rebuilt = dataclasses.replace(value, **changes)

    return rebuilt
