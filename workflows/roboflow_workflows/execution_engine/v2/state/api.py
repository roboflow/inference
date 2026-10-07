"""Managed state: atomic single-key operations in global and source scopes.

A block requests the engine's state through its constructor::

    class ZoneCounter(Block):
        def __init__(self, *, managed_state: ManagedState):
            self.state = managed_state

        def run(self, *, image):
            seen = self.state.source.incr("seen")     # this call's source
            self.state.global_.incr("seen_total")     # whole execution
            ...

Scopes of one ``ManagedState``::

    state.global_              one scope per namespace
    state.source               the current call's source (resolved per operation)
    state.at(index)            the source of one index of a batch call
    state.for_source("cam_a")  an explicit source

Each operation is one atomic single-key operation; nothing groups two
operations, so ``get`` followed by ``set`` is not atomic. Use ``incr`` or
``compare_and_set`` for read-modify-write.

``MISSING`` means "no stored value" and differs from ``None`` (JSON null)::

    state.global_.get("k")                    # None when missing or null
    state.global_.get("k", default=MISSING)   # MISSING only when missing
    state.global_.compare_and_set("k", MISSING, 1)   # create if absent
    state.global_.compare_and_set("k", 1, MISSING)   # delete if it holds 1

Values follow ``state.codec``: portable JSON only, compared by canonical
encoding, copied on the way in and out.

Declared defaults are initial values, written once with set-if-absent; after
that they are ordinary stored values (see ``ManagedState.with_defaults``).
"""

import copy
import threading
import uuid
from typing import Any, Dict, Mapping, Optional, Protocol, Set, Tuple

from roboflow_workflows.execution_engine.v2.state import _resolution
from roboflow_workflows.execution_engine.v2.state.codec import (
    decode_value,
    encode_value,
    global_storage_key,
    source_storage_key,
    validate_amount,
    validate_name,
)
from roboflow_workflows.execution_engine.v2.state.memory import InMemoryStateBackend

__all__ = [
    "MANAGED_STATE_RESOURCE",
    "MISSING",
    "ManagedState",
    "StateBackend",
    "StateScope",
]

MANAGED_STATE_RESOURCE = "managed_state"


class _Missing:
    _instance: Optional["_Missing"] = None

    def __new__(cls) -> "_Missing":
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False

    def __reduce__(self) -> str:
        return "MISSING"


MISSING: Any = _Missing()
"""No stored value. Distinct from ``None``, which is a stored JSON null."""


class StateBackend(Protocol):
    """Storage of encoded state values.

    Implemented by ``InMemoryStateBackend`` and ``RedisStateBackend``.
    Keys and values are already-encoded strings; ``None`` stands for a
    missing key. Every method is one atomic operation.
    """

    def get(self, key: str) -> Optional[str]:
        """Return the stored text or ``None``."""

    def set(self, key: str, value: str, *, only_if_absent: bool) -> bool:
        """Store a value; return whether it was stored."""

    def delete(self, key: str) -> bool:
        """Remove a key; return whether a value existed."""

    def incr(self, key: str, amount: int) -> int:
        """Add to a stored integer (missing counts as 0); return the new value."""

    def compare_and_set(
        self,
        key: str,
        expected: Optional[str],
        new: Optional[str],
    ) -> bool:
        """Replace or delete when the current value equals ``expected``."""

    def close(self) -> None:
        """Release backend resources."""


class StateScope:
    """Single-key atomic operations on one scope (global or one source).

    Obtain scopes from ``ManagedState``; do not construct them directly.
    """

    def __init__(
        self,
        backend: StateBackend,
        *,
        namespace: str,
        source_id: Optional[str],
        source_defaults: Optional["_SourceDefaults"] = None,
        lazy_source: bool = False,
    ) -> None:
        self._backend = backend
        self._namespace = namespace
        self._source_id = source_id
        self._source_defaults = source_defaults
        self._lazy_source = lazy_source

    @property
    def source_id(self) -> Optional[str]:
        """Source of this scope; ``None`` for the global scope.

        For ``state.source`` it is the current call's source.

        Raises:
            StateScopeError: When ``state.source`` cannot resolve one source.
        """
        source_id = self._resolved_source_id()

        return source_id

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value of ``key``.

        Args:
            key: State key.
            default: Returned when the key is missing. Pass ``MISSING`` to
                tell missing from ``None``.

        Returns:
            A fresh copy of the stored value, or ``default``.

        Raises:
            StateScopeError: When ``state.source`` cannot resolve one source.
            StateBackendError: When the backend fails.
        """
        storage_key = self._storage_key(key)
        stored = self._backend.get(storage_key)
        if stored is None:
            return default

        value = decode_value(stored)

        return value

    def set(self, key: str, value: Any, *, only_if_absent: bool = False) -> bool:
        """Store ``value`` under ``key``.

        Args:
            key: State key.
            value: Portable JSON value; it is copied.
            only_if_absent: Store only when no value is stored.

        Returns:
            Whether the value was stored.

        Raises:
            StateValueError: When ``value`` is not portable JSON.
            StateBackendError: When the backend fails; not applied.
            StateOutcomeUnknownError: When the reply was lost; maybe applied.
        """
        encoded = encode_value(value)
        storage_key = self._storage_key(key)
        stored = self._backend.set(storage_key, encoded, only_if_absent=only_if_absent)

        return stored

    def delete(self, key: str) -> bool:
        """Remove the stored value of ``key``; it stays missing.

        Returns:
            Whether a stored value existed.

        Raises:
            StateBackendError: When the backend fails; not applied.
            StateOutcomeUnknownError: When the reply was lost; maybe applied.
        """
        storage_key = self._storage_key(key)
        existed = self._backend.delete(storage_key)

        return existed

    def incr(self, key: str, amount: int = 1) -> int:
        """Atomically add ``amount`` to the integer under ``key``.

        A missing key counts as 0.

        Args:
            key: State key.
            amount: Signed 64-bit ``int`` (negative decrements).

        Returns:
            The new value.

        Raises:
            StateValueError: When ``amount`` is not a signed 64-bit ``int``.
            StateTypeError: When the current value is not an ``int``.
            StateOverflowError: When the result leaves the signed 64-bit
                range; the value is unchanged.
            StateBackendError: When the backend fails; not applied.
            StateOutcomeUnknownError: When the reply was lost; maybe applied.
                Never retried automatically: a retry could count twice.
        """
        validate_amount(amount)
        storage_key = self._storage_key(key)
        result = self._backend.incr(storage_key, amount)

        return result

    def compare_and_set(self, key: str, expected: Any, new: Any) -> bool:
        """Atomically replace the value of ``key`` if it equals ``expected``.

        Equality compares canonical encodings: ``1`` differs from ``1.0`` and
        ``True``.

        Args:
            key: State key.
            expected: Expected current value; ``MISSING`` expects no value.
            new: Value to store; ``MISSING`` deletes the key.

        Returns:
            Whether the value matched and the change was applied. Of several
            concurrent callers expecting the same value, at most one wins.

        Raises:
            StateValueError: When a value is not portable JSON.
            StateBackendError: When the backend fails; not applied.
            StateOutcomeUnknownError: When the reply was lost; maybe applied.
        """
        encoded_expected = None if expected is MISSING else encode_value(expected)
        encoded_new = None if new is MISSING else encode_value(new)
        storage_key = self._storage_key(key)
        applied = self._backend.compare_and_set(
            storage_key, encoded_expected, encoded_new
        )

        return applied

    def __repr__(self) -> str:
        if self._lazy_source:
            where = "source=<current call>"
        elif self._source_id is None:
            where = "global"
        else:
            where = f"source={self._source_id!r}"

        return f"StateScope(namespace={self._namespace!r}, {where})"

    def _resolved_source_id(self) -> Optional[str]:
        if self._lazy_source:
            source_id = _resolution.current_source_id()
            return source_id

        return self._source_id

    def _storage_key(self, key: str) -> str:
        validate_name(key, what="key")
        source_id = self._resolved_source_id()
        if source_id is None:
            storage_key = global_storage_key(self._namespace, key)
            return storage_key

        if self._source_defaults is not None:
            self._source_defaults.ensure_seeded(source_id)
        storage_key = source_storage_key(self._namespace, source_id, key)

        return storage_key


class ManagedState:
    """Engine-managed state shared by the blocks and handlers of an execution.

    The engine injects it as the constructor resource ``managed_state``. By
    default each session gets a private in-memory store namespaced by its
    session id. Pass your own object under ``managed_state`` to share or to
    use Redis::

        ManagedState()                                     # private, in memory
        ManagedState(RedisStateBackend("redis://127.0.0.1:6379/0"),
                     namespace="line-7")                   # shared by name

    Two ``ManagedState()`` objects never share data. Sharing needs the same
    backend and the same namespace; choosing Redis alone shares nothing,
    because a missing namespace is a fresh random one.

    Args:
        backend: Backend; ``None`` creates an in-memory backend this object
            owns and closes.
        namespace: Name isolating this state inside the backend; ``None``
            generates a unique one.

    Raises:
        StateValueError: When ``namespace`` is not a non-empty ``str``.
    """

    def __init__(
        self,
        backend: Optional[StateBackend] = None,
        *,
        namespace: Optional[str] = None,
    ) -> None:
        if namespace is None:
            namespace = uuid.uuid4().hex
        validate_name(namespace, what="namespace")

        self._owns_backend = backend is None
        self._backend: StateBackend = (
            InMemoryStateBackend() if backend is None else backend
        )
        self._namespace = namespace
        self._seeding = _Seeding(self._backend)
        self._source_defaults: Optional[_SourceDefaults] = None
        self._build_scopes()

    @property
    def namespace(self) -> str:
        """Namespace of this state inside its backend."""
        return self._namespace

    @property
    def backend(self) -> StateBackend:
        """Backend holding the values."""
        return self._backend

    @property
    def global_(self) -> StateScope:
        """Execution-global scope: one per namespace, shared by all sources."""
        return self._global_scope

    @property
    def source(self) -> StateScope:
        """Scope of the current block call's source.

        The source is resolved at each operation from the running call, so
        keeping this object across calls is safe. The runtime's
        ``ExecutionContext.source_id`` decides; operations raise
        ``StateScopeError`` outside a call, when the call has no source and
        in every batch-delivering call, even a batch of one or one whose
        members share a source (use ``at`` or ``for_source``).
        """
        return self._current_source_scope

    def at(self, index: Tuple[int, ...]) -> StateScope:
        """Return the scope of the source of one index of the current call.

        Args:
            index: Full logical index, e.g. ``batch.indices[i]``.

        Returns:
            Scope of that index's source.

        Raises:
            StateScopeError: Outside a call, when ``index`` is not one of the
                call's indices, or when it has no source.
        """
        source_id = _resolution.source_id_at(index)
        scope = self.for_source(source_id)

        return scope

    def for_source(self, source_id: str) -> StateScope:
        """Return the scope of an explicitly named source.

        Args:
            source_id: Source identity, as in ``SampleContext.source_id``.

        Returns:
            Scope of that source.

        Raises:
            StateValueError: When ``source_id`` is not a non-empty ``str``.
        """
        validate_name(source_id, what="source id")
        scope = StateScope(
            self._backend,
            namespace=self._namespace,
            source_id=source_id,
            source_defaults=self._source_defaults,
        )

        return scope

    def with_defaults(
        self,
        *,
        global_: Optional[Mapping[str, Any]] = None,
        source: Optional[Mapping[str, Any]] = None,
    ) -> "ManagedState":
        """Return a view of the same state that initializes declared defaults.

        Defaults are initial values, written with atomic set-if-absent: global
        ones now, source ones the first time the view touches each source.
        After that they are ordinary stored values: ``delete`` leaves the key
        missing and ``set(only_if_absent=True)`` sees the initialized value.

        Each key of each scope is initialized at most once per service (this
        object and all its views), so a later ``with_defaults`` never brings
        back a key deleted since, and the first initialized value wins. A new
        ``ManagedState`` on the same backend and namespace has its own
        bookkeeping and writes every default that is absent at that moment.
        Each key is seeded on its own; defaults are not one transaction.

        The view shares the backend, namespace and bookkeeping but does not
        own the backend: closing it closes nothing.

        Args:
            global_: Initial values of the global scope.
            source: Initial values of every source scope; merged over the
                source defaults of this object.

        Returns:
            New ``ManagedState`` view.

        Raises:
            StateValueError: When a key or value is not portable.
            StateBackendError: When writing a global default fails.
            StateOutcomeUnknownError: When a global default write's reply was
                lost.
        """
        encoded_global = _encoded_defaults(global_)
        encoded_source = _encoded_defaults(source)
        if self._source_defaults is not None:
            encoded_source = {**self._source_defaults.values, **encoded_source}

        for key, encoded in encoded_global.items():
            storage_key = global_storage_key(self._namespace, key)
            self._seeding.initialize(storage_key, encoded)

        view = copy.copy(self)
        view._owns_backend = False
        view._source_defaults = (
            _SourceDefaults(
                self._seeding, namespace=self._namespace, values=encoded_source
            )
            if encoded_source
            else None
        )
        view._build_scopes()

        return view

    def initialize_once(self, storage_key: str, value: Any) -> None:
        """Write an initial value once per service, with set-if-absent.

        Engine-internal seam for state-machine records
        (``codec.machine_storage_key``); blocks use ``with_defaults``. The
        bookkeeping is the one ``with_defaults`` uses, shared by this object
        and all its views and kept across session restarts: once initialized,
        a key is never written again by this call, even after it was deleted.
        A new service on the same namespace writes it if absent.

        Args:
            storage_key: Full storage key, e.g. from ``machine_storage_key``.
            value: Portable JSON initial value.

        Raises:
            StateValueError: When ``value`` is not portable JSON.
            StateBackendError: When the write fails; retried on the next call.
            StateOutcomeUnknownError: When the reply was lost; the next call
                tries set-if-absent again.
        """
        encoded = encode_value(value)
        self._seeding.initialize(storage_key, encoded)

    def close(self) -> None:
        """Close the backend if this object created it; otherwise do nothing."""
        if self._owns_backend:
            self._backend.close()

    def __repr__(self) -> str:
        backend_name = type(self._backend).__name__

        return f"ManagedState(namespace={self._namespace!r}, backend={backend_name})"

    def _build_scopes(self) -> None:
        self._global_scope = StateScope(
            self._backend,
            namespace=self._namespace,
            source_id=None,
        )
        self._current_source_scope = StateScope(
            self._backend,
            namespace=self._namespace,
            source_id=None,
            source_defaults=self._source_defaults,
            lazy_source=True,
        )


class _Seeding:
    """Storage keys one service has initialized, shared by its views.

    The lock is held while a seeding write runs, so no operation of the same
    service acts on a key before its initial value is written. It never
    wraps user code.
    """

    def __init__(self, backend: StateBackend) -> None:
        self._backend = backend
        self._seeded: Set[str] = set()
        self._lock = threading.Lock()

    def initialize(self, storage_key: str, encoded: str) -> None:
        # A failed write leaves the key unmarked, so the next touch tries
        # set-if-absent again.
        with self._lock:
            if storage_key in self._seeded:
                return

            self._backend.set(storage_key, encoded, only_if_absent=True)
            self._seeded.add(storage_key)


class _SourceDefaults:
    """Source defaults of one view; seeds each source on its first use."""

    def __init__(
        self, seeding: _Seeding, *, namespace: str, values: Mapping[str, str]
    ) -> None:
        self.values = values
        self._seeding = seeding
        self._namespace = namespace
        self._ready: Set[str] = set()

    def ensure_seeded(self, source_id: str) -> None:
        if source_id in self._ready:
            return

        for key, encoded in self.values.items():
            storage_key = source_storage_key(self._namespace, source_id, key)
            self._seeding.initialize(storage_key, encoded)
        self._ready.add(source_id)


def _encoded_defaults(defaults: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    encoded: Dict[str, str] = {}
    for key, value in (defaults or {}).items():
        validate_name(key, what="key")
        encoded[key] = encode_value(value)

    return encoded
