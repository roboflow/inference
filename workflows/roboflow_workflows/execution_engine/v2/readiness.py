"""Waiting for ``concurrent.futures.Future`` objects inside block results.

The engine hands results on only when they are ready: futures in the value
itself, in mapping values, list and tuple items and ``Batch`` contents are
replaced by their results, recursively (also inside a future's result).
The payloads of the engine's own result wrappers (``Selected`` and
``Selection``, see ``ResultWrapper``) are resolved too, in the same pass.
Arbitrary payload objects are never looked into.

Nothing is copied or changed in place. A container without futures is
returned as the same object; a container holding one is rebuilt, and the
caller's container stays as it was. Within one call, a container reached
twice is rebuilt once, so aliases stay aliases, also across wrappers::

    shared = [future]
    ready = resolve_futures({"left": shared, "right": Selected(index, shared)})
    ready["left"] is ready["right"].value     # True; shared still holds the future

Waiting cancels nothing: whoever submitted the work stays its owner. With
``reject_awaitables=True`` (phases use it) a coroutine or other awaitable
anywhere in the traversed result is refused. An unstarted coroutine found
there is closed, since nothing else can ever run it; a coroutine that has
already started, and any other awaitable, is left exactly as it is: its
owner remains responsible for finishing or cancelling it.

A failed future stops the resolution, and its exception is raised again,
unchanged, once the resolution has let go of the result: a retained
traceback keeps the failure, not the result's other values.
"""

import inspect
from collections.abc import Mapping
from concurrent.futures import CancelledError, Future
from typing import Any, Callable, Dict, List, Optional, Tuple

from roboflow_workflows.execution_engine.v2.data import Batch

__all__ = ["ResultWrapper", "resolve_futures"]


class ResultWrapper:
    """Base of the engine's result wrappers, whose payloads belong to the result.

    ``Selected`` and ``Selection`` carry output payloads next to member
    indices. Readiness reaches those payloads, and nothing else of the
    wrapper, through ``map_payloads``.
    """

    def map_payloads(self, function: Callable[[Any], Any]) -> "ResultWrapper":
        """Return the wrapper with ``function`` applied to its payloads.

        Args:
            function: Maps one payload, or one container of payloads.

        Returns:
            The wrapper itself when no payload changed, otherwise a new one
            with the same indices.
        """
        raise NotImplementedError


def resolve_futures(value: Any, *, reject_awaitables: bool = False) -> Any:
    """Return ``value`` with every reachable ``Future`` replaced by its result.

    Args:
        value: A block or phase result, or part of one.
        reject_awaitables: Refuse coroutines and other awaitables in the
            result instead of passing them on as payloads.

    Returns:
        The ready value; the same object when it holds no futures.

    Raises:
        TypeError: When ``reject_awaitables`` is set and the result holds an
            awaitable.
        Exception: Whatever a future raised.
    """
    resolution = _Resolution(reject_awaitables=reject_awaitables)
    try:
        resolved = resolution.resolve(value)
    finally:
        failure = resolution.finish()
    if failure is None:
        return resolved

    # The failure's traceback keeps this frame; it must not keep the result.
    value = resolved = None
    raise failure


def _failure_of(future: Future) -> Optional[BaseException]:
    """Wait for ``future``; return its exception without raising it here.

    Raising inside the resolution would put its frames, and through them the
    result being resolved, into the failure's traceback.
    """
    try:
        failure = future.exception()
    except CancelledError:
        failure = CancelledError()

    return failure


class _Resolution:
    """State of one ``resolve_futures`` call: resolved containers by identity."""

    def __init__(self, *, reject_awaitables: bool):
        self.reject_awaitables = reject_awaitables
        # Originals stay referenced while their ids are keys.
        self.resolved: Dict[int, Tuple[Any, Any]] = {}
        self.awaitables: List[Any] = []
        self.failure: Optional[BaseException] = None

    def resolve(self, value: Any) -> Any:
        if self.failure is not None:
            return value
        if isinstance(value, Future):
            failure = _failure_of(value)
            if failure is not None:
                self.failure = failure
                return value
            resolved = self.resolve(value.result())
            return resolved
        if self.reject_awaitables and inspect.isawaitable(value):
            self.awaitables.append(value)
            return value
        if not isinstance(value, (Batch, Mapping, list, tuple, ResultWrapper)):
            return value
        if id(value) in self.resolved:
            return self.resolved[id(value)][1]

        rebuilt = self._rebuild(value)
        self.resolved[id(value)] = (value, rebuilt)

        return rebuilt

    def _rebuild(self, value: Any) -> Any:
        """The container itself when no item changed, otherwise a new one."""
        if isinstance(value, ResultWrapper):
            rebuilt_wrapper = value.map_payloads(self.resolve)
            return rebuilt_wrapper
        if isinstance(value, Batch):
            content = [self.resolve(item) for item in value.content]
            if all(new is old for new, old in zip(content, value.content)):
                return value
            rebuilt = Batch(
                content,
                indices=value.indices,
                layout=value.layout,
                metadata=value.metadata,
                parent_index=value.parent_index,
            )
            return rebuilt
        if isinstance(value, Mapping):
            items = {key: self.resolve(item) for key, item in value.items()}
            if all(items[key] is item for key, item in value.items()):
                return value
            rebuilt_mapping = type(value)(items) if isinstance(value, dict) else items
            return rebuilt_mapping

        items = [self.resolve(item) for item in value]
        if all(new is old for new, old in zip(items, value)):
            return value
        rebuilt_sequence = type(value)(items)

        return rebuilt_sequence

    def finish(self) -> Optional[BaseException]:
        """End the resolution: close unstarted coroutines and forget the result.

        Returns:
            The failed future's exception, else a ``TypeError`` for rejected
            awaitables, else ``None``.
        """
        for awaitable in self.awaitables:
            if (
                inspect.iscoroutine(awaitable)
                and inspect.getcoroutinestate(awaitable) == inspect.CORO_CREATED
            ):
                awaitable.close()
        failure = self.failure
        if failure is None and self.awaitables:
            found = sorted({type(awaitable).__name__ for awaitable in self.awaitables})
            failure = TypeError(
                f"the result holds {' and '.join(found)}, which V2 does not await; "
                "block code is synchronous: return values or "
                "concurrent.futures.Future objects"
            )

        self.resolved.clear()
        self.awaitables.clear()
        self.failure = None

        return failure
