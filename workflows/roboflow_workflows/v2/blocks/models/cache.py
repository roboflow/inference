"""A small host-owned cache of loaded models, keyed by model and package id.

The host creates one ``ModelCache`` around one fixed loader and passes it to a
session as the ``models`` resource. A model block asks it for a
``ModelRequest`` (model id plus optional package id) on each call::

    get(request)
      supplied for this model id?  -> that instance, only on an exact request
      already loaded?              -> the cached instance
      another call loading it?     -> wait for that load, share its result
      otherwise                    -> load(request) outside the lock, cache it

A host with other loader options (device, API key, ...) uses another cache.
Identical concurrent misses load once; different requests load concurrently.
A failed load caches nothing: its waiters get ``ModelLoadError`` and a later
call loads again. The cache never closes, evicts or wraps a model and adds no
lock or copy around inference. Supplied and loaded models live as long as the
host keeps the cache; sessions, resets and blocks do not release them.
"""

import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

ModelLoader = Callable[["ModelRequest"], Any]
"""Loads the model of one request; must not return ``None``."""

# Per-request values of AutoModel.from_pretrained; a loader option cannot fix them.
_REQUEST_ARGUMENTS = ("model_id_or_path", "model_package_id")


@dataclass(frozen=True)
class ModelRequest:
    """The model a block asks for: a model id and an optional package id.

    Args:
        model_id: Model id or local path, as ``AutoModel`` accepts it.
        model_package_id: Exact model package to load; ``None`` leaves the
            choice to the loader.

    Raises:
        ValueError: When an id is not a non-empty string.
    """

    model_id: str
    model_package_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError(
                f"model_id must be a non-empty string, got {self.model_id!r}"
            )
        package = self.model_package_id
        if package is not None and (not isinstance(package, str) or not package):
            raise ValueError(
                f"model_package_id must be a non-empty string or None, got {package!r}"
            )

    def __str__(self) -> str:
        if self.model_package_id is None:
            return self.model_id

        return f"{self.model_id} (package {self.model_package_id!r})"


class ModelRequestConflict(LookupError):
    """A request does not match the instances supplied for its model id."""


class ModelLoadError(RuntimeError):
    """A load this call waited for failed; the original error is the cause."""


class ModelLoadRecursionError(RuntimeError):
    """A loader asked the cache for the request it is loading."""


def auto_model_loader(**options: Any) -> ModelLoader:
    """Create a loader calling ``inference_models.AutoModel.from_pretrained``.

    ``inference_models`` is imported on the first load, not here. Each load
    passes the request's model id and package id; ``options`` are forwarded
    unchanged to every load and validated by ``AutoModel`` itself.

    Args:
        **options: Fixed ``AutoModel.from_pretrained`` keyword arguments, e.g.
            ``device`` or ``api_key``.

    Returns:
        A loader for ``ModelCache``.

    Raises:
        ValueError: When ``options`` sets a per-request argument.
    """
    fixed = sorted(set(options) & set(_REQUEST_ARGUMENTS))
    if fixed:
        raise ValueError(
            f"{fixed} come from each ModelRequest; they cannot be loader options"
        )

    def load(request: ModelRequest) -> Any:
        from inference_models import AutoModel

        model = AutoModel.from_pretrained(
            request.model_id,
            model_package_id=request.model_package_id,
            **options,
        )

        return model

    return load


class _PendingLoad:
    """One in-flight load; waiters read its outcome after ``done`` is set."""

    def __init__(self) -> None:
        self.loading_thread = threading.get_ident()
        self.done = threading.Event()
        self.model: Any = None
        self.error: Optional[BaseException] = None


class ModelCache:
    """Host-owned models for model blocks, loaded at most once per request.

    Supplied instances take precedence and are never loaded. A model id with
    supplied instances is served only from them: a request must equal one of
    their requests. A model supplied under a bare model id has no verified
    package, so a request naming a package fails instead of ignoring it.

    Args:
        load: Loader of missing models; ``None`` uses ``auto_model_loader()``.
        supplied: Caller-loaded models keyed by ``ModelRequest``, or by a bare
            model id meaning ``ModelRequest(model_id)``.

    Raises:
        TypeError: When ``load`` is not callable, a key is neither a string nor
            a ``ModelRequest`` or a supplied model is ``None``.
        ValueError: When two keys name the same request.
    """

    def __init__(
        self,
        load: Optional[ModelLoader] = None,
        *,
        supplied: Mapping[Union[str, ModelRequest], Any] = MappingProxyType({}),
    ):
        if load is not None and not callable(load):
            raise TypeError(f"load must be callable, got {load!r}")

        self._load = load if load is not None else auto_model_loader()
        self._supplied = _supplied_by_request(supplied)
        self._supplied_ids = frozenset(request.model_id for request in self._supplied)
        self._lock = threading.Lock()
        self._loaded: Dict[ModelRequest, Any] = {}
        self._pending: Dict[ModelRequest, _PendingLoad] = {}

    def supplied_requests(self) -> Tuple[ModelRequest, ...]:
        """Return the requests of supplied models, in supply order."""
        requests = tuple(self._supplied)

        return requests

    def get(self, request: ModelRequest) -> Any:
        """Return the model of ``request``, loading it on its first miss.

        Args:
            request: Model and package the caller needs.

        Returns:
            The supplied or cached model; the same object for equal requests.

        Raises:
            ModelRequestConflict: When the model id was supplied, but not for
                this request.
            ModelLoadError: When the load this call waited for failed.
            ModelLoadRecursionError: When called by the loader for the request
                it is loading.
        """
        if request.model_id in self._supplied_ids:
            model = self._supplied_model(request)

            return model

        with self._lock:
            if request in self._loaded:
                return self._loaded[request]

            pending = self._pending.get(request)
            loads_here = pending is None
            if loads_here:
                pending = self._pending[request] = _PendingLoad()

        if not loads_here:
            model = _wait_for(pending, request=request)

            return model

        model = self._load_once(request, pending=pending)

        return model

    def _supplied_model(self, request: ModelRequest) -> Any:
        model = self._supplied.get(request)
        if model is not None:
            return model

        unverified = ModelRequest(request.model_id) in self._supplied
        if request.model_package_id is not None and unverified:
            reason = (
                "it was supplied without a package id, so its package is unverified"
            )
        else:
            same_id = [
                str(other)
                for other in self._supplied
                if other.model_id == request.model_id
            ]
            reason = f"it was supplied only as {same_id}"
        raise ModelRequestConflict(
            f"cannot serve model request {str(request)!r}: {reason}. Supply the "
            "model for this exact request or remove the package selection"
        )

    def _load_once(self, request: ModelRequest, *, pending: _PendingLoad) -> Any:
        try:
            model = self._load(request)
            if model is None:
                raise TypeError(
                    f"the loader returned None for model request {str(request)!r}"
                )
        except BaseException as error:
            with self._lock:
                del self._pending[request]
            pending.error = error
            pending.done.set()
            raise

        with self._lock:
            self._loaded[request] = model
            del self._pending[request]
        pending.model = model
        pending.done.set()

        return model


def _wait_for(pending: _PendingLoad, *, request: ModelRequest) -> Any:
    if pending.loading_thread == threading.get_ident():
        raise ModelLoadRecursionError(
            f"the loader of model request {str(request)!r} asked the cache for "
            "the same request; a loader must not depend on its own result"
        )

    pending.done.wait()
    if pending.error is not None:
        raise ModelLoadError(
            f"loading model request {str(request)!r} failed in a concurrent "
            "call; a later call loads it again"
        ) from pending.error

    return pending.model


def _supplied_by_request(
    supplied: Mapping[Union[str, ModelRequest], Any],
) -> Mapping[ModelRequest, Any]:
    if not isinstance(supplied, Mapping):
        raise TypeError(
            "supplied must be a mapping of model id or ModelRequest to a loaded "
            f"model, got {type(supplied).__name__}"
        )

    by_request: Dict[ModelRequest, Any] = {}
    for key, model in supplied.items():
        if isinstance(key, ModelRequest):
            request = key
        elif isinstance(key, str):
            request = ModelRequest(key)
        else:
            raise TypeError(
                f"a supplied model key must be a model id or ModelRequest, got {key!r}"
            )
        if model is None:
            raise TypeError(f"supplied model for {str(request)!r} is None")
        if request in by_request:
            raise ValueError(f"model request {str(request)!r} is supplied twice")
        by_request[request] = model

    return MappingProxyType(by_request)
