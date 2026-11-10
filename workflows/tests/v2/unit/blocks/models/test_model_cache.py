"""``ModelCache``: one load per request, supplied-model binding, failure retry.

Loaders here are plain functions returning sentinel objects; they test the
cache's bookkeeping, not real model loading. Concurrency tests synchronise
with events and join daemon threads with a timeout, so a deadlock fails the
test instead of hanging it.
"""

import sys
import threading
import types
from typing import Any, Callable, Dict, List

import pytest
from roboflow_workflows.v2.blocks.models.cache import (
    ModelCache,
    ModelLoadError,
    ModelLoadRecursionError,
    ModelRequest,
    ModelRequestConflict,
    auto_model_loader,
)

TIMEOUT = 5.0


class RecordingLoader:
    """Returns a new object per call and records the requests it loaded."""

    def __init__(self, before_return: Callable[[ModelRequest], None] = None):
        self.before_return = before_return
        self.requests: List[ModelRequest] = []
        self._lock = threading.Lock()

    def __call__(self, request: ModelRequest) -> Any:
        with self._lock:
            self.requests.append(request)
        if self.before_return is not None:
            self.before_return(request)
        model = types.SimpleNamespace(request=request)

        return model


def _in_threads(count: int, target: Callable[[], Any]) -> List[Any]:
    """Call ``target`` in ``count`` threads; return values or raised errors."""
    outcomes: List[Any] = [None] * count

    def call(index: int) -> None:
        try:
            outcomes[index] = target()
        except BaseException as error:
            outcomes[index] = error

    threads = [
        threading.Thread(target=call, args=(i,), daemon=True) for i in range(count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT)
    assert not any(thread.is_alive() for thread in threads), "threads deadlocked"

    return outcomes


def _wait_until_waiting(thread: threading.Thread) -> None:
    """Wait until ``thread`` blocks on another call's load of its request."""
    for _ in range(int(TIMEOUT / 0.005)):
        frame = sys._current_frames().get(thread.ident)
        while frame is not None:
            if frame.f_code.co_name == "_wait_for":
                return
            frame = frame.f_back
        threading.Event().wait(0.005)
    raise AssertionError("the thread never waited for the pending load")


def _each_thread_requests(cache: ModelCache, model_ids: List[str]) -> Callable[[], Any]:
    """A target giving each calling thread the next model id."""
    remaining = list(model_ids)
    lock = threading.Lock()

    def target() -> Any:
        with lock:
            model_id = remaining.pop(0)
        model = cache.get(ModelRequest(model_id))

        return model

    return target


def test_identical_concurrent_misses_load_once_and_share_the_model() -> None:
    started, release = threading.Event(), threading.Event()

    def hold(_: ModelRequest) -> None:
        started.set()
        assert release.wait(TIMEOUT)

    loader = RecordingLoader(before_return=hold)
    cache = ModelCache(loader)
    request = ModelRequest("yolo", "pkg-a")
    outcomes: Dict[str, Any] = {}

    def call(name: str) -> None:
        try:
            outcomes[name] = cache.get(ModelRequest("yolo", "pkg-a"))
        except BaseException as error:
            outcomes[name] = error

    loading = threading.Thread(target=call, args=("loading",), daemon=True)
    loading.start()
    assert started.wait(TIMEOUT)
    # Every other miss must be waiting on the pending load before it ends.
    waiting = [
        threading.Thread(target=call, args=(f"waiting{i}",), daemon=True)
        for i in range(7)
    ]
    for thread in waiting:
        thread.start()
    for thread in waiting:
        _wait_until_waiting(thread)
    release.set()
    for thread in [loading, *waiting]:
        thread.join(TIMEOUT)
    assert not any(thread.is_alive() for thread in [loading, *waiting])

    assert loader.requests == [request]
    assert len(outcomes) == 8
    assert all(outcome is outcomes["loading"] for outcome in outcomes.values())
    assert outcomes["loading"].request == request


def test_unrelated_requests_load_concurrently() -> None:
    started = {"a": threading.Event(), "b": threading.Event()}

    def meet(request: ModelRequest) -> None:
        # Each load waits for the other to start: serialised loads time out.
        started[request.model_id].set()
        other = "b" if request.model_id == "a" else "a"
        assert started[other].wait(TIMEOUT), "loads were serialised"

    cache = ModelCache(RecordingLoader(before_return=meet))

    first, second = _in_threads(2, _each_thread_requests(cache, ["a", "b"]))

    assert first.request == ModelRequest("a")
    assert second.request == ModelRequest("b")


def test_package_ids_are_separate_load_keys() -> None:
    loader = RecordingLoader()
    cache = ModelCache(loader)
    requests = [
        ModelRequest("yolo"),
        ModelRequest("yolo", "pkg-a"),
        ModelRequest("yolo", "pkg-b"),
    ]

    models = [cache.get(request) for request in requests]
    again = [cache.get(request) for request in requests]

    assert loader.requests == requests
    assert len({id(model) for model in models}) == 3
    assert all(first is second for first, second in zip(models, again))


def test_a_failed_load_releases_waiters_and_a_later_call_retries() -> None:
    started, release = threading.Event(), threading.Event()
    failures = [RuntimeError("weights unavailable")]

    def fail_first(_: ModelRequest) -> None:
        if failures:
            started.set()
            assert release.wait(TIMEOUT)
            raise failures.pop()

    loader = RecordingLoader(before_return=fail_first)
    cache = ModelCache(loader)
    request = ModelRequest("yolo")
    outcomes: Dict[str, Any] = {}

    def call(name: str) -> None:
        try:
            outcomes[name] = cache.get(request)
        except BaseException as error:
            outcomes[name] = error

    loading = threading.Thread(target=call, args=("loading",), daemon=True)
    loading.start()
    assert started.wait(TIMEOUT)
    waiting = threading.Thread(target=call, args=("waiting",), daemon=True)
    waiting.start()
    _wait_until_waiting(waiting)
    release.set()
    for thread in (loading, waiting):
        thread.join(TIMEOUT)
    assert not waiting.is_alive(), "the failed load did not release its waiter"

    assert str(outcomes["loading"]) == "weights unavailable"
    assert isinstance(outcomes["waiting"], ModelLoadError)
    assert outcomes["waiting"].__cause__ is outcomes["loading"]
    assert "'yolo'" in str(outcomes["waiting"])
    retried = cache.get(request)
    assert retried.request == request
    assert loader.requests == [request, request]


def test_a_loader_returning_none_fails_without_caching() -> None:
    results = [None, "model"]
    cache = ModelCache(lambda request: results.pop(0))

    with pytest.raises(TypeError, match="returned None for model request 'yolo'"):
        cache.get(ModelRequest("yolo"))

    assert cache.get(ModelRequest("yolo")) == "model"


def test_a_loader_asking_for_its_own_request_fails_instead_of_deadlocking() -> None:
    cache_holder: List[ModelCache] = []

    def recursive(request: ModelRequest) -> Any:
        return cache_holder[0].get(request)

    cache_holder.append(ModelCache(recursive))

    (outcome,) = _in_threads(1, lambda: cache_holder[0].get(ModelRequest("yolo")))

    assert isinstance(outcome, ModelLoadRecursionError)
    assert "'yolo'" in str(outcome)


def test_a_loader_may_load_a_different_request_it_depends_on() -> None:
    cache_holder: List[ModelCache] = []

    def with_dependency(request: ModelRequest) -> Any:
        if request.model_id == "pipeline":
            dependency = cache_holder[0].get(ModelRequest("backbone"))
            return ("pipeline", dependency)
        return ("model", request.model_id)

    cache_holder.append(ModelCache(with_dependency))

    (outcome,) = _in_threads(1, lambda: cache_holder[0].get(ModelRequest("pipeline")))

    assert outcome == ("pipeline", ("model", "backbone"))
    assert cache_holder[0].get(ModelRequest("backbone")) is outcome[1]


def test_supplied_models_are_served_for_their_exact_request_only() -> None:
    loader = RecordingLoader()
    bare, packaged = object(), object()
    cache = ModelCache(
        loader,
        supplied={"bare": bare, ModelRequest("packaged", "pkg-a"): packaged},
    )

    assert cache.get(ModelRequest("bare")) is bare
    assert cache.get(ModelRequest("packaged", "pkg-a")) is packaged
    assert cache.supplied_requests() == (
        ModelRequest("bare"),
        ModelRequest("packaged", "pkg-a"),
    )
    assert loader.requests == []


@pytest.mark.parametrize(
    "request_, message",
    [
        (ModelRequest("bare", "pkg-a"), "supplied without a package id"),
        (ModelRequest("packaged", "pkg-b"), r"supplied only as \[\"packaged"),
        (ModelRequest("packaged"), r"supplied only as \[\"packaged"),
    ],
)
def test_requests_conflicting_with_a_supplied_model_fail_without_loading(
    request_: ModelRequest, message: str
) -> None:
    loader = RecordingLoader()
    cache = ModelCache(
        loader,
        supplied={"bare": object(), ModelRequest("packaged", "pkg-a"): object()},
    )

    with pytest.raises(ModelRequestConflict, match=message):
        cache.get(request_)

    assert loader.requests == []


def test_other_model_ids_still_load_beside_supplied_models() -> None:
    loader = RecordingLoader()
    cache = ModelCache(loader, supplied={"bare": object()})

    model = cache.get(ModelRequest("other", "pkg-a"))

    assert model.request == ModelRequest("other", "pkg-a")


@pytest.mark.parametrize(
    "supplied, error, message",
    [
        ({"yolo": object(), ModelRequest("yolo"): object()}, ValueError, "twice"),
        ({"yolo": None}, TypeError, "is None"),
        ({3: object()}, TypeError, "model id or ModelRequest"),
        ([("yolo", object())], TypeError, "must be a mapping"),
    ],
)
def test_invalid_supplied_models_are_rejected(
    supplied: Any, error: type, message: str
) -> None:
    with pytest.raises(error, match=message):
        ModelCache(RecordingLoader(), supplied=supplied)


@pytest.mark.parametrize(
    "model_id, model_package_id", [("", None), (None, None), ("yolo", "")]
)
def test_requests_need_non_empty_ids(model_id: Any, model_package_id: Any) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        ModelRequest(model_id, model_package_id)


def test_auto_model_loader_imports_on_load_and_forwards_request_and_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    load = auto_model_loader(device="cpu", api_key="key")
    fake_package = types.ModuleType("inference_models")
    fake_package.AutoModel = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: calls.append((args, kwargs)) or "m"
    )
    # Installed after the loader exists: the import happens at load time.
    monkeypatch.setitem(sys.modules, "inference_models", fake_package)

    model = ModelCache(load).get(ModelRequest("yolo", "pkg-a"))

    assert model == "m"
    assert calls == [
        (("yolo",), {"model_package_id": "pkg-a", "device": "cpu", "api_key": "key"})
    ]


def test_auto_model_loader_keeps_request_fields_out_of_its_options() -> None:
    with pytest.raises(ValueError, match="model_package_id"):
        auto_model_loader(model_package_id="pkg-a")


def test_the_loader_must_be_callable() -> None:
    with pytest.raises(TypeError, match="callable"):
        ModelCache("AutoModel")
