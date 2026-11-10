"""Opt-in ownership preserves the existing raw-model resource contract."""

import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from roboflow_workflows.v2.blocks.models import managed
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest


class FakeManager:

    def __init__(self):
        self.calls = []
        self.backends = {}
        self.shutdowns = 0
        self.before_load = lambda: None

    def load(self, route, **kwargs):
        self.calls.append((route, kwargs))
        self.before_load()
        self.backends[route] = SimpleNamespace(model=SimpleNamespace(request=kwargs))

    def get_backend(self, route):
        return self.backends.get(route)

    def unload(self, route, **kwargs):
        del self.backends[route]

    def shutdown(self):
        self.shutdowns += 1
        self.backends.clear()


@pytest.fixture
def manager(monkeypatch):
    value = FakeManager()
    monkeypatch.setattr(managed, "_new_manager", lambda: value)
    return value


def owner():
    return managed.ManagedModels(api_key="host-key", device="cpu", namespace="tenant-a")


def test_exact_request_policy_identity_and_resource_type(manager):
    with owner() as models:
        first = models.preload(ModelRequest("model", "package-a"))
        assert models.preload(ModelRequest("model", "package-a")) is first
        second = models.preload(ModelRequest("model", "package-b"))
        assert second is not first
        plan = SimpleNamespace(
            create_session=lambda **kw: SimpleNamespace(
                close=lambda: None, resources=kw["resources"]
            )
        )
        session = models.create_session(plan)
        resource = session.resources["models"]
        assert isinstance(resource, ModelCache)
        assert resource.get(ModelRequest("model", "package-a")) is first
        route, options = manager.calls[0]
        assert "host-key" not in route and "tenant-a" not in route
        assert options == dict(
            api_key="host-key",
            device="cpu",
            model_id_or_path="model",
            model_package_id="package-a",
            backend="direct",
            pinned=True,
            warmup_iters=0,
            allow_untrusted_packages=False,
        )
    assert manager.shutdowns == 1
    with pytest.raises(RuntimeError, match="closing or closed"):
        resource.get(ModelRequest("model", "package-a"))


def test_same_request_concurrent_loads_are_single_flight(manager):
    started, release = (threading.Event(), threading.Event())

    def hold():
        started.set()
        assert release.wait(3)

    manager.before_load = hold
    models = owner()
    results = []
    errors = []

    def load():
        try:
            results.append(models.preload(ModelRequest("same", "package")))
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=load) for _ in range(5)]
    for thread in threads:
        thread.start()
    assert started.wait(3)
    release.set()
    for thread in threads:
        thread.join(3)
    assert not any((thread.is_alive() for thread in threads))
    assert not errors and len(results) == 5 and (len(manager.calls) == 1)
    assert all((result is results[0] for result in results))
    models.close()


def test_load_failure_can_retry_and_close(manager):

    def fail():
        raise ValueError("load failure")

    manager.before_load = fail
    models = owner()
    with pytest.raises(ValueError, match="load failure"):
        models.preload(ModelRequest("model"))
    manager.before_load = lambda: None
    models.preload(ModelRequest("model"))
    models.close()
    assert len(manager.calls) == 2 and manager.shutdowns == 1


def test_busy_session_refuses_shutdown_but_model_access_continues(manager):
    models = owner()
    busy = [True]

    def close():
        if busy[0]:
            raise RuntimeError("unfinished active run")

    session = models.create_session(
        SimpleNamespace(
            create_session=lambda **kw: SimpleNamespace(
                close=close, resource=kw["resources"]["models"]
            )
        )
    )
    with pytest.raises(RuntimeError, match="unfinished active"):
        models.close()
    assert manager.shutdowns == 0
    assert session.resource.get(ModelRequest("still-running")) is not None
    busy[0] = False
    models.close()
    models.close()
    assert manager.shutdowns == 1


def test_close_timeout_keeps_loading_owner_retryable(manager):
    started, release = (threading.Event(), threading.Event())

    def hold():
        started.set()
        assert release.wait(3)

    manager.before_load = hold
    models = owner()
    errors = []

    def load():
        try:
            models.preload(ModelRequest("pending"))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=load)
    thread.start()
    assert started.wait(3)
    with pytest.raises(TimeoutError):
        models.close(timeout=0.02)
    assert manager.shutdowns == 0
    release.set()
    thread.join(3)
    assert not thread.is_alive() and (not errors)
    models.close()
    assert manager.shutdowns == 1


def test_partial_session_close_is_idempotent_on_retry(manager):
    models = owner()
    calls = []
    busy = [True]

    def closed_first():
        calls.append("first")

    def second():
        if busy[0]:
            raise RuntimeError("busy")
        calls.append("second")

    for closer in [closed_first, second]:
        models.create_session(
            SimpleNamespace(
                create_session=lambda closer=closer, **kw: SimpleNamespace(close=closer)
            )
        )
    with pytest.raises(RuntimeError, match="busy"):
        models.close()
    assert calls == ["first"] and manager.shutdowns == 0
    busy[0] = False
    models.close()
    assert calls == ["first", "first", "second"] and manager.shutdowns == 1


def test_session_construction_failure_keeps_owner_closeable(manager):

    def fail(**kwargs):
        raise ValueError("construction")

    models = owner()
    with pytest.raises(ValueError, match="construction"):
        models.create_session(SimpleNamespace(create_session=fail))
    models.close()
    assert manager.shutdowns == 1


@pytest.mark.parametrize("resource", ["models", "core.models"])
def test_resource_override_refused(manager, resource):
    with owner() as models:
        with pytest.raises(ValueError, match="owns"):
            models.create_session(None, resources={resource: {}})


def test_distinct_owners_get_distinct_routes(manager):
    first, second = (owner(), owner())
    first.preload(ModelRequest("same", "pkg"))
    second.preload(ModelRequest("same", "pkg"))
    assert manager.calls[0][0] != manager.calls[1][0]
    first.close()
    second.close()


def test_optional_import_not_needed_by_module_or_legacy_cache(tmp_path):
    script = "\nimport importlib.abc, sys\nclass Reject(importlib.abc.MetaPathFinder):\n    def find_spec(self, fullname, path=None, target=None):\n        if fullname.split('.')[0] == 'inference_model_manager':\n            raise ImportError('optional dependency blocked')\nsys.meta_path.insert(0, Reject())\nfrom roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest\nfrom roboflow_workflows.v2.blocks.models.managed import ManagedModels\nassert ModelCache(supplied={'test': 7}).get(ModelRequest('test')) == 7\ntry: ManagedModels(api_key='', device='cpu')\nexcept ImportError as error: assert 'optional inference-model-manager' in str(error)\nelse: raise AssertionError('missing dependency accepted')\n"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(managed.__file__).resolve().parents[4]),
            environment.get("PYTHONPATH", ""),
        ]
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_close_waits_for_admitted_session_construction(manager):
    entered, release = (threading.Event(), threading.Event())
    sessions, failures, closed = ([], [], [])
    models = owner()

    def construct(**kwargs):
        entered.set()
        assert release.wait(3)
        return SimpleNamespace(close=lambda: closed.append(True))

    def create():
        try:
            sessions.append(
                models.create_session(SimpleNamespace(create_session=construct))
            )
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=create)
    thread.start()
    assert entered.wait(3)
    with pytest.raises(TimeoutError):
        models.close(timeout=0.01)
    assert not closed and manager.shutdowns == 0
    release.set()
    thread.join(3)
    assert not thread.is_alive() and (not failures) and (len(sessions) == 1)
    models.close()
    assert closed == [True] and manager.shutdowns == 1


def test_concurrent_close_is_refused_and_does_not_reopen_owner(manager):
    entered, release = (threading.Event(), threading.Event())
    failures = []
    models = owner()

    def close_session():
        entered.set()
        assert release.wait(3)

    models.create_session(
        SimpleNamespace(
            create_session=lambda **kwargs: SimpleNamespace(close=close_session)
        )
    )

    def close_owner():
        try:
            models.close()
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=close_owner)
    thread.start()
    assert entered.wait(3)
    with pytest.raises(RuntimeError, match="already in progress"):
        models.close()
    with pytest.raises(RuntimeError, match="closing or closed"):
        models.preload(ModelRequest("new"))
    release.set()
    thread.join(3)
    assert not thread.is_alive() and (not failures) and (manager.shutdowns == 1)


def test_shutdown_failure_requires_retry_before_new_work(manager):
    models = owner()
    models.preload(ModelRequest("model"))
    original_shutdown = manager.shutdown

    def fail():
        raise RuntimeError("shutdown failed")

    manager.shutdown = fail
    with pytest.raises(RuntimeError, match="shutdown failed"):
        models.close()
    assert manager.backends
    with pytest.raises(RuntimeError, match="closing or closed"):
        models.preload(ModelRequest("model"))
    manager.shutdown = original_shutdown
    models.close()
    assert not manager.backends and manager.shutdowns == 1


def test_cancelled_load_cleans_published_backend_and_can_retry(manager):
    original_load = manager.load

    def interrupted(route, **kwargs):
        original_load(route, **kwargs)
        raise KeyboardInterrupt("cancelled startup")

    manager.load = interrupted
    models = owner()
    with pytest.raises(KeyboardInterrupt):
        models.preload(ModelRequest("model"))
    assert not manager.backends
    manager.load = original_load
    models.preload(ModelRequest("model"))
    models.close()
    assert manager.shutdowns == 1


@pytest.mark.parametrize("device", ["cuda", "cuda:", "cuda:-1", "cuda:abc"])
def test_thread_relative_or_invalid_cuda_device_is_refused(manager, device):
    with pytest.raises(ValueError, match="explicit index"):
        managed.ManagedModels(api_key="", device=device)
    assert not manager.calls


def test_indexed_cuda_policy_is_forwarded_without_optional_torch_import(manager):
    with managed.ManagedModels(api_key="", device="cuda:2") as models:
        models.preload(ModelRequest("model"))
    assert manager.calls[0][1]["device"] == "cuda:2"


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false"])
def test_untrusted_package_policy_requires_boolean(manager, value):
    with pytest.raises(TypeError, match="must be a bool"):
        managed.ManagedModels(api_key="", device="cpu", allow_untrusted_packages=value)
    assert not manager.calls


def test_untrusted_package_policy_is_explicit_and_fixed_for_the_owner(manager):
    with managed.ManagedModels(
        api_key="", device="cpu", allow_untrusted_packages=True
    ) as models:
        models.preload(ModelRequest("model", "package-a"))
        models.preload(ModelRequest("model", "package-b"))
    assert len(manager.calls) == 2
    assert all(
        options["allow_untrusted_packages"] is True for _, options in manager.calls
    )
