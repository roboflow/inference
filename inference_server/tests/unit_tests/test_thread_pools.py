import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import anyio.to_thread
import pytest

from inference_server import configuration
from tests.unit_tests.legacy.conftest import FakeGateway
from tests.unit_tests.workflows.test_router import PASSTHROUGH_WF


def _read_setting(name, value, expression):
    env = {k: v for k, v in os.environ.items() if k != name}
    if value is not None:
        env[name] = value
    return subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore",
            "-c",
            f"from inference_server import configuration as c; print({expression})",
        ],
        env=env,
        capture_output=True,
        text=True,
    )


def test_threadpool_workers_is_unset_by_default():
    result = _read_setting(
        "HTTP_API_THREADPOOL_WORKERS", None, "c.HTTP_API_THREADPOOL_WORKERS"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "None"


def test_threadpool_workers_is_read_from_the_environment():
    result = _read_setting(
        "HTTP_API_THREADPOOL_WORKERS", "128", "c.HTTP_API_THREADPOOL_WORKERS"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "128"


@pytest.mark.parametrize("value", ["0", "-4", "many", "1.5", ""])
def test_threadpool_workers_rejects_non_positive_or_non_integer(value):
    result = _read_setting(
        "HTTP_API_THREADPOOL_WORKERS", value, "c.HTTP_API_THREADPOOL_WORKERS"
    )

    assert result.returncode != 0
    assert "HTTP_API_THREADPOOL_WORKERS must be a positive integer" in result.stderr


@pytest.mark.parametrize(
    "value,expected", [(None, "True"), ("True", "True"), ("False", "False")]
)
def test_shared_workflows_pool_switch(value, expected):
    result = _read_setting(
        "HTTP_API_SHARED_WORKFLOWS_THREAD_POOL_ENABLED",
        value,
        "c.WORKFLOWS_THREAD_POOL_ENABLED",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


def _stub_lifespan(monkeypatch):
    class _StubProxy:
        async def start(self):
            pass

        async def shutdown(self):
            pass

    monkeypatch.delenv("INFERENCE_PRELOAD_MODELS", raising=False)
    monkeypatch.setattr(
        "inference_model_manager.watchdogs.start_enabled_watchdogs", lambda: []
    )
    monkeypatch.setattr(
        "inference_server.gateway_resolver.resolve_gateway", lambda: _StubProxy()
    )


@pytest.fixture
def app_log():
    records = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Handler(level=logging.INFO)
    app_logger = logging.getLogger("inference_server.app")
    previous_level = app_logger.level
    app_logger.setLevel(logging.INFO)
    app_logger.addHandler(handler)
    yield records
    app_logger.removeHandler(handler)
    app_logger.setLevel(previous_level)


@pytest.mark.asyncio
async def test_limiter_is_untouched_when_the_setting_is_unset(monkeypatch, app_log):
    import inference_server.app as app_mod

    _stub_lifespan(monkeypatch)
    monkeypatch.setattr(configuration, "HTTP_API_THREADPOOL_WORKERS", None)
    limiter = anyio.to_thread.current_default_thread_limiter()
    before = limiter.total_tokens

    async with app_mod._lifespan(app_mod.app):
        during = limiter.total_tokens

    assert during == before == limiter.total_tokens
    assert not [m for m in app_log if "thread pool" in m]


@pytest.mark.asyncio
async def test_limiter_is_resized_when_the_setting_is_set(monkeypatch, app_log):
    import inference_server.app as app_mod

    _stub_lifespan(monkeypatch)
    monkeypatch.setattr(configuration, "HTTP_API_THREADPOOL_WORKERS", 128)
    limiter = anyio.to_thread.current_default_thread_limiter()
    before = limiter.total_tokens
    try:
        async with app_mod._lifespan(app_mod.app):
            during = limiter.total_tokens
    finally:
        limiter.total_tokens = before

    assert during == 128
    assert [m for m in app_log if "thread pool" in m] == [
        "HTTP API thread pool resized to 128 threads"
    ]


@pytest.mark.asyncio
async def test_shared_pool_exists_by_default(monkeypatch):
    import inference_server.app as app_mod

    _stub_lifespan(monkeypatch)

    async with app_mod._lifespan(app_mod.app):
        executor = app_mod.app.state.workflows_executor

    assert isinstance(executor, ThreadPoolExecutor)


@pytest.mark.asyncio
async def test_no_shared_pool_when_disabled(monkeypatch):
    import inference_server.app as app_mod

    _stub_lifespan(monkeypatch)
    monkeypatch.setattr(configuration, "WORKFLOWS_THREAD_POOL_ENABLED", False)

    async with app_mod._lifespan(app_mod.app):
        executor = app_mod.app.state.workflows_executor

    assert executor is None


def _captured_executors(legacy_client, monkeypatch):
    from inference_server.workflows import execution

    captured = []
    original = execution.run_workflow_sync

    def _capturing(**kwargs):
        captured.append(kwargs["executor"])
        return original(**kwargs)

    monkeypatch.setattr(execution, "run_workflow_sync", _capturing)
    response = legacy_client(FakeGateway()).post(
        "/workflows/run",
        json={"specification": PASSTHROUGH_WF, "inputs": {"x": 3}, "api_key": "k"},
    )
    assert response.status_code == 200, response.text

    return captured


def test_workflow_route_passes_the_shared_pool_by_default(legacy_client, monkeypatch):
    captured = _captured_executors(legacy_client, monkeypatch)

    assert len(captured) == 1 and isinstance(captured[0], ThreadPoolExecutor)


def test_workflow_route_passes_no_executor_when_disabled(legacy_client, monkeypatch):
    monkeypatch.setattr(configuration, "WORKFLOWS_THREAD_POOL_ENABLED", False)

    assert _captured_executors(legacy_client, monkeypatch) == [None]
