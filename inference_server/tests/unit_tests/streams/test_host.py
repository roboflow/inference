import ast
import asyncio
import logging
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from streamvision.stream.exceptions import MissingApiKeyError
from streamvision.stream_manager.manager_app.host import (
    PipelineHostDescriptor,
    create_pipeline_host,
)

from inference_server import configuration
from inference_server.streams import host as streams_host
from inference_server.streams.host import (
    SERVER_PIPELINE_HOST_DESCRIPTOR,
    ServerPipelineHost,
)
from inference_server.usage.observer import StreamUsageExecutionObserver
from inference_server.usage.rows import USAGE_SCOPE
from inference_server.workflows import host as workflows_host
from tests.unit_tests.legacy.conftest import FakeGateway

INLINE_SPECIFICATION = {"version": "1.0", "inputs": [], "steps": [], "outputs": []}


class _RecordingGateway(FakeGateway):
    async def start(self):
        self.calls.append(("start",))

    async def shutdown(self):
        self.calls.append(("shutdown",))


class _RecordingProfiler:
    def __init__(self):
        self.phases = []
        self.open_phases = []

    @contextmanager
    def profile_execution_phase(self, name, categories=None, metadata=None):
        self.phases.append((name, list(categories or [])))
        self.open_phases.append(name)
        try:
            yield
        finally:
            self.open_phases.remove(name)


@pytest.fixture
def gateway(monkeypatch):
    fake = _RecordingGateway()
    monkeypatch.setattr(
        "inference_server.gateway_resolver.resolve_gateway", lambda: fake
    )

    return fake


@pytest.fixture
def pipeline_host(gateway):
    host = ServerPipelineHost(gateway_kind="direct")
    yield host
    host.close()


def test_inline_specification_yields_gateway_backed_init_parameters(
    pipeline_host, gateway
):
    specification, init_parameters, step_error_handler = pipeline_host.prepare_workflow(
        workflow_specification=INLINE_SPECIFICATION,
        workspace_name=None,
        workflow_id=None,
        workflow_version_id=None,
        api_key="key-1",
        profiler=_RecordingProfiler(),
    )

    assert specification is INLINE_SPECIFICATION
    assert init_parameters["workflows_core.api_key"] == "key-1"
    assert init_parameters["workflows_core.model_manager"] is not None
    assert init_parameters["workflows_core.background_tasks"] is None
    assert init_parameters["workflows_core.disable_sinks"] is False
    assert "workflows_core.step_execution_mode" not in init_parameters
    assert step_error_handler is workflows_host.step_error_handler
    assert ("start",) in gateway.calls


def test_api_key_falls_back_to_the_server_default(pipeline_host, monkeypatch):
    monkeypatch.setattr(configuration, "DEFAULT_API_KEY", "env-key")

    _, init_parameters, _ = pipeline_host.prepare_workflow(
        workflow_specification=INLINE_SPECIFICATION,
        workspace_name=None,
        workflow_id=None,
        workflow_version_id=None,
        api_key=None,
        profiler=_RecordingProfiler(),
    )

    assert init_parameters["workflows_core.api_key"] == "env-key"


def test_missing_specification_and_ids_raise_the_legacy_value_error(pipeline_host):
    with pytest.raises(ValueError) as error:
        pipeline_host.prepare_workflow(
            workflow_specification=None,
            workspace_name="workspace",
            workflow_id=None,
            workflow_version_id=None,
            api_key="key-1",
            profiler=_RecordingProfiler(),
        )

    assert str(error.value) == (
        "Either (`workspace_name`, `workflow_id`) or `workflow_specification` "
        "must be provided."
    )


def test_named_workflow_without_api_key_raises_missing_api_key(
    pipeline_host, monkeypatch
):
    monkeypatch.setattr(configuration, "DEFAULT_API_KEY", None)

    with pytest.raises(MissingApiKeyError):
        pipeline_host.prepare_workflow(
            workflow_specification=None,
            workspace_name="workspace",
            workflow_id="workflow",
            workflow_version_id=None,
            api_key=None,
            profiler=_RecordingProfiler(),
        )


def test_named_workflow_is_fetched_inside_the_profiler_phase(
    pipeline_host, monkeypatch
):
    profiler = _RecordingProfiler()
    recorded = {}

    def fake_get_workflow_specification(**kwargs):
        recorded["kwargs"] = kwargs
        recorded["open_phases"] = list(profiler.open_phases)
        return {"fetched": True}

    monkeypatch.setattr(
        workflows_host, "get_workflow_specification", fake_get_workflow_specification
    )

    specification, init_parameters, _ = pipeline_host.prepare_workflow(
        workflow_specification=None,
        workspace_name="workspace",
        workflow_id="workflow",
        workflow_version_id="3",
        api_key="key-1",
        profiler=profiler,
    )

    assert specification == {"fetched": True}
    assert recorded["kwargs"] == {
        "api_key": "key-1",
        "workspace_id": "workspace",
        "workflow_id": "workflow",
        "workflow_version_id": "3",
        "use_cache": True,
    }
    assert recorded["open_phases"] == ["workflow_definition_fetching"]
    assert profiler.phases == [
        ("workflow_definition_fetching", ["inference_package_operation"])
    ]
    assert init_parameters["workflows_core.api_key"] == "key-1"


def test_close_shuts_the_gateway_and_stops_the_loop_thread(gateway):
    host = ServerPipelineHost(gateway_kind="direct")
    host.prepare_workflow(
        workflow_specification=INLINE_SPECIFICATION,
        workspace_name=None,
        workflow_id=None,
        workflow_version_id=None,
        api_key="key-1",
        profiler=_RecordingProfiler(),
    )
    thread = host._thread

    host.close()
    host.close()

    assert thread is not None
    assert not thread.is_alive()
    assert gateway.calls.count(("shutdown",)) == 1
    with pytest.raises(RuntimeError):
        host.prepare_workflow(
            workflow_specification=INLINE_SPECIFICATION,
            workspace_name=None,
            workflow_id=None,
            workflow_version_id=None,
            api_key="key-1",
            profiler=_RecordingProfiler(),
        )


def test_close_before_first_use_starts_nothing(gateway):
    host = ServerPipelineHost(gateway_kind="direct")

    host.close()

    assert host._thread is None
    assert gateway.calls == []


def test_gateway_kind_is_written_to_the_environment_only_when_absent(
    gateway, monkeypatch
):
    monkeypatch.delenv(configuration.INFERENCE_GATEWAY_ENV, raising=False)
    host = ServerPipelineHost(gateway_kind="descriptor-kind")
    host.prepare_workflow(
        workflow_specification=INLINE_SPECIFICATION,
        workspace_name=None,
        workflow_id=None,
        workflow_version_id=None,
        api_key="key-1",
        profiler=_RecordingProfiler(),
    )
    host.close()

    assert os.environ.get(configuration.INFERENCE_GATEWAY_ENV) == "descriptor-kind"

    monkeypatch.setenv(configuration.INFERENCE_GATEWAY_ENV, "inherited-kind")
    host = ServerPipelineHost(gateway_kind="descriptor-kind")
    host.prepare_workflow(
        workflow_specification=INLINE_SPECIFICATION,
        workspace_name=None,
        workflow_id=None,
        workflow_version_id=None,
        api_key="key-1",
        profiler=_RecordingProfiler(),
    )
    host.close()

    assert os.environ.get(configuration.INFERENCE_GATEWAY_ENV) == "inherited-kind"


def test_descriptor_resolves_to_a_server_pipeline_host(gateway):
    assert isinstance(SERVER_PIPELINE_HOST_DESCRIPTOR, PipelineHostDescriptor)
    assert SERVER_PIPELINE_HOST_DESCRIPTOR.factory == (
        "inference_server.streams.host:ServerPipelineHost"
    )
    assert SERVER_PIPELINE_HOST_DESCRIPTOR.settings == {
        "gateway_kind": os.environ.get(
            configuration.INFERENCE_GATEWAY_ENV, configuration.INFERENCE_GATEWAY_DEFAULT
        )
    }

    host = create_pipeline_host(SERVER_PIPELINE_HOST_DESCRIPTOR)
    host.close()

    assert isinstance(host, ServerPipelineHost)


def _loop_threads():
    return [
        thread
        for thread in threading.enumerate()
        if thread.name == "pipeline-host-loop" and thread.is_alive()
    ]


def _prepare(host):
    return host.prepare_workflow(
        workflow_specification=INLINE_SPECIFICATION,
        workspace_name=None,
        workflow_id=None,
        workflow_version_id=None,
        api_key="key-1",
        profiler=_RecordingProfiler(),
    )


class _FailingStartGateway(_RecordingGateway):
    async def start(self):
        raise RuntimeError("start failed")


class _BlockedStartGateway(_RecordingGateway):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.loop = None
        self.release = None

    async def start(self):
        self.loop = asyncio.get_running_loop()
        self.release = asyncio.Event()
        self.entered.set()
        await self.release.wait()


class _SlowShutdownGateway(_RecordingGateway):
    async def shutdown(self):
        await asyncio.sleep(2.0)


def test_failed_gateway_start_tears_down_and_the_host_retries(monkeypatch):
    bad, good = _FailingStartGateway(), _RecordingGateway()
    gateways = [bad, good]
    monkeypatch.setattr(
        "inference_server.gateway_resolver.resolve_gateway",
        lambda: gateways.pop(0),
    )
    host = ServerPipelineHost(gateway_kind="direct")

    with pytest.raises(RuntimeError, match="start failed"):
        _prepare(host)

    assert ("shutdown",) in bad.calls
    assert _loop_threads() == []

    _prepare(host)
    host.close()

    assert ("start",) in good.calls
    assert good.calls.count(("shutdown",)) == 1
    assert _loop_threads() == []


def test_failed_codec_binding_tears_the_gateway_down(gateway, monkeypatch):
    def failing_bind_loop(loop_bridge):
        raise RuntimeError("bind failed")

    monkeypatch.setattr(
        workflows_host.GUARDED_IMAGE_CODEC, "bind_loop", failing_bind_loop
    )
    host = ServerPipelineHost(gateway_kind="direct")

    with pytest.raises(RuntimeError, match="bind failed"):
        _prepare(host)

    assert gateway.calls.count(("shutdown",)) == 1
    assert _loop_threads() == []
    assert host._loop is None


@pytest.mark.timeout(20)
def test_close_during_a_pending_start_returns_and_the_starter_tears_down(
    monkeypatch,
):
    monkeypatch.setattr(streams_host, "CLOSE_TIMEOUT_S", 0.5)
    blocked = _BlockedStartGateway()
    monkeypatch.setattr(
        "inference_server.gateway_resolver.resolve_gateway", lambda: blocked
    )
    host = ServerPipelineHost(gateway_kind="direct")
    outcome = {}

    def starter():
        try:
            _prepare(host)
        except BaseException as error:
            outcome["error"] = error

    starter_thread = threading.Thread(target=starter)
    starter_thread.start()
    assert blocked.entered.wait(5)

    started_at = time.monotonic()
    host.close()
    elapsed = time.monotonic() - started_at

    assert elapsed < 2.0
    assert starter_thread.is_alive()

    blocked.loop.call_soon_threadsafe(blocked.release.set)
    starter_thread.join(10)

    assert not starter_thread.is_alive()
    assert isinstance(outcome["error"], RuntimeError)
    assert "closed" in str(outcome["error"])
    assert blocked.calls.count(("shutdown",)) == 1
    assert _loop_threads() == []


@pytest.mark.timeout(20)
def test_close_is_bounded_by_the_deadline_when_the_shutdown_is_slow(
    monkeypatch, caplog
):
    monkeypatch.setattr(streams_host, "CLOSE_TIMEOUT_S", 0.5)
    slow = _SlowShutdownGateway()
    monkeypatch.setattr(
        "inference_server.gateway_resolver.resolve_gateway", lambda: slow
    )
    host = ServerPipelineHost(gateway_kind="direct")
    _prepare(host)

    with caplog.at_level(logging.WARNING, logger=streams_host.logger.name):
        started_at = time.monotonic()
        host.close()
        elapsed = time.monotonic() - started_at

    assert elapsed < 1.5
    assert "Could not shut the pipeline gateway down" in caplog.text
    assert _loop_threads() == []


class _RecordingCollector:
    instances = []

    def __init__(self):
        self.calls = []
        self.events = None
        self.flush_delay = 0.0
        self.stop_timeouts = []
        _RecordingCollector.instances.append(self)

    def _note(self, call):
        self.calls.append(call)
        if self.events is not None:
            self.events.append(call)

    def start(self):
        self._note("start")

    def record_usage(self, **row):
        self._note("record_usage")

    def flush(self):
        time.sleep(self.flush_delay)
        self._note("flush")

    def stop(self, timeout=None):
        self.stop_timeouts.append(timeout)
        self._note("stop")

        return True


@pytest.fixture
def collectors(monkeypatch):
    _RecordingCollector.instances = []
    monkeypatch.setattr(
        "inference_server.usage.collector.UsageCollector", _RecordingCollector
    )

    return _RecordingCollector.instances


def _only_collector(collectors):
    assert len(collectors) == 1

    return collectors[0]


def test_collector_starts_with_the_gateway_and_is_bound_to_the_run(gateway, collectors):
    host = ServerPipelineHost(gateway_kind="direct")

    _, init_parameters, _ = _prepare(host)

    collector = _only_collector(collectors)
    assert collector.calls == ["start"]
    observer = init_parameters["workflows_core.execution_observer"]
    assert isinstance(observer, StreamUsageExecutionObserver)
    with observer.scope_binding():
        scope = USAGE_SCOPE.get()
    assert scope is observer.scope
    assert scope.collector is collector
    assert scope.api_key == "key-1"
    provider = init_parameters["workflows_core.model_manager"]
    assert provider._bridge._usage_scope is scope
    assert USAGE_SCOPE.get() is None
    observer.observe_workflow_run(
        workflow=None,
        runtime_parameters={},
        workflow_id="wf",
        fps=0,
        is_preview=False,
        run=lambda: None,
    )
    assert collector.calls == ["start", "record_usage"]
    host.close()


def test_one_collector_serves_every_workflow_of_the_host(gateway, collectors):
    host = ServerPipelineHost(gateway_kind="direct")

    _, first, _ = _prepare(host)
    _, second, _ = _prepare(host)
    host.close()

    assert len(collectors) == 1
    assert (
        first["workflows_core.execution_observer"]
        is not second["workflows_core.execution_observer"]
    )


def test_offline_mode_builds_and_starts_a_collector_like_online(
    gateway, collectors, monkeypatch
):
    monkeypatch.setattr(configuration, "LEGACY_OFFLINE_MODE", True)
    host = ServerPipelineHost(gateway_kind="direct")

    _, init_parameters, _ = _prepare(host)
    host.close()

    collector = _only_collector(collectors)
    assert collector.calls[0] == "start"
    assert isinstance(
        init_parameters["workflows_core.execution_observer"],
        StreamUsageExecutionObserver,
    )
    assert gateway.calls.count(("shutdown",)) == 1


def test_close_flushes_and_stops_the_collector_before_the_gateway_shutdown(
    collectors, monkeypatch
):
    events = []

    class _EventGateway(_RecordingGateway):
        async def shutdown(self):
            events.append("shutdown")

    monkeypatch.setattr(
        "inference_server.gateway_resolver.resolve_gateway", _EventGateway
    )
    host = ServerPipelineHost(gateway_kind="direct")
    _prepare(host)
    collector = _only_collector(collectors)
    collector.events = events

    host.close()
    host.close()

    assert events == ["flush", "stop", "shutdown"]
    assert collector.calls == ["start", "flush", "stop"]
    assert 0 < collector.stop_timeouts[0] <= streams_host.CLOSE_TIMEOUT_S
    assert _loop_threads() == []


@pytest.mark.timeout(20)
def test_close_with_a_slow_flush_stays_within_the_deadline(
    gateway, collectors, monkeypatch, caplog
):
    monkeypatch.setattr(streams_host, "CLOSE_TIMEOUT_S", 0.5)
    host = ServerPipelineHost(gateway_kind="direct")
    _prepare(host)
    collector = _only_collector(collectors)
    collector.flush_delay = 2.0

    with caplog.at_level(logging.WARNING, logger=streams_host.logger.name):
        started_at = time.monotonic()
        host.close()
        elapsed = time.monotonic() - started_at

    assert elapsed < 1.5
    assert "usage collector" in caplog.text
    deadline = time.monotonic() + 5.0
    while _loop_threads() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _loop_threads() == []


class _PartialStartCollector(_RecordingCollector):
    def __init__(self):
        super().__init__()
        self.running = False

    def start(self):
        self.running = True
        raise RuntimeError("partial start")

    def stop(self, timeout=None):
        self.running = False

        return super().stop(timeout)


def test_collector_failing_after_starting_is_stopped_and_the_host_retries(
    gateway, monkeypatch
):
    created = []

    def factory():
        collector = _PartialStartCollector() if not created else _RecordingCollector()
        created.append(collector)

        return collector

    monkeypatch.setattr("inference_server.usage.collector.UsageCollector", factory)
    host = ServerPipelineHost(gateway_kind="direct")

    with pytest.raises(RuntimeError, match="partial start"):
        _prepare(host)

    failed = created[0]
    assert failed.running is False
    assert failed.calls == ["stop"]
    assert gateway.calls.count(("shutdown",)) == 1
    assert _loop_threads() == []
    assert host._loop is None

    _prepare(host)
    host.close()

    assert len(created) == 2
    assert created[1].calls == ["start", "flush", "stop"]


def test_observer_receives_the_attribution_inputs(gateway, collectors, monkeypatch):
    fetched = {"fetched": True}
    monkeypatch.setattr(
        workflows_host, "get_workflow_specification", lambda **kwargs: fetched
    )
    host = ServerPipelineHost(gateway_kind="direct")

    def prepare(workflow_specification):
        _, init_parameters, _ = host.prepare_workflow(
            workflow_specification=workflow_specification,
            workspace_name="workspace",
            workflow_id="requested",
            workflow_version_id=None,
            api_key="key-1",
            profiler=_RecordingProfiler(),
        )

        return init_parameters["workflows_core.execution_observer"]

    inline = prepare(INLINE_SPECIFICATION)
    named = prepare(None)
    host.close()

    collector = _only_collector(collectors)
    for observer in (inline, named):
        assert observer.scope.collector is collector
        assert observer.scope.api_key == "key-1"
        assert observer.scope.billable is True


def test_failed_codec_binding_tears_the_collector_down(
    gateway, collectors, monkeypatch
):
    def failing_bind_loop(loop_bridge):
        raise RuntimeError("bind failed")

    monkeypatch.setattr(
        workflows_host.GUARDED_IMAGE_CODEC, "bind_loop", failing_bind_loop
    )
    host = ServerPipelineHost(gateway_kind="direct")

    with pytest.raises(RuntimeError, match="bind failed"):
        _prepare(host)

    collector = _only_collector(collectors)
    assert collector.calls == ["start", "stop"]
    assert gateway.calls.count(("shutdown",)) == 1
    assert _loop_threads() == []


def test_failed_collector_start_tears_the_gateway_down(gateway, monkeypatch):
    def failing_collector():
        raise RuntimeError("collector failed")

    monkeypatch.setattr(
        "inference_server.usage.collector.UsageCollector", failing_collector
    )
    host = ServerPipelineHost(gateway_kind="direct")

    with pytest.raises(RuntimeError, match="collector failed"):
        _prepare(host)

    assert gateway.calls.count(("shutdown",)) == 1
    assert _loop_threads() == []
    assert host._loop is None


def test_host_module_does_not_import_the_app():
    source = Path(streams_host.__file__).read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)

    assert "inference_server.app" not in imported

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import inference_server.streams.host; "
            "assert 'inference_server.app' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_importing_host_first_keeps_server_workflows_configuration():
    env = {**os.environ, "PROJECT": "roboflow-staging"}

    result = subprocess.run(
        [sys.executable, "-c", "import inference_server.streams.host"],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
