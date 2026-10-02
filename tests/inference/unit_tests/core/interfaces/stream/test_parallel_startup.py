"""Inert concurrency tests: no network, real models, or GPU execution."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from threading import Barrier, Event
from types import SimpleNamespace

import numpy as np
import pytest
from roboflow_workflows.execution_engine.v1.compiler import core as compiler
from roboflow_workflows.prototypes.block import (
    ModelExecutionLocation,
    ModelRequiredAction,
    roboflow_platform_model,
    third_party_model,
)
from streamvision.stream import pipeline as portable_pipeline_module
from streamvision.stream.parallel_startup import (
    _ParallelWorkflowStartup,
    _StartupResources,
)

from inference.core.interfaces.camera.entities import SourceProperties
from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
    VideoSource,
)
from inference.core.interfaces.stream import inference_pipeline as pipeline_module
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline


class _Manager(dict):
    def __init__(self, load=lambda: None, max_size=8):
        super().__init__()
        self.load = load
        self.max_size = max_size
        self.calls = []
        self.removed = []

    def add_model(self, model_id, api_key, **kwargs):
        self.calls.append((model_id, kwargs))
        self.load()
        self[model_id] = object()

    def remove(self, model_id, delete_from_disk=True):
        self.removed.append((model_id, delete_from_disk))
        del self[model_id]


class _Source:
    def __init__(self, start=lambda: None):
        self.operation = start
        self.started = 0
        self.closed = 0

    def start(self):
        self.started += 1
        self.operation()

    def terminate(self, **kwargs):
        self.closed += 1


def _definition(*dependencies):
    return SimpleNamespace(
        inputs=[],
        steps=[SimpleNamespace(discover_dependent_resources=lambda: dependencies)],
    )


def _startup(manager=None, source=None, cancelled=None, owns_manager=True, limit=1):
    manager = manager if manager is not None else _Manager()

    def cleanup():
        for model_id in list(manager.keys()):
            manager.remove(model_id, delete_from_disk=False)

    return _ParallelWorkflowStartup(
        pipeline=SimpleNamespace(
            _video_sources=[source or _Source()],
            _started_sources=[],
            _sources_startup_finished=Event(),
        ),
        resources=_StartupResources(
            provider=manager,
            available_capacity=max(0, manager.max_size - len(manager)),
            cleanup=cleanup if owns_manager else None,
        ),
        cancelled=cancelled,
        model_limit=limit,
        report=lambda *args: None,
    )


def _prepare(startup, *dependencies):
    startup._on_workflow_parsed(
        _definition(*dependencies), init_parameters={}, api_key="inert"
    )


def test_sources_models_and_real_graph_compilation_overlap(monkeypatch):
    barrier = Barrier(3, timeout=5)
    manager = _Manager(load=barrier.wait)
    source = _Source(start=barrier.wait)
    monkeypatch.setattr(
        pipeline_module, "prepare_video_sources", lambda **kwargs: [source]
    )
    original_graph = compiler.prepare_execution_graph

    def graph(**kwargs):
        barrier.wait()
        return original_graph(**kwargs)

    monkeypatch.setattr(compiler, "prepare_execution_graph", graph)
    workflow = {
        "version": "1.0",
        "inputs": [{"type": "InferenceImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
                "name": "detect_overlap",
                "images": "$inputs.image",
                "model_id": "inert/1",
            }
        ],
        "outputs": [],
    }
    pipeline = InferencePipeline.init_with_workflow(
        video_reference="inert",
        workflow_specification=workflow,
        model_manager=manager,
        parallel_startup=True,
    )
    try:
        assert manager.calls == [("inert/1", {})]
        assert bool(pipeline._parallel_startup.gated_manager)
        assert source.started == 1
        timings = pipeline.startup_phase_timings
        assert max(t["start_seconds"] for t in timings.values()) < min(
            t["end_seconds"] for t in timings.values()
        )
        monkeypatch.setattr(
            portable_pipeline_module, "multiplex_videos", lambda **kwargs: iter([])
        )
        assert list(pipeline._generate_frames()) == []
        assert source.started == 1
        # Cache hits must still discover dependencies for each manager/job.
        source.operation = lambda: None
        second_manager = _Manager()
        second = InferencePipeline.init_with_workflow(
            video_reference="inert",
            workflow_specification=workflow,
            model_manager=second_manager,
            parallel_startup=True,
        )
        second.join()
        assert second_manager.calls == [("inert/1", {})]
    finally:
        pipeline.join()
    assert manager.removed == []  # Caller retains ownership, including on join.


def test_default_pipeline_does_not_start_sources_or_load_models(monkeypatch):
    source = _Source()
    manager = _Manager()
    monkeypatch.setattr(
        pipeline_module, "prepare_video_sources", lambda **kwargs: [source]
    )
    pipeline = InferencePipeline.init_with_workflow(
        video_reference="inert",
        workflow_specification={
            "version": "1.0",
            "inputs": [],
            "steps": [],
            "outputs": [],
        },
        model_manager=manager,
    )
    pipeline.join()
    assert source.started == 0
    assert manager.calls == []


def test_admission_dedup_exclusions_and_registration_kwargs():
    manager = _Manager(max_size=2)
    startup = _startup(manager=manager, limit=5)
    _prepare(
        startup,
        third_party_model(provider="inert", model_id="external"),
        roboflow_platform_model(model_id="$inputs.model"),
        roboflow_platform_model(model_id="$steps.pick.model"),
        roboflow_platform_model(
            model_id="remote/1", execution_location=ModelExecutionLocation.REMOTE
        ),
        roboflow_platform_model(
            model_id="access/1", required_action=ModelRequiredAction.ACCESS
        ),
        roboflow_platform_model(
            model_id="a/1", model_registration_kwargs={"endpoint_type": "core"}
        ),
        roboflow_platform_model(model_id="a/1"),
        roboflow_platform_model(model_id="b/1"),
        roboflow_platform_model(model_id="c/1"),
    )
    startup._finish()
    assert manager.calls == [("a/1", {"endpoint_type": "core"}), ("b/1", {})]
    startup._close()
    assert manager.removed == [("a/1", False), ("b/1", False)]


def test_admission_preserves_existing_cache_entries():
    manager = _Manager(max_size=1)
    manager["resident"] = object()
    startup = _startup(manager=manager, owns_manager=False)
    _prepare(startup, roboflow_platform_model(model_id="new/1"))
    startup._finish()
    startup._close()
    assert manager.calls == []
    assert list(manager) == ["resident"]


@pytest.mark.parametrize("failure_phase", ["models", "sources", "compilation"])
def test_failure_drains_native_siblings_before_owned_cleanup(failure_phase):
    started, release = Event(), Event()
    failure = MemoryError("inert OOM")

    def blocked():
        started.set()
        assert release.wait(5)

    def fail():
        assert started.wait(5)
        raise failure

    source = _Source(start=fail if failure_phase == "sources" else blocked)
    manager = _Manager(load=blocked if failure_phase == "sources" else fail)
    manager["partial"] = object()
    startup = _startup(source=source, manager=manager)
    startup._start_sources()
    if failure_phase != "compilation":
        _prepare(startup, roboflow_platform_model(model_id="inert/1"))
    else:
        with pytest.raises(MemoryError):
            startup._phase("compilation", fail)
    assert started.wait(5)
    with ThreadPoolExecutor(1) as pool:
        closing = pool.submit(startup._close)
        assert not closing.done()
        assert manager.removed == []
        release.set()
        closing.result(timeout=5)
    assert source.closed == 1
    assert ("partial", False) in manager.removed
    startup._close()
    assert source.closed == 1


def test_cancellation_blocks_pipeline_use_but_allows_provider_teardown():
    cancelled = Event()
    startup = _startup(cancelled=cancelled)
    _prepare(startup, roboflow_platform_model(model_id="inert/1"))
    startup._finish()
    cancelled.set()
    with pytest.raises(CancelledError):
        startup._check_cancelled()
    # Runtime close/flush handlers still need the prepared provider after cancel.
    assert startup.gated_manager["inert/1"] is startup.manager["inert/1"]
    startup._close()
    assert startup.manager == {}


def test_manager_gate_blocks_until_model_is_complete():
    entered, release, attempting, accessed = Event(), Event(), Event(), Event()

    def load():
        entered.set()
        assert release.wait(5)

    startup = _startup(manager=_Manager(load=load))
    _prepare(startup, roboflow_platform_model(model_id="inert/1"))
    assert entered.wait(5)

    def read():
        attempting.set()
        value = startup.gated_manager["inert/1"]
        accessed.set()
        return value

    with ThreadPoolExecutor(1) as pool:
        reading = pool.submit(read)
        assert attempting.wait(5)
        assert not accessed.is_set()
        release.set()
        assert reading.result(timeout=5) is startup.manager["inert/1"]
    startup._finish()
    startup._close()


class _Producer:
    def __init__(self, *, is_file=False):
        self.is_file = is_file
        self.counter = 0
        self.open = True
        self.released = 0
        self.produced = Event()
        self.stop = Event()

    def isOpened(self):
        return self.open

    def initialize_source_properties(self, properties):
        pass

    def discover_source_properties(self):
        return SourceProperties(
            width=2,
            height=2,
            fps=30,
            total_frames=10 if self.is_file else 0,
            is_file=self.is_file,
        )

    def grab(self):
        if self.counter >= 10:
            self.produced.set()
            if not self.is_file:
                self.stop.wait(5)
            return False
        self.counter += 1
        return True

    def retrieve(self):
        return True, np.full((2, 2, 3), self.counter, dtype=np.uint8)

    def interrupt(self):
        self.stop.set()

    def release(self):
        self.released += 1
        self.open = False


def test_live_source_stays_bounded_fresh_and_reconnects_with_new_producer():
    producers = []

    def factory():
        producer = _Producer()
        producers.append(producer)
        return producer

    source = VideoSource.init(
        video_reference=factory,
        buffer_size=1,
        buffer_filling_strategy=BufferFillingStrategy.DROP_OLDEST,
        buffer_consumption_strategy=BufferConsumptionStrategy.EAGER,
    )
    startup = _startup(source=source)
    startup._start_sources()
    startup._finish()
    try:
        assert producers[0].produced.wait(5)
        assert source._frames_buffer.qsize() == 1
        assert source.read_frame(timeout=1).image[0, 0, 0] == 10
        source.restart(wait_on_frames_consumption=False, purge_frames_buffer=True)
        assert len(producers) == 2
        assert producers[0].released == 1
        assert producers[1].produced.wait(5)
    finally:
        startup._close()
    assert producers[1].released == 1


def test_file_source_keeps_order_during_early_capture():
    producer = _Producer(is_file=True)
    source = VideoSource.init(video_reference=lambda: producer, buffer_size=2)
    startup = _startup(source=source)
    startup._start_sources()
    startup._finish()
    try:
        frames = [source.read_frame(timeout=2).image[0, 0, 0] for _ in range(10)]
        assert frames == list(range(1, 11))
    finally:
        startup._close()
    assert producer.released == 1


def test_abort_unconsumed_file_with_one_frame_buffer_does_not_deadlock():
    producer = _Producer(is_file=True)
    source = VideoSource.init(video_reference=lambda: producer, buffer_size=1)
    startup = _startup(source=source)
    startup._start_sources()
    startup._finish()
    # No inference consumer has ever run. The capture thread can be waiting to
    # enqueue a frame and still needs room for its final end marker.
    with ThreadPoolExecutor(1) as pool:
        pool.submit(startup._close).result(timeout=3)
    from inference.core.interfaces.camera.video_source import POISON_PILL

    assert source._frames_buffer.get_nowait() == POISON_PILL
    source._frames_buffer.task_done()
    assert producer.released == 1


@pytest.mark.parametrize(
    "parameters,expected", [({}, "default/1"), ({"model": "selected/2"}, "selected/2")]
)
def test_fixed_model_parameters_and_defaults_use_validated_manifests(
    parameters, expected
):
    from roboflow_workflows.core_steps.models.roboflow.object_detection.v3 import (
        BlockManifest,
    )
    from roboflow_workflows.execution_engine.entities.base import WorkflowParameter
    from roboflow_workflows.execution_engine.v1.compiler.entities import (
        ParsedWorkflowDefinition,
    )

    definition = ParsedWorkflowDefinition(
        version="1.0",
        outputs=[],
        inputs=[
            WorkflowParameter(
                type="WorkflowParameter", name="model", default_value="default/1"
            )
        ],
        steps=[
            BlockManifest(
                type="roboflow_core/roboflow_object_detection_model@v3",
                name="detect",
                images="$inputs.image",
                model_id="$inputs.model",
            )
        ],
    )
    startup = _startup()
    startup._on_workflow_parsed(
        definition, init_parameters={}, api_key="inert", workflows_parameters=parameters
    )
    startup._finish()
    assert startup.manager.calls == [(expected, {})]
    startup._close()


def test_invalid_fixed_input_is_rejected_before_model_preparation():
    from roboflow_workflows.core_steps.models.roboflow.object_detection.v3 import (
        BlockManifest,
    )
    from roboflow_workflows.errors import RuntimeInputError
    from roboflow_workflows.execution_engine.entities.base import WorkflowParameter
    from roboflow_workflows.execution_engine.v1.compiler.entities import (
        ParsedWorkflowDefinition,
    )

    definition = ParsedWorkflowDefinition(
        version="1.0",
        outputs=[],
        inputs=[WorkflowParameter(type="WorkflowParameter", name="model")],
        steps=[
            BlockManifest(
                type="roboflow_core/roboflow_object_detection_model@v3",
                name="detect",
                images="$inputs.image",
                model_id="$inputs.model",
            )
        ],
    )
    startup = _startup()
    try:
        with pytest.raises(RuntimeInputError):
            startup._on_workflow_parsed(
                definition,
                init_parameters={},
                api_key="inert",
                workflows_parameters={"model": 42},
            )
        assert startup.manager.calls == []
    finally:
        startup._close()


def test_fixed_parameter_model_id_resolver_and_frame_inputs_excluded():
    from roboflow_workflows.execution_engine.entities.base import WorkflowParameter

    startup = _startup(limit=2)
    definition = _definition(
        roboflow_platform_model(
            model_id="$inputs.version", model_id_resolver=lambda value: "clip/" + value
        ),
        roboflow_platform_model(model_id="$inputs.image"),
    )
    definition.inputs = [
        WorkflowParameter(type="WorkflowParameter", name="version", default_value="v1"),
        WorkflowParameter(
            type="WorkflowParameter", name="image", default_value="wrong/1"
        ),
    ]
    definition.steps[0].model_fields = {}
    startup._on_workflow_parsed(
        definition, init_parameters={}, api_key="inert", frame_input_names=("image",)
    )
    startup._finish()
    assert startup.manager.calls == [("clip/v1", {})]
    startup._close()


def test_init_propagates_model_failure_and_preserves_external_manager(monkeypatch):
    source = _Source()
    error = MemoryError("synthetic allocation failure")

    def fail():
        raise error

    manager = _Manager(load=fail)
    manager["resident"] = object()
    monkeypatch.setattr(
        pipeline_module, "prepare_video_sources", lambda **kwargs: [source]
    )
    workflow = {
        "version": "1.0",
        "inputs": [{"type": "InferenceImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
                "name": "fail_model",
                "images": "$inputs.image",
                "model_id": "inert/1",
            }
        ],
        "outputs": [],
    }
    with pytest.raises(MemoryError) as caught:
        InferencePipeline.init_with_workflow(
            video_reference="inert",
            workflow_specification=workflow,
            model_manager=manager,
            parallel_startup=True,
        )
    assert caught.value is error
    assert source.closed == 1
    assert list(manager) == ["resident"]


def test_cancelled_returned_pipeline_cannot_start(monkeypatch):
    source = _Source()
    cancelled = Event()
    monkeypatch.setattr(
        pipeline_module, "prepare_video_sources", lambda **kwargs: [source]
    )
    pipeline = InferencePipeline.init_with_workflow(
        video_reference="inert",
        workflow_specification={
            "version": "1.0",
            "inputs": [],
            "steps": [],
            "outputs": [],
        },
        model_manager=_Manager(),
        parallel_startup=True,
        startup_cancel_event=cancelled,
    )
    cancelled.set()
    with pytest.raises(CancelledError):
        pipeline.start(use_main_thread=False)
    assert pipeline._inference_thread is None
    assert source.closed == 1


def test_memory_pressure_eviction_is_reported_without_extra_loading(
    caplog, monkeypatch
):
    import logging

    monkeypatch.setattr(logging.getLogger("inference"), "propagate", True)

    class EvictingManager(_Manager):
        def add_model(self, model_id, api_key, **kwargs):
            self.clear()
            super().add_model(model_id, api_key, **kwargs)

    manager = EvictingManager()
    startup = _startup(manager=manager, limit=2)
    with caplog.at_level("WARNING", logger="inference"):
        _prepare(
            startup,
            roboflow_platform_model(model_id="first/1"),
            roboflow_platform_model(model_id="second/1"),
        )
        startup._finish()
    assert manager.calls == [("first/1", {}), ("second/1", {})]
    assert "first/1" in caplog.text
    assert list(manager) == ["second/1"]
    startup._close()


def test_cleanup_failure_retains_resource_for_retry():
    source = _Source()
    startup = _startup(source=source)
    startup._start_sources()
    startup._finish()
    original = source.terminate

    def fail(**kwargs):
        raise RuntimeError("inert release failure")

    source.terminate = fail
    with pytest.raises(RuntimeError, match="resource cleanup failed"):
        startup._close()
    assert not startup._closed
    assert startup._attempted_sources == [source]
    source.terminate = original
    startup._close()
    assert startup._closed
    assert source.closed == 1


def test_cancellation_during_model_load_drains_before_cleanup():
    entered, release, cancelled = Event(), Event(), Event()

    def load():
        entered.set()
        assert release.wait(5)

    startup = _startup(manager=_Manager(load=load), cancelled=cancelled)
    _prepare(startup, roboflow_platform_model(model_id="late/1"))
    assert entered.wait(5)
    cancelled.set()
    with ThreadPoolExecutor(1) as pool:
        finish = pool.submit(startup._finish)
        assert not finish.done()
        release.set()
        with pytest.raises(CancelledError):
            finish.result(timeout=5)
    assert "late/1" in startup.manager
    startup._close()
    assert startup.manager == {}


def test_cleanup_before_capture_thread_begins_reading():
    from inference.core.interfaces.camera.video_source import (
        SOURCE_STATE_UPDATE_EVENT,
        VIDEO_CONSUMPTION_STARTED_EVENT,
        StreamState,
    )

    entered, release, terminating = Event(), Event(), Event()

    def status(update):
        if update.event_type == VIDEO_CONSUMPTION_STARTED_EVENT:
            entered.set()
            assert release.wait(5)
        if (
            update.event_type == SOURCE_STATE_UPDATE_EVENT
            and update.payload["new_state"] == StreamState.TERMINATING
        ):
            terminating.set()

    producer = _Producer()
    source = VideoSource.init(
        video_reference=lambda: producer, status_update_handlers=[status], buffer_size=1
    )
    startup = _startup(source=source)
    startup._start_sources()
    startup._finish()
    assert entered.wait(5)
    assert source.describe_source().state is StreamState.RUNNING
    with ThreadPoolExecutor(1) as pool:
        closing = pool.submit(startup._close)
        try:
            assert terminating.wait(2)
        finally:
            release.set()
        closing.result(timeout=3)
    assert producer.released == 1
    assert not source._stream_consumption_thread.is_alive()


def test_selected_cuda_device_and_sync_barrier_are_carried_to_workers(monkeypatch):
    import sys
    from contextlib import contextmanager
    from threading import local

    state, synced = local(), Event()

    @contextmanager
    def device(index):
        state.device = index
        yield

    def assert_device():
        assert state.device == 3

    def synchronize():
        assert_device()
        synced.set()

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_initialized=lambda: True,
            current_device=lambda: 3,
            device=device,
            synchronize=synchronize,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    startup = _startup(
        source=_Source(start=assert_device), manager=_Manager(load=assert_device)
    )
    startup._start_sources()
    _prepare(startup, roboflow_platform_model(model_id="inert/1"))
    startup._finish()
    assert synced.is_set()
    startup._close()


@pytest.mark.parametrize("limit,expected", [(0, []), (1, [("branch/1", {})])])
def test_conditional_branch_preloading_is_explicit_and_does_not_run_workflow(
    monkeypatch, limit, expected
):
    manager = _Manager()
    monkeypatch.setattr(
        pipeline_module, "prepare_video_sources", lambda **kwargs: [_Source()]
    )
    from roboflow_workflows.execution_engine.core import ExecutionEngine

    def forbidden_run(*args, **kwargs):
        raise AssertionError("Startup must not execute a workflow")

    monkeypatch.setattr(ExecutionEngine, "run", forbidden_run)
    workflow = {
        "version": "1.0",
        "inputs": [{"type": "InferenceImage", "name": "image"}],
        "steps": [
            {
                "type": "ContinueIf",
                "name": "never",
                "condition_statement": {
                    "type": "StatementGroup",
                    "statements": [
                        {
                            "type": "BinaryStatement",
                            "left_operand": {"type": "StaticOperand", "value": 0},
                            "comparator": {"type": "(Number) =="},
                            "right_operand": {"type": "StaticOperand", "value": 1},
                        }
                    ],
                },
                "evaluation_parameters": {},
                "next_steps": ["$steps.branch"],
            },
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
                "name": "branch",
                "images": "$inputs.image",
                "model_id": "branch/1",
            },
        ],
        "outputs": [],
    }
    pipeline = InferencePipeline.init_with_workflow(
        video_reference="inert",
        workflow_specification=workflow,
        model_manager=manager,
        parallel_startup=True,
        startup_model_limit=limit,
    )
    pipeline.join()
    assert manager.calls == expected


def test_partial_multiple_sources_are_closed_without_opening_remaining_sources():
    failure = RuntimeError("inert source failure")

    def fail():
        raise failure

    sources = [_Source(), _Source(start=fail), _Source()]
    startup = _startup()
    startup.pipeline._video_sources = sources
    startup._start_sources()
    with pytest.raises(RuntimeError) as caught:
        startup._finish()
    assert caught.value is failure
    startup._close()
    assert [source.started for source in sources] == [1, 1, 0]
    assert [source.closed for source in sources] == [1, 1, 0]


def test_oom_on_second_model_removes_first_owned_model_after_drain():
    manager = _Manager(max_size=2)

    def load():
        if len(manager.calls) == 2:
            raise MemoryError("inert second model OOM")

    manager.load = load
    startup = _startup(manager=manager, limit=2)
    _prepare(
        startup,
        roboflow_platform_model(model_id="first/1"),
        roboflow_platform_model(model_id="second/1"),
    )
    with pytest.raises(MemoryError):
        startup._finish()
    assert list(manager) == ["first/1"]
    startup._close()
    assert manager == {}
    assert manager.removed == [("first/1", False)]


def test_real_manager_reuses_prepared_instance_on_first_add_and_releases_owned_model(
    monkeypatch,
):
    from unittest.mock import MagicMock

    from inference.core.managers import base as base_module
    from inference.core.managers.base import ModelManager
    from inference.core.managers.decorators import fixed_size_cache as cache_module
    from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache

    monkeypatch.setattr(base_module, "MODELS_CACHE_AUTH_ENABLED", False)
    monkeypatch.setattr(cache_module, "MODELS_CACHE_AUTH_ENABLED", False)
    registry, model_class = MagicMock(), MagicMock()
    registry.get_model.return_value = model_class
    manager = WithFixedSizeCache(ModelManager(model_registry=registry), max_size=1)
    startup = _startup(manager=manager)
    _prepare(startup, roboflow_platform_model(model_id="same/1"))
    startup._finish()
    startup.gated_manager.add_model(model_id="same/1", api_key="inert")
    assert startup.gated_manager["same/1"] is model_class.return_value
    assert model_class.call_count == 1
    assert registry.get_model.call_count == 1
    startup._close()
    assert len(manager) == 0
    model_class.return_value.clear_cache.assert_called_once_with(delete_from_disk=False)


def test_profiler_failure_cannot_release_sources_before_executor_drain(monkeypatch):
    executors = []

    def make_executor(**kwargs):
        executor = ThreadPoolExecutor(**kwargs)
        executors.append(executor)
        return executor

    def fail_export(**kwargs):
        raise RuntimeError("inert export failure")

    source = _Source()
    monkeypatch.setattr(portable_pipeline_module, "ThreadPoolExecutor", make_executor)
    monkeypatch.setattr(
        pipeline_module, "prepare_video_sources", lambda **kwargs: [source]
    )
    monkeypatch.setattr(portable_pipeline_module, "on_pipeline_end", fail_export)
    pipeline = InferencePipeline.init_with_workflow(
        video_reference="inert",
        workflow_specification={
            "version": "1.0",
            "inputs": [],
            "steps": [],
            "outputs": [],
        },
        model_manager=_Manager(),
        parallel_startup=True,
    )
    entered, release = Event(), Event()

    def task():
        entered.set()
        assert release.wait(5)

    executors[0].submit(task)
    assert entered.wait(5)
    with ThreadPoolExecutor(1) as pool:
        joining = pool.submit(pipeline.join)
        assert not joining.done()
        assert source.closed == 0
        release.set()
        with pytest.raises(RuntimeError, match="export failure"):
            joining.result(timeout=5)
    assert source.closed == 1
