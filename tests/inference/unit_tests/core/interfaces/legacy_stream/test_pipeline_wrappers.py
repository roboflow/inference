"""WP-A02: the legacy `InferencePipeline` wrappers over the host-neutral pipeline.

Engine-facing assertions capture the arguments of the real
`ExecutionEngine.init` call (the engine itself is replaced by a recording
double), so they observe what the Execution Engine actually receives rather
than what a mocked delegate was asked to do. Only video decoding is kept out:
`prepare_video_sources` is patched where each class looks it up - the legacy
module (the historical `inference_pipeline` name) for the legacy class,
`inference.core.interfaces.stream.pipeline` for the neutral one - when a test
needs no real source.
"""

import asyncio
import dataclasses
import os
import sys
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.exceptions import CannotInitialiseModelError, MissingApiKeyError
from inference.core.interfaces.camera.entities import SourceProperties, VideoFrame
from inference.core.interfaces.camera.exceptions import EndOfStreamError
from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
    SourceMetadata,
    StreamState,
)
from inference.core.interfaces.legacy_stream import inference_pipeline as legacy_module
from inference.core.interfaces.legacy_stream.inference_pipeline import (
    InferencePipeline as LegacyInferencePipeline,
)
from inference.core.interfaces.legacy_stream.inference_pipeline import (
    PreparedWorkflow,
    prepare_workflow_for_pipeline,
)
from inference.core.interfaces.stream import pipeline as core_module
from inference.core.interfaces.stream.entities import ModelConfig
from inference.core.interfaces.stream.model_handlers.workflows import WorkflowRunner
from inference.core.interfaces.stream.pipeline import (
    InferencePipeline as CoreInferencePipeline,
)
from inference.core.interfaces.stream.sinks import active_learning_sink
from inference.core.interfaces.stream.watchdog import BasePipelineWatchDog
from inference.core.interfaces.stream_manager.manager_app.entities import (
    WebRTCOffer,
    WorkflowConfiguration,
)
from inference.core.interfaces.workflows_configuration import (
    server_workflows_configuration,
)
from inference.core.interfaces.workflows_execution_observer import (
    UsageTrackingExecutionObserver,
)
from inference.core.interfaces.workflows_models_provider import (
    ModelManagerModelsProvider,
)
from inference.core.interfaces.workflows_step_error_handlers import (
    SERVER_STEP_ERROR_HANDLERS,
)
from inference.core.managers.active_learning import BackgroundTaskActiveLearningManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.workflows.configuration import ensure_process_configuration_matches
from inference.core.workflows.errors import WorkflowEnvironmentConfigurationError
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.profiling.core import (
    BaseWorkflowsProfiler,
    NullWorkflowsProfiler,
)
from inference.core.workflows.prototypes.image_codec import reset_image_codec

MINIMAL_SPEC = {"version": "1.0", "inputs": [], "steps": [], "outputs": []}
QUEUE_SIZE_ENV = "INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE"


@pytest.fixture(autouse=True)
def _clean_image_codec_registry():
    # The binder installs the process-wide image codec; keep tests isolated.
    reset_image_codec()
    yield
    reset_image_codec()


class _RecordingEngine:
    """Stands in for the object `ExecutionEngine.init` returns."""

    def __init__(self) -> None:
        self.runs: List[Dict[str, Any]] = []

    def run(self, **kwargs) -> List[dict]:
        self.runs.append(kwargs)
        images = kwargs["runtime_parameters"]["image"]
        return [{"frame": index} for index in range(len(images))]


class _EngineInitCapture:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.engine = _RecordingEngine()

    def __call__(self, **kwargs) -> _RecordingEngine:
        self.calls.append(kwargs)
        return self.engine

    @property
    def kwargs(self) -> Dict[str, Any]:
        assert len(self.calls) == 1, self.calls
        return self.calls[0]


@pytest.fixture
def engine_init(monkeypatch) -> _EngineInitCapture:
    capture = _EngineInitCapture()
    monkeypatch.setattr(ExecutionEngine, "init", capture)
    return capture


@pytest.fixture
def video_sources(monkeypatch) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []

    def fake_prepare_video_sources(**kwargs):
        calls.append(kwargs)
        return []

    for module in (core_module, legacy_module):
        monkeypatch.setattr(module, "prepare_video_sources", fake_prepare_video_sources)
    return calls


class _DuckModelManager:
    """Class-shaped manager without `__workflows_bind__` (not a ModelManager)."""

    def add_model(self, *args, **kwargs):
        raise AssertionError("not exercised")

    def infer_from_request_sync(self, *args, **kwargs):
        raise AssertionError("not exercised")

    def __contains__(self, item) -> bool:
        return True


# ---------------------------------------------------------------------------
# Legacy init_with_workflow: what the Execution Engine and pipeline receive
# ---------------------------------------------------------------------------


def test_legacy_workflow_wrapper_forwards_every_argument(
    engine_init: _EngineInitCapture,
    video_sources: List[Dict[str, Any]],
) -> None:
    duck = _DuckModelManager()
    watchdog = BasePipelineWatchDog()
    status_handler = MagicMock()
    on_prediction = MagicMock()

    pipeline = LegacyInferencePipeline.init_with_workflow(
        video_reference=["a.mp4", "b.mp4"],
        workflow_specification=MINIMAL_SPEC,
        workflow_id="my-workflow",
        api_key="fresh-key",
        image_input_name="frame_in",
        workflows_parameters={"zone": [1, 2]},
        on_prediction=on_prediction,
        max_fps=7,
        watchdog=watchdog,
        status_update_handlers=[status_handler],
        source_buffer_filling_strategy=BufferFillingStrategy.DROP_OLDEST,
        source_buffer_consumption_strategy=BufferConsumptionStrategy.EAGER,
        video_source_properties={"fps": 15.0},
        disable_sinks=True,
        workflows_thread_pool_workers=3,
        execution_engine_thread_pool_workers=5,
        cancel_thread_pool_tasks_on_exit=False,
        video_metadata_input_name="meta_in",
        batch_collection_timeout=0.25,
        video_processing_mode="auto",
        max_staleness=0.75,
        profiling_directory="/tmp/some-profiles",
        serialize_results=True,
        predictions_queue_size=21,
        decoding_buffer_size=9,
        model_manager=duck,
        _is_preview=True,
        exec_session_id="session-1",
        workflows_dependencies_pre_init=["roboflow_platform_model"],
    )

    engine_kwargs = engine_init.kwargs
    assert engine_kwargs["workflow_definition"] is MINIMAL_SPEC
    assert engine_kwargs["workflow_id"] == "my-workflow"
    assert engine_kwargs["dependencies_pre_init"] == ["roboflow_platform_model"]
    init_parameters = engine_kwargs["init_parameters"]
    assert init_parameters["workflows_core.model_manager"]._model_manager is duck
    assert init_parameters["workflows_core.api_key"] == "fresh-key"
    assert init_parameters["workflows_core.disable_sinks"] is True
    blocks_executor = init_parameters["workflows_core.thread_pool_executor"]
    engine_executor = engine_kwargs["executor"]
    assert blocks_executor is not engine_executor
    assert blocks_executor._max_workers == 3
    assert engine_executor._max_workers == 5

    runner = pipeline._on_video_frame
    assert isinstance(runner, WorkflowRunner)
    assert runner._execution_engine is engine_init.engine
    assert runner._workflows_parameters == {"zone": [1, 2]}
    assert runner._image_input_name == "frame_in"
    assert runner._video_metadata_input_name == "meta_in"
    assert runner._serialize_results is True
    assert runner._is_preview is True

    assert len(video_sources) == 1
    assert video_sources[0]["video_reference"] == ["a.mp4", "b.mp4"]
    assert video_sources[0]["video_source_properties"] == {"fps": 15.0}
    assert (
        video_sources[0]["source_buffer_filling_strategy"]
        is BufferFillingStrategy.DROP_OLDEST
    )
    assert (
        video_sources[0]["source_buffer_consumption_strategy"]
        is BufferConsumptionStrategy.EAGER
    )
    assert video_sources[0]["decoding_buffer_size"] == 9
    assert (
        video_sources[0]["allow_tensor_frames"]
        is legacy_module.ENABLE_TENSOR_DATA_REPRESENTATION
    )
    assert pipeline._on_prediction is on_prediction
    assert pipeline._max_fps == 7
    assert pipeline._watchdog is watchdog
    assert pipeline._status_update_handlers[0] is status_handler
    assert pipeline._batch_collection_timeout == 0.25
    assert pipeline._collection_policy._max_staleness == 0.75
    assert pipeline._predictions_queue.maxsize == 21
    assert pipeline._stream_session_id == "session-1"
    assert pipeline._on_pipeline_start is None
    on_end = pipeline._on_pipeline_end
    assert on_end.keywords["thread_pool_executor"] is blocks_executor
    assert on_end.keywords["execution_engine_thread_pool_executor"] is engine_executor
    assert on_end.keywords["cancel_thread_pool_tasks_on_exit"] is False
    assert on_end.keywords["profiling_directory"] == "/tmp/some-profiles"
    assert on_end.keywords["profiler"] is engine_kwargs["profiler"]
    blocks_executor.shutdown()
    engine_executor.shutdown()


def test_legacy_workflow_wrapper_mutates_and_passes_the_callers_dict(
    engine_init: _EngineInitCapture, video_sources
) -> None:
    caller_parameters = {"my_plugin.setting": 42}

    LegacyInferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workflow_specification=MINIMAL_SPEC,
        api_key="fresh-key",
        model_manager=_DuckModelManager(),
        workflow_init_parameters=caller_parameters,
    )

    assert engine_init.kwargs["init_parameters"] is caller_parameters
    assert caller_parameters["my_plugin.setting"] == 42
    assert {
        "workflows_core.model_manager",
        "workflows_core.api_key",
        "workflows_core.execution_observer",
        "workflows_core.thread_pool_executor",
        "workflows_core.disable_sinks",
        "workflows_core.configuration",
        "workflows_core.image_codec",
    } <= set(caller_parameters)


def test_legacy_workflow_wrapper_overwrites_observer_but_keeps_bare_observer(
    engine_init: _EngineInitCapture, video_sources
) -> None:
    bare_observer = object()
    namespaced_observer = object()

    LegacyInferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workflow_specification=MINIMAL_SPEC,
        api_key="fresh-key",
        model_manager=_DuckModelManager(),
        workflow_init_parameters={
            "execution_observer": bare_observer,
            "workflows_core.execution_observer": namespaced_observer,
        },
    )

    init_parameters = engine_init.kwargs["init_parameters"]
    assert init_parameters["execution_observer"] is bare_observer
    assert isinstance(
        init_parameters["workflows_core.execution_observer"],
        UsageTrackingExecutionObserver,
    )


def test_legacy_workflow_wrapper_hands_the_binders_provider_to_the_engine(
    engine_init: _EngineInitCapture, video_sources, monkeypatch
) -> None:
    # The provider the explicit binder installs is the very object the engine
    # receives: the host-neutral pipeline must not rewrap it.
    installed = []
    real_binder = legacy_module.bind_model_manager_to_workflows

    def spying_binder(**kwargs):
        result = real_binder(**kwargs)
        installed.append(kwargs["init_parameters"]["workflows_core.model_manager"])
        return result

    monkeypatch.setattr(legacy_module, "bind_model_manager_to_workflows", spying_binder)
    duck = _DuckModelManager()

    LegacyInferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workflow_specification=MINIMAL_SPEC,
        api_key="fresh-key",
        model_manager=duck,
    )

    provider = engine_init.kwargs["init_parameters"]["workflows_core.model_manager"]
    assert installed == [provider]
    assert isinstance(provider, ModelManagerModelsProvider)
    assert provider._model_manager is duck
    assert not hasattr(duck, "__workflows_bind__")
    assert not hasattr(provider, "__workflows_bind__")


def test_legacy_workflow_wrapper_falls_back_to_api_key_setting(
    engine_init: _EngineInitCapture, video_sources, monkeypatch
) -> None:
    # Patched through the historical module name: it is the legacy module.
    import inference.core.interfaces.stream.inference_pipeline as historical

    monkeypatch.setattr(historical, "API_KEY", "env-key")

    LegacyInferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workflow_specification=MINIMAL_SPEC,
        model_manager=_DuckModelManager(),
    )

    assert engine_init.kwargs["init_parameters"]["workflows_core.api_key"] == "env-key"


def test_legacy_workflow_wrapper_requires_a_specification_or_named_workflow(
    engine_init: _EngineInitCapture,
) -> None:
    with pytest.raises(ValueError, match="workflow_specification"):
        LegacyInferencePipeline.init_with_workflow(
            video_reference="video.mp4",
            workspace_name="only-workspace",
            api_key="fresh-key",
        )

    assert engine_init.calls == []


def test_legacy_workflow_wrapper_needs_api_key_to_fetch_named_workflow(
    engine_init: _EngineInitCapture, monkeypatch
) -> None:
    import inference.core.roboflow_api as roboflow_api

    fetch = MagicMock()
    monkeypatch.setattr(roboflow_api, "get_workflow_specification", fetch)
    monkeypatch.setattr(legacy_module, "API_KEY", None)

    with pytest.raises(MissingApiKeyError):
        LegacyInferencePipeline.init_with_workflow(
            video_reference="video.mp4",
            workspace_name="my-workspace",
            workflow_id="my-workflow",
        )

    fetch.assert_not_called()
    assert engine_init.calls == []


def test_legacy_workflow_wrapper_fetches_named_workflow_inside_the_profiler(
    engine_init: _EngineInitCapture, video_sources, monkeypatch
) -> None:
    import inference.core.roboflow_api as roboflow_api

    fetched_spec = dict(MINIMAL_SPEC)
    fetch = MagicMock(return_value=fetched_spec)
    monkeypatch.setattr(roboflow_api, "get_workflow_specification", fetch)
    monkeypatch.setattr(legacy_module, "ENABLE_WORKFLOWS_PROFILING", True)
    monkeypatch.setattr(legacy_module, "WORKFLOWS_PROFILER_BUFFER_SIZE", 3)

    pipeline = LegacyInferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workspace_name="my-workspace",
        workflow_id="my-workflow",
        workflow_version_id="v7",
        use_workflow_definition_cache=False,
        api_key="fresh-key",
        model_manager=_DuckModelManager(),
    )

    fetch.assert_called_once_with(
        api_key="fresh-key",
        workspace_id="my-workspace",
        workflow_id="my-workflow",
        workflow_version_id="v7",
        use_cache=False,
    )
    engine_kwargs = engine_init.kwargs
    assert engine_kwargs["workflow_definition"] is fetched_spec
    assert engine_kwargs["workflow_id"] == "my-workflow"
    profiler = engine_kwargs["profiler"]
    assert isinstance(profiler, BaseWorkflowsProfiler)
    assert pipeline._on_pipeline_end.keywords["profiler"] is profiler
    assert "workflow_definition_fetching" in [
        event["name"] for event in profiler.export_trace()
    ]


def test_legacy_workflow_wrapper_prefers_inline_specification_over_fetch(
    engine_init: _EngineInitCapture, video_sources, monkeypatch
) -> None:
    import inference.core.roboflow_api as roboflow_api

    fetch = MagicMock()
    monkeypatch.setattr(roboflow_api, "get_workflow_specification", fetch)

    LegacyInferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workflow_specification=MINIMAL_SPEC,
        workspace_name="my-workspace",
        workflow_id="my-workflow",
        api_key="fresh-key",
        model_manager=_DuckModelManager(),
    )

    fetch.assert_not_called()
    assert engine_init.kwargs["workflow_definition"] is MINIMAL_SPEC
    assert engine_init.kwargs["workflow_id"] == "my-workflow"


def test_legacy_workflow_wrapper_uses_null_profiler_when_profiling_disabled(
    engine_init: _EngineInitCapture, video_sources, monkeypatch
) -> None:
    monkeypatch.setattr(legacy_module, "ENABLE_WORKFLOWS_PROFILING", False)

    LegacyInferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workflow_specification=MINIMAL_SPEC,
        api_key="fresh-key",
        model_manager=_DuckModelManager(),
    )

    assert isinstance(engine_init.kwargs["profiler"], NullWorkflowsProfiler)


def test_legacy_workflow_wrapper_builds_default_stack_only_without_manager(
    engine_init: _EngineInitCapture, video_sources, monkeypatch
) -> None:
    registry = MagicMock(wraps=legacy_module.RoboflowModelRegistry)
    monkeypatch.setattr(legacy_module, "RoboflowModelRegistry", registry)

    LegacyInferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workflow_specification=MINIMAL_SPEC,
        api_key="fresh-key",
        model_manager=_DuckModelManager(),
    )
    registry.assert_not_called()

    LegacyInferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workflow_specification=MINIMAL_SPEC,
        api_key="fresh-key",
    )
    registry.assert_called_once()
    default_manager = engine_init.calls[1]["init_parameters"][
        "workflows_core.model_manager"
    ]._model_manager
    assert isinstance(default_manager, WithFixedSizeCache)
    assert isinstance(
        default_manager.model_manager, BackgroundTaskActiveLearningManager
    )
    assert default_manager.max_size == legacy_module.MAX_ACTIVE_MODELS


@pytest.mark.parametrize(
    "handler_name", ["legacy", "extended_roboflow_errors", "not-a-known-handler"]
)
def test_legacy_workflow_wrapper_passes_the_resolved_step_error_handler(
    engine_init: _EngineInitCapture, video_sources, monkeypatch, handler_name: str
) -> None:
    monkeypatch.setenv("DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER", handler_name)

    LegacyInferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workflow_specification=MINIMAL_SPEC,
        api_key="fresh-key",
        model_manager=_DuckModelManager(),
    )

    handler = engine_init.kwargs["step_error_handler"]
    if handler_name in SERVER_STEP_ERROR_HANDLERS:
        assert handler is SERVER_STEP_ERROR_HANDLERS[handler_name]
    else:
        # Unknown names reach the engine unchanged; it rejects them itself.
        assert handler == handler_name


def test_legacy_workflow_wrapper_rejects_mismatched_configuration_before_engine(
    engine_init: _EngineInitCapture, video_sources
) -> None:
    # Same error the real engine raises for this configuration; the binder
    # raises it before touching the process-wide codec or the engine.
    not_a_configuration = object()
    with pytest.raises(WorkflowEnvironmentConfigurationError) as engine_error:
        ensure_process_configuration_matches(not_a_configuration)

    with pytest.raises(WorkflowEnvironmentConfigurationError) as wrapper_error:
        LegacyInferencePipeline.init_with_workflow(
            video_reference="video.mp4",
            workflow_specification=MINIMAL_SPEC,
            api_key="fresh-key",
            model_manager=_DuckModelManager(),
            workflow_init_parameters={
                "workflows_core.configuration": not_a_configuration
            },
        )

    assert str(wrapper_error.value) == str(engine_error.value)
    assert engine_init.calls == []


def test_legacy_workflow_wrapper_keeps_an_equal_explicit_configuration(
    engine_init: _EngineInitCapture, video_sources
) -> None:
    supplied = dataclasses.replace(server_workflows_configuration())

    LegacyInferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workflow_specification=MINIMAL_SPEC,
        api_key="fresh-key",
        model_manager=_DuckModelManager(),
        workflow_init_parameters={"workflows_core.configuration": supplied},
    )

    assert engine_init.kwargs["init_parameters"]["workflows_core.configuration"] is (
        supplied
    )


def test_legacy_workflow_wrapper_wraps_preparation_import_errors(
    engine_init: _EngineInitCapture, monkeypatch
) -> None:
    monkeypatch.setitem(
        sys.modules, "inference.core.interfaces.workflows_execution_observer", None
    )

    with pytest.raises(CannotInitialiseModelError) as error:
        LegacyInferencePipeline.init_with_workflow(
            video_reference="video.mp4",
            workflow_specification=MINIMAL_SPEC,
            api_key="fresh-key",
            model_manager=_DuckModelManager(),
        )

    assert isinstance(error.value.__cause__, ImportError)
    assert engine_init.calls == []


def test_legacy_workflow_wrapper_wraps_engine_import_errors(monkeypatch) -> None:
    import_error = ImportError("plugin dependency missing")

    def failing_init(**kwargs):
        raise import_error

    monkeypatch.setattr(ExecutionEngine, "init", failing_init)

    with pytest.raises(CannotInitialiseModelError) as error:
        LegacyInferencePipeline.init_with_workflow(
            video_reference="video.mp4",
            workflow_specification=MINIMAL_SPEC,
            api_key="fresh-key",
            model_manager=_DuckModelManager(),
        )

    assert error.value.__cause__ is import_error


def test_prepare_workflow_for_pipeline_is_the_wrappers_composition(
    monkeypatch,
) -> None:
    # The shared helper the stream manager's legacy host reuses (WP-A03).
    monkeypatch.setattr(legacy_module, "API_KEY", "env-key")
    caller_parameters: Dict[str, Any] = {}
    duck = _DuckModelManager()

    prepared = prepare_workflow_for_pipeline(
        workflow_specification=MINIMAL_SPEC,
        workspace_name=None,
        workflow_id=None,
        workflow_version_id=None,
        api_key=None,
        use_workflow_definition_cache=True,
        workflow_init_parameters=caller_parameters,
        model_manager=duck,
        profiler=NullWorkflowsProfiler.init(),
    )

    assert isinstance(prepared, PreparedWorkflow)
    assert prepared.workflow_specification is MINIMAL_SPEC
    assert prepared.workflow_init_parameters is caller_parameters
    assert caller_parameters["workflows_core.api_key"] == "env-key"
    assert caller_parameters["workflows_core.model_manager"]._model_manager is duck
    assert callable(prepared.step_error_handler) or isinstance(
        prepared.step_error_handler, str
    )
    # The pipeline, not the host, owns the executors and the sink policy.
    assert "workflows_core.thread_pool_executor" not in caller_parameters
    assert "workflows_core.disable_sinks" not in caller_parameters


# ---------------------------------------------------------------------------
# The host-neutral constructor
# ---------------------------------------------------------------------------


def test_core_workflow_constructor_passes_host_bindings_through_unchanged(
    engine_init: _EngineInitCapture, video_sources
) -> None:
    provider = object()
    handler = object()
    profiler = NullWorkflowsProfiler.init()
    init_parameters = {"workflows_core.model_manager": provider}

    pipeline = CoreInferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workflow_specification=MINIMAL_SPEC,
        workflow_init_parameters=init_parameters,
        step_error_handler=handler,
        profiler=profiler,
        disable_sinks=True,
    )

    engine_kwargs = engine_init.kwargs
    assert engine_kwargs["init_parameters"] is init_parameters
    assert engine_kwargs["step_error_handler"] is handler
    assert engine_kwargs["profiler"] is profiler
    # Only the pipeline-owned keys are added: no API key, observer, provider
    # wrapping or platform/codec/configuration bindings.
    assert set(init_parameters) == {
        "workflows_core.model_manager",
        "workflows_core.thread_pool_executor",
        "workflows_core.disable_sinks",
    }
    assert init_parameters["workflows_core.model_manager"] is provider
    assert init_parameters["workflows_core.disable_sinks"] is True
    assert type(pipeline) is CoreInferencePipeline
    assert not isinstance(pipeline, LegacyInferencePipeline)
    init_parameters["workflows_core.thread_pool_executor"].shutdown()
    engine_kwargs["executor"].shutdown()


def test_core_workflow_constructor_requires_host_resolved_inputs() -> None:
    with pytest.raises(TypeError):
        CoreInferencePipeline.init_with_workflow(
            video_reference="video.mp4",
            workflow_specification=MINIMAL_SPEC,
        )
    with pytest.raises(TypeError):
        # Keyword-only: the historical positional order is not accepted.
        CoreInferencePipeline.init_with_workflow("video.mp4", MINIMAL_SPEC, {}, None)


def test_core_workflow_constructor_builds_profiler_from_stream_configuration(
    engine_init: _EngineInitCapture, video_sources, monkeypatch
) -> None:
    monkeypatch.setattr(core_module, "ENABLE_WORKFLOWS_PROFILING", True)
    monkeypatch.setattr(core_module, "WORKFLOWS_PROFILER_BUFFER_SIZE", 2)
    # Legacy (core.env) settings do not steer the neutral pipeline.
    monkeypatch.setattr(legacy_module, "ENABLE_WORKFLOWS_PROFILING", False)

    CoreInferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workflow_specification=MINIMAL_SPEC,
        workflow_init_parameters={},
        step_error_handler=None,
    )

    assert isinstance(engine_init.kwargs["profiler"], BaseWorkflowsProfiler)


def test_legacy_factories_construct_the_class_they_are_called_on(
    engine_init: _EngineInitCapture, video_sources, monkeypatch
) -> None:
    monkeypatch.setattr(
        legacy_module, "get_model", lambda **kwargs: MagicMock(task_type="od")
    )

    custom = LegacyInferencePipeline.init_with_custom_logic(
        video_reference="video.mp4", on_video_frame=lambda frames: []
    )
    workflow = LegacyInferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workflow_specification=MINIMAL_SPEC,
        api_key="fresh-key",
        model_manager=_DuckModelManager(),
    )
    model = LegacyInferencePipeline.init(
        video_reference="video.mp4",
        model_id="some/1",
        api_key="fresh-key",
        active_learning_enabled=False,
    )
    core_custom = CoreInferencePipeline.init_with_custom_logic(
        video_reference="video.mp4", on_video_frame=lambda frames: []
    )

    for pipeline in (custom, workflow, model):
        assert type(pipeline) is LegacyInferencePipeline
        assert isinstance(pipeline, CoreInferencePipeline)
    assert type(core_custom) is CoreInferencePipeline
    assert not isinstance(core_custom, LegacyInferencePipeline)


def test_legacy_yolo_world_forwards_config_and_wraps_import_errors(
    video_sources, monkeypatch
) -> None:
    import inference.core.interfaces.stream.model_handlers.yolo_world as historical

    handler = MagicMock()
    build = MagicMock(return_value=handler)
    monkeypatch.setattr(historical, "build_yolo_world_inference_function", build)

    pipeline = LegacyInferencePipeline.init_with_yolo_world(
        video_reference="video.mp4", classes=["cat"], model_size="m", confidence=0.7
    )

    assert build.call_args.kwargs["model_id"] == "yolo_world/m"
    assert build.call_args.kwargs["classes"] == ["cat"]
    assert isinstance(build.call_args.kwargs["inference_config"], ModelConfig)
    assert build.call_args.kwargs["inference_config"].confidence == 0.7
    assert pipeline._on_video_frame is handler
    assert type(pipeline) is LegacyInferencePipeline

    monkeypatch.setitem(
        sys.modules,
        "inference.core.interfaces.legacy_stream.model_handlers.yolo_world",
        None,
    )
    with pytest.raises(CannotInitialiseModelError) as error:
        LegacyInferencePipeline.init_with_yolo_world(
            video_reference="video.mp4", classes=["cat"]
        )
    assert isinstance(error.value.__cause__, ImportError)


# ---------------------------------------------------------------------------
# Prediction-queue cap under stream-pipelined RF-DETR
# ---------------------------------------------------------------------------


def test_legacy_queue_cap_checks_the_environment_when_the_pipeline_is_built(
    video_sources, monkeypatch
) -> None:
    monkeypatch.setenv("RFDETR_PIPELINE_DEPTH", "2")
    monkeypatch.delenv(QUEUE_SIZE_ENV, raising=False)
    # Whatever the configuration captured at import time is irrelevant here.
    monkeypatch.setattr(core_module, "PREDICTIONS_QUEUE_SIZE_EXPLICIT", True)

    capped = LegacyInferencePipeline.init_with_custom_logic(
        video_reference="video.mp4",
        on_video_frame=lambda frames: [],
        predictions_queue_size=64,
    )
    monkeypatch.setenv(QUEUE_SIZE_ENV, "64")
    explicit = LegacyInferencePipeline.init_with_custom_logic(
        video_reference="video.mp4",
        on_video_frame=lambda frames: [],
        predictions_queue_size=64,
    )

    assert capped._predictions_queue.maxsize == 4
    assert explicit._predictions_queue.maxsize == 64


@pytest.mark.parametrize("explicit, expected_size", [(False, 4), (True, 64)])
def test_core_queue_cap_uses_the_configured_presence_flag(
    video_sources, monkeypatch, explicit: bool, expected_size: int
) -> None:
    monkeypatch.setenv("RFDETR_PIPELINE_DEPTH", "2")
    # The process environment is not consulted by the neutral pipeline.
    monkeypatch.setenv(QUEUE_SIZE_ENV, "64")
    monkeypatch.setattr(core_module, "PREDICTIONS_QUEUE_SIZE_EXPLICIT", explicit)

    pipeline = CoreInferencePipeline.init_with_custom_logic(
        video_reference="video.mp4",
        on_video_frame=lambda frames: [],
        predictions_queue_size=64,
    )

    assert pipeline._predictions_queue.maxsize == expected_size


def test_queue_cap_is_not_applied_without_stream_pipelined_rfdetr(
    video_sources, monkeypatch
) -> None:
    monkeypatch.delenv("RFDETR_PIPELINE_DEPTH", raising=False)
    monkeypatch.delenv(QUEUE_SIZE_ENV, raising=False)

    pipeline = LegacyInferencePipeline.init_with_custom_logic(
        video_reference="video.mp4",
        on_video_frame=lambda frames: [],
        predictions_queue_size=64,
    )

    assert pipeline._predictions_queue.maxsize == 64


# ---------------------------------------------------------------------------
# Frames reach the WorkflowRunner unchanged
# ---------------------------------------------------------------------------


class _FileSourceStub:
    def __init__(self, frames: List[VideoFrame]) -> None:
        self._frames = list(frames)
        self.source_id = 0

    def start(self) -> None:
        pass

    def terminate(
        self, wait_on_frames_consumption: bool = True, purge_frames_buffer: bool = False
    ) -> None:
        pass

    def describe_source(self) -> SourceMetadata:
        return SourceMetadata(
            source_properties=SourceProperties(
                width=4, height=4, total_frames=len(self._frames), is_file=True, fps=25
            ),
            source_reference="stub.mp4",
            buffer_size=8,
            state=StreamState.RUNNING,
            buffer_filling_strategy=None,
            buffer_consumption_strategy=None,
            source_id=self.source_id,
        )

    def read_frame(self, timeout: Optional[float] = None) -> VideoFrame:
        if not self._frames:
            raise EndOfStreamError()
        return self._frames.pop(0)


def _frame(image: Any, frame_id: int) -> VideoFrame:
    return VideoFrame(
        image=image, frame_id=frame_id, frame_timestamp=datetime.now(), source_id=0
    )


def test_workflow_pipeline_hands_frame_pixels_to_the_engine_unchanged(
    engine_init: _EngineInitCapture, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    array = np.zeros((4, 4, 3), dtype=np.uint8)
    tensor = torch.zeros((3, 4, 4), dtype=torch.uint8)
    source = _FileSourceStub([_frame(array, 1), _frame(tensor, 2)])
    monkeypatch.setattr(
        legacy_module, "prepare_video_sources", lambda **kwargs: [source]
    )
    sink_calls = []

    pipeline = LegacyInferencePipeline.init_with_workflow(
        video_reference="stub.mp4",
        workflow_specification=MINIMAL_SPEC,
        api_key="fresh-key",
        model_manager=_DuckModelManager(),
        on_prediction=lambda predictions, frame: sink_calls.append(
            (predictions, frame)
        ),
    )
    pipeline.start(use_main_thread=True)
    pipeline.join()

    runs = engine_init.engine.runs
    assert len(runs) == 2
    assert runs[0]["runtime_parameters"]["image"][0]["value"] is array
    assert runs[1]["runtime_parameters"]["image"][0]["value"] is tensor
    assert len(sink_calls) == 2
    assert sink_calls[0][1].image is array
    assert sink_calls[1][1].image is tensor


# ---------------------------------------------------------------------------
# Active learning sink: structural injection
# ---------------------------------------------------------------------------


def test_active_learning_sink_accepts_any_batch_registrar() -> None:
    class _Registrar:
        def __init__(self) -> None:
            self.calls = []

        def register_batch(self, **kwargs) -> None:
            self.calls.append(kwargs)

    registrar = _Registrar()
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    active_learning_sink(
        predictions=[{"predictions": []}, None],
        video_frame=[_frame(image, 1), None],
        active_learning_middleware=registrar,
        model_type="object-detection",
        disable_preproc_auto_orient=True,
    )

    assert len(registrar.calls) == 1
    call = registrar.calls[0]
    assert call["inference_inputs"][0] is image
    assert call["predictions"] == [{"predictions": []}]
    assert call["prediction_type"] == "object-detection"
    assert call["disable_preproc_auto_orient"] is True


def test_model_config_stays_canonical_in_stream_entities() -> None:
    from inference.core.interfaces.legacy_stream.model_handlers import (
        roboflow_models,
        yolo_world,
    )

    assert ModelConfig.__module__ == "inference.core.interfaces.stream.entities"
    assert legacy_module.ModelConfig is ModelConfig
    assert roboflow_models.ModelConfig is ModelConfig
    assert yolo_world.ModelConfig is ModelConfig


# ---------------------------------------------------------------------------
# In-process WebRTC worker: the model manager reaches the engine at every layer
# ---------------------------------------------------------------------------


def _webrtc_workflow_configuration() -> WorkflowConfiguration:
    # Declares the "image" output the video-track variant streams back.
    return WorkflowConfiguration(
        type="WorkflowConfiguration",
        workflow_specification={
            **MINIMAL_SPEC,
            "outputs": [{"type": "JsonField", "name": "image", "selector": "$x"}],
        },
    )


def _provider_manager(engine_init: _EngineInitCapture) -> Any:
    provider = engine_init.kwargs["init_parameters"]["workflows_core.model_manager"]
    assert isinstance(provider, ModelManagerModelsProvider)
    return provider._model_manager


def test_webrtc_frame_processor_passes_model_manager_to_the_engine(
    engine_init: _EngineInitCapture,
) -> None:
    from inference.core.interfaces.webrtc_worker.webrtc import VideoFrameProcessor

    duck = _DuckModelManager()

    VideoFrameProcessor(
        asyncio_loop=MagicMock(),
        workflow_configuration=_webrtc_workflow_configuration(),
        api_key="api-key",
        model_manager=duck,
        has_video_track=False,
    )

    assert _provider_manager(engine_init) is duck
    assert engine_init.kwargs["init_parameters"]["workflows_core.api_key"] == "api-key"


def test_webrtc_video_track_passes_model_manager_to_the_engine(
    engine_init: _EngineInitCapture,
) -> None:
    from inference.core.interfaces.webrtc_worker.webrtc import (
        VideoTransformTrackWithLoop,
    )

    duck = _DuckModelManager()

    VideoTransformTrackWithLoop(
        asyncio_loop=MagicMock(),
        workflow_configuration=_webrtc_workflow_configuration(),
        api_key="api-key",
        model_manager=duck,
    )

    assert _provider_manager(engine_init) is duck


@pytest.mark.parametrize("stream_output", [[], ["image"]])
def test_webrtc_peer_connection_init_passes_model_manager_to_the_engine(
    monkeypatch, stream_output: List[str]
) -> None:
    # The engine records its inputs and then fails with an error the worker
    # reports through `send_answer`, so nothing past pipeline creation runs.
    from inference.core.interfaces.webrtc_worker.entities import WebRTCWorkerRequest
    from inference.core.interfaces.webrtc_worker.webrtc import (
        init_rtc_peer_connection_with_loop,
    )

    captured = []

    def recording_then_failing_init(**kwargs):
        captured.append(kwargs)
        raise MissingApiKeyError("stop after engine initialisation")

    monkeypatch.setattr(ExecutionEngine, "init", recording_then_failing_init)
    duck = _DuckModelManager()
    answers = []
    request = WebRTCWorkerRequest(
        api_key="api-key",
        workflow_configuration=_webrtc_workflow_configuration(),
        webrtc_offer=WebRTCOffer(type="offer", sdp="v=0"),
        stream_output=stream_output,
        processing_timeout=None,
    )

    asyncio.run(
        init_rtc_peer_connection_with_loop(
            webrtc_request=request,
            send_answer=answers.append,
            asyncio_loop=MagicMock(),
            model_manager=duck,
        )
    )

    assert len(captured) == 1
    provider = captured[0]["init_parameters"]["workflows_core.model_manager"]
    assert provider._model_manager is duck
    assert [answer.exception_type for answer in answers] == ["MissingApiKeyError"]


# ---------------------------------------------------------------------------
# Composition points reachable through the historical module names
# ---------------------------------------------------------------------------


def test_historical_names_are_the_legacy_modules() -> None:
    import inference
    import inference.core.interfaces.stream.inference_pipeline as historical_pipeline
    import inference.core.interfaces.stream.stream as historical_stream

    assert historical_pipeline is legacy_module
    assert historical_pipeline.InferencePipeline is LegacyInferencePipeline
    assert inference.InferencePipeline is LegacyInferencePipeline
    assert historical_pipeline.SinkMode is core_module.SinkMode
    assert (
        historical_pipeline.INFERENCE_THREAD_FINISHED_EVENT
        is core_module.INFERENCE_THREAD_FINISHED_EVENT
    )
    assert (
        historical_pipeline.send_inference_pipeline_status_update
        is core_module.send_inference_pipeline_status_update
    )
    assert inference.Stream is historical_stream.Stream
    assert historical_stream.Stream.__module__ == (
        "inference.core.interfaces.legacy_stream.stream"
    )


def test_historical_module_keeps_its_baseline_public_names() -> None:
    import json
    from pathlib import Path

    import inference.core.interfaces.stream.inference_pipeline as historical

    inventory = json.loads(
        (
            Path(__file__).resolve().parents[3] / "streams_compat_inventory.json"
        ).read_text()
    )
    for entry in inventory["modules"]:
        if not entry["retained_host_exception"]:
            continue
        module = __import__(entry["module"], fromlist=["_"])
        missing = [
            name
            for name in entry["public_top_level_names"]
            if not hasattr(module, name)
        ]
        assert missing == [], entry["module"]
    assert historical.os is os
    assert historical.partial is partial


HISTORICAL_PIPELINE = "inference.core.interfaces.stream.inference_pipeline"


def test_historical_profiler_patch_controls_fetch_and_engine_profiler(
    engine_init: _EngineInitCapture, video_sources, monkeypatch
) -> None:
    import inference.core.roboflow_api as roboflow_api

    class _PatchedProfiler(BaseWorkflowsProfiler):
        pass

    monkeypatch.setattr(
        roboflow_api,
        "get_workflow_specification",
        MagicMock(return_value=dict(MINIMAL_SPEC)),
    )
    # The patch targets development/stream_interface/run_workflow_on_video.py
    # has always used.
    monkeypatch.setattr(f"{HISTORICAL_PIPELINE}.ENABLE_WORKFLOWS_PROFILING", True)
    monkeypatch.setattr(f"{HISTORICAL_PIPELINE}.WORKFLOWS_PROFILER_BUFFER_SIZE", 3)
    monkeypatch.setattr(
        f"{HISTORICAL_PIPELINE}.BaseWorkflowsProfiler", _PatchedProfiler
    )

    pipeline = LegacyInferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workspace_name="my-workspace",
        workflow_id="my-workflow",
        api_key="fresh-key",
        model_manager=_DuckModelManager(),
    )

    profiler = engine_init.kwargs["profiler"]
    assert type(profiler) is _PatchedProfiler
    assert profiler._runs_buffer.maxlen == 3
    assert "workflow_definition_fetching" in [
        event["name"] for event in profiler.export_trace()
    ]
    assert pipeline._on_pipeline_end.keywords["profiler"] is profiler


def test_historical_null_profiler_patch_controls_engine_profiler(
    engine_init: _EngineInitCapture, video_sources, monkeypatch
) -> None:
    class _PatchedNullProfiler(NullWorkflowsProfiler):
        pass

    monkeypatch.setattr(f"{HISTORICAL_PIPELINE}.ENABLE_WORKFLOWS_PROFILING", False)
    monkeypatch.setattr(
        f"{HISTORICAL_PIPELINE}.NullWorkflowsProfiler", _PatchedNullProfiler
    )

    LegacyInferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workflow_specification=MINIMAL_SPEC,
        api_key="fresh-key",
        model_manager=_DuckModelManager(),
    )

    assert type(engine_init.kwargs["profiler"]) is _PatchedNullProfiler


def test_historical_composition_patches_steer_legacy_pipelines_only(
    monkeypatch,
) -> None:
    legacy_sources: List[Dict[str, Any]] = []
    core_sources: List[Dict[str, Any]] = []
    multiplexed: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        f"{HISTORICAL_PIPELINE}.prepare_video_sources",
        lambda **kwargs: legacy_sources.append(kwargs) or [],
    )
    monkeypatch.setattr(
        core_module,
        "prepare_video_sources",
        lambda **kwargs: core_sources.append(kwargs) or [],
    )
    monkeypatch.setattr(
        f"{HISTORICAL_PIPELINE}.ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING", True
    )
    monkeypatch.setattr(
        core_module, "ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING", False
    )
    monkeypatch.setattr(
        core_module,
        "multiplex_videos",
        lambda **kwargs: multiplexed.append(kwargs) or iter([]),
    )

    legacy = LegacyInferencePipeline.init_with_custom_logic(
        video_reference="video.mp4", on_video_frame=lambda frames: [], max_fps=5
    )
    core = CoreInferencePipeline.init_with_custom_logic(
        video_reference="video.mp4", on_video_frame=lambda frames: [], max_fps=5
    )
    list(legacy._generate_frames())
    list(core._generate_frames())

    # Frame dropping moves the FPS limit into the source (desired_source_fps)
    # and out of the multiplexer (max_fps).
    assert [call["desired_source_fps"] for call in legacy_sources] == [5]
    assert [call["desired_source_fps"] for call in core_sources] == [None]
    assert [call["max_fps"] for call in multiplexed] == [None, 5]


def test_historical_tensor_setting_steers_legacy_workflow_pipelines_only(
    engine_init: _EngineInitCapture, video_sources, monkeypatch
) -> None:
    monkeypatch.setattr(
        f"{HISTORICAL_PIPELINE}.ENABLE_TENSOR_DATA_REPRESENTATION", True
    )
    monkeypatch.setattr(core_module, "ENABLE_TENSOR_DATA_REPRESENTATION", False)

    LegacyInferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workflow_specification=MINIMAL_SPEC,
        api_key="fresh-key",
        model_manager=_DuckModelManager(),
    )
    CoreInferencePipeline.init_with_workflow(
        video_reference="video.mp4",
        workflow_specification=MINIMAL_SPEC,
        workflow_init_parameters={},
        step_error_handler=None,
    )

    assert [call["allow_tensor_frames"] for call in video_sources] == [True, False]


def test_historical_sinks_names_include_the_concrete_middleware_class() -> None:
    import json
    from pathlib import Path

    from inference.core.active_learning.middlewares import (
        ActiveLearningMiddleware as ConcreteMiddleware,
    )
    from inference.core.interfaces.stream import sinks
    from inference.core.interfaces.stream.sinks import ActiveLearningMiddleware

    assert ActiveLearningMiddleware is ConcreteMiddleware
    assert ActiveLearningMiddleware is not sinks.ActiveLearningBatchRegistrar
    inventory = json.loads(
        (
            Path(__file__).resolve().parents[3] / "streams_compat_inventory.json"
        ).read_text()
    )
    entry = next(
        entry
        for entry in inventory["modules"]
        if entry["module"] == "inference.core.interfaces.stream.sinks"
    )
    missing = [
        name for name in entry["public_top_level_names"] if not hasattr(sinks, name)
    ]
    assert missing == []


_SINKS_EXPORT_SCRIPT = """
import importlib
import inspect
import os
import sys

{first}
import inference.core.interfaces.stream.sinks as sinks
from inference.core.active_learning.middlewares import ActiveLearningMiddleware

assert sinks.ActiveLearningMiddleware is ActiveLearningMiddleware
assert "inference.core.interfaces.legacy_stream.inference_pipeline" not in sys.modules
assert sinks.__spec__.origin == sinks.__file__
assert sinks.__file__.endswith(os.path.join("stream", "sinks.py"))
assert "def active_learning_sink" in inspect.getsource(sinks)
reloaded = importlib.reload(sinks)
assert reloaded is sys.modules["inference.core.interfaces.stream.sinks"] is sinks
assert sinks.ActiveLearningMiddleware is ActiveLearningMiddleware
assert sum(
    bool(getattr(finder, "_roboflow_workflows_compat_finder", False))
    for finder in sys.meta_path
) == 1
"""

_BOOTSTRAP_STAYS_LIGHT = """
import inference.core
assert "inference.core.interfaces.stream.sinks" not in sys.modules
assert "inference.core.active_learning.middlewares" not in sys.modules
"""


@pytest.mark.parametrize(
    "first",
    [
        "",
        _BOOTSTRAP_STAYS_LIGHT,
        "import inference.core.active_learning.middlewares",
        "import inference.core.interfaces.stream.pipeline",
    ],
    ids=["sinks-first", "core-bootstrap-first", "middleware-first", "pipeline-first"],
)
def test_historical_sinks_export_in_a_fresh_interpreter(first: str) -> None:
    import subprocess
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[6]
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "DISABLE_VERSION_CHECK": "True",
            "PYTHONPATH": os.pathsep.join(
                [str(repo_root / "workflows"), str(repo_root / "inference_models")]
            ),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", _SINKS_EXPORT_SCRIPT.format(first=first)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stderr[-4000:]
