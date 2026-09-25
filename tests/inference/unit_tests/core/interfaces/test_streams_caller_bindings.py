"""WP-A00 caller-binding characterization for `InferencePipeline.init_with_workflow`.

Freezes the CURRENT override/mutation contract (plan §6's acceptance matrix:
"always replace namespaced model manager, api_key and observer; preserve
explicit configuration/platform/codec/resolver values") and the current
binding behavior of the four callers that reach it, exactly as they exist
today: a direct user call, the stream manager, the in-process WebRTC worker,
and a duck model manager with no `__workflows_bind__` hook.

Per the A00 scope note in the plan: this does NOT assert that production
already routes through a `PipelineHost` or `bind_model_manager_to_workflows`
(A02/A03 introduce those) - it captures what `init_with_workflow` itself does
today, at inference/core/interfaces/stream/inference_pipeline.py:758-779, so
later work packages can diff their new composition point against this.
"""

import dataclasses
from multiprocessing import Queue
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
import supervision as sv

from inference.core.active_learning.middlewares import (
    NullActiveLearningMiddleware,
    ThreadingActiveLearningMiddleware,
)
from inference.core.entities.requests.inference import ObjectDetectionInferenceRequest
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
)
from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
)
from inference.core.interfaces.roboflow_platform_client import (
    workflows_platform_bindings,
)
from inference.core.interfaces.stream import (
    inference_pipeline as inference_pipeline_module,
)
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.stream.sinks import active_learning_sink, multi_sink
from inference.core.interfaces.stream_manager.manager_app import (
    inference_pipeline_manager,
)
from inference.core.interfaces.stream_manager.manager_app.entities import (
    CommandType,
    InitialisePipelinePayload,
    VideoConfiguration,
    WorkflowConfiguration,
)
from inference.core.interfaces.stream_manager.manager_app.inference_pipeline_manager import (
    InferencePipelineManager,
)
from inference.core.interfaces.webrtc_worker.webrtc import VideoFrameProcessor
from inference.core.interfaces.workflows_configuration import (
    server_workflows_configuration,
)
from inference.core.interfaces.workflows_execution_observer import (
    UsageTrackingExecutionObserver,
)
from inference.core.interfaces.workflows_image_codec import GUARDED_IMAGE_CODEC
from inference.core.interfaces.workflows_models_provider import (
    ModelManagerModelsProvider,
)
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.roboflow.object_detection.v1 import (
    BlockManifest,
    RoboflowObjectDetectionModelBlockV1,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
)
from inference.core.workflows.execution_engine.v1.compiler.steps_initialiser import (
    initialise_step,
)
from inference.core.workflows.prototypes.image_codec import reset_image_codec

MINIMAL_SPEC = {"version": "1.0", "inputs": [], "steps": [], "outputs": []}
MODEL_ID = "some-project/1"
OBJECT_DETECTION_BLOCK = "roboflow_core/roboflow_object_detection_model@v1"


@pytest.fixture(autouse=True)
def _clean_image_codec_registry():
    # `bind_image_codec` sets a process-wide codec once; leaving one behind
    # would make a later test's own install (or override check) fail with
    # `WorkflowEnvironmentConfigurationError` instead of testing what it
    # means to test. Same pattern as test_direct_caller_bindings.py.
    reset_image_codec()
    yield
    reset_image_codec()


def _capture_engine_init(monkeypatch) -> list:
    captured = []

    def fake_init(**kwargs):
        engine = MagicMock()
        captured.append(kwargs)
        return engine

    monkeypatch.setattr(ExecutionEngine, "init", fake_init)
    return captured


class _DuckModelManager:
    """A plugin-shaped manager duck - NOT a `ModelManager` subclass, so it has
    no `__workflows_bind__` hook (`ModelManager.__workflows_bind__` at
    inference/core/managers/base.py:97; a real `ModelManager`, or a
    `spec=ModelManager` mock, would have one).
    """

    def add_model(self, *args, **kwargs):
        raise AssertionError("not exercised by these binding-identity tests")

    def infer_from_request_sync(self, *args, **kwargs):
        raise AssertionError("not exercised by these binding-identity tests")

    def __contains__(self, item) -> bool:
        return True


def _fake_manager_duck() -> _DuckModelManager:
    duck = _DuckModelManager()
    assert not hasattr(duck, "__workflows_bind__")
    return duck


class _WorkingDuckModelManager:
    """Same shape as `_DuckModelManager`, but the manager methods actually
    work, so a real workflow block can be driven end-to-end through the
    provider it gets wrapped in - proving forwarding, not just identity.
    """

    def __init__(self):
        self.add_model_calls = []
        self.infer_calls = []

    def add_model(self, *args, **kwargs):
        self.add_model_calls.append(kwargs)

    def infer_from_request_sync(self, *args, **kwargs):
        self.infer_calls.append(kwargs)
        return ObjectDetectionInferenceResponse(
            image=InferenceResponseImage(width=4, height=4), predictions=[]
        )

    def __contains__(self, item) -> bool:
        return True


def _run_real_detection_block_with_duck(init_parameters: dict, duck) -> list:
    """Adapts `test_direct_caller_bindings.py`'s `_run_real_detection_block` to
    a plain (non-Mock) duck: same real block, built by the compiler out of the
    caller's own `init_parameters`, asserted against the duck's own recorded
    calls instead of Mock assertions.
    """
    init_parameters = dict(init_parameters)
    # init_with_workflow never sets this key itself - it leaves
    # step_execution_mode to ExecutionEngine.init's own env-derived default
    # (core.py's _retrieve_step_execution_mode), which our ExecutionEngine.init
    # stub bypasses. Supplying the same default here stands in for that step.
    init_parameters.setdefault(
        "workflows_core.step_execution_mode", StepExecutionMode.LOCAL
    )
    initialised = initialise_step(
        step_manifest=BlockManifest.model_construct(
            name="detection", type=OBJECT_DETECTION_BLOCK
        ),
        block_specification=BlockSpecification(
            block_source="workflows_core",
            identifier=OBJECT_DETECTION_BLOCK,
            block_class=RoboflowObjectDetectionModelBlockV1,
            manifest_class=BlockManifest,
        ),
        explicit_init_parameters=init_parameters,
        initializers={},
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="image"),
        numpy_image=np.zeros((4, 4, 3), dtype=np.uint8),
    )
    result = initialised.step.run(
        images=Batch(content=[image], indices=[(0,)]),
        model_id=MODEL_ID,
        class_agnostic_nms=False,
        class_filter=None,
        confidence=0.4,
        iou_threshold=0.3,
        max_detections=300,
        max_candidates=3000,
        disable_active_learning=None,
        active_learning_target_dataset=None,
    )
    assert duck.add_model_calls == [{"model_id": MODEL_ID, "api_key": "fresh-key"}]
    request = duck.infer_calls[0]["request"]
    assert isinstance(request, ObjectDetectionInferenceRequest)
    assert request.model_id == MODEL_ID
    assert request.api_key == "fresh-key"
    assert request.confidence == 0.4
    assert len(result) == 1
    assert isinstance(result[0]["predictions"], sv.Detections)
    return result


# ---------------------------------------------------------------------------
# Caller 1: a direct user call to InferencePipeline.init_with_workflow
# ---------------------------------------------------------------------------


def test_user_wrapper_always_replaces_manager_api_key_and_observer(
    monkeypatch,
) -> None:
    captured = _capture_engine_init(monkeypatch)
    duck = _fake_manager_duck()
    conflicting_observer = object()
    caller_supplied = {
        # A caller who pre-populates these namespaced keys must still lose:
        # the pipeline unconditionally overwrites them (plain `=`, not
        # `setdefault`) at inference_pipeline.py:758-767.
        "workflows_core.model_manager": object(),
        "workflows_core.api_key": "stale-key",
        "workflows_core.execution_observer": conflicting_observer,
        "workflows_core.disable_sinks": True,
    }

    InferencePipeline.init_with_workflow(
        video_reference="rtsp://irrelevant",
        workflow_specification=MINIMAL_SPEC,
        api_key="fresh-key",
        model_manager=duck,
        disable_sinks=False,
        workflow_init_parameters=caller_supplied,
    )

    assert len(captured) == 1
    init_parameters = captured[0]["init_parameters"]
    provider = init_parameters["workflows_core.model_manager"]
    assert isinstance(provider, ModelManagerModelsProvider)
    assert provider._model_manager is duck
    assert init_parameters["workflows_core.api_key"] == "fresh-key"
    assert init_parameters["workflows_core.execution_observer"] is not (
        conflicting_observer
    )
    assert isinstance(
        init_parameters["workflows_core.execution_observer"],
        UsageTrackingExecutionObserver,
    )
    assert init_parameters["workflows_core.disable_sinks"] is False


def test_user_wrapper_preserves_explicit_configuration_and_codec_overrides(
    monkeypatch,
) -> None:
    captured = _capture_engine_init(monkeypatch)
    # An equal-but-distinct configuration: the real engine accepts it, so its
    # identity is what reaches ExecutionEngine.init. (Since WP-A02 the wrapper
    # calls bind_model_manager_to_workflows, which validates a caller-supplied
    # configuration before the codec side effect - an invalid one now raises
    # the engine's own error before this stubbed engine is reached; see
    # legacy_stream/test_pipeline_wrappers.py.)
    explicit_configuration = dataclasses.replace(server_workflows_configuration())
    explicit_codec = object()
    caller_supplied = {
        # `.setdefault(...)` at :777-779 and `bind_image_codec`'s own
        # `setdefault` (workflows_image_codec.py:106-108): an explicit value
        # here must survive untouched, unlike the always-replaced keys above.
        "workflows_core.configuration": explicit_configuration,
        "workflows_core.image_codec": explicit_codec,
    }

    InferencePipeline.init_with_workflow(
        video_reference="rtsp://irrelevant",
        workflow_specification=MINIMAL_SPEC,
        api_key="fresh-key",
        model_manager=_fake_manager_duck(),
        workflow_init_parameters=caller_supplied,
    )

    init_parameters = captured[0]["init_parameters"]
    assert init_parameters["workflows_core.configuration"] is explicit_configuration
    assert init_parameters["workflows_core.image_codec"] is explicit_codec
    # bind_image_codec makes the caller's own object process-wide, not the
    # module default, precisely because it was explicit.
    assert GUARDED_IMAGE_CODEC is not explicit_codec


def test_user_wrapper_default_model_manager_is_still_wrapped_in_the_provider(
    monkeypatch,
) -> None:
    # No model_manager passed at all - the pipeline constructs its own
    # default stack (BackgroundTaskActiveLearningManager + WithFixedSizeCache)
    # and wraps THAT, not None, in the provider.
    captured = _capture_engine_init(monkeypatch)

    InferencePipeline.init_with_workflow(
        video_reference="rtsp://irrelevant",
        workflow_specification=MINIMAL_SPEC,
        api_key="fresh-key",
    )

    provider = captured[0]["init_parameters"]["workflows_core.model_manager"]
    assert isinstance(provider, ModelManagerModelsProvider)
    assert provider._model_manager is not None


def test_user_wrapper_fills_in_platform_bindings_via_setdefault_when_absent(
    monkeypatch,
) -> None:
    captured = _capture_engine_init(monkeypatch)

    InferencePipeline.init_with_workflow(
        video_reference="rtsp://irrelevant",
        workflow_specification=MINIMAL_SPEC,
        api_key="fresh-key",
        model_manager=_fake_manager_duck(),
    )

    init_parameters = captured[0]["init_parameters"]
    for key, value in workflows_platform_bindings().items():
        assert init_parameters[key] is value, key


def test_user_wrapper_preserves_explicit_platform_binding_override(monkeypatch) -> None:
    captured = _capture_engine_init(monkeypatch)
    explicit_resolver = object()

    InferencePipeline.init_with_workflow(
        video_reference="rtsp://irrelevant",
        workflow_specification=MINIMAL_SPEC,
        api_key="fresh-key",
        model_manager=_fake_manager_duck(),
        workflow_init_parameters={
            "workflows_core.inner_workflow_spec_resolver": explicit_resolver,
        },
    )

    init_parameters = captured[0]["init_parameters"]
    assert (
        init_parameters["workflows_core.inner_workflow_spec_resolver"]
        is explicit_resolver
    )


# ---------------------------------------------------------------------------
# Caller 2: the stream manager (InferencePipelineManager._initialise_pipeline)
# ---------------------------------------------------------------------------


def _assembly_init_payload() -> dict:
    payload = InitialisePipelinePayload(
        video_configuration=VideoConfiguration(
            type="VideoConfiguration",
            video_reference="rtsp://128.0.0.1",
            source_buffer_filling_strategy=BufferFillingStrategy.DROP_OLDEST,
            source_buffer_consumption_strategy=BufferConsumptionStrategy.EAGER,
        ),
        processing_configuration=WorkflowConfiguration(
            type="WorkflowConfiguration", workflow_specification=MINIMAL_SPEC
        ),
        api_key="<MY-API-KEY>",
    ).dict()
    payload["type"] = CommandType.INIT
    return payload


@pytest.mark.timeout(30)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_manager_initialisation_does_not_inject_its_own_model_manager(
    pipeline_init_mock: MagicMock,
    monkeypatch,
) -> None:
    # The manager relies entirely on init_with_workflow's own default
    # construction path; it never builds or injects a ModelManager of its
    # own (contrast with the WebRTC worker chain below, which does).
    # WP-A03: the manager's legacy host now runs that path - the same
    # preparation helper init_with_workflow uses - and hands its result to the
    # host-neutral pipeline, so the request's API key and the absent model
    # manager are observed at the helper.
    prepare_spy = MagicMock(
        wraps=inference_pipeline_module.prepare_workflow_for_pipeline
    )
    monkeypatch.setattr(
        inference_pipeline_module, "prepare_workflow_for_pipeline", prepare_spy
    )
    pipeline_init_mock.return_value = MagicMock()
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        pipeline_id="p", command_queue=command_queue, responses_queue=responses_queue
    )
    command_queue.put(("1", _assembly_init_payload()))
    command_queue.put(("2", {"type": CommandType.TERMINATE}))

    manager.run()

    assert "model_manager" not in pipeline_init_mock.call_args.kwargs
    assert prepare_spy.call_args.kwargs["model_manager"] is None
    assert prepare_spy.call_args.kwargs["api_key"] == "<MY-API-KEY>"
    init_parameters = pipeline_init_mock.call_args.kwargs["workflow_init_parameters"]
    assert init_parameters["workflows_core.api_key"] == "<MY-API-KEY>"
    assert isinstance(
        init_parameters["workflows_core.model_manager"], ModelManagerModelsProvider
    )


# ---------------------------------------------------------------------------
# Caller 3: the in-process WebRTC worker chain
# ---------------------------------------------------------------------------


def test_webrtc_worker_chain_passes_its_model_manager_through_unchanged(
    monkeypatch,
) -> None:
    pipeline_init_mock = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(InferencePipeline, "init_with_workflow", pipeline_init_mock)
    duck = _fake_manager_duck()
    workflow_configuration = WorkflowConfiguration(
        type="WorkflowConfiguration", workflow_specification=MINIMAL_SPEC
    )

    VideoFrameProcessor(
        asyncio_loop=MagicMock(),
        workflow_configuration=workflow_configuration,
        api_key="api-key",
        model_manager=duck,
        has_video_track=False,
    )

    assert pipeline_init_mock.call_args.kwargs["model_manager"] is duck


# ---------------------------------------------------------------------------
# Caller 4: a no-hook duck model manager, through the real wrapper
# ---------------------------------------------------------------------------


def test_no_hook_duck_manager_is_wrapped_by_identity_not_copied(monkeypatch) -> None:
    captured = _capture_engine_init(monkeypatch)
    duck = _fake_manager_duck()

    InferencePipeline.init_with_workflow(
        video_reference="rtsp://irrelevant",
        workflow_specification=MINIMAL_SPEC,
        api_key="fresh-key",
        model_manager=duck,
    )

    provider = captured[0]["init_parameters"]["workflows_core.model_manager"]
    assert isinstance(provider, ModelManagerModelsProvider)
    assert provider._model_manager is duck


def test_no_hook_duck_manager_with_real_calls_drives_a_real_detection_block(
    monkeypatch,
) -> None:
    # test_no_hook_duck_manager_is_wrapped_by_identity_not_copied above only
    # proves the provider holds the SAME object; it never proves the
    # provider forwards a working duck's calls correctly, because its duck
    # raises on every call. This drives a real block through the real
    # provider, backed by a duck that actually works.
    captured = _capture_engine_init(monkeypatch)
    duck = _WorkingDuckModelManager()

    InferencePipeline.init_with_workflow(
        video_reference="rtsp://irrelevant",
        workflow_specification=MINIMAL_SPEC,
        api_key="fresh-key",
        model_manager=duck,
    )

    init_parameters = captured[0]["init_parameters"]
    provider = init_parameters["workflows_core.model_manager"]
    assert isinstance(provider, ModelManagerModelsProvider)
    assert provider._model_manager is duck
    _run_real_detection_block_with_duck(init_parameters, duck)


# ---------------------------------------------------------------------------
# Caller 5: InferencePipeline.init - active learning on/off and model alias
# resolution, exercised for real (only get_model, init_with_custom_logic, and
# ThreadingActiveLearningMiddleware.init are mocked; the latter because it
# calls prepare_active_learning_configuration, a network call).
# ---------------------------------------------------------------------------


def test_init_forces_active_learning_off_and_skips_resolution_when_api_key_missing(
    monkeypatch,
) -> None:
    fake_model = MagicMock(task_type="object-detection")
    monkeypatch.setattr(
        inference_pipeline_module, "get_model", lambda **kwargs: fake_model
    )
    al_init_mock = MagicMock()
    monkeypatch.setattr(ThreadingActiveLearningMiddleware, "init", al_init_mock)
    custom_logic_mock = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(InferencePipeline, "init_with_custom_logic", custom_logic_mock)
    on_prediction = lambda *a, **k: None

    InferencePipeline.init(
        video_reference="rtsp://irrelevant",
        model_id="yolov8n-640",
        on_prediction=on_prediction,
        api_key=None,
        active_learning_enabled=True,
    )

    al_init_mock.assert_not_called()
    kwargs = custom_logic_mock.call_args.kwargs
    assert kwargs["on_prediction"] is on_prediction
    assert isinstance(
        kwargs["on_pipeline_start"].__self__, NullActiveLearningMiddleware
    )
    assert isinstance(kwargs["on_pipeline_end"].__self__, NullActiveLearningMiddleware)


def test_init_enables_active_learning_and_resolves_model_alias_when_flag_true(
    monkeypatch,
) -> None:
    fake_model = MagicMock(task_type="object-detection")
    monkeypatch.setattr(
        inference_pipeline_module, "get_model", lambda **kwargs: fake_model
    )
    fake_middleware = MagicMock()
    al_init_mock = MagicMock(return_value=fake_middleware)
    monkeypatch.setattr(ThreadingActiveLearningMiddleware, "init", al_init_mock)
    custom_logic_mock = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(InferencePipeline, "init_with_custom_logic", custom_logic_mock)
    on_prediction = lambda *a, **k: None

    InferencePipeline.init(
        video_reference="rtsp://irrelevant",
        model_id="yolov8n-640",  # a registered alias, see inference/models/aliases.py
        on_prediction=on_prediction,
        api_key="real-key",
        active_learning_enabled=True,
    )

    assert al_init_mock.call_args.kwargs["model_id"] == "coco/3"
    assert al_init_mock.call_args.kwargs["target_dataset"] == "coco"
    assert al_init_mock.call_args.kwargs["api_key"] == "real-key"
    kwargs = custom_logic_mock.call_args.kwargs
    wrapped = kwargs["on_prediction"]
    assert wrapped.func is multi_sink
    assert wrapped.keywords["sinks"][0] is on_prediction
    al_sink = wrapped.keywords["sinks"][1]
    assert al_sink.func is active_learning_sink
    assert al_sink.keywords["active_learning_middleware"] is fake_middleware
    assert al_sink.keywords["model_type"] == "object-detection"
    assert kwargs["on_pipeline_start"] is fake_middleware.start_registration_thread
    assert kwargs["on_pipeline_end"] is fake_middleware.stop_registration_thread


def test_init_uses_active_learning_env_default_when_flag_omitted(monkeypatch) -> None:
    fake_model = MagicMock(task_type="object-detection")
    monkeypatch.setattr(
        inference_pipeline_module, "get_model", lambda **kwargs: fake_model
    )
    monkeypatch.setattr(inference_pipeline_module, "ACTIVE_LEARNING_ENABLED", False)
    al_init_mock = MagicMock()
    monkeypatch.setattr(ThreadingActiveLearningMiddleware, "init", al_init_mock)
    custom_logic_mock = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(InferencePipeline, "init_with_custom_logic", custom_logic_mock)

    InferencePipeline.init(
        video_reference="rtsp://irrelevant",
        model_id="yolov8n-640",
        api_key="real-key",
    )

    al_init_mock.assert_not_called()
    assert custom_logic_mock.call_args.kwargs["on_prediction"] is None
