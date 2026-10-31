"""Raw `ModelManager` compatibility: `ExecutionEngine.init(...)` may take a
manager (or a `ModelManagerDecorator` around it) as
`workflows_core.model_manager` and receive the historical server bindings for
free through the class-level `__workflows_bind__` hook.

Companion coverage:

* pure hook shape without server imports:
  `tests/workflows/unit_tests/execution_engine/test_model_manager_bind_hook.py`
* explicit-provider path at composition roots and image codec matrix:
  `test_direct_caller_bindings.py`, `test_image_codec_binding.py`
* per-service depth (usage rows, platform bindings, handler selection) lives
  in the plan's named suites - this file does not duplicate them.
"""

import dataclasses
from typing import Any, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.entities.requests.inference import ObjectDetectionInferenceRequest
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.interfaces.roboflow_platform_client import (
    SERVER_PLATFORM_CLIENT,
    SERVER_WORKSPACE_RESOLVER,
    default_inner_workflow_spec_resolver,
    workflows_platform_bindings,
)
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
from inference.core.interfaces.workflows_step_error_handlers import (
    extended_roboflow_errors_handler,
    legacy_step_error_handler,
    resolve_step_error_handler,
)
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.base import ModelManagerDecorator
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import WorkflowEnvironmentConfigurationError
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.v1.core import ExecutionEngineV1
from inference.core.workflows.prototypes.image_codec import (
    WorkflowsLocalImageCodec,
    get_image_codec,
    reset_image_codec,
)

TRIVIAL_WORKFLOW = {"version": "1.0", "inputs": [], "steps": [], "outputs": []}
API_KEY = "raw-manager-key"
MODEL_ID = "some-project/1"

OBJECT_DETECTION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "model_id"},
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": 0.4,
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.model.predictions",
        }
    ],
}


@pytest.fixture(autouse=True)
def _clean_codec_registry():
    # `bind_model_manager_to_workflows` installs the guarded codec process-wide;
    # leaving that state behind would make later tests' installs conflict, as
    # documented on `set_image_codec`.
    reset_image_codec()
    yield
    reset_image_codec()


class _RawTestManager(ModelManager):
    """Minimal real subclass of `ModelManager` so `type(mm)` inherits the hook.

    A `MagicMock(spec=ModelManager)` would NOT trigger it: the mock's type is
    `MagicMock`, and lookup is class-level to keep permissive mocks from opting
    in. Skips the full manager init and stubs only what the tests reach - and
    deliberately not the `run_*` methods the provider exists to add.
    """

    def __init__(self, inference_response: Optional[Any] = None) -> None:  # noqa: D401
        # Skip `ModelManager.__init__` (no registry, no shared blob cache).
        response = inference_response or ObjectDetectionInferenceResponse(
            image=InferenceResponseImage(width=64, height=64), predictions=[]
        )
        self.add_model = MagicMock()  # type: ignore[assignment]
        self.infer_from_request_sync = MagicMock(  # type: ignore[assignment]
            return_value=response
        )
        # `ModelManagerDecorator.add_model` short-circuits when `model_id in
        # self` and forwards to `record_request_metadata`; the real method
        # reads state the skipped `__init__` never set, so we stub it.
        self.record_request_metadata = MagicMock()  # type: ignore[assignment]

    def __contains__(self, model_id: str) -> bool:
        return True


def _fake_manager() -> _RawTestManager:
    manager = _RawTestManager()
    assert not hasattr(manager, "run_object_detection")
    return manager


def _v1(engine: ExecutionEngine) -> ExecutionEngineV1:
    inner = engine._engine
    assert isinstance(inner, ExecutionEngineV1)
    return inner


def _stored(engine: ExecutionEngine) -> dict:
    return _v1(engine)._compiled_workflow.init_parameters


# Wrap-and-fill: identity of every service the hook installs.


def test_raw_manager_becomes_provider_with_server_bindings():
    manager = _fake_manager()
    engine = ExecutionEngine.init(
        workflow_definition=TRIVIAL_WORKFLOW,
        init_parameters={
            "workflows_core.model_manager": manager,
            "workflows_core.api_key": API_KEY,
        },
    )
    parameters = _stored(engine)

    provider = parameters["workflows_core.model_manager"]
    assert isinstance(provider, ModelManagerModelsProvider)
    assert provider._model_manager is manager

    for key, value in workflows_platform_bindings().items():
        assert parameters[key] is value, key
    assert parameters["workflows_core.platform_client"] is SERVER_PLATFORM_CLIENT
    assert parameters["workflows_core.workspace_resolver"] is SERVER_WORKSPACE_RESOLVER
    assert (
        parameters["workflows_core.inner_workflow_spec_resolver"]
        is default_inner_workflow_spec_resolver
    )

    assert parameters["workflows_core.image_codec"] is GUARDED_IMAGE_CODEC
    assert get_image_codec() is GUARDED_IMAGE_CODEC

    assert isinstance(
        parameters["workflows_core.execution_observer"], UsageTrackingExecutionObserver
    )
    assert parameters["workflows_core.configuration"] is (
        server_workflows_configuration()
    )
    assert _v1(engine)._step_error_handler is resolve_step_error_handler()


def test_decorated_manager_is_wrapped_not_unwrapped():
    # Wrap `self`, not the inner manager: the decorator's cache / locking /
    # active-learning overrides must survive.
    inner = _fake_manager()
    decorated = ModelManagerDecorator(model_manager=inner)
    engine = ExecutionEngine.init(
        workflow_definition=TRIVIAL_WORKFLOW,
        init_parameters={"workflows_core.model_manager": decorated},
    )
    provider = _stored(engine)["workflows_core.model_manager"]
    assert isinstance(provider, ModelManagerModelsProvider)
    assert provider._model_manager is decorated


# Explicit values / overrides survive; lifetime is per-engine.


def test_explicit_platform_client_survives_hook():
    manager = _fake_manager()
    platform_client = object()
    engine = ExecutionEngine.init(
        workflow_definition=TRIVIAL_WORKFLOW,
        init_parameters={
            "workflows_core.model_manager": manager,
            "workflows_core.platform_client": platform_client,
        },
    )
    assert _stored(engine)["workflows_core.platform_client"] is platform_client


def test_explicit_configuration_survives_hook():
    manager = _fake_manager()
    explicit_config = server_workflows_configuration()
    engine = ExecutionEngine.init(
        workflow_definition=TRIVIAL_WORKFLOW,
        init_parameters={
            "workflows_core.model_manager": manager,
            "workflows_core.configuration": explicit_config,
        },
    )
    assert _stored(engine)["workflows_core.configuration"] is explicit_config


def test_bare_execution_observer_survives_hook():
    # A bare `execution_observer` reaches `_resolve_execution_observer`
    # and is republished under the namespaced key; a hook-installed namespaced
    # default would silently hide it.
    manager = _fake_manager()
    observer = object()
    engine = ExecutionEngine.init(
        workflow_definition=TRIVIAL_WORKFLOW,
        init_parameters={
            "workflows_core.model_manager": manager,
            "execution_observer": observer,
        },
    )
    parameters = _stored(engine)
    assert parameters["workflows_core.execution_observer"] is observer
    assert parameters["execution_observer"] is observer


def test_dynamic_block_execution_observer_override_survives_hook():
    # A dynamic-block override wins for custom-Python blocks without
    # replacing the server default for core blocks.
    manager = _fake_manager()
    dynamic_observer = object()
    engine = ExecutionEngine.init(
        workflow_definition=TRIVIAL_WORKFLOW,
        init_parameters={
            "workflows_core.model_manager": manager,
            "dynamic_workflows_blocks.execution_observer": dynamic_observer,
        },
    )
    parameters = _stored(engine)
    assert parameters["dynamic_workflows_blocks.execution_observer"] is dynamic_observer
    assert isinstance(
        parameters["workflows_core.execution_observer"], UsageTrackingExecutionObserver
    )
    assert parameters["workflows_core.execution_observer"] is not dynamic_observer


def test_bare_observer_factory_is_called_once_per_engine():
    manager = _fake_manager()

    class _Observer:
        pass

    build_calls = {"count": 0}

    def _factory():
        build_calls["count"] += 1
        return _Observer()

    engine_a = ExecutionEngine.init(
        workflow_definition=TRIVIAL_WORKFLOW,
        init_parameters={
            "workflows_core.model_manager": manager,
            "execution_observer": _factory,
        },
    )
    engine_b = ExecutionEngine.init(
        workflow_definition=TRIVIAL_WORKFLOW,
        init_parameters={
            "workflows_core.model_manager": manager,
            "execution_observer": _factory,
        },
    )
    # `_resolve_execution_observer` calls callables exactly once per engine.
    assert build_calls["count"] == 2
    obs_a = _stored(engine_a)["workflows_core.execution_observer"]
    obs_b = _stored(engine_b)["workflows_core.execution_observer"]
    assert isinstance(obs_a, _Observer) and isinstance(obs_b, _Observer)
    assert obs_a is not obs_b


def test_reuse_of_manager_and_dict_across_engines_stays_isolated():
    manager = _fake_manager()
    caller_dict = {
        "workflows_core.model_manager": manager,
        "workflows_core.api_key": API_KEY,
    }
    engine_a = ExecutionEngine.init(
        workflow_definition=TRIVIAL_WORKFLOW, init_parameters=caller_dict
    )
    engine_b = ExecutionEngine.init(
        workflow_definition=TRIVIAL_WORKFLOW, init_parameters=caller_dict
    )
    provider_a = _stored(engine_a)["workflows_core.model_manager"]
    provider_b = _stored(engine_b)["workflows_core.model_manager"]
    assert isinstance(provider_a, ModelManagerModelsProvider)
    assert isinstance(provider_b, ModelManagerModelsProvider)
    assert provider_a is not provider_b
    assert provider_a._model_manager is manager
    assert provider_b._model_manager is manager
    # A fresh usage-tracking observer per engine.
    assert (
        _stored(engine_a)["workflows_core.execution_observer"]
        is not _stored(engine_b)["workflows_core.execution_observer"]
    )
    # Caller dict was not mutated.
    assert set(caller_dict) == {
        "workflows_core.model_manager",
        "workflows_core.api_key",
    }


# Configuration mismatch: refused BEFORE codec is installed.


def test_mismatched_configuration_is_rejected_before_codec_is_installed():
    manager = _fake_manager()
    server_config = server_workflows_configuration()
    wrong_config = dataclasses.replace(
        server_config,
        engine=dataclasses.replace(
            server_config.engine,
            max_inner_workflow_depth=server_config.engine.max_inner_workflow_depth
            + 999,
        ),
    )
    custom_codec = WorkflowsLocalImageCodec()
    codec_before = get_image_codec()
    with pytest.raises(WorkflowEnvironmentConfigurationError):
        ExecutionEngine.init(
            workflow_definition=TRIVIAL_WORKFLOW,
            init_parameters={
                "workflows_core.model_manager": manager,
                "workflows_core.configuration": wrong_config,
                "workflows_core.image_codec": custom_codec,
            },
        )
    # Codec registry must NOT have been touched by the failed init.
    assert get_image_codec() is codec_before
    engine = ExecutionEngine.init(
        workflow_definition=TRIVIAL_WORKFLOW,
        init_parameters={"workflows_core.model_manager": manager},
    )
    assert _stored(engine)["workflows_core.image_codec"] is GUARDED_IMAGE_CODEC


# step_error_handler semantics with a concrete server manager.


@pytest.mark.parametrize(
    "kwargs, expected_handler",
    [
        ({}, extended_roboflow_errors_handler),
        ({"step_error_handler": "legacy"}, legacy_step_error_handler),
        (
            {"step_error_handler": "extended_roboflow_errors"},
            extended_roboflow_errors_handler,
        ),
        ({"step_error_handler": None}, None),
    ],
)
def test_handler_semantics(monkeypatch, kwargs, expected_handler):
    monkeypatch.delenv("DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER", raising=False)
    manager = _fake_manager()
    engine = ExecutionEngine.init(
        workflow_definition=TRIVIAL_WORKFLOW,
        init_parameters={"workflows_core.model_manager": manager},
        **kwargs,
    )
    assert _v1(engine)._step_error_handler is expected_handler


def test_explicit_callable_handler_keeps_identity():
    manager = _fake_manager()

    def _my_handler(step_name: str, error: Exception) -> None:
        return None

    engine = ExecutionEngine.init(
        workflow_definition=TRIVIAL_WORKFLOW,
        init_parameters={"workflows_core.model_manager": manager},
        step_error_handler=_my_handler,
    )
    assert _v1(engine)._step_error_handler is _my_handler


def test_unknown_handler_name_with_raw_manager_still_raises():
    manager = _fake_manager()
    with pytest.raises(WorkflowEnvironmentConfigurationError):
        ExecutionEngine.init(
            workflow_definition=TRIVIAL_WORKFLOW,
            init_parameters={"workflows_core.model_manager": manager},
            step_error_handler="does-not-exist",
        )


# End-to-end: engine.run reaches the raw manager through the hook-wired provider.


def _detection_response() -> ObjectDetectionInferenceResponse:
    return ObjectDetectionInferenceResponse(
        image=InferenceResponseImage(width=64, height=64),
        predictions=[
            ObjectDetectionPrediction(
                x=32.0,
                y=32.0,
                width=10.0,
                height=10.0,
                confidence=0.9,
                **{"class": "raw-manager", "class_id": 0},
            )
        ],
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_engine_run_reaches_raw_manager_through_wired_provider(wrapped):
    # Compatibility contract: engine.run for a real workflow, with raw AND
    # decorated managers, reaches the manager via the hook-installed provider.
    inner = _RawTestManager(inference_response=_detection_response())
    manager = ModelManagerDecorator(model_manager=inner) if wrapped else inner
    engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW,
        init_parameters={
            "workflows_core.model_manager": manager,
            "workflows_core.api_key": API_KEY,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
    )
    provider = _stored(engine)["workflows_core.model_manager"]
    assert isinstance(provider, ModelManagerModelsProvider)
    assert provider._model_manager is manager
    # The image codec installed by the hook drives both the engine's
    # deserializer and later reference re-loads: we exercise the deserializer
    # via engine.run's numpy input.
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    result = engine.run(
        runtime_parameters={
            "image": image,
            "model_id": MODEL_ID,
        }
    )
    # The decorator short-circuits to `record_request_metadata` when the
    # model is already present; the raw manager takes `add_model`. Either
    # way, the translated request must reach the inner manager.
    if wrapped:
        inner.record_request_metadata.assert_called_once()
    else:
        inner.add_model.assert_called_once_with(model_id=MODEL_ID, api_key=API_KEY)
    call = inner.infer_from_request_sync.call_args
    request = call.kwargs.get("request") or call.args[1]
    assert isinstance(request, ObjectDetectionInferenceRequest)
    assert request.model_id == MODEL_ID
    assert request.api_key == API_KEY
    predictions = result[0]["predictions"]
    assert len(predictions) == 1


# V1 entry point check.


def test_v1_direct_entry_point_also_fires_hook():
    manager = _fake_manager()
    v1_engine = ExecutionEngineV1.init(
        workflow_definition=TRIVIAL_WORKFLOW,
        init_parameters={"workflows_core.model_manager": manager},
    )
    parameters = v1_engine._compiled_workflow.init_parameters
    provider = parameters["workflows_core.model_manager"]
    assert isinstance(provider, ModelManagerModelsProvider)
    assert provider._model_manager is manager
    assert v1_engine._step_error_handler is resolve_step_error_handler()
