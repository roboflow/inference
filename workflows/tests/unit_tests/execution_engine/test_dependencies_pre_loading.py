"""
Tests for pre-loading of declared dependent resources at Execution Engine
init time (`dependencies_pre_init`) and on the first run (for dependencies
declared through `$inputs.<name>` selectors).
"""

import inspect
import json
import typing
from unittest.mock import MagicMock, NonCallableMagicMock

import networkx as nx
import numpy as np
import pytest
from roboflow_workflows.core_steps.models.roboflow.object_detection.v3 import (
    BlockManifest as ObjectDetectionV3Manifest,
)
from roboflow_workflows.core_steps.models.roboflow.object_detection.v3_tensor import (
    BlockManifest as ObjectDetectionV3TensorManifest,
)
from roboflow_workflows.errors import (
    RuntimeInputError,
    WorkflowEnvironmentConfigurationError,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    complete_discovery,
    declaration_unavailable_problem,
    incomplete_discovery,
)
from roboflow_workflows.execution_engine.v1 import core as ee_core
from roboflow_workflows.execution_engine.v1.compiler.entities import (
    CompiledWorkflow,
    ParsedWorkflowDefinition,
)
from roboflow_workflows.execution_engine.v1.compiler.utils import (
    deduce_blocks_dependencies,
)
from roboflow_workflows.execution_engine.v1.core import (
    ExecutionEngineV1,
    _parse_dependencies_pre_init,
    _pre_load_roboflow_platform_models,
    _resolve_and_pre_load_runtime_dependencies,
    _retrieve_init_parameter,
    _retrieve_step_execution_mode,
)
from roboflow_workflows.prototypes.block import (
    DependentResource,
    DependentResourceType,
    ModelExecutionLocation,
    ModelRequiredAction,
    StepExecutionMode,
    WorkflowBlockManifest,
    roboflow_platform_model,
    roboflow_platform_project,
    third_party_model,
)
from roboflow_workflows.prototypes.models_provider import ModelsProvider


def _object_detection_manifest(name: str, model_id: str) -> ObjectDetectionV3Manifest:
    return ObjectDetectionV3Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_object_detection_model@v3",
            "name": name,
            "images": "$inputs.image",
            "model_id": model_id,
        }
    )


def _compiled_workflow_with_steps(steps) -> CompiledWorkflow:
    return CompiledWorkflow(
        workflow_definition=ParsedWorkflowDefinition(
            version="1.0",
            inputs=[],
            steps=steps,
            outputs=[],
        ),
        execution_graph=nx.DiGraph(),
        steps={},
        input_substitutions=[],
        workflow_json={},
        init_parameters={},
    )


# ---------------------------------------------------------------------------
# deduce_blocks_dependencies
# ---------------------------------------------------------------------------


def test_deduce_blocks_dependencies_aggregates_declarations_of_all_steps() -> None:
    compiled_workflow = _compiled_workflow_with_steps(
        steps=[
            _object_detection_manifest(name="a", model_id="my_project/3"),
            _object_detection_manifest(name="b", model_id="$inputs.model"),
        ]
    )

    dependencies = deduce_blocks_dependencies(compiled_workflow=compiled_workflow)

    assert dependencies == [
        roboflow_platform_model(model_id="my_project/3"),
        roboflow_platform_model(model_id="$inputs.model"),
    ]


# ---------------------------------------------------------------------------
# _parse_dependencies_pre_init
# ---------------------------------------------------------------------------


def test_parse_dependencies_pre_init_accepts_supported_value() -> None:
    assert _parse_dependencies_pre_init(["roboflow_platform_model"]) == {
        DependentResourceType.ROBOFLOW_PLATFORM_MODEL
    }


def test_parse_dependencies_pre_init_accepts_hyphenated_value() -> None:
    assert _parse_dependencies_pre_init(["roboflow-platform-model"]) == {
        DependentResourceType.ROBOFLOW_PLATFORM_MODEL
    }


def test_parse_dependencies_pre_init_rejects_unknown_value() -> None:
    with pytest.raises(WorkflowEnvironmentConfigurationError):
        _parse_dependencies_pre_init(["some-unknown-resource"])


def test_parse_dependencies_pre_init_rejects_unsupported_resource_type() -> None:
    with pytest.raises(WorkflowEnvironmentConfigurationError):
        _parse_dependencies_pre_init(["third_party_model"])


# ---------------------------------------------------------------------------
# _retrieve_init_parameter
# ---------------------------------------------------------------------------


def test_retrieve_init_parameter_prefers_workflows_core_prefixed_key() -> None:
    assert (
        _retrieve_init_parameter(
            init_parameters={
                "workflows_core.api_key": "prefixed",
                "api_key": "bare",
            },
            parameter_name="api_key",
        )
        == "prefixed"
    )


def test_retrieve_init_parameter_falls_back_to_bare_key_and_calls_callables() -> None:
    assert (
        _retrieve_init_parameter(
            init_parameters={"api_key": lambda: "from-callable"},
            parameter_name="api_key",
        )
        == "from-callable"
    )


def test_retrieve_init_parameter_returns_none_when_missing() -> None:
    assert (
        _retrieve_init_parameter(init_parameters={}, parameter_name="model_manager")
        is None
    )


def test_retrieve_step_execution_mode_uses_explicit_init_parameter() -> None:
    assert (
        _retrieve_step_execution_mode(
            init_parameters={
                "workflows_core.step_execution_mode": StepExecutionMode.REMOTE
            }
        )
        is StepExecutionMode.REMOTE
    )


def test_retrieve_step_execution_mode_coerces_string_values() -> None:
    assert (
        _retrieve_step_execution_mode(init_parameters={"step_execution_mode": "remote"})
        is StepExecutionMode.REMOTE
    )


def test_retrieve_step_execution_mode_falls_back_to_environment_default() -> None:
    assert isinstance(
        _retrieve_step_execution_mode(init_parameters={}), StepExecutionMode
    )


# ---------------------------------------------------------------------------
# _pre_load_roboflow_platform_models
# ---------------------------------------------------------------------------


def test_pre_load_registers_concrete_execution_models_only_once() -> None:
    model_manager = MagicMock()

    pending = _pre_load_roboflow_platform_models(
        dependencies=[
            roboflow_platform_model(model_id="my_project/3"),
            roboflow_platform_model(model_id="my_project/3"),
            roboflow_platform_model(model_id="other_project/1"),
        ],
        model_manager=model_manager,
        api_key="api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    assert pending == []
    assert model_manager.add_model.call_count == 2
    model_manager.add_model.assert_any_call(model_id="my_project/3", api_key="api-key")
    model_manager.add_model.assert_any_call(
        model_id="other_project/1", api_key="api-key"
    )


def test_pre_load_skips_non_model_access_only_and_remote_dependencies() -> None:
    model_manager = MagicMock()

    pending = _pre_load_roboflow_platform_models(
        dependencies=[
            roboflow_platform_project(project_url="my_dataset"),
            third_party_model(provider="openai", model_id="gpt-4o"),
            roboflow_platform_model(
                model_id="monitored/1",
                required_action=ModelRequiredAction.ACCESS,
            ),
            roboflow_platform_model(
                model_id="remote_only/1",
                execution_location=ModelExecutionLocation.REMOTE,
            ),
        ],
        model_manager=model_manager,
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    assert pending == []
    model_manager.add_model.assert_not_called()


def test_pre_load_honours_remote_step_execution_mode() -> None:
    model_manager = MagicMock()

    pending = _pre_load_roboflow_platform_models(
        dependencies=[
            roboflow_platform_model(model_id="env_defined/1"),
            roboflow_platform_model(model_id="$inputs.model"),
            roboflow_platform_model(
                model_id="local_only/1",
                execution_location=ModelExecutionLocation.LOCAL,
            ),
        ],
        model_manager=model_manager,
        api_key="api-key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    # ENVIRONMENT_DEFINED dependencies resolve to remote execution — nothing
    # to pre-load and nothing pending; LOCAL-declared execution still loads.
    assert pending == []
    model_manager.add_model.assert_called_once_with(
        model_id="local_only/1", api_key="api-key"
    )


def test_pre_load_defers_input_selectors_and_drops_step_output_selectors() -> None:
    model_manager = MagicMock()
    input_fed = roboflow_platform_model(model_id="$inputs.model")

    pending = _pre_load_roboflow_platform_models(
        dependencies=[
            input_fed,
            roboflow_platform_model(model_id="$steps.parser.model_id"),
        ],
        model_manager=model_manager,
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    assert pending == [input_fed]
    model_manager.add_model.assert_not_called()


# ---------------------------------------------------------------------------
# _resolve_and_pre_load_runtime_dependencies
# ---------------------------------------------------------------------------


def test_runtime_resolution_registers_models_for_concrete_input_values() -> None:
    model_manager = MagicMock()

    _resolve_and_pre_load_runtime_dependencies(
        pending_dependencies=[
            roboflow_platform_model(model_id="$inputs.model"),
            roboflow_platform_model(model_id="$inputs.other_model"),
        ],
        runtime_parameters={"model": "my_project/3"},
        model_manager=model_manager,
        api_key="api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    model_manager.add_model.assert_called_once_with(
        model_id="my_project/3", api_key="api-key"
    )


def test_runtime_resolution_ignores_non_string_and_selector_values() -> None:
    model_manager = MagicMock()

    _resolve_and_pre_load_runtime_dependencies(
        pending_dependencies=[
            roboflow_platform_model(model_id="$inputs.model"),
            roboflow_platform_model(model_id="$inputs.other_model"),
        ],
        runtime_parameters={"model": 42, "other_model": "$inputs.something"},
        model_manager=model_manager,
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    model_manager.add_model.assert_not_called()


def test_runtime_resolution_only_considers_input_selector_dependencies() -> None:
    model_manager = MagicMock()

    _resolve_and_pre_load_runtime_dependencies(
        pending_dependencies=[
            roboflow_platform_model(model_id="$steps.parser.model_id"),
            roboflow_platform_model(model_id="concrete_project/3"),
        ],
        # An input named like the step-output's last chunk must NOT be
        # accidentally matched.
        runtime_parameters={"model_id": "sneaky_project/1"},
        model_manager=model_manager,
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    model_manager.add_model.assert_not_called()


def test_runtime_resolution_follows_the_same_eligibility_logic_as_init() -> None:
    model_manager = MagicMock()

    _resolve_and_pre_load_runtime_dependencies(
        pending_dependencies=[
            roboflow_platform_model(model_id="$inputs.model"),
            roboflow_platform_model(
                model_id="$inputs.local_model",
                execution_location=ModelExecutionLocation.LOCAL,
            ),
            roboflow_platform_model(
                model_id="$inputs.monitored_model",
                required_action=ModelRequiredAction.ACCESS,
            ),
        ],
        runtime_parameters={
            "model": "env_defined/1",
            "local_model": "local_only/1",
            "monitored_model": "monitored/1",
        },
        model_manager=model_manager,
        api_key="api-key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    # Under REMOTE mode only the LOCAL-declared dependency pulls weights;
    # ENVIRONMENT_DEFINED and ACCESS-only entries are filtered exactly as at
    # init time.
    model_manager.add_model.assert_called_once_with(
        model_id="local_only/1", api_key="api-key"
    )


def test_pre_load_forwards_model_registration_kwargs() -> None:
    model_manager = MagicMock()

    pending = _pre_load_roboflow_platform_models(
        dependencies=[
            roboflow_platform_model(
                model_id="clip/ViT-B-32",
                model_registration_kwargs={"endpoint_type": "core-model"},
            ),
        ],
        model_manager=model_manager,
        api_key="api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    assert pending == []
    model_manager.add_model.assert_called_once_with(
        model_id="clip/ViT-B-32", api_key="api-key", endpoint_type="core-model"
    )


def test_runtime_resolution_forwards_model_registration_kwargs() -> None:
    model_manager = MagicMock()

    _resolve_and_pre_load_runtime_dependencies(
        pending_dependencies=[
            roboflow_platform_model(
                model_id="$inputs.variant",
                model_id_resolver=lambda version: f"clip/{version}",
                model_registration_kwargs={"endpoint_type": "core-model"},
            ),
        ],
        runtime_parameters={"variant": "ViT-B-16"},
        model_manager=model_manager,
        api_key="api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    model_manager.add_model.assert_called_once_with(
        model_id="clip/ViT-B-16", api_key="api-key", endpoint_type="core-model"
    )


def test_pre_load_warns_when_registered_models_were_evicted(caplog) -> None:
    model_manager = MagicMock()
    # Second registration evicts the first — only the last model remains.
    model_manager.__contains__ = lambda self, model_id: model_id == "other_project/1"

    with caplog.at_level("WARNING"):
        _pre_load_roboflow_platform_models(
            dependencies=[
                roboflow_platform_model(model_id="my_project/3"),
                roboflow_platform_model(model_id="other_project/1"),
            ],
            model_manager=model_manager,
            api_key="api-key",
            step_execution_mode=StepExecutionMode.LOCAL,
        )

    assert model_manager.add_model.call_count == 2
    assert "my_project/3" in caplog.text
    assert "no longer present in the model manager" in caplog.text


def test_pre_load_does_not_warn_when_all_models_are_present(caplog) -> None:
    model_manager = MagicMock()
    model_manager.__contains__ = lambda self, model_id: True

    with caplog.at_level("WARNING"):
        _pre_load_roboflow_platform_models(
            dependencies=[
                roboflow_platform_model(model_id="my_project/3"),
                roboflow_platform_model(model_id="other_project/1"),
            ],
            model_manager=model_manager,
            api_key="api-key",
            step_execution_mode=StepExecutionMode.LOCAL,
        )

    assert "no longer present in the model manager" not in caplog.text


def test_runtime_resolution_warns_when_registered_model_was_evicted(caplog) -> None:
    model_manager = MagicMock()
    model_manager.__contains__ = lambda self, model_id: False

    with caplog.at_level("WARNING"):
        _resolve_and_pre_load_runtime_dependencies(
            pending_dependencies=[
                roboflow_platform_model(model_id="$inputs.model"),
            ],
            runtime_parameters={"model": "my_project/3"},
            model_manager=model_manager,
            api_key="api-key",
            step_execution_mode=StepExecutionMode.LOCAL,
        )

    assert "my_project/3" in caplog.text
    assert "no longer present in the model manager" in caplog.text


def test_runtime_resolution_applies_attached_model_id_resolver() -> None:
    model_manager = MagicMock()

    _resolve_and_pre_load_runtime_dependencies(
        pending_dependencies=[
            roboflow_platform_model(
                model_id="$inputs.variant",
                model_id_resolver=lambda version: f"clip/{version}",
            ),
        ],
        runtime_parameters={"variant": "ViT-B-16"},
        model_manager=model_manager,
        api_key="api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    model_manager.add_model.assert_called_once_with(
        model_id="clip/ViT-B-16", api_key="api-key"
    )


def test_runtime_resolution_skips_dependency_when_resolver_returns_none() -> None:
    model_manager = MagicMock()
    catalog = {"known-label": "provider/known"}

    _resolve_and_pre_load_runtime_dependencies(
        pending_dependencies=[
            roboflow_platform_model(
                model_id="$inputs.model",
                model_id_resolver=lambda label: catalog.get(label),
            ),
            roboflow_platform_model(model_id="$inputs.other_model"),
        ],
        runtime_parameters={
            "model": "statically-unresolvable-label",
            "other_model": "my_project/3",
        },
        model_manager=model_manager,
        api_key="api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # A resolver returning None declares the value statically unresolvable —
    # the dependency is skipped without failing the run (execution resolves
    # it); other dependencies still pre-load.
    model_manager.add_model.assert_called_once_with(
        model_id="my_project/3", api_key="api-key"
    )


def test_runtime_resolution_raises_runtime_input_error_when_resolver_fails() -> None:
    model_manager = MagicMock()
    catalog = {"known-label": "provider/known"}

    with pytest.raises(RuntimeInputError):
        _resolve_and_pre_load_runtime_dependencies(
            pending_dependencies=[
                roboflow_platform_model(
                    model_id="$inputs.model",
                    model_id_resolver=lambda label: catalog[label],
                ),
            ],
            runtime_parameters={"model": "unknown-label"},
            model_manager=model_manager,
            api_key=None,
            step_execution_mode=StepExecutionMode.LOCAL,
        )

    model_manager.add_model.assert_not_called()


# ---------------------------------------------------------------------------
# End-to-end init() wiring
# ---------------------------------------------------------------------------

WORKFLOW_WITH_LITERAL_AND_INPUT_FED_MODELS = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "model"},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v3",
            "name": "static_model",
            "images": "$inputs.image",
            "model_id": "my_project/3",
        },
        {
            "type": "roboflow_core/roboflow_object_detection_model@v3",
            "name": "dynamic_model",
            "images": "$inputs.image",
            "model_id": "$inputs.model",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "static_predictions",
            "selector": "$steps.static_model.predictions",
        },
        {
            "type": "JsonField",
            "name": "dynamic_predictions",
            "selector": "$steps.dynamic_model.predictions",
        },
    ],
}


def test_execution_engine_init_pre_loads_declared_models() -> None:
    # NonCallable: a bare MagicMock is callable, and callable init parameters
    # are treated as factories (mirroring the steps initialiser semantics) —
    # a real ModelManager instance is not callable.
    model_manager = NonCallableMagicMock()

    engine = ExecutionEngineV1.init(
        workflow_definition=WORKFLOW_WITH_LITERAL_AND_INPUT_FED_MODELS,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": "api-key",
        },
        dependencies_pre_init=["roboflow_platform_model"],
    )

    model_manager.add_model.assert_called_once_with(
        model_id="my_project/3", api_key="api-key"
    )
    assert [
        dependency.metadata.model_id
        for dependency in engine._pending_runtime_dependencies
    ] == ["$inputs.model"]
    assert engine._pending_dependencies_resolution_attempted is False


def test_invalid_runtime_input_does_not_consume_the_preload_attempt() -> None:
    model_manager = NonCallableMagicMock()
    engine = ExecutionEngineV1.init(
        workflow_definition=WORKFLOW_WITH_LITERAL_AND_INPUT_FED_MODELS,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": "api-key",
        },
        dependencies_pre_init=["roboflow_platform_model"],
    )
    model_manager.add_model.reset_mock()

    # `model` fails manifest validation (int where a model id string is
    # expected) — validation must fire before preload resolution.
    with pytest.raises(RuntimeInputError):
        engine.run(
            runtime_parameters={
                "image": np.zeros((64, 64, 3), dtype=np.uint8),
                "model": 42,
            }
        )

    assert engine._pending_dependencies_resolution_attempted is False
    model_manager.add_model.assert_not_called()


def test_execution_engine_init_with_remote_execution_mode_pre_loads_nothing() -> None:
    model_manager = NonCallableMagicMock()

    engine = ExecutionEngineV1.init(
        workflow_definition=WORKFLOW_WITH_LITERAL_AND_INPUT_FED_MODELS,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": "api-key",
            "workflows_core.step_execution_mode": StepExecutionMode.REMOTE,
        },
        dependencies_pre_init=["roboflow_platform_model"],
    )

    model_manager.add_model.assert_not_called()
    assert engine._pending_runtime_dependencies == []


def test_execution_engine_init_without_pre_init_dependencies_does_not_touch_manager() -> (
    None
):
    model_manager = NonCallableMagicMock()

    engine = ExecutionEngineV1.init(
        workflow_definition=WORKFLOW_WITH_LITERAL_AND_INPUT_FED_MODELS,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": "api-key",
        },
    )

    model_manager.add_model.assert_not_called()
    assert engine._pending_runtime_dependencies == []


def test_preloading_helpers_are_typed_against_the_port() -> None:
    # The engine drives the model manager itself (add_model / __contains__).
    # Typing it `Any` hid that dependency from import-based tooling.
    for fn in (
        ee_core._pre_load_roboflow_platform_models,
        ee_core._resolve_and_pre_load_runtime_dependencies,
        ee_core._verify_pre_loaded_models_presence,
    ):
        annotation = inspect.signature(fn).parameters["model_manager"].annotation
        assert annotation is ModelsProvider, (fn.__name__, annotation)


# ---------------------------------------------------------------------------
# preloadable=False: known models the generic preloader must not register
# ---------------------------------------------------------------------------


def _non_preloadable(model_id: str):
    return roboflow_platform_model(
        model_id,
        required_action=ModelRequiredAction.EXECUTION,
        execution_location=ModelExecutionLocation.LOCAL,
        preloadable=False,
    )


class _RecordingResolver:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, value: str):
        self.calls.append(value)
        return value


def test_preloadable_defaults_to_true_for_existing_callers() -> None:
    # positional arguments keep their meaning; the new flag is keyword-only
    resource = roboflow_platform_model(
        "my_project/3",
        ModelRequiredAction.EXECUTION,
        ModelExecutionLocation.LOCAL,
    )

    assert resource.metadata.preloadable is True
    assert resource.metadata.execution_location is ModelExecutionLocation.LOCAL
    assert (
        inspect.signature(roboflow_platform_model).parameters["preloadable"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


def test_preloadable_is_internal_and_leaves_serialized_shapes_unchanged() -> None:
    # given
    ordinary = roboflow_platform_model(
        "sam2video/small", execution_location=ModelExecutionLocation.LOCAL
    )
    internal = _non_preloadable("sam2video/small")

    # then - same wire shape, same schema, same equality and hash
    assert (
        internal.to_dict()
        == ordinary.to_dict()
        == {
            "resource_type": "roboflow_platform_model",
            "metadata": {
                "model_id": "sam2video/small",
                "required_action": "execution",
                "execution_location": "local",
            },
        }
    )
    assert internal.model_dump(mode="json") == ordinary.model_dump(mode="json")
    assert "preloadable" not in internal.model_dump_json()
    assert "preloadable" not in json.dumps(type(internal.metadata).model_json_schema())
    assert internal == ordinary
    assert hash(internal) == hash(ordinary)
    assert "preloadable" not in repr(internal)


def test_pre_load_skips_non_preloadable_literal_and_input_dependencies() -> None:
    # given
    model_manager = MagicMock()
    resolver = _RecordingResolver()
    input_fed = roboflow_platform_model(
        "$inputs.tracker",
        execution_location=ModelExecutionLocation.LOCAL,
        model_id_resolver=resolver,
        preloadable=False,
    )

    # when
    pending = _pre_load_roboflow_platform_models(
        dependencies=[_non_preloadable("sam2video/small"), input_fed],
        model_manager=model_manager,
        api_key="api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    _resolve_and_pre_load_runtime_dependencies(
        pending_dependencies=[input_fed],
        runtime_parameters={"tracker": "sam2video/large"},
        model_manager=model_manager,
        api_key="api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # then - never pending, never resolved, never registered
    assert pending == []
    assert resolver.calls == []
    model_manager.add_model.assert_not_called()
    model_manager.__contains__.assert_not_called()


def test_non_preloadable_declaration_does_not_suppress_an_ordinary_one() -> None:
    # given - two steps declare the same id; only one opts out
    model_manager = MagicMock()

    # when
    pending = _pre_load_roboflow_platform_models(
        dependencies=[
            _non_preloadable("shared/1"),
            roboflow_platform_model(model_id="shared/1"),
            _non_preloadable("$inputs.model"),
            roboflow_platform_model(model_id="$inputs.model"),
        ],
        model_manager=model_manager,
        api_key="api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    _resolve_and_pre_load_runtime_dependencies(
        pending_dependencies=[
            _non_preloadable("$inputs.model"),
            *pending,
        ],
        runtime_parameters={"model": "runtime/2"},
        model_manager=model_manager,
        api_key="api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # then
    assert pending == [roboflow_platform_model(model_id="$inputs.model")]
    assert all(dependency.metadata.preloadable for dependency in pending)
    assert model_manager.add_model.call_count == 2
    model_manager.add_model.assert_any_call(model_id="shared/1", api_key="api-key")
    model_manager.add_model.assert_any_call(model_id="runtime/2", api_key="api-key")


def test_ordinary_local_declarations_are_still_preloaded() -> None:
    # given
    model_manager = MagicMock()

    # when
    pending = _pre_load_roboflow_platform_models(
        dependencies=[
            roboflow_platform_model(
                "local/1", execution_location=ModelExecutionLocation.LOCAL
            )
        ],
        model_manager=model_manager,
        api_key="api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # then
    assert pending == []
    model_manager.add_model.assert_called_once_with(
        model_id="local/1", api_key="api-key"
    )


def test_real_video_manifest_declarations_are_never_preloaded() -> None:
    # given - the declarations the streaming video blocks really make
    from roboflow_workflows.core_steps.models.foundation.segment_anything2_video.v1 import (
        BlockManifest as SAM2VideoManifest,
    )
    from roboflow_workflows.core_steps.models.foundation.segment_anything3_video.v1 import (
        BlockManifest as SAM3VideoManifest,
    )
    from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1 import (
        BlockManifest as ActionRecognitionManifest,
    )

    dependencies = [
        *SAM2VideoManifest.model_validate(
            {
                "type": "roboflow_core/segment_anything_2_video@v1",
                "name": "sam2",
                "images": "$inputs.image",
            }
        ).discover_dependent_resources(),
        *SAM3VideoManifest.model_validate(
            {
                "type": "roboflow_core/sam3_video@v1",
                "name": "sam3",
                "images": "$inputs.image",
                "class_names": ["person"],
                "model_id": "$inputs.sam3_model",
            }
        ).discover_dependent_resources(),
        *ActionRecognitionManifest.model_validate(
            {
                "type": "roboflow_core/roboflow_action_recognition_model@v1",
                "name": "actions",
                "images": "$inputs.image",
                "model_id": "my-actions/2",
            }
        ).discover_dependent_resources(),
    ]
    model_manager = MagicMock()

    # when
    pending = _pre_load_roboflow_platform_models(
        dependencies=dependencies,
        model_manager=model_manager,
        api_key="api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # then - declared (visible to introspection) but never registered
    assert [dependency.metadata.model_id for dependency in dependencies] == [
        "sam2video/small",
        "$inputs.sam3_model",
        "my-actions/2",
    ]
    assert pending == []
    model_manager.add_model.assert_not_called()


# ---------------------------------------------------------------------------
# Discovery[DependentResource] declarations reach the pre-loader as items
# ---------------------------------------------------------------------------


def _unavailable_resources_problem(step_name: str):
    return declaration_unavailable_problem(
        node_id=f"$steps.{step_name}", declaration="resources"
    )


def _declare_resources_per_step(monkeypatch, declarations: dict) -> None:
    # Every object-detection step answers with the declaration registered
    # under its own step name. Both manifests are patched: the loader
    # registers the tensor-native one when tensor representation is enabled.
    def discover_dependent_resources(self):
        return declarations[self.name]

    for manifest_class in (ObjectDetectionV3Manifest, ObjectDetectionV3TensorManifest):
        monkeypatch.setattr(
            manifest_class,
            "discover_dependent_resources",
            discover_dependent_resources,
        )


def _init_engine_with_pre_loading(model_manager) -> ExecutionEngineV1:
    engine = ExecutionEngineV1.init(
        workflow_definition=WORKFLOW_WITH_LITERAL_AND_INPUT_FED_MODELS,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": "api-key",
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        dependencies_pre_init=["roboflow_platform_model"],
    )

    return engine


def test_deduce_blocks_dependencies_flattens_discovery_declarations_into_items(
    monkeypatch,
) -> None:
    # given
    literal_resource = roboflow_platform_model(model_id="my_project/3")
    input_fed_resource = roboflow_platform_model(model_id="$inputs.model")
    _declare_resources_per_step(
        monkeypatch,
        declarations={
            "a": complete_discovery([literal_resource]),
            "b": incomplete_discovery(
                [input_fed_resource], [_unavailable_resources_problem("b")]
            ),
        },
    )
    compiled_workflow = _compiled_workflow_with_steps(
        steps=[
            _object_detection_manifest(name="a", model_id="my_project/3"),
            _object_detection_manifest(name="b", model_id="$inputs.model"),
        ]
    )

    # when
    dependencies = deduce_blocks_dependencies(compiled_workflow=compiled_workflow)

    # then - resources, never the (field, value) tuples of the Discovery model
    assert all(isinstance(dependency, DependentResource) for dependency in dependencies)
    assert dependencies == [literal_resource, input_fed_resource]


def test_deduce_blocks_dependencies_skips_unknown_and_empty_declarations(
    monkeypatch,
) -> None:
    # given
    _declare_resources_per_step(monkeypatch, declarations={"a": None, "b": []})
    compiled_workflow = _compiled_workflow_with_steps(
        steps=[
            _object_detection_manifest(name="a", model_id="my_project/3"),
            _object_detection_manifest(name="b", model_id="my_project/4"),
        ]
    )

    # when
    dependencies = deduce_blocks_dependencies(compiled_workflow=compiled_workflow)

    # then
    assert dependencies == []


def test_execution_engine_init_pre_loads_known_items_of_incomplete_discovery(
    monkeypatch,
) -> None:
    # given
    resolver = _RecordingResolver()
    _declare_resources_per_step(
        monkeypatch,
        declarations={
            "static_model": incomplete_discovery(
                [
                    roboflow_platform_model(
                        model_id="my_project/3",
                        model_registration_kwargs={"endpoint_type": "custom"},
                    )
                ],
                [_unavailable_resources_problem("static_model")],
            ),
            "dynamic_model": incomplete_discovery(
                [
                    roboflow_platform_model(
                        model_id="$inputs.model", model_id_resolver=resolver
                    )
                ],
                [_unavailable_resources_problem("dynamic_model")],
            ),
        },
    )
    model_manager = NonCallableMagicMock()

    # when
    engine = _init_engine_with_pre_loading(model_manager=model_manager)

    # then - the known literal model is registered with its kwargs intact and
    # the input-fed one is deferred with its resolver intact
    model_manager.add_model.assert_called_once_with(
        model_id="my_project/3", api_key="api-key", endpoint_type="custom"
    )
    assert len(engine._pending_runtime_dependencies) == 1
    pending_dependency = engine._pending_runtime_dependencies[0]
    assert isinstance(pending_dependency, DependentResource)
    assert pending_dependency.metadata.model_id == "$inputs.model"
    assert pending_dependency.metadata.model_id_resolver is resolver


def test_resolver_of_incomplete_discovery_item_is_applied_on_runtime_resolution(
    monkeypatch,
) -> None:
    # given
    resolver = _RecordingResolver()
    _declare_resources_per_step(
        monkeypatch,
        declarations={
            "static_model": complete_discovery([]),
            "dynamic_model": incomplete_discovery(
                [
                    roboflow_platform_model(
                        model_id="$inputs.model",
                        model_id_resolver=resolver,
                        model_registration_kwargs={"endpoint_type": "custom"},
                    )
                ],
                [_unavailable_resources_problem("dynamic_model")],
            ),
        },
    )
    model_manager = NonCallableMagicMock()
    engine = _init_engine_with_pre_loading(model_manager=model_manager)
    model_manager.add_model.assert_not_called()

    # when
    _resolve_and_pre_load_runtime_dependencies(
        pending_dependencies=engine._pending_runtime_dependencies,
        runtime_parameters={"model": "my_project/7"},
        model_manager=model_manager,
        api_key="api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # then
    assert resolver.calls == ["my_project/7"]
    model_manager.add_model.assert_called_once_with(
        model_id="my_project/7", api_key="api-key", endpoint_type="custom"
    )


def test_execution_engine_init_tolerates_incomplete_discovery_without_items(
    monkeypatch,
) -> None:
    # given - the reported reproduction: an empty incomplete discovery used to
    # hand the Discovery model's (field, value) tuples to the pre-loader
    _declare_resources_per_step(
        monkeypatch,
        declarations={
            "static_model": incomplete_discovery(
                [], [_unavailable_resources_problem("static_model")]
            ),
            "dynamic_model": incomplete_discovery(
                [], [_unavailable_resources_problem("dynamic_model")]
            ),
        },
    )
    model_manager = NonCallableMagicMock()

    # when
    engine = _init_engine_with_pre_loading(model_manager=model_manager)

    # then
    model_manager.add_model.assert_not_called()
    assert engine._pending_runtime_dependencies == []


def test_execution_engine_init_pre_loads_complete_discovery_items(monkeypatch) -> None:
    # given
    _declare_resources_per_step(
        monkeypatch,
        declarations={
            "static_model": complete_discovery(
                [roboflow_platform_model(model_id="my_project/3")]
            ),
            "dynamic_model": complete_discovery(
                [roboflow_platform_model(model_id="$inputs.model")]
            ),
        },
    )
    model_manager = NonCallableMagicMock()

    # when
    engine = _init_engine_with_pre_loading(model_manager=model_manager)

    # then
    model_manager.add_model.assert_called_once_with(
        model_id="my_project/3", api_key="api-key"
    )
    assert [
        dependency.metadata.model_id
        for dependency in engine._pending_runtime_dependencies
    ] == ["$inputs.model"]


@pytest.mark.parametrize(
    "declarations",
    [
        {"static_model": None, "dynamic_model": None},
        {"static_model": [], "dynamic_model": []},
        {"static_model": complete_discovery([]), "dynamic_model": None},
    ],
    ids=["unknown", "legacy-empty-list", "complete-empty-discovery-and-unknown"],
)
def test_execution_engine_init_pre_loads_nothing_when_nothing_is_known(
    monkeypatch, declarations: dict
) -> None:
    # given
    _declare_resources_per_step(monkeypatch, declarations=declarations)
    model_manager = NonCallableMagicMock()

    # when
    engine = _init_engine_with_pre_loading(model_manager=model_manager)

    # then
    model_manager.add_model.assert_not_called()
    assert engine._pending_runtime_dependencies == []


def test_execution_engine_init_pre_loads_legacy_list_declarations(monkeypatch) -> None:
    # given
    _declare_resources_per_step(
        monkeypatch,
        declarations={
            "static_model": [
                roboflow_platform_model(
                    model_id="my_project/3",
                    model_registration_kwargs={"endpoint_type": "custom"},
                )
            ],
            "dynamic_model": [roboflow_platform_model(model_id="$inputs.model")],
        },
    )
    model_manager = NonCallableMagicMock()

    # when
    engine = _init_engine_with_pre_loading(model_manager=model_manager)

    # then
    model_manager.add_model.assert_called_once_with(
        model_id="my_project/3", api_key="api-key", endpoint_type="custom"
    )
    assert [
        dependency.metadata.model_id
        for dependency in engine._pending_runtime_dependencies
    ] == ["$inputs.model"]


def test_discover_dependent_resources_annotation_admits_discovery() -> None:
    annotation = typing.get_type_hints(
        WorkflowBlockManifest.discover_dependent_resources
    )["return"]

    assert set(typing.get_args(annotation)) == {
        typing.List[DependentResource],
        Discovery[DependentResource],
        type(None),
    }
