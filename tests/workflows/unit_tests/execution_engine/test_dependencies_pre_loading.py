"""
Tests for pre-loading of declared dependent resources at Execution Engine
init time (`dependencies_pre_init`) and on the first run (for dependencies
declared through `$inputs.<name>` selectors).
"""

from unittest.mock import MagicMock, NonCallableMagicMock

import networkx as nx
import pytest

from inference.core.workflows.core_steps.models.roboflow.object_detection.v3 import (
    BlockManifest as ObjectDetectionV3Manifest,
)
from inference.core.workflows.errors import (
    RuntimeInputError,
    WorkflowEnvironmentConfigurationError,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    CompiledWorkflow,
    ParsedWorkflowDefinition,
)
from inference.core.workflows.execution_engine.v1.compiler.utils import (
    deduce_blocks_dependencies,
)
from inference.core.workflows.execution_engine.v1.core import (
    ExecutionEngineV1,
    _parse_dependencies_pre_init,
    _pre_load_roboflow_platform_models,
    _resolve_and_pre_load_runtime_dependencies,
    _retrieve_init_parameter,
    _retrieve_step_execution_mode,
)
from inference.core.workflows.prototypes.block import (
    DependentResourceType,
    ModelExecutionLocation,
    ModelRequiredAction,
    StepExecutionMode,
    roboflow_platform_model,
    roboflow_platform_project,
    third_party_model,
)


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
