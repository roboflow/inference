"""`describe_workflow_workload` - the public API, end to end on real core blocks
and structurally compiled dynamic blocks.

Core block declarations (operations / restrictions / resources) are owned by
the block authors and evolve independently; these tests pin graph shape,
dimensionality, model inventory derived from `discover_dependent_resources`
(a long-standing hook) and the declarations of dynamic blocks (owned here).
"""

import copy
import os
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest
from packaging.version import Version
from roboflow_workflows.errors import (
    InvalidReferenceTargetError,
    WorkflowDefinitionError,
    WorkflowExecutionEngineVersionError,
    WorkflowSyntaxError,
)
from roboflow_workflows.execution_engine.entities.workload import (
    ModelMetadata,
    ModelMetadataLookup,
    ModelMetadataProvider,
    WorkOperation,
)
from roboflow_workflows.execution_engine.introspection.workload import (
    describe_workflow_workload,
)
from roboflow_workflows.execution_engine.introspection.workload_entities import (
    WorkflowIntrospection,
)
from roboflow_workflows.execution_engine.v1.compiler import core as compiler_core
from roboflow_workflows.execution_engine.v1.compiler.core import COMPILATION_CACHE
from roboflow_workflows.execution_engine.v1.core import EXECUTION_ENGINE_V1_VERSION
from roboflow_workflows.execution_engine.v1.dynamic_blocks import (
    block_scaffolding,
    modal_executor,
)
from roboflow_workflows.execution_engine.v1.inner_workflow.reference_resolution import (
    WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER,
)
from roboflow_workflows.prototypes.models_provider import ModelsProvider

from tests.unit_tests.execution_engine.dynamic_blocs._workspace_resolver_stub import (
    StubResolver,
)

OBJECT_DETECTION_MODEL = "roboflow_core/roboflow_object_detection_model@v3"
CLASSIFICATION_MODEL = "roboflow_core/roboflow_classification_model@v2"
DYNAMIC_CROP = "roboflow_core/dynamic_crop@v1"
DIMENSION_COLLAPSE = "roboflow_core/dimension_collapse@v1"
CLASSES_REPLACEMENT = "roboflow_core/detections_classes_replacement@v1"
CONTINUE_IF = "roboflow_core/continue_if@v1"
INNER_WORKFLOW = "roboflow_core/inner_workflow@v1"

CONDITION_ON_PREDICTIONS = {
    "type": "StatementGroup",
    "statements": [
        {
            "type": "BinaryStatement",
            "left_operand": {
                "type": "DynamicOperand",
                "operand_name": "predictions",
                "operations": [{"type": "SequenceLength"}],
            },
            "comparator": {"type": "(Number) >="},
            "right_operand": {"type": "StaticOperand", "value": 1},
        }
    ],
}


def _image_input(name: str = "image") -> dict:
    return {"type": "WorkflowImage", "name": name}


def _model(name: str, images: str = "$inputs.image", model_id: str = "my_project/3"):
    return {
        "type": OBJECT_DETECTION_MODEL,
        "name": name,
        "images": images,
        "model_id": model_id,
    }


def _crop(name: str, images: str, predictions: str, block_type: str = DYNAMIC_CROP):
    return {
        "type": block_type,
        "name": name,
        "images": images,
        "predictions": predictions,
    }


def _output(name: str, selector: str) -> dict:
    return {"type": "JsonField", "name": name, "selector": selector}


def _definition(
    steps: List[dict],
    outputs: Optional[List[dict]] = None,
    inputs: Optional[List[dict]] = None,
    dynamic_blocks_definitions: Optional[List[dict]] = None,
    version: str = "1.0",
) -> dict:
    definition: Dict[str, Any] = {
        "version": version,
        "inputs": inputs if inputs is not None else [_image_input()],
        "steps": steps,
        "outputs": outputs or [],
    }
    if dynamic_blocks_definitions is not None:
        definition["dynamic_blocks_definitions"] = dynamic_blocks_definitions
    return definition


def _inputless_dynamic_block(block_type: str = "Inputless") -> dict:
    return {
        "type": "DynamicBlockDefinition",
        "manifest": {
            "type": "ManifestDescription",
            "block_type": block_type,
            "inputs": {},
            "outputs": {
                "output": {"type": "DynamicOutputDefinition", "kind": ["string"]}
            },
        },
        "code": {
            "type": "PythonCode",
            "run_function_code": "def run(self):\n    return {'output': 'x'}\n",
        },
    }


def _side_effect_dynamic_block(marker_path: str, block_type: str = "SideEffect"):
    code = (
        "import pathlib\n"
        f"pathlib.Path({marker_path!r}).write_text('executed')\n"
        "raise RuntimeError('must never run during inspection')\n\n"
        "def run(self, predictions):\n    return {'output': predictions}\n"
    )
    return {
        "type": "DynamicBlockDefinition",
        "manifest": {
            "type": "ManifestDescription",
            "block_type": block_type,
            "inputs": {
                "predictions": {
                    "type": "DynamicInputDefinition",
                    "selector_types": ["step_output"],
                    "selector_data_kind": {
                        "step_output": ["object_detection_prediction"]
                    },
                }
            },
            "outputs": {
                "output": {
                    "type": "DynamicOutputDefinition",
                    "kind": ["object_detection_prediction"],
                }
            },
            "tensor_compatibility": "tensor_native",
        },
        "code": {"type": "PythonCode", "run_function_code": code},
    }


def _step(introspection: WorkflowIntrospection, node_id: str):
    matching = [step for step in introspection.steps if step.node_id == node_id]
    assert len(matching) == 1
    return matching[0]


def _dims(introspection: WorkflowIntrospection) -> Dict[str, tuple]:
    return {
        step.node_id: (step.input_dimensionality, step.output_dimensionality)
        for step in introspection.steps
    }


class RecordingProvider:
    def __init__(self, answers: Optional[Dict[Any, Any]] = None) -> None:
        self.answers = answers or {}
        self.calls: List[Any] = []

    def resolve_model_metadata(self, provider: str, model_id: str):
        self.calls.append((provider, model_id))
        answer = self.answers.get((provider, model_id))
        if isinstance(answer, Exception):
            raise answer
        if answer is None:
            return ModelMetadataLookup(status="unavailable")
        return answer


# ---------------------------------------------------------------------------
# graph and dimensionality on real blocks
# ---------------------------------------------------------------------------


def test_expansion_nesting_reduction_and_alias_normalisation() -> None:
    # given
    definition = _definition(
        steps=[
            _model("model"),
            _crop(
                "crop",
                images="$inputs.image",
                predictions="$steps.model.predictions",
                block_type="DynamicCrop",
            ),
            _model("model_on_crops", images="$steps.crop.crops"),
            _crop(
                "nested_crop",
                images="$steps.crop.crops",
                predictions="$steps.model_on_crops.predictions",
            ),
            {
                "type": DIMENSION_COLLAPSE,
                "name": "collapse",
                "data": "$steps.crop.crops",
            },
        ],
        outputs=[_output("nested", "$steps.nested_crop.crops")],
    )

    # when
    introspection = describe_workflow_workload(workflow_definition=definition)

    # then
    assert _dims(introspection) == {
        "$steps.model": (1, 1),
        "$steps.crop": (1, 2),
        "$steps.model_on_crops": (2, 2),
        "$steps.nested_crop": (2, 3),
        "$steps.collapse": (2, 1),
    }
    assert _step(introspection, "$steps.crop").block_type == DYNAMIC_CROP
    assert _step(introspection, "$steps.crop").accepts_batch_input is True
    assert introspection.summary.steps_by_dimensionality == {1: 2, 2: 3}
    assert introspection.summary.max_dimensionality == 3
    assert [(node.id, node.kind) for node in introspection.nodes] == [
        ("$inputs.image", "input"),
        ("$steps.model", "step"),
        ("$steps.crop", "step"),
        ("$steps.model_on_crops", "step"),
        ("$steps.nested_crop", "step"),
        ("$steps.collapse", "step"),
        ("$outputs.nested", "output"),
    ]
    assert [(e.source, e.target, e.kind) for e in introspection.edges] == [
        ("$inputs.image", "$steps.crop", "data"),
        ("$inputs.image", "$steps.model", "data"),
        ("$steps.crop", "$steps.collapse", "data"),
        ("$steps.crop", "$steps.model_on_crops", "data"),
        ("$steps.crop", "$steps.nested_crop", "data"),
        ("$steps.model", "$steps.crop", "data"),
        ("$steps.model_on_crops", "$steps.nested_crop", "data"),
        ("$steps.nested_crop", "$outputs.nested", "data"),
    ]
    assert introspection.execution_engine_version == str(EXECUTION_ENGINE_V1_VERSION)


def test_mixed_depth_inputs_with_reference_property() -> None:
    definition = _definition(
        steps=[
            _model("model"),
            _crop(
                "crop", images="$inputs.image", predictions="$steps.model.predictions"
            ),
            {
                "type": CLASSIFICATION_MODEL,
                "name": "classifier",
                "images": "$steps.crop.crops",
                "model_id": "breeds/1",
            },
            {
                "type": CLASSES_REPLACEMENT,
                "name": "replacement",
                "object_detection_predictions": "$steps.model.predictions",
                "classification_predictions": "$steps.classifier.predictions",
            },
        ],
        outputs=[_output("predictions", "$steps.replacement.predictions")],
    )
    introspection = describe_workflow_workload(workflow_definition=definition)
    assert _dims(introspection) == {
        "$steps.model": (1, 1),
        "$steps.crop": (1, 2),
        "$steps.classifier": (2, 2),
        "$steps.replacement": (1, 1),
    }
    assert introspection.summary.steps_by_dimensionality == {1: 3, 2: 1}
    assert introspection.summary.max_dimensionality == 2


def test_control_only_scalar_step_and_inputless_step() -> None:
    definition = _definition(
        steps=[
            _model("model"),
            {
                "type": CONTINUE_IF,
                "name": "gate",
                "condition_statement": CONDITION_ON_PREDICTIONS,
                "evaluation_parameters": {"predictions": "$steps.model.predictions"},
                "next_steps": ["$steps.gated"],
            },
            {"type": "Inputless", "name": "gated"},
            {"type": "Inputless", "name": "free"},
        ],
        outputs=[_output("out", "$steps.gated.output")],
        dynamic_blocks_definitions=[_inputless_dynamic_block()],
    )
    introspection = describe_workflow_workload(workflow_definition=definition)
    assert _dims(introspection) == {
        "$steps.model": (1, 1),
        "$steps.gate": (1, 1),
        "$steps.gated": (1, 1),
        "$steps.free": (0, 0),
    }
    assert _step(introspection, "$steps.gate").block_type == CONTINUE_IF
    control = [(e.source, e.target) for e in introspection.edges if e.kind == "control"]
    assert control == [("$steps.gate", "$steps.gated")]
    assert ("$steps.model", "$steps.gate", "data") in [
        (e.source, e.target, e.kind) for e in introspection.edges
    ]
    assert introspection.summary.steps_by_dimensionality == {0: 1, 1: 3}
    gated = _step(introspection, "$steps.gated")
    assert gated.operations.items == [WorkOperation.CUSTOM_PYTHON]
    assert gated.operations.complete is False


def test_no_step_input_to_output_workflow() -> None:
    definition = _definition(steps=[], outputs=[_output("echo", "$inputs.image")])
    introspection = describe_workflow_workload(workflow_definition=definition)
    assert introspection.steps == []
    assert introspection.summary.steps_by_dimensionality == {}
    assert introspection.summary.max_dimensionality == 1
    assert [(e.source, e.target, e.kind) for e in introspection.edges] == [
        ("$inputs.image", "$outputs.echo", "data")
    ]


# ---------------------------------------------------------------------------
# model inventory
# ---------------------------------------------------------------------------


def test_model_inventory_dedupes_and_reports_selectors() -> None:
    definition = _definition(
        inputs=[_image_input(), {"type": "WorkflowParameter", "name": "model"}],
        steps=[
            _model("a", model_id="shared/1"),
            _model("b", model_id="shared/1"),
            _model("c", model_id="other/2"),
            _model("dynamic", model_id="$inputs.model"),
        ],
    )
    models = describe_workflow_workload(workflow_definition=definition).summary.models
    assert [(m.provider, m.model_id, m.used_by_steps) for m in models.items] == [
        ("roboflow", "other/2", ["$steps.c"]),
        ("roboflow", "shared/1", ["$steps.a", "$steps.b"]),
    ]
    assert models.complete is False
    assert models.unknown_reasons == ["unresolved_model_selector:$steps.dynamic"]
    assert all(m.metadata_status == "unavailable" for m in models.items)


def test_metadata_provider_is_called_once_per_pair_and_never_with_selectors() -> None:
    definition = _definition(
        inputs=[_image_input(), {"type": "WorkflowParameter", "name": "model"}],
        steps=[
            _model("a", model_id="shared/1"),
            _model("b", model_id="shared/1"),
            _model("dynamic", model_id="$inputs.model"),
        ],
    )
    provider = RecordingProvider(
        answers={
            ("roboflow", "shared/1"): ModelMetadataLookup(
                status="available", metadata=ModelMetadata(model_type="yolov8n")
            )
        }
    )
    assert isinstance(provider, ModelMetadataProvider)
    introspection = describe_workflow_workload(
        workflow_definition=definition, model_metadata_provider=provider
    )
    assert provider.calls == [("roboflow", "shared/1")]
    (entry,) = introspection.summary.models.items
    assert entry.metadata_status == "available"
    assert entry.metadata == ModelMetadata(model_type="yolov8n")


# ---------------------------------------------------------------------------
# inner workflows
# ---------------------------------------------------------------------------


def _child_definition() -> dict:
    return _definition(
        steps=[
            _model("model", model_id="child/1"),
            _crop(
                "crop", images="$inputs.image", predictions="$steps.model.predictions"
            ),
        ],
        outputs=[_output("crops", "$steps.crop.crops")],
    )


def test_embedded_inner_workflow_is_counted_once_under_generated_names() -> None:
    definition = _definition(
        steps=[
            {
                "type": INNER_WORKFLOW,
                "name": "inner",
                "workflow_definition": _child_definition(),
                "parameter_bindings": {"image": "$inputs.image"},
            }
        ],
        outputs=[_output("crops", "$steps.inner.crops")],
    )
    frozen = copy.deepcopy(definition)
    introspection = describe_workflow_workload(workflow_definition=definition)
    assert _dims(introspection) == {
        "$steps.inner__model": (1, 1),
        "$steps.inner__crop": (1, 2),
    }
    assert "$steps.inner" not in {node.id for node in introspection.nodes}
    assert [
        (m.model_id, m.used_by_steps) for m in introspection.summary.models.items
    ] == [("child/1", ["$steps.inner__model"])]
    assert definition == frozen


def test_saved_inner_workflow_resolved_through_injected_resolver() -> None:
    calls = []

    def resolver(workspace_id, workflow_id, workflow_version_id, init_parameters):
        calls.append((workspace_id, workflow_id, workflow_version_id))
        return _child_definition()

    definition = _definition(
        steps=[
            {
                "type": INNER_WORKFLOW,
                "name": "inner",
                "workflow_workspace_id": "ws",
                "workflow_id": "saved",
                "parameter_bindings": {"image": "$inputs.image"},
            }
        ],
        outputs=[_output("crops", "$steps.inner.crops")],
    )
    introspection = describe_workflow_workload(
        workflow_definition=definition,
        init_parameters={WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER: resolver},
    )
    assert calls == [("ws", "saved", None)]
    assert set(_dims(introspection)) == {"$steps.inner__model", "$steps.inner__crop"}


def test_remote_dispatch_inner_workflow_stays_opaque() -> None:
    definition = _definition(
        steps=[
            _model("model"),
            {
                "type": INNER_WORKFLOW,
                "name": "dispatch",
                "execution_mode": "remote_dispatch",
                "workflow_definition": {
                    "version": "1.0",
                    "inputs": [_image_input()],
                    "dynamic_blocks_definitions": [
                        _inputless_dynamic_block("RemoteOnly")
                    ],
                    "steps": [{"type": "not_installed/block@v1", "name": "x"}],
                    "outputs": [],
                },
                "parameter_bindings": {"image": "$inputs.image"},
            },
        ],
        outputs=[_output("predictions", "$steps.model.predictions")],
    )
    introspection = describe_workflow_workload(workflow_definition=definition)
    dispatch = _step(introspection, "$steps.dispatch")
    assert dispatch.block_type == INNER_WORKFLOW
    for discovery in (dispatch.resources, dispatch.operations, dispatch.restrictions):
        assert discovery.complete is False
        assert (
            "remote_dispatch_child_opaque:$steps.dispatch" in discovery.unknown_reasons
        )
    assert introspection.summary.models.complete is False
    assert [m.model_id for m in introspection.summary.models.items] == ["my_project/3"]
    assert set(_dims(introspection)) == {"$steps.model", "$steps.dispatch"}


# ---------------------------------------------------------------------------
# dynamic blocks and inertness through the public API
# ---------------------------------------------------------------------------


@pytest.fixture
def marker_path(empty_directory: str) -> str:
    return os.path.join(empty_directory, "marker.txt")


def test_dynamic_block_declarations_and_inertness(marker_path: str) -> None:
    # given
    definition = _definition(
        steps=[
            _model("model"),
            {
                "type": "SideEffect",
                "name": "custom",
                "predictions": "$steps.model.predictions",
            },
        ],
        outputs=[_output("out", "$steps.custom.output")],
        dynamic_blocks_definitions=[_side_effect_dynamic_block(marker_path)],
    )
    frozen = copy.deepcopy(definition)
    resolver = StubResolver(workspace="ws")
    models_provider = mock.MagicMock(spec=ModelsProvider)
    cache_before = dict(COMPILATION_CACHE._cache)

    # when
    with mock.patch.object(
        compiler_core, "initialise_steps"
    ) as initialise_steps, mock.patch.object(
        block_scaffolding, "create_dynamic_module"
    ) as create_dynamic_module, mock.patch.object(
        block_scaffolding, "exec", create=True
    ) as exec_spy, mock.patch.object(
        modal_executor, "validate_code_in_modal"
    ) as validate_code_in_modal:
        introspection = describe_workflow_workload(
            workflow_definition=definition,
            init_parameters={
                "workflows_core.api_key": "secret",
                "workflows_core.workspace_resolver": resolver,
                "workflows_core.model_manager": models_provider,
            },
        )

    # then
    initialise_steps.assert_not_called()
    create_dynamic_module.assert_not_called()
    exec_spy.assert_not_called()
    validate_code_in_modal.assert_not_called()
    models_provider.add_model.assert_not_called()
    assert resolver.calls == []
    assert not os.path.exists(marker_path)
    assert definition == frozen
    assert dict(COMPILATION_CACHE._cache) == cache_before
    custom = _step(introspection, "$steps.custom")
    assert custom.block_type == "SideEffect"
    assert custom.operations.items == [WorkOperation.CUSTOM_PYTHON]
    assert custom.operations.unknown_reasons == [
        "custom_python_internal_operations_unknown:$steps.custom"
    ]
    assert {item.code for item in custom.restrictions.items} == {
        "custom_python_execution_disabled",
        "tensor_native_requires_tensor_representation",
        "tensor_native_unsupported_in_modal",
    }
    assert custom.resources.unknown_reasons == ["step_resources_unknown:$steps.custom"]
    assert "step_resources_unknown:$steps.custom" in (
        introspection.summary.models.unknown_reasons
    )
    assert introspection.summary.models.complete is False


# ---------------------------------------------------------------------------
# versions, errors, serialization, stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "execution_engine_version",
    [None, "1.0.0", "1.5.0", Version("1.0.0"), str(EXECUTION_ENGINE_V1_VERSION)],
)
def test_supported_engine_versions(execution_engine_version) -> None:
    introspection = describe_workflow_workload(
        workflow_definition=_definition(steps=[_model("model")]),
        execution_engine_version=execution_engine_version,
    )
    assert introspection.execution_engine_version == str(EXECUTION_ENGINE_V1_VERSION)


@pytest.mark.parametrize("execution_engine_version", ["2.0.0", Version("0.9.0")])
def test_unsupported_engine_versions_raise(execution_engine_version) -> None:
    with pytest.raises(WorkflowExecutionEngineVersionError):
        describe_workflow_workload(
            workflow_definition=_definition(steps=[_model("model")]),
            execution_engine_version=execution_engine_version,
        )


def test_unparsable_engine_version_raises() -> None:
    with pytest.raises(WorkflowExecutionEngineVersionError):
        describe_workflow_workload(
            workflow_definition=_definition(steps=[_model("model")]),
            execution_engine_version="not-a-version",
        )


def test_definition_version_selects_engine() -> None:
    with pytest.raises(WorkflowExecutionEngineVersionError):
        describe_workflow_workload(
            workflow_definition=_definition(steps=[_model("model")], version="2.0")
        )


def test_non_dict_definition_raises() -> None:
    with pytest.raises(WorkflowDefinitionError):
        describe_workflow_workload(workflow_definition=["not", "a", "dict"])


def test_malformed_definition_raises_existing_error_type() -> None:
    definition = _definition(steps=[{"type": "not_installed/block@v1", "name": "x"}])
    with pytest.raises(WorkflowSyntaxError):
        describe_workflow_workload(workflow_definition=definition)


def test_invalid_selector_raises_existing_error_type() -> None:
    definition = _definition(
        steps=[_model("model")],
        outputs=[_output("out", "$steps.missing.predictions")],
    )
    with pytest.raises(InvalidReferenceTargetError):
        describe_workflow_workload(workflow_definition=definition)


def test_result_roundtrips_json_and_serialises_type_fields() -> None:
    definition = _definition(
        steps=[
            _model("model"),
            _crop(
                "crop", images="$inputs.image", predictions="$steps.model.predictions"
            ),
        ],
        outputs=[_output("crops", "$steps.crop.crops")],
    )
    introspection = describe_workflow_workload(workflow_definition=definition)
    roundtrip = WorkflowIntrospection.model_validate_json(
        introspection.model_dump_json()
    )
    assert roundtrip == introspection
    dumped = introspection.model_dump(mode="json")
    assert dumped["summary"]["steps_by_dimensionality"] == {"1": 2}
    assert dumped["summary"]["models"]["items"][0]["type"] == "model_summary"
    assert dumped["steps"][0]["resources"]["items"][0]["type"] == "dependent_resource"
    assert dumped["steps"][0]["resources"]["items"][0]["metadata"]["type"] == (
        "roboflow_platform_model"
    )


def test_output_is_stable_across_runs() -> None:
    definition = _definition(
        inputs=[
            {
                "type": "WorkflowBatchInput",
                "name": "crops",
                "kind": ["image"],
                "dimensionality": 2,
            }
        ],
        steps=[
            _model("model", images="$inputs.crops"),
            _crop(
                "crop", images="$inputs.crops", predictions="$steps.model.predictions"
            ),
        ],
        outputs=[_output("crops", "$steps.crop.crops")],
    )
    first = describe_workflow_workload(workflow_definition=definition)
    second = describe_workflow_workload(workflow_definition=copy.deepcopy(definition))
    assert first.model_dump() == second.model_dump()
    assert first is not second
    assert _dims(first) == {"$steps.model": (2, 2), "$steps.crop": (2, 3)}


# ---------------------------------------------------------------------------
# blank literal identifiers through the public API (deep-review F1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blank_model_id", ["", "   "])
def test_blank_platform_model_id_does_not_break_introspection(
    blank_model_id: str,
) -> None:
    definition = _definition(
        steps=[_model("blank", model_id=blank_model_id), _model("valid")],
        outputs=[_output("predictions", "$steps.valid.predictions")],
    )
    provider = RecordingProvider()
    introspection = describe_workflow_workload(
        workflow_definition=definition, model_metadata_provider=provider
    )
    models = introspection.summary.models
    assert models.complete is False
    assert models.unknown_reasons == ["blank_model_identifier:$steps.blank"]
    assert [(m.model_id, m.used_by_steps) for m in models.items] == [
        ("my_project/3", ["$steps.valid"])
    ]
    assert provider.calls == [("roboflow", "my_project/3")]
    blank = _step(introspection, "$steps.blank")
    assert blank.resources.items[0].metadata.model_id == blank_model_id


@pytest.mark.parametrize("blank_base_url", ["", "   "])
def test_blank_openai_compatible_base_url_does_not_break_introspection(
    blank_base_url: str,
) -> None:
    definition = _definition(
        steps=[
            {
                "type": "roboflow_core/openai_compatible@v1",
                "name": "llm",
                "base_url": blank_base_url,
                "model_name": "gpt-x",
                "prompt": "describe {image}",
                "prompt_parameters": {"image": "$inputs.image"},
            }
        ],
    )
    provider = RecordingProvider()
    introspection = describe_workflow_workload(
        workflow_definition=definition, model_metadata_provider=provider
    )
    models = introspection.summary.models
    assert models.items == []
    assert models.complete is False
    assert "blank_model_identifier:$steps.llm" in models.unknown_reasons
    assert provider.calls == []
    llm = _step(introspection, "$steps.llm")
    third_party = [
        item
        for item in llm.resources.items
        if item.resource_type.value == "third_party_model"
    ]
    assert len(third_party) == 1
    assert third_party[0].metadata.model_id == "gpt-x"
    assert not third_party[0].metadata.provider.strip()
