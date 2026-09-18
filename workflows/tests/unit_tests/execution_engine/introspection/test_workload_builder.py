"""`build_workflow_introspection` - graph, per-step facts, inventory, summary.

Fixtures are built from test-local manifests through the real graph
constructor (`prepare_execution_graph`), so every dimensionality pinned here is
the compiler's own answer, and the declarations are fully under the test's
control (core block declarations evolve independently).
"""

from typing import Any, Dict, List, Literal, Optional, Type, Union

import pytest
from pydantic import ConfigDict, Field
from roboflow_workflows.core_steps.flow_control.inner_workflow.v1 import (
    BlockManifest as InnerWorkflowManifest,
)
from roboflow_workflows.core_steps.flow_control.inner_workflow.v1 import (
    InnerWorkflowBlockV1,
)
from roboflow_workflows.execution_engine.entities.base import (
    JsonField,
    OutputDefinition,
    WorkflowBatchInput,
    WorkflowImage,
    WorkflowParameter,
)
from roboflow_workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    IMAGE_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    STRING_KIND,
    Selector,
    StepOutputImageSelector,
    StepOutputSelector,
    StepSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    ModelMetadata,
    ModelMetadataLookup,
    ModelMetadataProvider,
    RestrictionCondition,
    RestrictionMetadata,
    Runtime,
    Severity,
    StepExecutionMode,
    WorkOperation,
    complete_discovery,
    incomplete_discovery,
)
from roboflow_workflows.execution_engine.introspection.utils import get_full_type_name
from roboflow_workflows.execution_engine.introspection.workload_entities import (
    WorkflowIntrospection,
)
from roboflow_workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
    ParsedWorkflowDefinition,
    StepNode,
    StructuralCompilationResult,
)
from roboflow_workflows.execution_engine.v1.compiler.graph_constructor import (
    prepare_execution_graph,
)
from roboflow_workflows.execution_engine.v1.core import EXECUTION_ENGINE_V1_VERSION
from roboflow_workflows.execution_engine.v1.introspection.workload import (
    build_workflow_introspection,
)
from roboflow_workflows.prototypes.block import (
    DependentResource,
    ModelRequiredAction,
    WorkflowBlock,
    WorkflowBlockManifest,
    roboflow_platform_model,
    roboflow_platform_project,
    third_party_model,
)

# ---------------------------------------------------------------------------
# test-local blocks
# ---------------------------------------------------------------------------

GPU_RESTRICTION = RestrictionMetadata(
    code="requires_gpu_for_local_execution",
    severity=Severity.HARD,
    when=RestrictionCondition(
        runtimes=[Runtime.SELF_HOSTED_CPU],
        step_execution_modes=[StepExecutionMode.LOCAL],
    ),
)


class ModelManifest(WorkflowBlockManifest):
    model_config = ConfigDict(protected_namespaces=())
    type: Literal["test/model@v1", "TestModelAlias"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector]
    model_id: Union[WorkflowParameterSelector(kind=[ROBOFLOW_MODEL_ID_KIND]), str]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            )
        ]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        return [roboflow_platform_model(model_id=self.model_id)]

    def discover_work_operations(self):
        return [WorkOperation.MODEL_INFERENCE]

    def discover_portable_restrictions(self):
        return [GPU_RESTRICTION]


class ClassifierManifest(WorkflowBlockManifest):
    model_config = ConfigDict(protected_namespaces=())
    type: Literal["test/classifier@v1"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector]
    model_id: str = "classifier/1"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="predictions", kind=[CLASSIFICATION_PREDICTION_KIND])
        ]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        return [roboflow_platform_model(model_id=self.model_id)]

    def discover_work_operations(self):
        return [WorkOperation.MODEL_INFERENCE]

    def discover_portable_restrictions(self):
        return []


class ThirdPartyManifest(WorkflowBlockManifest):
    type: Literal["test/third_party@v1"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector]
    provider: str = "openai"
    model: str = "gpt-x"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="result", kind=[STRING_KIND])]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        return [third_party_model(provider=self.provider, model_id=self.model)]

    def discover_work_operations(self):
        return [WorkOperation.MODEL_INFERENCE, WorkOperation.EXTERNAL_REQUEST]

    def discover_portable_restrictions(self):
        return []


class ProjectSinkManifest(WorkflowBlockManifest):
    type: Literal["test/project_sink@v1"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector]
    project_url: str = "workspace/project"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="status", kind=[BOOLEAN_KIND])]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        return [
            roboflow_platform_project(project_url=self.project_url),
            roboflow_platform_model(
                model_id="monitored/1", required_action=ModelRequiredAction.ACCESS
            ),
        ]

    def discover_work_operations(self):
        return [WorkOperation.STORAGE_WRITE, WorkOperation.EXTERNAL_REQUEST]

    def discover_portable_restrictions(self):
        return []


class UnknownManifest(WorkflowBlockManifest):
    """No declaration at all - the unannotated plugin case."""

    type: Literal["test/unknown@v1"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="image", kind=[IMAGE_KIND])]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]


class NoopManifest(WorkflowBlockManifest):
    type: Literal["test/noop@v1"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="image", kind=[IMAGE_KIND])]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        return []

    def discover_work_operations(self):
        return []

    def discover_portable_restrictions(self):
        return []


class FailingHooksManifest(WorkflowBlockManifest):
    type: Literal["test/failing@v1"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="image", kind=[IMAGE_KIND])]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        raise RuntimeError("resources hook exploded")

    def discover_work_operations(self):
        raise RuntimeError("operations hook exploded")

    def discover_portable_restrictions(self):
        raise RuntimeError("restrictions hook exploded")


class GarbageHooksManifest(WorkflowBlockManifest):
    type: Literal["test/garbage@v1"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="image", kind=[IMAGE_KIND])]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    def discover_dependent_resources(self):
        return ["not-a-resource"]

    def discover_work_operations(self):
        return ["not-an-operation"]

    def discover_portable_restrictions(self):
        return [{"code": 1}]


class ExplicitDiscoveryManifest(WorkflowBlockManifest):
    type: Literal["test/explicit@v1"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="image", kind=[IMAGE_KIND])]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    def discover_dependent_resources(self):
        return incomplete_discovery(
            [roboflow_platform_model(model_id="explicit/1")],
            [f"explicit_partial:$steps.{self.name}"],
        )

    def discover_work_operations(self):
        return incomplete_discovery(
            [WorkOperation.CUSTOM_PYTHON],
            [f"custom_python_internal_operations_unknown:$steps.{self.name}"],
        )

    def discover_portable_restrictions(self):
        return complete_discovery([])


class CropManifest(WorkflowBlockManifest):
    type: Literal["test/crop@v1"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector]
    predictions: StepOutputSelector(kind=[OBJECT_DETECTION_PREDICTION_KIND])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="crops", kind=[IMAGE_KIND])]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images", "predictions"]

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return 1

    def discover_dependent_resources(self):
        return []

    def discover_work_operations(self):
        return [WorkOperation.IMAGE_CROP]

    def discover_portable_restrictions(self):
        return []


class CollapseManifest(WorkflowBlockManifest):
    type: Literal["test/collapse@v1"]
    data: StepOutputSelector()

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output", kind=[LIST_OF_VALUES_KIND])]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["data"]

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return -1

    def discover_dependent_resources(self):
        return []

    def discover_work_operations(self):
        return [WorkOperation.DATA_AGGREGATION]

    def discover_portable_restrictions(self):
        return []


class ReplacementManifest(WorkflowBlockManifest):
    type: Literal["test/replacement@v1"]
    object_detection_predictions: StepOutputSelector(
        kind=[OBJECT_DETECTION_PREDICTION_KIND]
    )
    classification_predictions: StepOutputSelector(
        kind=[CLASSIFICATION_PREDICTION_KIND]
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            )
        ]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["object_detection_predictions", "classification_predictions"]

    @classmethod
    def get_input_dimensionality_offsets(cls) -> Dict[str, int]:
        return {"object_detection_predictions": 0, "classification_predictions": 1}

    @classmethod
    def get_dimensionality_reference_property(cls) -> Optional[str]:
        return "object_detection_predictions"

    def discover_dependent_resources(self):
        return []

    def discover_work_operations(self):
        return [WorkOperation.DETECTION_PROCESSING]

    def discover_portable_restrictions(self):
        return []


class ScalarSourceManifest(WorkflowBlockManifest):
    """Input-less step producing a scalar."""

    type: Literal["test/scalar_source@v1"]
    text: str = "constant"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output", kind=[STRING_KIND])]

    def discover_dependent_resources(self):
        return []

    def discover_work_operations(self):
        return [WorkOperation.DATA_TRANSFORMATION]

    def discover_portable_restrictions(self):
        return []


class BatchOnlyManifest(WorkflowBlockManifest):
    """Batch-only property: a scalar plugged in gets auto-batch-casted."""

    type: Literal["test/batch_only@v1"]
    data: StepOutputSelector()

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output", kind=[STRING_KIND])]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["data"]

    def discover_dependent_resources(self):
        return []

    def discover_work_operations(self):
        return [WorkOperation.DATA_TRANSFORMATION]

    def discover_portable_restrictions(self):
        return []


class ConditionManifest(WorkflowBlockManifest):
    type: Literal["test/continue_if@v1"]
    evaluation_parameters: Dict[str, Selector()] = Field(default_factory=dict)
    next_steps: List[StepSelector]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []

    def discover_dependent_resources(self):
        return []

    def discover_work_operations(self):
        return [WorkOperation.FLOW_CONTROL]

    def discover_portable_restrictions(self):
        return []


class SwitchManifest(WorkflowBlockManifest):
    type: Literal["test/switch@v1"]
    value: Selector()
    cases: Dict[str, StepSelector]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []

    def discover_dependent_resources(self):
        return []

    def discover_work_operations(self):
        return [WorkOperation.FLOW_CONTROL]

    def discover_portable_restrictions(self):
        return []


class GateWithOutputManifest(WorkflowBlockManifest):
    """Flow control that also produces data - lets one node pair carry both a
    data and a control edge."""

    type: Literal["test/gate_with_output@v1"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector]
    next_steps: List[StepSelector]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="image", kind=[IMAGE_KIND])]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    def discover_dependent_resources(self):
        return []

    def discover_work_operations(self):
        return [WorkOperation.FLOW_CONTROL]

    def discover_portable_restrictions(self):
        return []


TEST_PLUGIN = "test_workload_plugin"


def _specification(
    manifest_class: Type[WorkflowBlockManifest],
    block_source: str = TEST_PLUGIN,
    block_class: Optional[Type[WorkflowBlock]] = None,
) -> BlockSpecification:
    if block_class is None:
        block_class = type(
            f"{manifest_class.__name__}Block",
            (WorkflowBlock,),
            {
                "get_manifest": classmethod(lambda cls: manifest_class),
                "run": lambda self, *args, **kwargs: None,
            },
        )
    return BlockSpecification(
        block_source=block_source,
        identifier=get_full_type_name(selected_type=block_class),
        block_class=block_class,
        manifest_class=manifest_class,
    )


AVAILABLE_BLOCKS = [
    _specification(manifest_class)
    for manifest_class in [
        ModelManifest,
        ClassifierManifest,
        ThirdPartyManifest,
        ProjectSinkManifest,
        UnknownManifest,
        NoopManifest,
        FailingHooksManifest,
        GarbageHooksManifest,
        ExplicitDiscoveryManifest,
        CropManifest,
        CollapseManifest,
        ReplacementManifest,
        ScalarSourceManifest,
        BatchOnlyManifest,
        ConditionManifest,
        SwitchManifest,
        GateWithOutputManifest,
    ]
] + [
    _specification(
        InnerWorkflowManifest,
        block_source="roboflow_core",
        block_class=InnerWorkflowBlockV1,
    )
]


def _compile(
    inputs: List[Any],
    steps: List[WorkflowBlockManifest],
    outputs: Optional[List[JsonField]] = None,
) -> StructuralCompilationResult:
    definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=inputs,
        steps=steps,
        outputs=outputs or [],
    )
    return StructuralCompilationResult(
        execution_graph=prepare_execution_graph(workflow_definition=definition),
        parsed_workflow_definition=definition,
        available_blocks=AVAILABLE_BLOCKS,
    )


def _image_input(name: str = "image") -> WorkflowImage:
    return WorkflowImage(type="WorkflowImage", name=name)


def _model(name: str, images: str = "$inputs.image", model_id: str = "project/1"):
    return ModelManifest(
        type="test/model@v1", name=name, images=images, model_id=model_id
    )


def _output(name: str, selector: str) -> JsonField:
    return JsonField(type="JsonField", name=name, selector=selector)


def _step(introspection: WorkflowIntrospection, node_id: str):
    matching = [step for step in introspection.steps if step.node_id == node_id]
    assert len(matching) == 1, f"expected exactly one StepMetadata for {node_id}"
    return matching[0]


def _step_node(result: StructuralCompilationResult, node_id: str) -> StepNode:
    return result.execution_graph.nodes[node_id]["node_compilation_output"]


def _assert_reference_depth_consistent(
    result: StructuralCompilationResult, introspection: WorkflowIntrospection
) -> None:
    """Both derivations of the effective input depth must agree."""
    for step in introspection.steps:
        node = _step_node(result, step.node_id)
        expected = (
            step.output_dimensionality
            - node.step_manifest.get_output_dimensionality_offset()
        )
        assert step.input_dimensionality == node.reference_dimensionality
        assert step.input_dimensionality == expected, step.node_id


# ---------------------------------------------------------------------------
# nodes and edges
# ---------------------------------------------------------------------------


def test_nodes_are_inputs_steps_outputs_in_definition_order() -> None:
    # given
    result = _compile(
        inputs=[
            _image_input("image"),
            WorkflowParameter(type="WorkflowParameter", name="model"),
        ],
        steps=[_model("model", model_id="$inputs.model"), _model("second")],
        outputs=[
            _output("predictions", "$steps.model.predictions"),
            _output("more", "$steps.second.predictions"),
        ],
    )

    # when
    introspection = build_workflow_introspection(compilation_result=result)

    # then
    assert [(node.id, node.kind) for node in introspection.nodes] == [
        ("$inputs.image", "input"),
        ("$inputs.model", "input"),
        ("$steps.model", "step"),
        ("$steps.second", "step"),
        ("$outputs.predictions", "output"),
        ("$outputs.more", "output"),
    ]
    assert not any("super-input" in node.id for node in introspection.nodes)


def test_data_edges_cover_inputs_steps_and_outputs() -> None:
    # given
    result = _compile(
        inputs=[_image_input()],
        steps=[
            _model("model"),
            CropManifest(
                type="test/crop@v1",
                name="crop",
                images="$inputs.image",
                predictions="$steps.model.predictions",
            ),
        ],
        outputs=[_output("crops", "$steps.crop.crops")],
    )

    # when
    introspection = build_workflow_introspection(compilation_result=result)

    # then
    assert [(e.source, e.target, e.kind) for e in introspection.edges] == [
        ("$inputs.image", "$steps.crop", "data"),
        ("$inputs.image", "$steps.model", "data"),
        ("$steps.crop", "$outputs.crops", "data"),
        ("$steps.model", "$steps.crop", "data"),
    ]


def test_control_edges_for_condition_and_switch() -> None:
    # given
    result = _compile(
        inputs=[_image_input()],
        steps=[
            _model("model"),
            ConditionManifest(
                type="test/continue_if@v1",
                name="gate",
                evaluation_parameters={"predictions": "$steps.model.predictions"},
                next_steps=["$steps.on_true"],
            ),
            SwitchManifest(
                type="test/switch@v1",
                name="switch",
                value="$steps.model.predictions",
                cases={"a": "$steps.on_a", "b": "$steps.on_b"},
            ),
            ScalarSourceManifest(type="test/scalar_source@v1", name="on_true"),
            ScalarSourceManifest(type="test/scalar_source@v1", name="on_a"),
            ScalarSourceManifest(type="test/scalar_source@v1", name="on_b"),
        ],
    )

    # when
    introspection = build_workflow_introspection(compilation_result=result)

    # then
    control = [(e.source, e.target) for e in introspection.edges if e.kind == "control"]
    assert control == [
        ("$steps.gate", "$steps.on_true"),
        ("$steps.switch", "$steps.on_a"),
        ("$steps.switch", "$steps.on_b"),
    ]
    data = [(e.source, e.target) for e in introspection.edges if e.kind == "data"]
    assert data == [
        ("$inputs.image", "$steps.model"),
        ("$steps.model", "$steps.gate"),
        ("$steps.model", "$steps.switch"),
    ]


def test_same_node_pair_can_carry_data_and_control_edges() -> None:
    # given
    result = _compile(
        inputs=[_image_input()],
        steps=[
            GateWithOutputManifest(
                type="test/gate_with_output@v1",
                name="gate",
                images="$inputs.image",
                next_steps=["$steps.consumer"],
            ),
            UnknownManifest(
                type="test/unknown@v1", name="consumer", images="$steps.gate.image"
            ),
        ],
    )

    # when
    introspection = build_workflow_introspection(compilation_result=result)

    # then
    assert [(e.source, e.target, e.kind) for e in introspection.edges] == [
        ("$inputs.image", "$steps.gate", "data"),
        ("$steps.gate", "$steps.consumer", "control"),
        ("$steps.gate", "$steps.consumer", "data"),
    ]


def test_edges_never_expose_lineage_or_selectors() -> None:
    result = _compile(
        inputs=[_image_input()],
        steps=[_model("model")],
        outputs=[_output("predictions", "$steps.model.predictions")],
    )
    introspection = build_workflow_introspection(compilation_result=result)
    for edge in introspection.edges:
        assert set(edge.model_dump().keys()) == {"type", "source", "target", "kind"}


# ---------------------------------------------------------------------------
# per-step facts
# ---------------------------------------------------------------------------


def test_block_type_is_canonical_even_when_alias_used() -> None:
    result = _compile(
        inputs=[_image_input()],
        steps=[
            ModelManifest(
                type="TestModelAlias",
                name="model",
                images="$inputs.image",
                model_id="project/1",
            )
        ],
    )
    introspection = build_workflow_introspection(compilation_result=result)
    assert _step(introspection, "$steps.model").block_type == "test/model@v1"


def test_accepts_batch_input_is_normalised_bool() -> None:
    result = _compile(
        inputs=[_image_input()],
        steps=[
            _model("model"),
            ScalarSourceManifest(type="test/scalar_source@v1", name="scalar"),
        ],
    )
    introspection = build_workflow_introspection(compilation_result=result)
    assert _step(introspection, "$steps.model").accepts_batch_input is True
    assert _step(introspection, "$steps.scalar").accepts_batch_input is False


def test_dimensionality_expansion_nesting_and_reduction() -> None:
    # given: model (1->1), crop (1->2), model on crops (2->2), nested crop
    # (2->3), collapse of the first crops (2->1, executor depth 1)
    result = _compile(
        inputs=[_image_input()],
        steps=[
            _model("model"),
            CropManifest(
                type="test/crop@v1",
                name="crop",
                images="$inputs.image",
                predictions="$steps.model.predictions",
            ),
            _model("model_on_crops", images="$steps.crop.crops"),
            CropManifest(
                type="test/crop@v1",
                name="nested_crop",
                images="$steps.crop.crops",
                predictions="$steps.model_on_crops.predictions",
            ),
            CollapseManifest(
                type="test/collapse@v1", name="collapse", data="$steps.crop.crops"
            ),
        ],
        outputs=[_output("nested", "$steps.nested_crop.crops")],
    )

    # when
    introspection = build_workflow_introspection(compilation_result=result)

    # then
    dims = {
        step.node_id: (step.input_dimensionality, step.output_dimensionality)
        for step in introspection.steps
    }
    assert dims == {
        "$steps.model": (1, 1),
        "$steps.crop": (1, 2),
        "$steps.model_on_crops": (2, 2),
        "$steps.nested_crop": (2, 3),
        "$steps.collapse": (2, 1),
    }
    collapse_node = _step_node(result, "$steps.collapse")
    assert collapse_node.step_execution_dimensionality == 1
    assert collapse_node.reference_dimensionality == 2, "reference != executor depth"
    _assert_reference_depth_consistent(result, introspection)
    assert introspection.summary.steps_by_dimensionality == {1: 2, 2: 3}
    assert introspection.summary.max_dimensionality == 3


def test_mixed_depth_inputs_with_explicit_reference_property() -> None:
    # given: detections at depth 1, classifications at depth 2
    result = _compile(
        inputs=[_image_input()],
        steps=[
            _model("model"),
            CropManifest(
                type="test/crop@v1",
                name="crop",
                images="$inputs.image",
                predictions="$steps.model.predictions",
            ),
            ClassifierManifest(
                type="test/classifier@v1", name="classifier", images="$steps.crop.crops"
            ),
            ReplacementManifest(
                type="test/replacement@v1",
                name="replacement",
                object_detection_predictions="$steps.model.predictions",
                classification_predictions="$steps.classifier.predictions",
            ),
        ],
        outputs=[_output("predictions", "$steps.replacement.predictions")],
    )

    # when
    introspection = build_workflow_introspection(compilation_result=result)

    # then
    replacement = _step(introspection, "$steps.replacement")
    assert (replacement.input_dimensionality, replacement.output_dimensionality) == (
        1,
        1,
    )
    classifier = _step(introspection, "$steps.classifier")
    assert (classifier.input_dimensionality, classifier.output_dimensionality) == (
        2,
        2,
    )
    _assert_reference_depth_consistent(result, introspection)
    assert introspection.summary.steps_by_dimensionality == {1: 3, 2: 1}
    assert introspection.summary.max_dimensionality == 2


def test_scalar_step_with_auto_batch_casted_input_reports_reference_depth_zero() -> (
    None
):
    # given: a scalar plugged into a batch-only property is auto-batch-casted,
    # so the executor runs the step at depth 1 - but the compiled reference
    # depth (and output depth) stay 0.
    result = _compile(
        inputs=[_image_input()],
        steps=[
            ScalarSourceManifest(type="test/scalar_source@v1", name="scalar"),
            BatchOnlyManifest(
                type="test/batch_only@v1", name="casted", data="$steps.scalar.output"
            ),
        ],
        outputs=[_output("out", "$steps.casted.output")],
    )

    # when
    introspection = build_workflow_introspection(compilation_result=result)

    # then
    casted_node = _step_node(result, "$steps.casted")
    assert casted_node.step_execution_dimensionality == 1
    casted = _step(introspection, "$steps.casted")
    assert (casted.input_dimensionality, casted.output_dimensionality) == (0, 0)
    scalar = _step(introspection, "$steps.scalar")
    assert (scalar.input_dimensionality, scalar.output_dimensionality) == (0, 0)
    _assert_reference_depth_consistent(result, introspection)
    assert introspection.summary.steps_by_dimensionality == {0: 2}
    assert introspection.summary.max_dimensionality == 1, "input node depth counts"


def test_control_only_scalar_step_takes_control_lineage_depth() -> None:
    # given: an input-less step gated by a condition evaluated on depth-1 data
    result = _compile(
        inputs=[_image_input()],
        steps=[
            _model("model"),
            ConditionManifest(
                type="test/continue_if@v1",
                name="gate",
                evaluation_parameters={"predictions": "$steps.model.predictions"},
                next_steps=["$steps.gated"],
            ),
            ScalarSourceManifest(type="test/scalar_source@v1", name="gated"),
            ScalarSourceManifest(type="test/scalar_source@v1", name="free"),
        ],
        outputs=[_output("out", "$steps.gated.output")],
    )

    # when
    introspection = build_workflow_introspection(compilation_result=result)

    # then
    gated = _step(introspection, "$steps.gated")
    assert (gated.input_dimensionality, gated.output_dimensionality) == (1, 1)
    free = _step(introspection, "$steps.free")
    assert (free.input_dimensionality, free.output_dimensionality) == (0, 0)
    gate = _step(introspection, "$steps.gate")
    assert (gate.input_dimensionality, gate.output_dimensionality) == (1, 1)
    _assert_reference_depth_consistent(result, introspection)
    assert introspection.summary.steps_by_dimensionality == {0: 1, 1: 3}


def test_no_step_input_to_output_workflow() -> None:
    result = _compile(
        inputs=[_image_input()],
        steps=[],
        outputs=[_output("echo", "$inputs.image")],
    )
    introspection = build_workflow_introspection(compilation_result=result)
    assert introspection.steps == []
    assert [(e.source, e.target, e.kind) for e in introspection.edges] == [
        ("$inputs.image", "$outputs.echo", "data")
    ]
    assert introspection.summary.steps_by_dimensionality == {}
    assert introspection.summary.max_dimensionality == 1
    assert introspection.summary.models.complete is True
    assert introspection.summary.models.items == []


def test_batch_input_with_declared_dimensionality_two() -> None:
    result = _compile(
        inputs=[
            WorkflowBatchInput(
                type="WorkflowBatchInput",
                name="crops",
                kind=[IMAGE_KIND],
                dimensionality=2,
            )
        ],
        steps=[_model("model", images="$inputs.crops")],
        outputs=[_output("predictions", "$steps.model.predictions")],
    )
    introspection = build_workflow_introspection(compilation_result=result)
    model = _step(introspection, "$steps.model")
    assert (model.input_dimensionality, model.output_dimensionality) == (2, 2)
    assert introspection.summary.max_dimensionality == 2


# ---------------------------------------------------------------------------
# declarations
# ---------------------------------------------------------------------------


def test_unannotated_block_yields_incomplete_declarations() -> None:
    result = _compile(
        inputs=[_image_input()],
        steps=[
            UnknownManifest(type="test/unknown@v1", name="x", images="$inputs.image")
        ],
    )
    step = _step(build_workflow_introspection(compilation_result=result), "$steps.x")
    assert step.resources == Discovery[DependentResource](
        items=[], complete=False, unknown_reasons=["step_resources_unknown:$steps.x"]
    )
    assert step.operations.model_dump() == {
        "type": "discovery",
        "items": [],
        "complete": False,
        "unknown_reasons": ["step_operations_unknown:$steps.x"],
    }
    assert step.restrictions.model_dump() == {
        "type": "discovery",
        "items": [],
        "complete": False,
        "unknown_reasons": ["step_restrictions_unknown:$steps.x"],
    }


def test_empty_declarations_are_complete_known_absence() -> None:
    result = _compile(
        inputs=[_image_input()],
        steps=[NoopManifest(type="test/noop@v1", name="x", images="$inputs.image")],
    )
    step = _step(build_workflow_introspection(compilation_result=result), "$steps.x")
    for discovery in (step.resources, step.operations, step.restrictions):
        assert discovery.complete is True
        assert discovery.items == []
        assert discovery.unknown_reasons == []


def test_declared_items_are_kept_verbatim_including_selectors_and_access() -> None:
    result = _compile(
        inputs=[
            _image_input(),
            WorkflowParameter(type="WorkflowParameter", name="model"),
        ],
        steps=[
            _model("model", model_id="$inputs.model"),
            ProjectSinkManifest(
                type="test/project_sink@v1", name="sink", images="$inputs.image"
            ),
        ],
    )
    introspection = build_workflow_introspection(compilation_result=result)
    model = _step(introspection, "$steps.model")
    assert model.resources.complete is True
    assert model.resources.items == [roboflow_platform_model(model_id="$inputs.model")]
    assert model.operations.items == [WorkOperation.MODEL_INFERENCE]
    assert model.restrictions.items == [GPU_RESTRICTION]
    sink = _step(introspection, "$steps.sink")
    assert sink.resources.complete is True
    assert set(sink.resources.items) == {
        roboflow_platform_project(project_url="workspace/project"),
        roboflow_platform_model(
            model_id="monitored/1", required_action=ModelRequiredAction.ACCESS
        ),
    }
    access_entry = [
        item
        for item in sink.resources.items
        if item.resource_type.value == "roboflow_platform_model"
    ][0]
    assert access_entry.metadata.required_action is ModelRequiredAction.ACCESS
    assert sink.operations.items == [
        WorkOperation.EXTERNAL_REQUEST,
        WorkOperation.STORAGE_WRITE,
    ], "sorted by enum value"


def test_explicit_discovery_declarations_pass_through() -> None:
    result = _compile(
        inputs=[_image_input()],
        steps=[
            ExplicitDiscoveryManifest(
                type="test/explicit@v1", name="custom", images="$inputs.image"
            )
        ],
    )
    step = _step(
        build_workflow_introspection(compilation_result=result), "$steps.custom"
    )
    assert step.resources.complete is False
    assert step.resources.items == [roboflow_platform_model(model_id="explicit/1")]
    assert step.resources.unknown_reasons == ["explicit_partial:$steps.custom"]
    assert step.operations.items == [WorkOperation.CUSTOM_PYTHON]
    assert step.operations.unknown_reasons == [
        "custom_python_internal_operations_unknown:$steps.custom"
    ]
    assert step.restrictions.complete is True


def test_failing_hooks_become_incomplete_declarations_not_errors() -> None:
    result = _compile(
        inputs=[_image_input()],
        steps=[
            FailingHooksManifest(
                type="test/failing@v1", name="x", images="$inputs.image"
            )
        ],
    )
    step = _step(build_workflow_introspection(compilation_result=result), "$steps.x")
    assert step.resources.unknown_reasons == [
        "discover_dependent_resources_failed:$steps.x"
    ]
    assert step.operations.unknown_reasons == [
        "discover_work_operations_failed:$steps.x"
    ]
    assert step.restrictions.unknown_reasons == [
        "discover_portable_restrictions_failed:$steps.x"
    ]
    for discovery in (step.resources, step.operations, step.restrictions):
        assert discovery.complete is False
        assert discovery.items == []


def test_garbage_hook_answers_become_incomplete_declarations() -> None:
    result = _compile(
        inputs=[_image_input()],
        steps=[
            GarbageHooksManifest(
                type="test/garbage@v1", name="x", images="$inputs.image"
            )
        ],
    )
    step = _step(build_workflow_introspection(compilation_result=result), "$steps.x")
    assert step.resources.unknown_reasons == [
        "discover_dependent_resources_failed:$steps.x"
    ]
    assert step.operations.unknown_reasons == [
        "discover_work_operations_failed:$steps.x"
    ]
    assert step.restrictions.unknown_reasons == [
        "discover_portable_restrictions_failed:$steps.x"
    ]


# ---------------------------------------------------------------------------
# model inventory
# ---------------------------------------------------------------------------


def test_five_independent_model_branches_yield_five_entries() -> None:
    result = _compile(
        inputs=[_image_input()],
        steps=[_model(f"model_{i}", model_id=f"project/{i}") for i in range(5)],
    )
    models = build_workflow_introspection(compilation_result=result).summary.models
    assert models.complete is True
    assert {(m.provider, m.model_id, tuple(m.used_by_steps)) for m in models.items} == {
        ("roboflow", f"project/{i}", (f"$steps.model_{i}",)) for i in range(5)
    }
    assert all(m.metadata_status == "unavailable" for m in models.items)
    assert all(m.metadata is None for m in models.items)


def test_shared_model_id_across_two_steps_is_one_entry() -> None:
    result = _compile(
        inputs=[_image_input()],
        steps=[
            _model("b", model_id="shared/1"),
            _model("a", model_id="shared/1"),
            ThirdPartyManifest(
                type="test/third_party@v1", name="llm", images="$inputs.image"
            ),
        ],
    )
    models = build_workflow_introspection(compilation_result=result).summary.models
    assert models.complete is True
    assert [(m.provider, m.model_id, m.used_by_steps) for m in models.items] == [
        ("openai", "gpt-x", ["$steps.llm"]),
        ("roboflow", "shared/1", ["$steps.a", "$steps.b"]),
    ]


def test_shared_ancestor_model_is_counted_once() -> None:
    result = _compile(
        inputs=[_image_input()],
        steps=[
            _model("model", model_id="project/1"),
            CropManifest(
                type="test/crop@v1",
                name="crop_a",
                images="$inputs.image",
                predictions="$steps.model.predictions",
            ),
            CropManifest(
                type="test/crop@v1",
                name="crop_b",
                images="$inputs.image",
                predictions="$steps.model.predictions",
            ),
        ],
    )
    models = build_workflow_introspection(compilation_result=result).summary.models
    assert [(m.model_id, m.used_by_steps) for m in models.items] == [
        ("project/1", ["$steps.model"])
    ]


def test_projects_and_access_only_models_are_not_fabricated_but_access_is_inventory() -> (
    None
):
    result = _compile(
        inputs=[_image_input()],
        steps=[
            ProjectSinkManifest(
                type="test/project_sink@v1", name="sink", images="$inputs.image"
            )
        ],
    )
    models = build_workflow_introspection(compilation_result=result).summary.models
    # the project is not a model; the ACCESS model reference IS an inventory
    # entry (inventory != execution claim; per-step resources keep the action)
    assert [(m.provider, m.model_id) for m in models.items] == [
        ("roboflow", "monitored/1")
    ]
    assert models.complete is True


def test_selector_model_ids_stay_per_step_and_make_inventory_incomplete() -> None:
    result = _compile(
        inputs=[
            _image_input(),
            WorkflowParameter(type="WorkflowParameter", name="model"),
        ],
        steps=[
            _model("dynamic", model_id="$inputs.model"),
            _model("static", model_id="project/1"),
        ],
    )
    introspection = build_workflow_introspection(compilation_result=result)
    models = introspection.summary.models
    assert models.complete is False
    assert models.unknown_reasons == ["unresolved_model_selector:$steps.dynamic"]
    assert [(m.model_id, m.used_by_steps) for m in models.items] == [
        ("project/1", ["$steps.static"])
    ]
    dynamic = _step(introspection, "$steps.dynamic")
    assert dynamic.resources.items == [
        roboflow_platform_model(model_id="$inputs.model")
    ]


def test_unknown_step_resources_propagate_to_inventory_reasons() -> None:
    result = _compile(
        inputs=[_image_input()],
        steps=[
            _model("model", model_id="project/1"),
            UnknownManifest(
                type="test/unknown@v1", name="mystery", images="$inputs.image"
            ),
            ExplicitDiscoveryManifest(
                type="test/explicit@v1", name="partial", images="$inputs.image"
            ),
        ],
    )
    models = build_workflow_introspection(compilation_result=result).summary.models
    assert models.complete is False
    assert models.unknown_reasons == [
        "explicit_partial:$steps.partial",
        "step_resources_unknown:$steps.mystery",
    ]
    assert [(m.model_id, m.used_by_steps) for m in models.items] == [
        ("explicit/1", ["$steps.partial"]),
        ("project/1", ["$steps.model"]),
    ]


class RecordingProvider:
    def __init__(self, answers: Dict[Any, Any]) -> None:
        self.answers = answers
        self.calls: List[Any] = []

    def resolve_model_metadata(self, provider: str, model_id: str):
        self.calls.append((provider, model_id))
        answer = self.answers.get((provider, model_id))
        if isinstance(answer, Exception):
            raise answer
        if answer is None:
            return ModelMetadataLookup(status="unavailable")
        return answer


def test_recording_provider_satisfies_protocol() -> None:
    assert isinstance(RecordingProvider(answers={}), ModelMetadataProvider)


def test_metadata_provider_called_once_per_unique_literal_reference() -> None:
    # given
    result = _compile(
        inputs=[
            _image_input(),
            WorkflowParameter(type="WorkflowParameter", name="model"),
        ],
        steps=[
            _model("a", model_id="shared/1"),
            _model("b", model_id="shared/1"),
            _model("selector", model_id="$inputs.model"),
            _model("disabled", model_id="disabled/1"),
            _model("partial", model_id="partial/1"),
            _model("broken", model_id="broken/1"),
            ThirdPartyManifest(
                type="test/third_party@v1", name="llm", images="$inputs.image"
            ),
        ],
    )
    provider = RecordingProvider(
        answers={
            ("roboflow", "shared/1"): ModelMetadataLookup(
                status="available",
                metadata=ModelMetadata(
                    model_type="yolov8n", model_variant="640", task_type="od"
                ),
            ),
            ("roboflow", "disabled/1"): ModelMetadataLookup(status="disabled"),
            ("roboflow", "partial/1"): ModelMetadataLookup(
                status="available", metadata=ModelMetadata(task_type="od")
            ),
            ("roboflow", "broken/1"): RuntimeError("lookup failed"),
        }
    )

    # when
    introspection = build_workflow_introspection(
        compilation_result=result, model_metadata_provider=provider
    )

    # then
    assert sorted(provider.calls) == [
        ("openai", "gpt-x"),
        ("roboflow", "broken/1"),
        ("roboflow", "disabled/1"),
        ("roboflow", "partial/1"),
        ("roboflow", "shared/1"),
    ]
    assert len(provider.calls) == 5, "one call per unique pair, never a selector"
    by_id = {m.model_id: m for m in introspection.summary.models.items}
    assert by_id["shared/1"].metadata_status == "available"
    assert by_id["shared/1"].metadata.model_type == "yolov8n"
    assert by_id["shared/1"].used_by_steps == ["$steps.a", "$steps.b"]
    assert by_id["disabled/1"].metadata_status == "disabled"
    assert by_id["disabled/1"].metadata is None
    assert by_id["partial/1"].metadata_status == "available"
    assert by_id["partial/1"].metadata == ModelMetadata(task_type="od")
    assert by_id["broken/1"].metadata_status == "unavailable"
    assert by_id["gpt-x"].metadata_status == "unavailable"
    assert introspection.summary.models.complete is False, "selector, not lookups"
    assert introspection.summary.models.unknown_reasons == [
        "unresolved_model_selector:$steps.selector"
    ]


def test_lookup_outcomes_never_change_inventory_completeness() -> None:
    result = _compile(
        inputs=[_image_input()],
        steps=[_model("broken", model_id="broken/1")],
    )
    provider = RecordingProvider(
        answers={("roboflow", "broken/1"): RuntimeError("boom")}
    )
    models = build_workflow_introspection(
        compilation_result=result, model_metadata_provider=provider
    ).summary.models
    assert models.complete is True
    assert models.items[0].metadata_status == "unavailable"


# ---------------------------------------------------------------------------
# remote dispatch opacity
# ---------------------------------------------------------------------------


def test_remote_dispatch_inner_workflow_is_opaque() -> None:
    # given
    result = _compile(
        inputs=[_image_input()],
        steps=[
            _model("model", model_id="project/1"),
            InnerWorkflowManifest(
                type="roboflow_core/inner_workflow@v1",
                name="dispatch",
                execution_mode="remote_dispatch",
                workflow_definition={
                    "version": "1.0",
                    "inputs": [{"type": "WorkflowImage", "name": "image"}],
                    "steps": [
                        {
                            "type": "not_installed/block@v1",
                            "name": "child",
                            "image": "$inputs.image",
                        }
                    ],
                    "outputs": [],
                },
                parameter_bindings={"image": "$inputs.image"},
            ),
        ],
    )

    # when
    introspection = build_workflow_introspection(compilation_result=result)

    # then
    dispatch = _step(introspection, "$steps.dispatch")
    assert dispatch.block_type == "roboflow_core/inner_workflow@v1"
    for discovery in (dispatch.resources, dispatch.operations, dispatch.restrictions):
        assert discovery.complete is False
        assert (
            "remote_dispatch_child_opaque:$steps.dispatch" in discovery.unknown_reasons
        )
    models = introspection.summary.models
    assert models.complete is False
    assert "remote_dispatch_child_opaque:$steps.dispatch" in models.unknown_reasons
    assert [(m.model_id, m.used_by_steps) for m in models.items] == [
        ("project/1", ["$steps.model"])
    ]


# ---------------------------------------------------------------------------
# response properties
# ---------------------------------------------------------------------------


def test_response_carries_engine_version_and_roundtrips_json() -> None:
    result = _compile(
        inputs=[_image_input()],
        steps=[
            _model("model"),
            CropManifest(
                type="test/crop@v1",
                name="crop",
                images="$inputs.image",
                predictions="$steps.model.predictions",
            ),
        ],
        outputs=[_output("crops", "$steps.crop.crops")],
    )
    introspection = build_workflow_introspection(compilation_result=result)
    assert introspection.execution_engine_version == str(EXECUTION_ENGINE_V1_VERSION)
    roundtrip = WorkflowIntrospection.model_validate_json(
        introspection.model_dump_json()
    )
    assert roundtrip == introspection
    dumped = introspection.model_dump(mode="json")
    assert dumped["type"] == "workflow_introspection"
    assert dumped["summary"]["steps_by_dimensionality"] == {"1": 2}
    assert all(step["type"] == "step_metadata" for step in dumped["steps"])


def test_output_is_stable_across_runs_with_different_lineage_ids() -> None:
    def build() -> WorkflowIntrospection:
        result = _compile(
            inputs=[
                WorkflowBatchInput(
                    type="WorkflowBatchInput",
                    name="crops",
                    kind=[IMAGE_KIND],
                    dimensionality=2,
                )
            ],
            steps=[
                _model("model", images="$inputs.crops"),
                CropManifest(
                    type="test/crop@v1",
                    name="crop",
                    images="$inputs.crops",
                    predictions="$steps.model.predictions",
                ),
            ],
            outputs=[_output("crops", "$steps.crop.crops")],
        )
        return build_workflow_introspection(compilation_result=result)

    first, second = build(), build()
    assert first.model_dump() == second.model_dump()
    assert first is not second


def test_builder_returns_fresh_objects_per_call() -> None:
    result = _compile(inputs=[_image_input()], steps=[_model("model")])
    first = build_workflow_introspection(compilation_result=result)
    second = build_workflow_introspection(compilation_result=result)
    assert first == second
    assert first is not second
    assert first.summary is not second.summary


# ---------------------------------------------------------------------------
# blank literal identifiers (deep-review F1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blank_model_id", ["", "   "])
def test_blank_platform_model_id_is_declared_but_never_inventoried(
    blank_model_id: str,
) -> None:
    # given
    result = _compile(
        inputs=[_image_input()],
        steps=[
            _model("blank", model_id=blank_model_id),
            _model("valid", model_id="project/1"),
        ],
    )
    provider = RecordingProvider(answers={})

    # when
    introspection = build_workflow_introspection(
        compilation_result=result, model_metadata_provider=provider
    )

    # then - kept per step as declared, excluded from the inventory, no lookup
    blank = _step(introspection, "$steps.blank")
    assert blank.resources.complete is True
    assert blank.resources.items[0].metadata.model_id == blank_model_id
    models = introspection.summary.models
    assert models.complete is False
    assert models.unknown_reasons == ["blank_model_identifier:$steps.blank"]
    assert [(m.provider, m.model_id, m.used_by_steps) for m in models.items] == [
        ("roboflow", "project/1", ["$steps.valid"])
    ]
    assert provider.calls == [("roboflow", "project/1")]


@pytest.mark.parametrize(
    "provider_value, model_value",
    [("", "gpt-x"), ("  ", "gpt-x"), ("openai", ""), ("openai", " ")],
)
def test_blank_third_party_provider_or_model_is_never_inventoried(
    provider_value: str, model_value: str
) -> None:
    result = _compile(
        inputs=[_image_input()],
        steps=[
            ThirdPartyManifest(
                type="test/third_party@v1",
                name="llm",
                images="$inputs.image",
                provider=provider_value,
                model=model_value,
            )
        ],
    )
    provider = RecordingProvider(answers={})
    introspection = build_workflow_introspection(
        compilation_result=result, model_metadata_provider=provider
    )
    models = introspection.summary.models
    assert models.items == []
    assert models.complete is False
    assert models.unknown_reasons == ["blank_model_identifier:$steps.llm"]
    assert provider.calls == []
    llm = _step(introspection, "$steps.llm")
    assert llm.resources.items[0].metadata.provider == provider_value
    assert llm.resources.items[0].metadata.model_id == model_value
