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
    DiscoveryProblem,
    DiscoveryProblemCode,
    ModelMetadata,
    ModelMetadataLookup,
    ModelMetadataProvider,
    RestrictionCondition,
    RestrictionMetadata,
    Runtime,
    RuntimeRestriction,
    Severity,
    StepExecutionMode,
    WorkOperation,
    complete_discovery,
    custom_python_internals_unknown_problem,
    declaration_failed_problem,
    declaration_unavailable_problem,
    environment_filtered_declaration_problem,
    incomplete_discovery,
    invalid_resource_identifier_problem,
    opaque_remote_workflow_problem,
    unresolved_selector_problem,
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
    actual_restrictions_of,
    roboflow_platform_model,
    roboflow_platform_project,
    third_party_model,
)

# ---------------------------------------------------------------------------
# test-local blocks
# ---------------------------------------------------------------------------

# What a block AUTHORS ...
GPU_RESTRICTION = RuntimeRestriction(
    code="requires_gpu_for_local_execution",
    severity=Severity.HARD,
    note="Requires a GPU; local execution loads a model that needs CUDA.",
    applies_to_runtimes=[Runtime.SELF_HOSTED_CPU],
    applies_to_step_execution_modes=[StepExecutionMode.LOCAL],
)

# ... and what the workload document carries for it.
GPU_RESTRICTION_METADATA = RestrictionMetadata(
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

    def get_actual_restrictions(self, *, ignore_environment_restrictions: bool = False):
        return actual_restrictions_of(
            declared=[GPU_RESTRICTION],
            node_id=f"$steps.{self.name}",
            ignore_environment_restrictions=ignore_environment_restrictions,
        )


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

    def get_actual_restrictions(self, *, ignore_environment_restrictions: bool = False):
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


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

    def get_actual_restrictions(self, *, ignore_environment_restrictions: bool = False):
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


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

    def get_actual_restrictions(self, *, ignore_environment_restrictions: bool = False):
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


class DuplicateModelManifest(WorkflowBlockManifest):
    """Declares the SAME model twice (execution + access) - one step, one
    model, two resource entries."""

    model_config = ConfigDict(protected_namespaces=())
    type: Literal["test/duplicate_model@v1"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector]
    model_id: str = "dup/1"

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
        return [
            roboflow_platform_model(model_id=self.model_id),
            roboflow_platform_model(
                model_id=self.model_id, required_action=ModelRequiredAction.ACCESS
            ),
        ]

    def discover_work_operations(self):
        return [WorkOperation.MODEL_INFERENCE]

    def get_actual_restrictions(self, *, ignore_environment_restrictions: bool = False):
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


class TwoModelsManifest(WorkflowBlockManifest):
    """One step referencing two DIFFERENT models."""

    model_config = ConfigDict(protected_namespaces=())
    type: Literal["test/two_models@v1"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector]
    primary_model_id: str = "primary/1"
    secondary_model_id: str = "secondary/1"

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
        return [
            roboflow_platform_model(model_id=self.primary_model_id),
            roboflow_platform_model(model_id=self.secondary_model_id),
        ]

    def discover_work_operations(self):
        return [WorkOperation.MODEL_INFERENCE]

    def get_actual_restrictions(self, *, ignore_environment_restrictions: bool = False):
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


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

    def get_actual_restrictions(self, *, ignore_environment_restrictions: bool = False):
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


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

    def get_actual_restrictions(self, *, ignore_environment_restrictions: bool = False):
        raise RuntimeError("restrictions hook exploded")


SECRET_IN_HOOK_EXCEPTION = "sk-live-super-secret-token"


class SecretLeakingHooksManifest(WorkflowBlockManifest):
    """Every hook raises with a secret in the message: nothing the block put
    into the exception may reach the response."""

    type: Literal["test/secret_leak@v1"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="image", kind=[IMAGE_KIND])]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        raise RuntimeError(f"cannot reach registry with {SECRET_IN_HOOK_EXCEPTION}")

    def discover_work_operations(self):
        raise RuntimeError(f"Authorization: Bearer {SECRET_IN_HOOK_EXCEPTION}")

    def get_actual_restrictions(self, *, ignore_environment_restrictions: bool = False):
        raise RuntimeError(f"db://user:{SECRET_IN_HOOK_EXCEPTION}@host/db")


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

    def get_actual_restrictions(self, *, ignore_environment_restrictions: bool = False):
        return [{"code": 1}]


class PartiallyKnownModelsManifest(WorkflowBlockManifest):
    """A block's own Discovery: a prior problem, a literal model and a
    selector-fed model in one declaration."""

    type: Literal["test/partially_known@v1"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="image", kind=[IMAGE_KIND])]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    def discover_dependent_resources(self):
        return incomplete_discovery(
            [
                roboflow_platform_model(model_id="known/1"),
                roboflow_platform_model(model_id="$inputs.model"),
            ],
            [
                declaration_unavailable_problem(
                    node_id=f"$steps.{self.name}", declaration="resources"
                )
            ],
        )


RESOLVER_CALLS: List[str] = []


def _recording_resolver(value: str) -> str:
    RESOLVER_CALLS.append(value)
    return f"resolved/{value}"


class ResolverModelManifest(WorkflowBlockManifest):
    """A legacy list with a selector-fed model carrying a resolver aid."""

    type: Literal["test/resolver_model@v1"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="image", kind=[IMAGE_KIND])]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        return [
            roboflow_platform_model(
                model_id="$inputs.model", model_id_resolver=_recording_resolver
            )
        ]


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
            [
                DiscoveryProblem(
                    code=DiscoveryProblemCode.UNRESOLVED_SELECTOR,
                    description="A second model is chosen by a runtime value.",
                    details={
                        "node_id": f"$steps.{self.name}",
                        "declaration": "resources",
                        "field": "secondary_model",
                        "selector": "$inputs.secondary",
                    },
                )
            ],
        )

    def discover_work_operations(self):
        return incomplete_discovery(
            [WorkOperation.CUSTOM_PYTHON],
            [
                custom_python_internals_unknown_problem(
                    node_id=f"$steps.{self.name}", declaration="operations"
                )
            ],
        )

    def get_actual_restrictions(self, *, ignore_environment_restrictions: bool = False):
        return actual_restrictions_of(
            declared=complete_discovery([]),
            node_id=f"$steps.{self.name}",
            ignore_environment_restrictions=ignore_environment_restrictions,
        )


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

    def get_actual_restrictions(self, *, ignore_environment_restrictions: bool = False):
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


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

    def get_actual_restrictions(self, *, ignore_environment_restrictions: bool = False):
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


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

    def get_actual_restrictions(self, *, ignore_environment_restrictions: bool = False):
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


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

    def get_actual_restrictions(self, *, ignore_environment_restrictions: bool = False):
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


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

    def get_actual_restrictions(self, *, ignore_environment_restrictions: bool = False):
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


class BatchOnlyModelManifest(WorkflowBlockManifest):
    """Batch-only step that declares a model: with a scalar plugged in it is
    auto-batch-casted, so its compiled reference depth is 0."""

    model_config = ConfigDict(protected_namespaces=())
    type: Literal["test/batch_only_model@v1"]
    data: StepOutputSelector()
    model_id: str = "project/1"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            )
        ]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["data"]

    def discover_dependent_resources(self):
        return [roboflow_platform_model(model_id=self.model_id)]

    def discover_work_operations(self):
        return [WorkOperation.MODEL_INFERENCE]

    def get_actual_restrictions(self, *, ignore_environment_restrictions: bool = False):
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


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

    def get_actual_restrictions(self, *, ignore_environment_restrictions: bool = False):
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


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

    def get_actual_restrictions(self, *, ignore_environment_restrictions: bool = False):
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


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

    def get_actual_restrictions(self, *, ignore_environment_restrictions: bool = False):
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


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
        DuplicateModelManifest,
        TwoModelsManifest,
        UnknownManifest,
        NoopManifest,
        FailingHooksManifest,
        SecretLeakingHooksManifest,
        GarbageHooksManifest,
        ExplicitDiscoveryManifest,
        PartiallyKnownModelsManifest,
        ResolverModelManifest,
        CropManifest,
        CollapseManifest,
        ReplacementManifest,
        ScalarSourceManifest,
        BatchOnlyManifest,
        BatchOnlyModelManifest,
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


def _assert_model_histograms_consistent(introspection: WorkflowIntrospection) -> None:
    """Every inventory entry's histogram is exactly the input depths of its
    `used_by_steps`, one count per step, and sums to their number."""
    depth_of = {step.node_id: step.input_dimensionality for step in introspection.steps}
    for model in introspection.summary.models.items:
        expected: Dict[int, int] = {}
        for step_id in model.used_by_steps:
            expected[depth_of[step_id]] = expected.get(depth_of[step_id], 0) + 1
        assert model.steps_by_dimensionality == dict(sorted(expected.items())), (
            model.provider,
            model.model_id,
        )
        assert sum(model.steps_by_dimensionality.values()) == len(model.used_by_steps)
        assert list(model.steps_by_dimensionality) == sorted(
            model.steps_by_dimensionality
        )


def _model_histograms(introspection: WorkflowIntrospection) -> Dict[Any, Dict]:
    return {
        (model.provider, model.model_id): model.steps_by_dimensionality
        for model in introspection.summary.models.items
    }


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
    # `model` (depth 1) and `model_on_crops` (depth 2) share `project/1`
    assert _model_histograms(introspection) == {("roboflow", "project/1"): {1: 1, 2: 1}}
    _assert_model_histograms_consistent(introspection)


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
    assert _model_histograms(introspection) == {
        ("roboflow", "project/1"): {1: 1},
        ("roboflow", "classifier/1"): {2: 1},
    }
    _assert_model_histograms_consistent(introspection)


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
    assert _model_histograms(introspection) == {("roboflow", "project/1"): {1: 1}}


def test_model_referenced_by_a_scalar_step_reports_depth_zero() -> None:
    # given: a model on a depth-2 batch input plus a model-declaring step fed
    # by a scalar-producing step (auto-batch-casted, so executor depth 1 but
    # compiled reference depth 0) - the per-model histogram must use the same
    # reference depth as `StepMetadata.input_dimensionality`
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
            _model("deep", images="$inputs.crops", model_id="project/1"),
            ScalarSourceManifest(type="test/scalar_source@v1", name="scalar"),
            BatchOnlyModelManifest(
                type="test/batch_only_model@v1",
                name="on_scalar",
                data="$steps.scalar.output",
                model_id="project/1",
            ),
        ],
    )

    # when
    introspection = build_workflow_introspection(compilation_result=result)

    # then
    assert _step_node(result, "$steps.on_scalar").step_execution_dimensionality == 1
    assert _step(introspection, "$steps.on_scalar").input_dimensionality == 0
    assert _step(introspection, "$steps.deep").input_dimensionality == 2
    assert _model_histograms(introspection) == {("roboflow", "project/1"): {0: 1, 2: 1}}
    assert introspection.summary.models.items[0].used_by_steps == [
        "$steps.deep",
        "$steps.on_scalar",
    ]
    _assert_model_histograms_consistent(introspection)


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
        items=[],
        complete=False,
        unknown_reasons=[
            declaration_unavailable_problem(
                node_id="$steps.x",
                declaration="resources",
                block_type="test/unknown@v1",
            )
        ],
    )
    assert step.operations.model_dump(mode="json") == {
        "type": "discovery_v1",
        "items": [],
        "complete": False,
        "unknown_reasons": [
            {
                "type": "discovery_problem_v1",
                "code": "declaration_unavailable",
                "description": (
                    "Step `$steps.x` does not declare its operations, so they "
                    "are unknown rather than absent."
                ),
                "details": {
                    "node_id": "$steps.x",
                    "declaration": "operations",
                    "block_type": "test/unknown@v1",
                },
            }
        ],
    }
    # restrictions fall back to the legacy `get_restrictions()` - here the
    # inherited one - so the reason names that source
    assert step.restrictions.unknown_reasons == [
        environment_filtered_declaration_problem(
            node_id="$steps.x",
            declaration="restrictions",
            block_type="test/unknown@v1",
        )
    ]


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
    # the item is kept verbatim, but its identity is only known at run time
    assert model.resources.complete is False
    assert model.resources.unknown_reasons == [
        unresolved_selector_problem(
            node_id="$steps.model",
            declaration="resources",
            field="model_id",
            selector="$inputs.model",
            resource_type="roboflow_platform_model",
        )
    ]
    assert model.resources.items == [roboflow_platform_model(model_id="$inputs.model")]
    assert model.operations.items == [WorkOperation.MODEL_INFERENCE]
    assert model.restrictions.items == [GPU_RESTRICTION_METADATA]
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
    # the block's own problem object survives the builder untouched
    assert step.resources.unknown_reasons[0].code is (
        DiscoveryProblemCode.UNRESOLVED_SELECTOR
    )
    assert step.resources.unknown_reasons[0].details == {
        "node_id": "$steps.custom",
        "declaration": "resources",
        "field": "secondary_model",
        "selector": "$inputs.secondary",
    }
    assert step.operations.items == [WorkOperation.CUSTOM_PYTHON]
    assert step.operations.unknown_reasons == [
        custom_python_internals_unknown_problem(
            node_id="$steps.custom", declaration="operations"
        )
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
        declaration_failed_problem(
            node_id="$steps.x", declaration="resources", block_type="test/failing@v1"
        )
    ]
    assert step.operations.unknown_reasons == [
        declaration_failed_problem(
            node_id="$steps.x", declaration="operations", block_type="test/failing@v1"
        )
    ]
    assert step.restrictions.unknown_reasons == [
        declaration_failed_problem(
            node_id="$steps.x",
            declaration="restrictions",
            block_type="test/failing@v1",
        )
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
    for declaration, discovery in (
        ("resources", step.resources),
        ("operations", step.operations),
        ("restrictions", step.restrictions),
    ):
        assert discovery.unknown_reasons == [
            declaration_failed_problem(
                node_id="$steps.x",
                declaration=declaration,
                block_type="test/garbage@v1",
            )
        ]


def test_failing_hook_never_leaks_the_exception_text() -> None:
    result = _compile(
        inputs=[_image_input()],
        steps=[
            SecretLeakingHooksManifest(
                type="test/secret_leak@v1", name="x", images="$inputs.image"
            )
        ],
    )
    step = _step(build_workflow_introspection(compilation_result=result), "$steps.x")
    rendered = step.resources.model_dump_json() + step.operations.model_dump_json()
    rendered += step.restrictions.model_dump_json()

    assert SECRET_IN_HOOK_EXCEPTION not in rendered
    assert "Traceback" not in rendered
    for discovery in (step.resources, step.operations, step.restrictions):
        assert [reason.code for reason in discovery.unknown_reasons] == [
            DiscoveryProblemCode.DECLARATION_FAILED
        ]


# ---------------------------------------------------------------------------
# model inventory
# ---------------------------------------------------------------------------


def test_five_independent_model_branches_yield_five_entries() -> None:
    result = _compile(
        inputs=[_image_input()],
        steps=[_model(f"model_{i}", model_id=f"project/{i}") for i in range(5)],
    )
    introspection = build_workflow_introspection(compilation_result=result)
    models = introspection.summary.models
    assert models.complete is True
    assert {(m.provider, m.model_id, tuple(m.used_by_steps)) for m in models.items} == {
        ("roboflow", f"project/{i}", (f"$steps.model_{i}",)) for i in range(5)
    }
    assert all(m.steps_by_dimensionality == {1: 1} for m in models.items)
    assert all(m.metadata_status == "unavailable" for m in models.items)
    assert all(m.metadata is None for m in models.items)
    _assert_model_histograms_consistent(introspection)


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
    introspection = build_workflow_introspection(compilation_result=result)
    models = introspection.summary.models
    assert models.complete is True
    assert [(m.provider, m.model_id, m.used_by_steps) for m in models.items] == [
        ("openai", "gpt-x", ["$steps.llm"]),
        ("roboflow", "shared/1", ["$steps.a", "$steps.b"]),
    ]
    # two steps at the SAME depth accumulate into one bucket
    assert [m.steps_by_dimensionality for m in models.items] == [{1: 1}, {1: 2}]
    _assert_model_histograms_consistent(introspection)


def test_shared_model_across_mixed_depths_reports_one_count_per_depth() -> None:
    # given: `shared/1` at depth 1 (image), twice at depth 2 (crops)
    result = _compile(
        inputs=[_image_input()],
        steps=[
            _model("model", model_id="shared/1"),
            CropManifest(
                type="test/crop@v1",
                name="crop",
                images="$inputs.image",
                predictions="$steps.model.predictions",
            ),
            _model("first_on_crops", images="$steps.crop.crops", model_id="shared/1"),
            _model("second_on_crops", images="$steps.crop.crops", model_id="shared/1"),
        ],
    )

    # when
    introspection = build_workflow_introspection(compilation_result=result)

    # then
    (model,) = introspection.summary.models.items
    assert model.used_by_steps == [
        "$steps.first_on_crops",
        "$steps.model",
        "$steps.second_on_crops",
    ]
    assert model.steps_by_dimensionality == {1: 1, 2: 2}
    assert sum(model.steps_by_dimensionality.values()) == len(model.used_by_steps)
    assert introspection.summary.steps_by_dimensionality == {1: 2, 2: 2}
    _assert_model_histograms_consistent(introspection)


def test_duplicate_resources_for_one_model_in_one_step_count_that_step_once() -> None:
    # given: the step declares `dup/1` twice (execution + access)
    result = _compile(
        inputs=[_image_input()],
        steps=[
            DuplicateModelManifest(
                type="test/duplicate_model@v1", name="twice", images="$inputs.image"
            )
        ],
    )

    # when
    introspection = build_workflow_introspection(compilation_result=result)

    # then - both declarations are kept per step, the inventory counts one step
    step = _step(introspection, "$steps.twice")
    assert len(step.resources.items) == 2
    assert {item.metadata.required_action for item in step.resources.items} == {
        ModelRequiredAction.ACCESS,
        ModelRequiredAction.EXECUTION,
    }
    (model,) = introspection.summary.models.items
    assert (model.provider, model.model_id) == ("roboflow", "dup/1")
    assert model.used_by_steps == ["$steps.twice"]
    assert model.steps_by_dimensionality == {1: 1}
    assert introspection.summary.models.complete is True
    _assert_model_histograms_consistent(introspection)


def test_one_step_referencing_two_models_counts_once_for_each() -> None:
    # given: one step at depth 2 referencing two different models, plus a
    # depth-1 step sharing one of them
    result = _compile(
        inputs=[_image_input()],
        steps=[
            _model("model", model_id="primary/1"),
            CropManifest(
                type="test/crop@v1",
                name="crop",
                images="$inputs.image",
                predictions="$steps.model.predictions",
            ),
            TwoModelsManifest(
                type="test/two_models@v1", name="pair", images="$steps.crop.crops"
            ),
        ],
    )

    # when
    introspection = build_workflow_introspection(compilation_result=result)

    # then
    assert [
        (m.model_id, m.used_by_steps, m.steps_by_dimensionality)
        for m in introspection.summary.models.items
    ] == [
        ("primary/1", ["$steps.model", "$steps.pair"], {1: 1, 2: 1}),
        ("secondary/1", ["$steps.pair"], {2: 1}),
    ]
    _assert_model_histograms_consistent(introspection)


def test_same_model_id_under_different_providers_stays_separate() -> None:
    # given: two third-party steps with the same model id, different providers
    result = _compile(
        inputs=[_image_input()],
        steps=[
            ThirdPartyManifest(
                type="test/third_party@v1",
                name="first",
                images="$inputs.image",
                provider="openai",
                model="shared-id",
            ),
            ThirdPartyManifest(
                type="test/third_party@v1",
                name="second",
                images="$inputs.image",
                provider="anthropic",
                model="shared-id",
            ),
            ThirdPartyManifest(
                type="test/third_party@v1",
                name="third",
                images="$inputs.image",
                provider="anthropic",
                model="shared-id",
            ),
        ],
    )

    # when
    introspection = build_workflow_introspection(compilation_result=result)

    # then
    assert [
        (m.provider, m.model_id, m.used_by_steps, m.steps_by_dimensionality)
        for m in introspection.summary.models.items
    ] == [
        ("anthropic", "shared-id", ["$steps.second", "$steps.third"], {1: 2}),
        ("openai", "shared-id", ["$steps.first"], {1: 1}),
    ]
    _assert_model_histograms_consistent(introspection)


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
    assert [
        (m.model_id, m.used_by_steps, m.steps_by_dimensionality) for m in models.items
    ] == [("project/1", ["$steps.model"], {1: 1})]


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
    # and it is counted in the histogram like any other reference
    assert [(m.provider, m.model_id) for m in models.items] == [
        ("roboflow", "monitored/1")
    ]
    assert models.items[0].used_by_steps == ["$steps.sink"]
    assert models.items[0].steps_by_dimensionality == {1: 1}
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
    assert models.unknown_reasons == [
        unresolved_selector_problem(
            node_id="$steps.dynamic",
            declaration="resources",
            field="model_id",
            selector="$inputs.model",
            resource_type="roboflow_platform_model",
        )
    ]
    # only the known reference contributes; the unresolved one is neither an
    # entry nor a count anywhere
    assert [
        (m.model_id, m.used_by_steps, m.steps_by_dimensionality) for m in models.items
    ] == [("project/1", ["$steps.static"], {1: 1})]
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
    # both step problems are propagated with the context they carry - the
    # declaration-level unknown AND the block's own unresolved selector
    assert [
        (reason.code, reason.details.get("node_id"))
        for reason in models.unknown_reasons
    ] == [
        (DiscoveryProblemCode.DECLARATION_UNAVAILABLE, "$steps.mystery"),
        (DiscoveryProblemCode.UNRESOLVED_SELECTOR, "$steps.partial"),
    ]
    assert models.unknown_reasons[0] == declaration_unavailable_problem(
        node_id="$steps.mystery",
        declaration="resources",
        block_type="test/unknown@v1",
    )
    # unknown resources make the inventory incomplete but never fabricate a
    # count; the known references keep their histograms
    assert [
        (m.model_id, m.used_by_steps, m.steps_by_dimensionality) for m in models.items
    ] == [
        ("explicit/1", ["$steps.partial"], {1: 1}),
        ("project/1", ["$steps.model"], {1: 1}),
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
    assert by_id["shared/1"].steps_by_dimensionality == {1: 2}
    # the histogram comes from compilation, never from the metadata provider
    assert all(
        m.steps_by_dimensionality == {1: 1}
        for m in by_id.values()
        if m.model_id != "shared/1"
    )
    _assert_model_histograms_consistent(introspection)
    assert by_id["disabled/1"].metadata_status == "disabled"
    assert by_id["disabled/1"].metadata is None
    assert by_id["partial/1"].metadata_status == "available"
    assert by_id["partial/1"].metadata == ModelMetadata(task_type="od")
    assert by_id["broken/1"].metadata_status == "unavailable"
    assert by_id["gpt-x"].metadata_status == "unavailable"
    assert introspection.summary.models.complete is False, "selector, not lookups"
    assert introspection.summary.models.unknown_reasons == [
        unresolved_selector_problem(
            node_id="$steps.selector",
            declaration="resources",
            field="model_id",
            selector="$inputs.model",
            resource_type="roboflow_platform_model",
        )
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
    for declaration, discovery in (
        ("resources", dispatch.resources),
        ("operations", dispatch.operations),
        ("restrictions", dispatch.restrictions),
    ):
        assert discovery.complete is False
        assert (
            opaque_remote_workflow_problem(
                node_id="$steps.dispatch", declaration=declaration
            )
            in discovery.unknown_reasons
        )
    models = introspection.summary.models
    assert models.complete is False
    assert (
        opaque_remote_workflow_problem(
            node_id="$steps.dispatch", declaration="resources"
        )
        in models.unknown_reasons
    )
    assert [
        (m.model_id, m.used_by_steps, m.steps_by_dimensionality) for m in models.items
    ] == [("project/1", ["$steps.model"], {1: 1})]


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
    assert dumped["type"] == "workflow_introspection_v1"
    assert "schema_version" not in dumped
    assert dumped["summary"]["steps_by_dimensionality"] == {"1": 2}
    assert dumped["summary"]["models"]["items"][0]["steps_by_dimensionality"] == {
        "1": 1
    }
    assert all(step["type"] == "step_metadata_v1" for step in dumped["steps"])


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
    assert blank.resources.complete is False
    assert blank.resources.unknown_reasons == [
        invalid_resource_identifier_problem(
            node_id="$steps.blank",
            declaration="resources",
            field="model_id",
            resource_type="roboflow_platform_model",
        )
    ]
    assert blank.resources.items[0].metadata.model_id == blank_model_id
    assert _step(introspection, "$steps.valid").resources.complete is True
    models = introspection.summary.models
    assert models.complete is False
    assert models.unknown_reasons == [
        invalid_resource_identifier_problem(
            node_id="$steps.blank",
            declaration="resources",
            field="model_id",
            resource_type="roboflow_platform_model",
        )
    ]
    # the problem names the field, never the invalid value
    assert set(models.unknown_reasons[0].details) == {
        "node_id",
        "declaration",
        "field",
        "resource_type",
    }
    assert [
        (m.provider, m.model_id, m.used_by_steps, m.steps_by_dimensionality)
        for m in models.items
    ] == [("roboflow", "project/1", ["$steps.valid"], {1: 1})]
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
    # one problem per blank identity field, naming that field
    assert models.unknown_reasons == [
        invalid_resource_identifier_problem(
            node_id="$steps.llm",
            declaration="resources",
            field="model_id" if not model_value.strip() else "provider",
            resource_type="third_party_model",
        )
    ]
    assert provider.calls == []
    llm = _step(introspection, "$steps.llm")
    assert llm.resources.items[0].metadata.provider == provider_value
    assert llm.resources.items[0].metadata.model_id == model_value


def test_two_blank_identity_fields_yield_one_problem_each() -> None:
    result = _compile(
        inputs=[_image_input()],
        steps=[
            ThirdPartyManifest(
                type="test/third_party@v1",
                name="llm",
                images="$inputs.image",
                provider="",
                model="  ",
            )
        ],
    )

    models = build_workflow_introspection(compilation_result=result).summary.models

    assert models.unknown_reasons == [
        invalid_resource_identifier_problem(
            node_id="$steps.llm",
            declaration="resources",
            field="model_id",
            resource_type="third_party_model",
        ),
        invalid_resource_identifier_problem(
            node_id="$steps.llm",
            declaration="resources",
            field="provider",
            resource_type="third_party_model",
        ),
    ]


def test_two_steps_with_the_same_selector_stay_two_problems() -> None:
    result = _compile(
        inputs=[
            _image_input(),
            WorkflowParameter(type="WorkflowParameter", name="model"),
        ],
        steps=[
            _model("first", model_id="$inputs.model"),
            _model("second", model_id="$inputs.model"),
        ],
    )

    models = build_workflow_introspection(compilation_result=result).summary.models

    assert [reason.details["node_id"] for reason in models.unknown_reasons] == [
        "$steps.first",
        "$steps.second",
    ]
    assert {reason.code for reason in models.unknown_reasons} == {
        DiscoveryProblemCode.UNRESOLVED_SELECTOR
    }


def test_inventory_problem_order_does_not_depend_on_step_order() -> None:
    def build(step_names: List[str]) -> List[DiscoveryProblem]:
        result = _compile(
            inputs=[
                _image_input(),
                WorkflowParameter(type="WorkflowParameter", name="model"),
            ],
            steps=[_model(name, model_id="$inputs.model") for name in step_names],
        )
        introspection = build_workflow_introspection(compilation_result=result)
        return list(introspection.summary.models.unknown_reasons)

    assert build(["alpha", "beta"]) == build(["beta", "alpha"])


# ---------------------------------------------------------------------------
# per-step resource identity completeness
# ---------------------------------------------------------------------------


def _problem_keys(problems: List[DiscoveryProblem]) -> List[str]:
    return sorted(problem.model_dump_json() for problem in problems)


def test_fully_literal_resources_stay_complete_per_step() -> None:
    # given
    result = _compile(
        inputs=[_image_input()],
        steps=[
            _model("model", model_id="project/1"),
            ThirdPartyManifest(
                type="test/third_party@v1", name="llm", images="$inputs.image"
            ),
            ProjectSinkManifest(
                type="test/project_sink@v1", name="sink", images="$inputs.image"
            ),
        ],
    )

    # when
    introspection = build_workflow_introspection(compilation_result=result)

    # then
    for step in introspection.steps:
        assert step.resources.complete is True, step.node_id
        assert step.resources.unknown_reasons == []
    assert introspection.summary.models.complete is True


@pytest.mark.parametrize(
    "provider_value, model_value, expected_fields",
    [
        ("$inputs.provider", "gpt-x", [("provider", "$inputs.provider")]),
        ("openai", "$inputs.model", [("model_id", "$inputs.model")]),
        (
            "$inputs.provider",
            "$inputs.model",
            [("model_id", "$inputs.model"), ("provider", "$inputs.provider")],
        ),
    ],
)
def test_selector_fed_third_party_identity_makes_step_resources_incomplete(
    provider_value: str, model_value: str, expected_fields: list
) -> None:
    # given
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

    # when
    introspection = build_workflow_introspection(
        compilation_result=result, model_metadata_provider=provider
    )

    # then - the item is kept, one problem per selector-valued field
    llm = _step(introspection, "$steps.llm")
    expected = [
        unresolved_selector_problem(
            node_id="$steps.llm",
            declaration="resources",
            field=field,
            selector=selector,
            resource_type="third_party_model",
        )
        for field, selector in expected_fields
    ]
    assert llm.resources.complete is False
    assert llm.resources.items == [
        third_party_model(provider=provider_value, model_id=model_value)
    ]
    assert llm.resources.unknown_reasons == expected
    # the model inventory carries the same problems once each, with no lookup
    models = introspection.summary.models
    assert models.items == []
    assert models.complete is False
    assert models.unknown_reasons == expected
    assert provider.calls == []


@pytest.mark.parametrize(
    "project_url, expected_problem",
    [
        (
            "$inputs.project",
            unresolved_selector_problem(
                node_id="$steps.sink",
                declaration="resources",
                field="project_url",
                selector="$inputs.project",
                resource_type="roboflow_platform_project",
            ),
        ),
        (
            "",
            invalid_resource_identifier_problem(
                node_id="$steps.sink",
                declaration="resources",
                field="project_url",
                resource_type="roboflow_platform_project",
            ),
        ),
        (
            "   ",
            invalid_resource_identifier_problem(
                node_id="$steps.sink",
                declaration="resources",
                field="project_url",
                resource_type="roboflow_platform_project",
            ),
        ),
    ],
)
def test_unknown_project_identity_is_step_incomplete_but_not_model_incomplete(
    project_url: str, expected_problem: DiscoveryProblem
) -> None:
    # given
    result = _compile(
        inputs=[_image_input()],
        steps=[
            ProjectSinkManifest(
                type="test/project_sink@v1",
                name="sink",
                images="$inputs.image",
                project_url=project_url,
            )
        ],
    )
    provider = RecordingProvider(answers={})

    # when
    introspection = build_workflow_introspection(
        compilation_result=result, model_metadata_provider=provider
    )

    # then - the step keeps both items and reports the project problem
    sink = _step(introspection, "$steps.sink")
    assert sink.resources.complete is False
    assert sink.resources.unknown_reasons == [expected_problem]
    assert set(sink.resources.items) == {
        roboflow_platform_project(project_url=project_url),
        roboflow_platform_model(
            model_id="monitored/1", required_action=ModelRequiredAction.ACCESS
        ),
    }
    # the fully known model inventory stays complete
    models = introspection.summary.models
    assert models.complete is True
    assert models.unknown_reasons == []
    assert [(m.provider, m.model_id) for m in models.items] == [
        ("roboflow", "monitored/1")
    ]
    assert provider.calls == [("roboflow", "monitored/1")]


def test_project_uncertainty_does_not_hide_other_inventory_problems() -> None:
    # given
    result = _compile(
        inputs=[
            _image_input(),
            WorkflowParameter(type="WorkflowParameter", name="model"),
        ],
        steps=[
            ProjectSinkManifest(
                type="test/project_sink@v1",
                name="sink",
                images="$inputs.image",
                project_url="$inputs.project",
            ),
            _model("dynamic", model_id="$inputs.model"),
            UnknownManifest(
                type="test/unknown@v1", name="mystery", images="$inputs.image"
            ),
        ],
    )

    # when
    models = build_workflow_introspection(compilation_result=result).summary.models

    # then - model and declaration problems propagate; the project one does not
    assert models.complete is False
    assert models.unknown_reasons == [
        declaration_unavailable_problem(
            node_id="$steps.mystery",
            declaration="resources",
            block_type="test/unknown@v1",
        ),
        unresolved_selector_problem(
            node_id="$steps.dynamic",
            declaration="resources",
            field="model_id",
            selector="$inputs.model",
            resource_type="roboflow_platform_model",
        ),
    ]
    assert [m.model_id for m in models.items] == ["monitored/1"]


def test_identity_problems_are_added_next_to_prior_declared_reasons() -> None:
    # given
    result = _compile(
        inputs=[
            _image_input(),
            WorkflowParameter(type="WorkflowParameter", name="model"),
        ],
        steps=[
            PartiallyKnownModelsManifest(
                type="test/partially_known@v1",
                name="partial",
                images="$inputs.image",
            )
        ],
    )
    prior_problem = declaration_unavailable_problem(
        node_id="$steps.partial", declaration="resources"
    )
    selector_problem = unresolved_selector_problem(
        node_id="$steps.partial",
        declaration="resources",
        field="model_id",
        selector="$inputs.model",
        resource_type="roboflow_platform_model",
    )

    # when
    introspection = build_workflow_introspection(compilation_result=result)

    # then - items and the prior reason are kept; the identity problem is added
    partial = _step(introspection, "$steps.partial")
    assert partial.resources.complete is False
    assert {item.metadata.model_id for item in partial.resources.items} == {
        "known/1",
        "$inputs.model",
    }
    assert _problem_keys(partial.resources.unknown_reasons) == _problem_keys(
        [prior_problem, selector_problem]
    )
    # the literal model of the partially known step is still inventoried, and
    # the inventory carries each reason exactly once
    models = introspection.summary.models
    assert [m.model_id for m in models.items] == ["known/1"]
    assert _problem_keys(models.unknown_reasons) == _problem_keys(
        [prior_problem, selector_problem]
    )


def test_identity_problems_never_run_the_model_id_resolver() -> None:
    # given
    RESOLVER_CALLS.clear()
    result = _compile(
        inputs=[
            _image_input(),
            WorkflowParameter(type="WorkflowParameter", name="model"),
        ],
        steps=[
            ResolverModelManifest(
                type="test/resolver_model@v1",
                name="resolver",
                images="$inputs.image",
            )
        ],
    )

    # when
    introspection = build_workflow_introspection(compilation_result=result)

    # then
    assert RESOLVER_CALLS == []
    step = _step(introspection, "$steps.resolver")
    assert step.resources.complete is False
    assert step.resources.items[0].metadata.model_id == "$inputs.model"
    assert [reason.code for reason in step.resources.unknown_reasons] == [
        DiscoveryProblemCode.UNRESOLVED_SELECTOR
    ]
