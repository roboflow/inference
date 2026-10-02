"""Introspection of alternative implementations: declared, selected and workload."""

from typing import Any

import pytest
from roboflow_workflows.execution_engine.entities.workload import (
    DiscoveryProblemCode,
    WorkOperation,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.implementations import Implementation
from roboflow_workflows.execution_engine.v2.introspection import (
    describe_catalogue,
    describe_workflow,
    discover_workload,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.execution_engine.v2.phases import phase
from roboflow_workflows.execution_engine.v2.plan import CompiledWorkflow, CompileOptions
from roboflow_workflows.execution_engine.v2.targets import Target

TORCH = Target(frozenset({"cpu", "torch"}))


class Torch(Implementation):
    """Declares its own weights; construction would load a model."""

    name = "torch"
    requires = ("cpu", "torch")

    def __init__(self, *, weights: Any):
        raise AssertionError("introspection constructed an implementation")

    @phase
    def prepare(self, *, value: float) -> float:
        return value

    @phase
    def forward(self, *, prepare: float, model_id: str) -> dict:
        return {"out": prepare}

    def run(self, *, value: float, model_id: str) -> dict:
        return self.forward(prepare=self.prepare(value=value), model_id=model_id)

    @classmethod
    def discover_dependent_resources(cls, params: BlockParams) -> list:
        return [
            DependentResource(resource_type="torch_weights", identifier=params.model_id)
        ]


class Reference(Implementation):
    """Overrides nothing, so the logical block's hooks answer."""

    name = "reference"

    def __init__(self):
        raise AssertionError("introspection constructed an implementation")

    def run(self, *, value: float, model_id: str) -> dict:
        return {"out": value}


class Infer(Block):
    type = "test/infer@v1"
    outputs = {"out": Output(FLOAT_KIND)}
    implementations = (Torch, Reference)

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        model_id: str | Ref(STRING_KIND) = "default-model"

    @classmethod
    def discover_dependent_resources(cls, params: BlockParams) -> list:
        return [
            DependentResource(
                resource_type="reference_table", identifier=params.model_id
            )
        ]

    @classmethod
    def discover_work_operations(cls, params: BlockParams) -> list:
        return [WorkOperation.MODEL_INFERENCE]


def compile_plan(
    target: Target, *, model_id: Any = "$inputs.model"
) -> CompiledWorkflow:
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "values", "kind": ["float"]},
            {
                "type": "WorkflowParameter",
                "name": "model",
                "kind": ["string"],
                "default_value": "resnet18",
            },
        ],
        "steps": [
            {
                "type": "test/infer@v1",
                "name": "infer",
                "value": "$inputs.values",
                "model_id": model_id,
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "out", "selector": "$steps.infer.out"}
        ],
    }
    plan = compile_workflow(
        definition,
        catalogue=Catalogue([Infer], namespace="test"),
        options=CompileOptions(target=target, block_execution="phases"),
    )

    return plan


def test_catalogue_lists_implementations_in_selection_order_without_constructing() -> (
    None
):
    # when
    (block,) = describe_catalogue(Catalogue([Infer], namespace="test"))["blocks"]

    # then
    torch, reference = block["implementations"]
    assert "resources" not in block
    assert [torch["name"], reference["name"]] == ["torch", "reference"]
    assert torch["requires"] == ["cpu", "torch"]
    assert [resource["name"] for resource in torch["resources"]] == ["weights"]
    assert torch["phases"]["result"] == "forward"
    assert reference["requires"] == []
    assert reference["phases"] is None


def test_workflow_description_explains_the_selection_and_execution() -> None:
    # when
    described = describe_workflow(compile_plan(Target.cpu()))

    # then
    (step,) = described["steps"]
    assert described["target"] == ["cpu"]
    assert described["block_execution"] == "phases"
    assert step["implementation"]["name"] == "reference"
    assert step["implementation"]["considered"] == [
        {"name": "torch", "missing": ["torch"], "selected": False},
        {"name": "reference", "missing": [], "selected": True},
    ]
    assert step["execution"] == "run"


def test_phased_selection_is_described_with_its_graph() -> None:
    # when
    (step,) = describe_workflow(compile_plan(TORCH))["steps"]

    # then
    assert step["implementation"]["name"] == "torch"
    assert step["execution"] == "phases"
    assert [item["name"] for item in step["implementation"]["phases"]["phases"]] == [
        "prepare",
        "forward",
    ]


def test_workload_uses_the_selected_hook_and_keeps_a_root_default_unresolved() -> None:
    # when
    workload = discover_workload(compile_plan(TORCH)).step("$steps.infer")

    # then: the input default may be overridden, so it is no constant
    assert workload.implementation == "torch"
    assert workload.execution == "phases"
    assert [resource.name for resource in workload.constructor_resources] == ["weights"]
    assert [item.resource_type for item in workload.resources.items] == [
        "torch_weights"
    ]
    assert [item.identifier for item in workload.resources.items] == ["$inputs.model"]
    assert workload.resources.complete is False
    assert [reason.code for reason in workload.resources.unknown_reasons] == [
        DiscoveryProblemCode.UNRESOLVED_SELECTOR
    ]


def test_workload_falls_back_to_the_contract_hooks() -> None:
    # when
    workload = discover_workload(compile_plan(Target.cpu(), model_id="table-v1")).step(
        "$steps.infer"
    )

    # then
    assert workload.implementation == "reference"
    assert [item.resource_type for item in workload.resources.items] == [
        "reference_table"
    ]
    assert workload.resources.complete is True
    assert workload.operations.items == [WorkOperation.MODEL_INFERENCE]


def test_undeclared_restrictions_stay_unknown_for_every_implementation() -> None:
    for target in (TORCH, Target.cpu()):
        # when
        restrictions = (
            discover_workload(compile_plan(target)).step("$steps.infer").restrictions
        )

        # then
        assert restrictions.complete is False
        assert [reason.code for reason in restrictions.unknown_reasons] == [
            DiscoveryProblemCode.DECLARATION_UNAVAILABLE
        ]


def test_workload_description_names_implementation_and_constructor_resources() -> None:
    described = discover_workload(compile_plan(TORCH)).describe()["steps"][
        "$steps.infer"
    ]

    assert described["implementation"] == "torch"
    assert described["execution"] == "phases"
    assert described["constructor_resources"] == [
        {"name": "weights", "required": True, "annotation": "Any"}
    ]


@pytest.mark.parametrize("target", [TORCH, Target.cpu()])
def test_plan_description_never_constructs(target: Target) -> None:
    plan = compile_plan(target)

    described = plan.describe()

    assert described["steps"][0]["implementation"]["name"] in {"torch", "reference"}
