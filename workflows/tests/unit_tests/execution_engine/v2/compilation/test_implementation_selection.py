"""Compile-time implementation selection, plan validation and selected construction."""

import dataclasses
from typing import Any, Dict, List

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
    spec_of,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    MutationConflictError,
)
from roboflow_workflows.execution_engine.v2.implementations import Implementation
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.execution_engine.v2.phases import phase
from roboflow_workflows.execution_engine.v2.plan import CompiledWorkflow, CompileOptions
from roboflow_workflows.execution_engine.v2.resources import Factory
from roboflow_workflows.execution_engine.v2.targets import (
    ImplementationChoice,
    Target,
    UnsupportedTargetError,
    consider,
)

CPU = Target.cpu()
FAST = Target(frozenset({"cpu", "fast-lib"}))
CONSTRUCTED: List[str] = []


class Accelerated(Implementation):
    """Preferred when the target has ``fast-lib``; needs an accelerator resource."""

    name = "accelerated"
    requires = ("cpu", "fast-lib")

    def __init__(self, *, accelerator: Any):
        CONSTRUCTED.append(self.name)
        self.accelerator = accelerator

    def run(self, *, value: float, factor: float) -> dict:
        return {"scaled": value * factor, "by": self.name}


class Portable(Implementation):
    """Runs anywhere with a CPU; a two-phase graph."""

    name = "portable"
    requires = ("cpu",)

    def __init__(self, *, offset: float = 0.0):
        CONSTRUCTED.append(self.name)
        self.offset = offset

    @phase
    def multiply(self, *, value: float, factor: float) -> float:
        return value * factor + self.offset

    @phase
    def pack(self, *, multiply: float) -> dict:
        return {"scaled": multiply, "by": self.name}

    def run(self, *, value: float, factor: float) -> dict:
        return self.pack(multiply=self.multiply(value=value, factor=factor))


class Scale(Block):
    """Logical contract with two implementations, preferred first."""

    type = "test/scale@v1"
    outputs = {"scaled": Output(FLOAT_KIND), "by": Output(STRING_KIND)}
    implementations = (Accelerated, Portable)

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        factor: float | Ref(FLOAT_KIND) = 2.0


class CudaOnly(Implementation):
    name = "cuda"
    requires = ("cuda", "torch")

    def __init__(self):
        raise AssertionError("an unsupported implementation was constructed")

    def run(self, *, value: float) -> dict:
        return {"out": value}


class GpuBlock(Block):
    type = "test/gpu@v1"
    outputs = {"out": Output(FLOAT_KIND)}
    implementations = (CudaOnly,)

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)


class InPlace(Implementation):
    name = "in-place"

    def run(self, *, value: float) -> dict:
        return {"out": value}


class Stamp(Block):
    """Its implementation declares nothing; the logical ``mutates`` still holds."""

    type = "test/stamp@v1"
    outputs = {"out": Output(FLOAT_KIND)}
    mutates = ("value",)
    implementations = (InPlace,)

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)


class Double(Block):
    """An ordinary block may declare phases itself: one class, no contract split."""

    type = "test/double@v1"
    outputs = {"out": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    @phase
    def doubled(self, *, value: float) -> float:
        return value * 2

    @phase
    def result(self, *, doubled: float) -> dict:
        return {"out": doubled}

    def run(self, *, value: float) -> dict:
        return self.result(doubled=self.doubled(value=value))


@pytest.fixture(autouse=True)
def clear_constructed() -> None:
    CONSTRUCTED.clear()


def catalogue(**providers: Any) -> Catalogue:
    return Catalogue(
        [Scale, GpuBlock, Stamp, Double], namespace="test", providers=providers
    )


def definition(*, factor: Any = "$inputs.factor", child: bool = False) -> dict:
    steps: List[Dict[str, Any]] = [
        {
            "type": "test/scale@v1",
            "name": "scale",
            "value": "$inputs.values",
            "factor": factor,
        }
    ]
    outputs = [
        {"type": "JsonField", "name": "scaled", "selector": "$steps.scale.scaled"},
        {"type": "JsonField", "name": "by", "selector": "$steps.scale.by"},
    ]
    if child:
        steps.append(
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "child",
                "workflow_definition": {
                    "version": "2.0",
                    "inputs": [
                        {"type": "WorkflowBatchInput", "name": "v", "kind": ["float"]}
                    ],
                    "steps": [
                        {"type": "test/scale@v1", "name": "scale", "value": "$inputs.v"}
                    ],
                    "outputs": [
                        {
                            "type": "JsonField",
                            "name": "by",
                            "selector": "$steps.scale.by",
                        }
                    ],
                },
                "parameter_bindings": {"v": "$inputs.values"},
            }
        )
        outputs.append(
            {"type": "JsonField", "name": "child_by", "selector": "$steps.child.by"}
        )
    workflow = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "values", "kind": ["float"]},
            {
                "type": "WorkflowParameter",
                "name": "factor",
                "kind": ["float"],
                "default_value": 3.0,
            },
        ],
        "steps": steps,
        "outputs": outputs,
    }

    return workflow


def compile_for(target: Target, **options: Any) -> CompiledWorkflow:
    plan = compile_workflow(
        definition(child=True),
        catalogue=catalogue(),
        options=CompileOptions(target=target, **options),
    )

    return plan


def test_default_cpu_target_selects_the_first_fitting_implementation() -> None:
    # when
    plan = compile_for(CPU)

    # then: accelerated is listed first but needs fast-lib, which the target lacks
    step = plan.step(("scale",))
    assert step.selected.name == "portable"
    assert step.implementation.considered == (
        ("accelerated", ("fast-lib",)),
        ("portable", ()),
    )
    assert plan.options.target == Target(frozenset({"cpu"}))
    assert CONSTRUCTED == []


def test_declared_order_wins_when_several_implementations_fit() -> None:
    # when
    plan = compile_for(FAST)

    # then
    assert plan.step(("scale",)).selected.name == "accelerated"
    assert plan.step(("scale",)).implementation.considered == (
        ("accelerated", ()),
        ("portable", ()),
    )


def test_nested_steps_are_selected_for_the_same_target() -> None:
    # when
    cpu_plan = compile_for(CPU)
    fast_plan = compile_for(FAST)

    # then
    assert cpu_plan.step(("child", "scale")).selected.name == "portable"
    assert fast_plan.step(("child", "scale")).selected.name == "accelerated"
    assert fast_plan.step(("child", "scale")).implementation.target == FAST


def test_unsupported_target_fails_at_compile_time_and_names_every_reason() -> None:
    # given
    gpu = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "values", "kind": ["float"]}],
        "steps": [{"type": "test/gpu@v1", "name": "gpu", "value": "$inputs.values"}],
        "outputs": [{"type": "JsonField", "name": "out", "selector": "$steps.gpu.out"}],
    }

    # when
    with pytest.raises(UnsupportedTargetError) as raised:
        compile_workflow(
            gpu,
            catalogue=catalogue(),
            options=CompileOptions(target=Target(frozenset({"cpu", "torch"}))),
        )

    # then
    assert raised.value.step_path == ("gpu",)
    assert raised.value.considered == (("cuda", ("cuda",)),)
    assert "'cuda' needs ['cuda']" in str(raised.value)
    assert "['cpu', 'torch']" in str(raised.value)


def test_session_constructs_and_resolves_resources_of_the_selected_class_only() -> None:
    # given: factories record their calls; the unselected one must never run
    created: List[str] = []

    def accelerator() -> str:
        created.append("accelerator")
        return "gpu-handle"

    def offset() -> float:
        created.append("offset")
        return 0.5

    plan = compile_workflow(
        definition(),
        catalogue=catalogue(accelerator=Factory(accelerator), offset=Factory(offset)),
    )

    # when
    session = plan.create_session()

    # then
    assert created == ["offset"]
    assert CONSTRUCTED == ["portable"]
    assert type(session.instances[("scale",)]) is Portable
    assert list(session.resources[("scale",)]) == ["offset"]
    assert session.run({"values": [1.0, 2.0]}).rows() == [
        {"scaled": 3.5, "by": "portable"},
        {"scaled": 6.5, "by": "portable"},
    ]


def test_selected_resources_keep_the_logical_namespace_and_type_for_lookup() -> None:
    # given: only namespaced caller resources are offered
    plan = compile_workflow(
        definition(), catalogue=catalogue(), options=CompileOptions(target=FAST)
    )

    # when
    session = plan.create_session(resources={"test.accelerator": "scoped-handle"})

    # then
    assert session.instances[("scale",)].accelerator == "scoped-handle"
    assert CONSTRUCTED == ["accelerated"]


def test_runtime_configuration_does_not_influence_the_compiled_selection() -> None:
    # given: the same step with a literal factor and with a selector-fed one
    literal = compile_workflow(definition(factor=10.0), catalogue=catalogue())
    fed = compile_workflow(definition(), catalogue=catalogue())
    session = fed.create_session()

    # when: the caller overrides the input default per run
    first = session.run({"values": [1.0], "factor": 1.0}).rows()
    second = session.run({"values": [1.0], "factor": 100.0}).rows()

    # then
    assert (
        literal.step(("scale",)).implementation == fed.step(("scale",)).implementation
    )
    assert first == [{"scaled": 1.0, "by": "portable"}]
    assert second == [{"scaled": 100.0, "by": "portable"}]
    assert CONSTRUCTED == ["portable"]


def test_phase_mode_records_phases_only_for_a_selected_graph() -> None:
    # when
    cpu = compile_for(CPU, block_execution="phases")
    fast = compile_for(FAST, block_execution="phases")
    default = compile_for(CPU)

    # then: accelerated has no phases, so it keeps run in phase mode
    assert cpu.step(("scale",)).execution == "phases"
    assert cpu.step(("child", "scale")).execution == "phases"
    assert fast.step(("scale",)).execution == "run"
    assert default.step(("scale",)).execution == "run"


def test_phase_mode_and_run_mode_return_the_same_rows() -> None:
    # given
    inputs = {"values": [1.0, 2.0], "factor": 4.0}
    run_plan = compile_for(CPU)
    phase_plan = compile_for(CPU, block_execution="phases")

    # when
    run_result = run_plan.create_session().run(inputs)
    phase_result = phase_plan.create_session().run(inputs)

    # then: the same rows, and only the phase plan actually ran the graph
    def phases_of(result) -> list:
        return [
            (tuple(event["step"]), event["phase"])
            for event in result.trace
            if event["event"] == "phase"
        ]

    assert phase_result.rows() == run_result.rows()
    assert run_result.rows()[0] == {
        "scaled": 4.0,
        "by": "portable",
        "child_by": "portable",
    }
    assert phases_of(run_result) == []
    assert phases_of(phase_result)[:2] == [
        (("scale",), "multiply"),
        (("scale",), "pack"),
    ]
    assert (("child", "scale"), "pack") in phases_of(phase_result)


def test_ordinary_phased_block_runs_its_default_implementation_in_phase_mode() -> None:
    # given
    workflow = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "values", "kind": ["float"]}],
        "steps": [
            {"type": "test/double@v1", "name": "double", "value": "$inputs.values"}
        ],
        "outputs": [
            {"type": "JsonField", "name": "out", "selector": "$steps.double.out"}
        ],
    }

    # when
    plan = compile_workflow(
        workflow,
        catalogue=catalogue(),
        options=CompileOptions(block_execution="phases"),
    )
    session = plan.create_session()

    # then
    step = plan.step(("double",))
    assert step.selected.name == "default"
    assert step.selected.implementation_class is Double
    assert step.implementation.considered == (("default", ()),)
    assert step.execution == "phases"
    assert type(session.instances[("double",)]) is Double
    assert session.run({"values": [1.5]}).rows() == [{"out": 3.0}]


def test_mutation_analysis_uses_the_logical_declaration_of_the_contract() -> None:
    # given: the selected implementation declares nothing about mutation
    shared = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "values", "kind": ["float"]}],
        "steps": [
            {"type": "test/stamp@v1", "name": "stamp", "value": "$inputs.values"},
            {"type": "test/scale@v1", "name": "scale", "value": "$inputs.values"},
        ],
        "outputs": [
            {"type": "JsonField", "name": "a", "selector": "$steps.stamp.out"},
            {"type": "JsonField", "name": "b", "selector": "$steps.scale.scaled"},
        ],
    }

    # when
    warned = compile_workflow(shared, catalogue=catalogue())
    with pytest.raises(MutationConflictError) as raised:
        compile_workflow(
            shared,
            catalogue=catalogue(),
            options=CompileOptions(mutation_conflicts="error"),
        )

    # then
    assert len(warned.warnings) == 1
    assert "$steps.stamp" in warned.warnings[0]
    assert raised.value.step_path == ("stamp",)


@pytest.fixture
def plan() -> CompiledWorkflow:
    return compile_for(FAST)


def test_contract_step_without_a_choice_is_rejected(plan: CompiledWorkflow) -> None:
    step = plan.step(("scale",))

    with pytest.raises(ContractError, match="needs an ImplementationChoice"):
        dataclasses.replace(step, implementation=None)


def test_forged_choice_of_a_later_fitting_implementation_is_rejected(
    plan: CompiledWorkflow,
) -> None:
    # given: portable also fits FAST, but accelerated is declared first
    step = plan.step(("scale",))
    portable = spec_of(Scale).implementations[1]
    forged = ImplementationChoice(
        spec=portable, target=FAST, considered=consider(spec_of(Scale), target=FAST)
    )

    # then
    with pytest.raises(ContractError, match="is not the selection for target"):
        dataclasses.replace(step, implementation=forged)


def test_choice_for_a_target_without_any_fit_is_rejected(
    plan: CompiledWorkflow,
) -> None:
    step = plan.step(("scale",))
    bare = Target(frozenset({"tpu"}))
    forged = ImplementationChoice(
        spec=step.selected,
        target=bare,
        considered=consider(spec_of(Scale), target=bare),
    )

    with pytest.raises(ContractError, match="cannot run on target"):
        dataclasses.replace(step, implementation=forged)


def test_choice_of_another_block_is_rejected(plan: CompiledWorkflow) -> None:
    step = plan.step(("scale",))
    gpu = spec_of(GpuBlock)
    target = Target(frozenset({"cuda", "torch"}))
    foreign = ImplementationChoice(
        spec=gpu.implementations[0],
        target=target,
        considered=consider(gpu, target=target),
    )

    with pytest.raises(ContractError):
        dataclasses.replace(step, implementation=foreign)


def test_unknown_or_unsupported_execution_mode_is_rejected(
    plan: CompiledWorkflow,
) -> None:
    step = plan.step(("scale",))

    with pytest.raises(ContractError, match="execution must be one of"):
        dataclasses.replace(step, execution="threads")
    with pytest.raises(ContractError, match="declares no phases"):
        dataclasses.replace(step, execution="phases")


def test_plan_rejects_choices_made_for_another_target(plan: CompiledWorkflow) -> None:
    with pytest.raises(ContractError, match="but the plan targets"):
        dataclasses.replace(plan, options=CompileOptions(target=CPU))


def test_plan_rejects_execution_that_differs_from_its_options() -> None:
    plan = compile_for(CPU)

    with pytest.raises(ContractError, match="records execution 'run'"):
        dataclasses.replace(plan, options=CompileOptions(block_execution="phases"))


def test_compile_options_validate_target_and_mode() -> None:
    with pytest.raises(ContractError, match="target must be a Target"):
        CompileOptions(target={"cpu"})
    with pytest.raises(ContractError, match="block_execution must be one of"):
        CompileOptions(block_execution="parallel")
