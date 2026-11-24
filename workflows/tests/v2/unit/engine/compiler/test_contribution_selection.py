"""Compiling and running a workflow with host-supplied implementations.

One workflow definition is compiled against a public catalogue and against the
same catalogue with contributions. Covers selection by target and quality,
plan validation, resource resolution, construction and plan comparison.
"""

import dataclasses
import sys
from typing import Any, List

import pytest
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.v2.engine import (
    Block,
    BlockParams,
    Catalogue,
    CompiledWorkflow,
    CompileOptions,
    Factory,
    Implementation,
    ImplementationContribution,
    Output,
    Ref,
    Target,
    compile_workflow,
    contract_digest,
    spec_of,
)
from roboflow_workflows.v2.engine.errors import (
    ContractError,
    GraphUpdateError,
    ResourceError,
)
from roboflow_workflows.v2.engine.updates import compare_plans

CPU = Target.cpu()
FAST = Target(frozenset({"cpu", "fast-kernel"}))
CONSTRUCTED: List[str] = []


class Reference(Implementation):
    name = "reference"
    requires = ("cpu",)

    def __init__(self) -> None:
        CONSTRUCTED.append(self.name)

    def run(self, *, value: float, factor: float) -> dict:
        return {"scaled": value * factor, "by": self.name}


class Scale(Block):
    """Public logical block with one public implementation."""

    type = "probe/scale@v1"
    outputs = {"scaled": Output(FLOAT_KIND), "by": Output(STRING_KIND)}
    implementations = (Reference,)

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        factor: float = 2.0


class Offset(Block):
    """Public ordinary block: the class is its own ``default`` implementation."""

    type = "probe/offset@v1"
    outputs = {"shifted": Output(FLOAT_KIND), "by": Output(STRING_KIND)}

    def __init__(self) -> None:
        CONSTRUCTED.append("default")

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value: float) -> dict:
        return {"shifted": value + 1.0, "by": "default"}


def fast(name: str = "fast", quality: tuple = ()) -> type:
    class Fast(Implementation):
        """Same output as the reference; needs a resource and a capability."""

        requires = ("cpu", "fast-kernel")

        def __init__(self, *, kernel_cache: Any):
            CONSTRUCTED.append(self.name)
            self.kernel_cache = kernel_cache

        def run(self, *, value: float, factor: float) -> dict:
            return {"scaled": value * factor, "by": self.name}

    Fast.name = name
    Fast.quality = quality

    return Fast


class FastOffset(Implementation):
    name = "fast-offset"
    requires = ("fast-kernel",)

    def __init__(self) -> None:
        CONSTRUCTED.append(self.name)

    def run(self, *, value: float) -> dict:
        return {"shifted": value + 1.0, "by": self.name}


FAST_SCALE = fast()


@pytest.fixture(autouse=True)
def clear_constructed() -> None:
    CONSTRUCTED.clear()


def contribution(
    *implementations: type, block: type = Scale, version: str = "1.0.0"
) -> ImplementationContribution:
    declared = ImplementationContribution(
        block_type=block.type,
        contract=contract_digest(block),
        implementations=implementations or (FAST_SCALE,),
        provider="example-provider",
        version=version,
    )

    return declared


def public() -> Catalogue:
    catalogue = Catalogue([Scale, Offset], namespace="probe")

    return catalogue


def contributed(*contributions: ImplementationContribution) -> Catalogue:
    catalogue = public().with_contributions(contributions or [contribution()])

    return catalogue


def definition(*, step_quality: Any = None, child: bool = False) -> dict:
    steps = [
        {"type": Scale.type, "name": "scale", "value": "$inputs.values"},
        {"type": Offset.type, "name": "offset", "value": "$steps.scale.scaled"},
    ]
    outputs = [
        {"type": "JsonField", "name": "scaled", "selector": "$steps.scale.scaled"},
        {"type": "JsonField", "name": "by", "selector": "$steps.scale.by"},
        {"type": "JsonField", "name": "shifted", "selector": "$steps.offset.shifted"},
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
                        {"type": Scale.type, "name": "scale", "value": "$inputs.v"}
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
    workflow = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "values", "kind": ["float"]}],
        "steps": steps,
        "outputs": outputs,
    }
    if step_quality is not None:
        workflow["execution"] = {"step_quality": {"$steps.scale": step_quality}}

    return workflow


def compiled(
    catalogue: Catalogue,
    target: Target = CPU,
    *,
    step_quality: Any = None,
    child: bool = False,
    **options: Any,
) -> CompiledWorkflow:
    plan = compile_workflow(
        definition(step_quality=step_quality, child=child),
        catalogue=catalogue,
        options=CompileOptions(target=target, **options),
    )

    return plan


def reasons(diff: Any) -> list:
    found = sorted((change.name, change.reason) for change in diff.breaking)

    return found


# --- same definition, with and without contributions -----------------------


def test_public_only_plan_is_unchanged_by_the_contribution_api() -> None:
    plan = compiled(public(), FAST)

    assert plan.step(("scale",)).selected.name == "reference"
    assert plan.step(("scale",)).selected.origin is None
    assert "origin" not in plan.describe()["steps"][0]["implementation"]
    assert plan.describe() == compiled(public().with_contributions([]), FAST).describe()


def test_fitting_target_runs_the_contribution_with_the_public_contract() -> None:
    reference = compiled(public(), FAST)
    plan = compiled(contributed(), FAST)

    step = plan.step(("scale",))
    assert step.selected.name == "fast"
    assert step.implementation.considered == (("fast", ()), ("reference", ()))
    assert step.outputs.keys() == reference.step(("scale",)).outputs.keys()
    assert [output.name for output in plan.outputs] == [
        output.name for output in reference.outputs
    ]
    described = plan.describe()["steps"][0]["implementation"]
    assert described["name"] == "fast"
    assert described["origin"] == dataclasses.asdict(step.selected.origin)
    assert dataclasses.asdict(step.selected.origin) == {
        "provider": "example-provider",
        "version": "1.0.0",
        "contract": contract_digest(Scale),
    }
    session = plan.create_session(resources={"kernel_cache": "cache"})
    assert session.run({"values": [1.0, 2.0]}).rows() == [
        {"scaled": 2.0, "by": "fast", "shifted": 3.0},
        {"scaled": 4.0, "by": "fast", "shifted": 5.0},
    ]
    assert CONSTRUCTED == ["fast", "default"]


def test_target_without_the_capability_selects_the_public_reference_visibly() -> None:
    plan = compiled(contributed(), CPU)

    step = plan.step(("scale",))
    assert step.selected.name == "reference"
    assert step.implementation.considered == (
        ("fast", ("fast-kernel",)),
        ("reference", ()),
    )
    session = plan.create_session()
    assert session.run({"values": [1.0]}).rows()[0]["by"] == "reference"
    assert "fast" not in CONSTRUCTED


def test_ordinary_block_runs_its_contribution_or_its_default() -> None:
    catalogue = contributed(contribution(FastOffset, block=Offset))

    cpu_plan = compiled(catalogue, CPU)
    fast_plan = compiled(catalogue, FAST)

    assert cpu_plan.step(("offset",)).selected.name == "default"
    assert fast_plan.step(("offset",)).selected.name == "fast-offset"
    rows = fast_plan.create_session().run({"values": [1.0]}).rows()
    assert rows == [{"scaled": 2.0, "by": "reference", "shifted": 3.0}]
    assert CONSTRUCTED == ["reference", "fast-offset"]


def test_nested_steps_select_from_the_same_catalogue() -> None:
    plan = compiled(contributed(), FAST, child=True)

    assert plan.step(("child", "scale")).selected.name == "fast"
    assert compiled(contributed(), CPU, child=True).step(
        ("child", "scale")
    ).selected.name == ("reference")


# --- resources ---------------------------------------------------------------


def test_only_the_selected_implementation_resolves_resources() -> None:
    created = []

    def cache() -> str:
        created.append("cache")
        return "cache"

    cpu_session = compiled(contributed(), CPU).create_session(
        resources={"kernel_cache": Factory(cache)}
    )
    assert created == []

    with pytest.raises(ResourceError, match="kernel_cache"):
        compiled(contributed(), FAST).create_session()

    fast_session = compiled(contributed(), FAST).create_session(
        resources={"probe.kernel_cache": Factory(cache)}
    )
    assert created == ["cache"]
    assert fast_session.instances[("scale",)].kernel_cache == "cache"
    assert cpu_session.instances[("scale",)].__class__ is Reference


# --- quality ---------------------------------------------------------------


class PublicFast(Implementation):
    name = "public-fast"
    requires = ("cpu",)
    quality = ("fast",)

    def run(self, *, value: float) -> dict:
        return {"graded": value, "by": self.name}


class PublicAccurate(Implementation):
    name = "public-accurate"
    requires = ("cpu",)
    quality = ("accurate",)

    def run(self, *, value: float) -> dict:
        return {"graded": value, "by": self.name}


class Graded(Block):
    """Public block whose own implementations serve two quality labels."""

    type = "probe/graded@v1"
    outputs = {"graded": Output(FLOAT_KIND), "by": Output(STRING_KIND)}
    implementations = (PublicAccurate, PublicFast)

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)


def graded_contribution(quality: tuple) -> ImplementationContribution:
    class ContributedFast(Implementation):
        name = "contributed-fast"
        requires = ("cpu", "fast-kernel")

        def run(self, *, value: float) -> dict:
            return {"graded": value, "by": self.name}

    ContributedFast.quality = quality
    declared = ImplementationContribution(
        block_type=Graded.type,
        contract=contract_digest(Graded),
        implementations=(ContributedFast,),
        provider="example-provider",
        version="1.0.0",
    )

    return declared


def graded_plan(
    catalogue: Catalogue, target: Target, *, step_quality: Any = "fast"
) -> str:
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "values", "kind": ["float"]}],
        "steps": [{"type": Graded.type, "name": "graded", "value": "$inputs.values"}],
        "outputs": [
            {"type": "JsonField", "name": "by", "selector": "$steps.graded.by"}
        ],
    }
    if step_quality is not None:
        definition["execution"] = {"step_quality": {"$steps.graded": step_quality}}
    plan = compile_workflow(
        definition, catalogue=catalogue, options=CompileOptions(target=target)
    )
    selected = plan.step(("graded",)).selected.name

    return selected


def graded_catalogue() -> Catalogue:
    catalogue = Catalogue([Graded], namespace="probe")

    return catalogue


def test_a_shared_quality_label_compiles_with_and_without_the_contribution() -> None:
    contributed_catalogue = graded_catalogue().with_contributions(
        [graded_contribution(("fast",))]
    )

    assert graded_plan(graded_catalogue(), FAST) == "public-fast"
    assert graded_plan(contributed_catalogue, FAST) == "contributed-fast"
    assert graded_plan(contributed_catalogue, CPU) == "public-fast"


def test_a_contribution_only_quality_label_is_rejected() -> None:
    problem = graded_catalogue().contribution_problem(
        graded_contribution(("fast", "turbo"))
    )

    assert "quality label(s) ['turbo'] are not served" in problem
    assert "['accurate', 'fast']" in problem


def test_quality_labels_on_a_block_without_labels_are_rejected() -> None:
    labelled = contribution(fast("labelled", quality=("fast",)))

    assert "quality label(s) ['fast']" in public().contribution_problem(labelled)


def test_a_fitting_unlabelled_contribution_is_the_default() -> None:
    # Contributions must be exact replacements: they win wherever they fit,
    # unless a quality request selects a public implementation first.
    unlabelled = graded_catalogue().with_contributions([graded_contribution(())])

    assert graded_plan(unlabelled, FAST) == "public-fast"
    assert graded_plan(unlabelled, FAST, step_quality=None) == "contributed-fast"


# --- plan validation ---------------------------------------------------------


def test_a_contributed_choice_is_invalid_against_the_public_description() -> None:
    step = compiled(contributed(), FAST).step(("scale",))

    with pytest.raises(ContractError, match="is not the selection"):
        dataclasses.replace(step, spec=spec_of(Scale))


def test_compiling_imports_and_constructs_nothing() -> None:
    catalogue = contributed()
    compiled(catalogue, FAST)
    modules = set(sys.modules)

    compiled(catalogue, FAST)

    assert set(sys.modules) == modules
    assert CONSTRUCTED == []


# --- plan comparison and live updates ----------------------------------------


def test_switching_to_a_contribution_needs_a_reset() -> None:
    diff = compare_plans(compiled(public(), FAST), compiled(contributed(), FAST))

    assert diff.kind == "reset"
    assert ("$steps.scale", "implementation_changed") in reasons(diff)


def test_a_provider_version_change_of_the_same_class_needs_a_reset() -> None:
    old = compiled(contributed(contribution(version="1.0.0")), FAST)
    new = compiled(contributed(contribution(version="1.1.0")), FAST)

    diff = compare_plans(old, new)

    assert old.step(("scale",)).selected.implementation_class is (
        new.step(("scale",)).selected.implementation_class
    )
    assert diff.kind == "reset"
    assert reasons(diff) == [
        ("$steps.scale", "contract_changed"),
        ("$steps.scale", "implementation_changed"),
    ]


def test_identical_reactivation_preserves_the_graph() -> None:
    old = compiled(contributed(contribution()), FAST)
    new = compiled(contributed(contribution()), FAST)

    assert old.step(("scale",)).spec is not new.step(("scale",)).spec
    assert compare_plans(old, new).kind == "preserve"


def test_an_unselected_contribution_change_still_resets_its_block() -> None:
    # Accepted conservative rule: the block's implementation list is compared.
    old = compiled(public(), CPU)
    new = compiled(contributed(), CPU)

    diff = compare_plans(old, new)

    assert new.step(("scale",)).selected.name == "reference"
    assert diff.kind == "reset"
    assert reasons(diff) == [("$steps.scale", "contract_changed")]


def test_a_running_session_switches_provider_version_only_by_reset() -> None:
    old = compiled(contributed(contribution(version="1.0.0")), FAST)
    new = compiled(contributed(contribution(version="1.1.0")), FAST)
    session = old.create_session(resources={"kernel_cache": "cache"})
    first = session.instances[("scale",)]

    with pytest.raises(GraphUpdateError):
        session.update(new)
    receipt = session.update(new, reset=True)

    assert receipt.reset
    assert session.instances[("scale",)] is not first
    assert session.run({"values": [1.0]}).rows()[0]["by"] == "fast"


def test_a_running_session_keeps_its_instance_on_identical_reactivation() -> None:
    session = compiled(contributed(), FAST).create_session(
        resources={"kernel_cache": "cache"}
    )
    first = session.instances[("scale",)]

    receipt = session.update(compiled(contributed(), FAST))

    assert not receipt.reset
    assert session.instances[("scale",)] is first
