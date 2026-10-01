"""Targets and the deterministic selection of a block's implementation."""

import pytest
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
    spec_of,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    WorkflowCompileError,
)
from roboflow_workflows.execution_engine.v2.implementations import Implementation
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.targets import (
    ImplementationChoice,
    Target,
    UnsupportedTargetError,
    check_choice,
    default_implementation,
    select_implementation,
)


class Mps(Implementation):
    name = "torch-mps"
    requires = ("torch", "mps")

    def run(self, *, value: float) -> dict:
        return {"out": value}


class Cpu(Implementation):
    name = "torch-cpu"
    requires = ("cpu", "torch")

    def run(self, *, value: float) -> dict:
        return {"out": value}


class Anywhere(Implementation):
    name = "anywhere"

    def run(self, *, value: float) -> dict:
        return {"out": value}


class Classify(Block):
    type = "test/classify@v1"
    outputs = {"out": Output(FLOAT_KIND)}
    implementations = (Mps, Cpu)

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)


class Reversed(Block):
    """Same implementations in the opposite preference order."""

    type = "test/reversed@v1"
    outputs = {"out": Output(FLOAT_KIND)}
    implementations = (Cpu, Mps)

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)


class Unconstrained(Block):
    type = "test/unconstrained@v1"
    outputs = {"out": Output(FLOAT_KIND)}
    implementations = (Anywhere,)

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)


class Ordinary(Block):
    type = "test/ordinary@v1"
    outputs = {"out": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value: float) -> dict:
        return {"out": value}


def test_target_normalizes_capabilities_and_cpu_is_the_default() -> None:
    assert Target(["torch", "cpu", "cpu"]).capabilities == frozenset({"cpu", "torch"})
    assert Target.cpu() == Target(frozenset({"cpu"}))
    assert Target(("mps", "cpu")).describe() == ["cpu", "mps"]


@pytest.mark.parametrize("capabilities", ["cpu", ["cpu", ""], [1], [[]], [{}], [set()]])
def test_target_rejects_a_single_string_and_non_names(capabilities) -> None:
    with pytest.raises(ContractError):
        Target(capabilities)


def test_missing_requirements_are_sorted() -> None:
    assert Target.cpu().missing(("torch", "mps", "cpu")) == ("mps", "torch")


def test_selection_follows_declared_order_not_names_or_registration() -> None:
    # given
    target = Target(frozenset({"cpu", "torch", "mps"}))

    # when
    preferred = select_implementation(
        spec_of(Classify), target=target, step_path=("a",)
    )
    reversed_ = select_implementation(
        spec_of(Reversed), target=target, step_path=("b",)
    )

    # then
    assert preferred.name == "torch-mps"
    assert reversed_.name == "torch-cpu"


def test_choice_explains_every_alternative() -> None:
    # when
    choice = select_implementation(
        spec_of(Classify), target=Target({"cpu", "torch"}), step_path=("classify",)
    )

    # then
    assert choice.spec is spec_of(Classify).implementations[1]
    assert choice.describe() == {
        "name": "torch-cpu",
        "target": ["cpu", "torch"],
        "considered": [
            {"name": "torch-mps", "missing": ["mps"], "selected": False},
            {"name": "torch-cpu", "missing": [], "selected": True},
        ],
    }


def test_implementation_without_requirements_fits_an_empty_target() -> None:
    choice = select_implementation(
        spec_of(Unconstrained), target=Target(frozenset()), step_path=("x",)
    )

    assert choice.name == "anywhere"


def test_unsupported_target_is_a_compile_error_with_reasons() -> None:
    # when
    with pytest.raises(UnsupportedTargetError) as raised:
        select_implementation(
            spec_of(Classify), target=Target.cpu(), step_path=("child", "classify")
        )

    # then
    assert isinstance(raised.value, WorkflowCompileError)
    assert raised.value.step_path == ("child", "classify")
    assert raised.value.considered == (
        ("torch-mps", ("mps", "torch")),
        ("torch-cpu", ("torch",)),
    )
    assert str(raised.value) == (
        "$steps.child/classify (test/classify@v1) has no implementation for target "
        "['cpu']: 'torch-mps' needs ['mps', 'torch']; 'torch-cpu' needs ['torch']"
    )


def test_ordinary_block_is_its_own_default_implementation() -> None:
    # when
    default = default_implementation(spec_of(Ordinary))
    choice = select_implementation(
        spec_of(Ordinary), target=Target.cpu(), step_path=("o",)
    )

    # then
    assert default.name == "default"
    assert default.implementation_class is Ordinary
    assert choice.spec is default
    check_choice(spec_of(Ordinary), None)
    check_choice(spec_of(Ordinary), choice)


def test_contract_block_has_no_default_implementation() -> None:
    with pytest.raises(ContractError, match="needs an ImplementationChoice"):
        default_implementation(spec_of(Classify))


def test_check_choice_rejects_a_changed_explanation() -> None:
    # given
    target = Target({"cpu", "torch"})
    honest = select_implementation(spec_of(Classify), target=target, step_path=())
    altered = ImplementationChoice(
        spec=honest.spec, target=target, considered=(("torch-cpu", ()),)
    )

    # then
    check_choice(spec_of(Classify), honest)
    with pytest.raises(ContractError, match="is not the selection"):
        check_choice(spec_of(Classify), altered)
    with pytest.raises(ContractError, match="must be an ImplementationChoice"):
        check_choice(spec_of(Classify), "torch-cpu")
