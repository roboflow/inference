"""Declaring the optional ``run_many`` hook of an implementation.

Synthetic blocks only: what ``ImplementationSpec`` records, which hook
signatures are accepted, the reserved name and ``InvocationBatchError``.
Execution is covered in ``integration/engine/runtime/test_run_many``.
"""

import dataclasses
from typing import Any, Callable

import pytest
from roboflow_workflows.v2.data.kinds import FLOAT_KIND
from roboflow_workflows.v2.engine import (
    Block,
    BlockParams,
    Catalogue,
    Implementation,
    ImplementationContribution,
    Output,
    Ref,
    contract_digest,
    spec_of,
)
from roboflow_workflows.v2.engine.errors import (
    ContractError,
    DeclarationError,
    InvocationBatchError,
    _is_invocation_position,
)
from roboflow_workflows.v2.engine.implementations import (
    RESERVED_PHASE_NAMES,
    ImplementationSpec,
)
from roboflow_workflows.v2.engine.phases import phase


def contract(*implementations: type, **body: Any) -> type:
    namespace = {
        "type": "test/run_many/contract@v1",
        "prunable": True,
        "outputs": {"scaled": Output(FLOAT_KIND)},
        "Params": type(
            "Params", (BlockParams,), {"__annotations__": {"value": Ref(FLOAT_KIND)}}
        ),
        "implementations": implementations,
        **body,
    }
    return type("Contract", (Block,), namespace)


def implementation(name: str = "impl", **body: Any) -> type:
    namespace = {"name": name, "run": lambda self, *, value: {"scaled": value}, **body}
    return type(f"Impl_{name}", (Implementation,), namespace)


def hook(self, *, invocations):
    return [{"scaled": row["value"]} for row in invocations]


def selected(block: type) -> ImplementationSpec:
    return spec_of(block).implementations[0]


# --- what the spec records ------------------------------------------------------


def test_spec_records_a_declared_hook_and_describes_it() -> None:
    block = contract(implementation("many", run_many=hook))

    spec = selected(block)

    assert spec.run_many is True
    assert spec.describe()["run_many"] is True


def test_spec_without_a_hook_is_unchanged() -> None:
    spec = selected(contract(implementation("scalar")))

    assert spec.run_many is False
    assert "run_many" not in spec.describe()


def test_hand_built_spec_and_ordinary_block_default_to_no_hook() -> None:
    class Ordinary(Block):
        """An ordinary block is its own default; it does not opt in."""

        type = "test/run_many/ordinary@v1"
        outputs = {"scaled": Output(FLOAT_KIND)}

        class Params(BlockParams):
            value: Ref(FLOAT_KIND)

        def run(self, *, value) -> dict:
            return {"scaled": value}

        def run_many(self, *, invocations) -> list:
            return []

    hand_built = ImplementationSpec(
        name="hand",
        implementation_class=Ordinary,
        requires=frozenset(),
        resources=(),
        phases=None,
    )

    assert hand_built.run_many is False
    assert selected(Ordinary).run_many is False


def test_subclass_may_inherit_or_remove_the_hook() -> None:
    many = implementation("many", run_many=hook)
    inherited = type("Inherited", (many,), {"name": "inherited"})
    removed = type("Removed", (many,), {"name": "removed", "run_many": None})

    assert selected(contract(inherited)).run_many is True
    assert selected(contract(removed)).run_many is False


def test_contribution_keeps_the_hook_and_not_the_digest() -> None:
    public = contract(implementation("reference"))
    contributed = Catalogue([public]).with_contributions(
        [
            ImplementationContribution(
                block_type=public.type,
                contract=contract_digest(public),
                implementations=(implementation("many", run_many=hook),),
                provider="example-provider",
                version="1.0.0",
            )
        ]
    )
    spec = contributed.entry(public.type).spec

    added, own = spec.implementations
    assert (added.name, added.run_many, added.origin.provider) == (
        "many",
        True,
        "example-provider",
    )
    assert own.run_many is False
    assert contract_digest(spec) == contract_digest(public)
    assert contract_digest(contract(implementation("many", run_many=hook))) == (
        contract_digest(public)
    )


def test_equal_specs_differ_only_by_the_hook() -> None:
    spec = selected(contract(implementation("scalar")))

    assert dataclasses.replace(spec, run_many=True) != spec


# --- accepted and rejected hooks -------------------------------------------------


def keyword_only(self, *, invocations):
    return []


def positional_or_keyword(self, invocations):
    return []


def any_keyword(self, **kwargs):
    return []


def optional_extra(self, *, invocations, chunk=8):
    return []


@pytest.mark.parametrize(
    "method", [keyword_only, positional_or_keyword, any_keyword, optional_extra]
)
def test_hook_taking_invocations_by_keyword_is_accepted(method: Callable) -> None:
    assert selected(contract(implementation("many", run_many=method))).run_many


def no_invocations(self):
    return []


def other_name(self, *, rows):
    return []


def required_extra(self, *, invocations, chunk):
    return []


def positional_only(self, invocations, /):
    return []


def variadic(self, *invocations):
    return []


async def coroutine(self, *, invocations):
    return []


def generator(self, *, invocations):
    yield {}


async def async_generator(self, *, invocations):
    yield {}


@pytest.mark.parametrize(
    "method, message",
    [
        (no_invocations, r"does not accept Params field\(s\) \['invocations'\]"),
        (other_name, "requires 'rows', which is not a Params field"),
        (required_extra, "requires 'chunk', which is not a Params field"),
        (positional_only, "'invocations' must be passable by keyword"),
        (variadic, "'invocations' must be passable by keyword"),
        (coroutine, "not a coroutine or generator"),
        (generator, "not a coroutine or generator"),
        (async_generator, "not a coroutine or generator"),
        (staticmethod(keyword_only), "plain instance method, got staticmethod"),
        (classmethod(keyword_only), "plain instance method, got classmethod"),
        (3, "plain instance method, got int"),
    ],
)
def test_malformed_hook_fails_the_block_declaration(method: Any, message: str) -> None:
    with pytest.raises(DeclarationError, match=message) as raised:
        contract(implementation("bad", run_many=method))

    assert "implementation Impl_bad" in str(raised.value)


def test_hook_does_not_replace_the_required_run() -> None:
    without_run = type(
        "WithoutRun", (Implementation,), {"name": "many", "run_many": hook}
    )

    with pytest.raises(DeclarationError, match=r"does not implement run\(\)"):
        contract(without_run)


def test_hook_name_is_reserved_against_phases() -> None:
    def run_many(self, *, value):
        return value

    assert "run_many" in RESERVED_PHASE_NAMES
    with pytest.raises(DeclarationError, match=r"phase\(s\) \['run_many'\] shadow"):
        implementation("phased", run_many=phase(run_many))
    with pytest.raises(DeclarationError, match=r"phase\(s\) \['run_many'\] shadow"):
        type(
            "PhasedBlock",
            (Block,),
            {
                "type": "test/run_many/phased_block@v1",
                "outputs": {"scaled": Output(FLOAT_KIND)},
                "run_many": phase(run_many),
            },
        )


# --- InvocationBatchError -------------------------------------------------------


def test_invocation_batch_error_keeps_a_valid_position_and_its_cause() -> None:
    original = ValueError("row failed")

    try:
        try:
            raise original
        except ValueError as error:
            raise InvocationBatchError(2) from error
    except InvocationBatchError as raised:
        caught = raised

    assert caught.position == 2
    assert caught.__cause__ is original
    assert "position 2" in str(caught)


@pytest.mark.parametrize("position", [True, False, -1, 1.0, "1", None])
def test_invocation_batch_error_rejects_a_malformed_position(position: Any) -> None:
    with pytest.raises(ContractError, match="non-negative int"):
        InvocationBatchError(position)


@pytest.mark.parametrize(
    "position, size, valid",
    [
        (0, 1, True),
        (2, 3, True),
        (3, 3, False),
        (0, 0, False),
        (5, None, True),
        (True, 3, False),
        (-1, 3, False),
        (1.0, 3, False),
    ],
)
def test_position_check_includes_the_upper_bound_when_given(
    position: Any, size: Any, valid: bool
) -> None:
    assert _is_invocation_position(position, size=size) is valid
