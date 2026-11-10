"""``retains_inputs`` and ``retains_outputs``: declaration, description, digest.

Both are tuples of names on the logical block, empty by default. Inputs name
selector-bound fields; outputs name outputs, checked when the class is
created or, for configured outputs, once ``describe_outputs`` resolved them.
"""

import dataclasses
from typing import Any, Mapping

import pytest
from roboflow_workflows.v2.data.kinds import DICTIONARY_KIND, FLOAT_KIND
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
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    CatalogueError,
    DeclarationError,
    ParamsValidationError,
)

from tests.v2.unit.engine.authoring.test_nullable_refs import (
    R1_PROBE_DIGEST,
    probe_block,
)


class Undeclared(Block):
    """Abstract base (no ``type``): one data field, one literal field, one output."""

    outputs = {"value": Output(DICTIONARY_KIND, source="value")}

    class Params(BlockParams):
        value: Ref(DICTIONARY_KIND)
        key: str = "k"

    def run(self, *, value: Any, key: str) -> dict:
        return {"value": value}


def declare(**attributes: Any) -> type:
    """A concrete ``Undeclared`` with extra class attributes."""
    declared = type(
        "Declared", (Undeclared,), {"type": "retained/declared@v1", **attributes}
    )

    return declared


class Picked(Block):
    """Configured outputs: one per name in ``names``; keeps ``value``."""

    type = "retained/picked@v1"
    output_fields = ("names",)
    retains_outputs = ("value",)

    class Params(BlockParams):
        names: list[str]
        trigger: Ref()

    @classmethod
    def describe_outputs(cls, params: BlockParams) -> Mapping[str, Output]:
        return {name: Output() for name in params.names}

    def run(self, *, names: list, trigger: Any) -> dict:
        return {name: {} for name in names}


# --- defaults and description ------------------------------------------------------


def test_existing_blocks_keep_nothing_by_default() -> None:
    spec = spec_of(declare())

    assert (spec.retains_inputs, spec.retains_outputs) == ((), ())
    assert spec.describe()["retains_inputs"] == []
    assert spec.describe()["retains_outputs"] == []


def test_declared_names_are_recorded_and_described_per_direction() -> None:
    # One name, two directions: the field "value" and the output "value".
    spec = spec_of(declare(retains_inputs=["value"], retains_outputs=("value",)))

    assert (spec.retains_inputs, spec.retains_outputs) == (("value",), ("value",))
    assert spec.describe()["retains_inputs"] == ["value"]
    assert spec.describe()["retains_outputs"] == ["value"]


@pytest.mark.parametrize(
    ("attributes", "message"),
    [
        ({"retains_inputs": "value"}, "retains_inputs must be a tuple of field names"),
        ({"retains_inputs": ("missing",)}, "retains_inputs names unknown field"),
        ({"retains_inputs": ("key",)}, "'key', which accepts no data selector"),
        (
            {"retains_outputs": "value"},
            "retains_outputs must be a tuple of output names",
        ),
        ({"retains_outputs": ("missing",)}, r"\['missing'\], which are not outputs"),
        ({"retains_outputs": ("a.b",)}, "'a.b', which is not an output name"),
        ({"retains_outputs": (1,)}, "1, which is not an output name"),
    ],
    ids=[
        "inputs-string",
        "inputs-unknown",
        "inputs-literal",
        "outputs-string",
        "outputs-unknown",
        "outputs-not-a-name",
        "outputs-not-a-string",
    ],
)
def test_invalid_names_are_rejected_at_class_creation(
    attributes: dict, message: str
) -> None:
    with pytest.raises(DeclarationError, match=message):
        declare(**attributes)


# --- configured outputs ---------------------------------------------------------------


def configured(names: list) -> dict:
    return {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "x"}],
        "steps": [
            {
                "type": Picked.type,
                "name": "pick",
                "names": names,
                "trigger": "$inputs.x",
            }
        ],
        "outputs": [],
    }


def test_configured_outputs_are_checked_once_resolved() -> None:
    spec = spec_of(Picked)
    params = spec.validate_params({"names": ["value", "other"], "trigger": "$inputs.x"})

    assert sorted(spec.resolve_outputs(params)) == ["other", "value"]
    with pytest.raises(DeclarationError, match=r"describe_outputs\(\) retains_outputs"):
        spec.resolve_outputs(
            spec.validate_params({"names": ["other"], "trigger": "$inputs.x"})
        )


def test_a_configured_step_without_its_kept_output_does_not_compile() -> None:
    catalogue = Catalogue([Picked], namespace="retained")

    compile_workflow(configured(["value"]), catalogue=catalogue)
    with pytest.raises(ParamsValidationError, match="configures invalid outputs"):
        compile_workflow(configured(["other"]), catalogue=catalogue)


# --- effects ------------------------------------------------------------------------


def test_a_kept_input_is_an_effect_so_the_block_cannot_be_prunable() -> None:
    with pytest.raises(DeclarationError, match="declares retains_inputs"):
        declare(retains_inputs=("value",), prunable=True)


def test_a_kept_output_alone_allows_a_prunable_block() -> None:
    spec = spec_of(declare(retains_outputs=("value",), prunable=True))

    assert spec.prunable


# --- the logical block owns the declarations ----------------------------------------


class Reference(Implementation):
    name = "reference"

    def run(self, *, value: float) -> dict:
        return {"value": value}


@pytest.mark.parametrize("attribute", ["retains_inputs", "retains_outputs"])
def test_an_implementation_cannot_restate_retention(attribute: str) -> None:
    restating = type("Restating", (Reference,), {attribute: ()})

    with pytest.raises(DeclarationError, match="restates contract attribute"):

        class Logical(Block):
            type = "retained/logical@v1"
            outputs = {"value": Output(FLOAT_KIND)}
            implementations = (restating,)

            class Params(BlockParams):
                value: Ref(FLOAT_KIND)


def test_a_block_listing_implementations_declares_retention_itself() -> None:
    class Logical(Block):
        type = "retained/logical@v1"
        outputs = {"value": Output(FLOAT_KIND)}
        retains_inputs = ("value",)
        retains_outputs = ("value",)
        implementations = (Reference,)

        class Params(BlockParams):
            value: Ref(FLOAT_KIND)

    spec = spec_of(Logical)

    assert (spec.retains_inputs, spec.retains_outputs) == (("value",), ("value",))


# --- contract digest and contributions ---------------------------------------------


def test_empty_declarations_keep_the_published_digest() -> None:
    spec = spec_of(probe_block())

    assert contract_digest(spec) == R1_PROBE_DIGEST
    assert contract_digest(dataclasses.replace(spec, retains_inputs=())) == (
        R1_PROBE_DIGEST
    )
    assert contract_digest(declare()) == contract_digest(
        declare(retains_inputs=(), retains_outputs=())
    )


def test_each_declared_direction_changes_the_digest() -> None:
    spec = spec_of(probe_block())
    kept_input = dataclasses.replace(spec, retains_inputs=("whole",))
    kept_output = dataclasses.replace(spec, retains_outputs=("out",))

    digests = {
        contract_digest(spec),
        contract_digest(kept_input),
        contract_digest(kept_output),
        contract_digest(dataclasses.replace(kept_input, retains_outputs=("out",))),
    }

    assert len(digests) == 4
    assert contract_digest(spec).startswith("v1:sha256:")


class Faster(Implementation):
    name = "faster"
    requires = ("fast-kernel",)

    def run(self, *, value: Any, key: str) -> dict:
        return {"value": value}


def test_a_contribution_pinned_before_retention_was_declared_is_rejected() -> None:
    before = declare()
    after = declare(retains_inputs=("value",))
    stale = ImplementationContribution(
        block_type=after.type,
        contract=contract_digest(before),
        implementations=(Faster,),
        provider="example-provider",
        version="1.0.0",
    )
    current = dataclasses.replace(stale, contract=contract_digest(after))
    catalogue = Catalogue([after], namespace="retained")

    assert "was built for contract" in catalogue.contribution_problem(stale)
    with pytest.raises(CatalogueError, match="was built for contract"):
        catalogue.with_contributions([stale])
    assert catalogue.contribution_problem(current) is None
    assert [
        item.name
        for item in catalogue.with_contributions([current])
        .entry(after.type)
        .spec.implementations
    ] == ["faster", "default"]
