"""Catalogue-level rules of host-supplied implementations (``contributions``).

Synthetic blocks only: validation, the contract digest, ordering, merging and
descriptions. Compilation and sessions are covered in
``test_contribution_selection``.
"""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Literal, Union

import pytest
from pydantic import BaseModel, Field
from roboflow_workflows.v2.data.kinds import (
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.engine import (
    Block,
    BlockParams,
    Catalogue,
    Event,
    Implementation,
    ImplementationContribution,
    ImplementationOrigin,
    Output,
    Ref,
    contract_digest,
    spec_of,
)
from roboflow_workflows.v2.engine.contributions import (
    CONTRACT_DIGEST_VERSION,
)
from roboflow_workflows.v2.engine.errors import CatalogueError

from tests.v2.support.paths import WORKFLOWS_ROOT

WORKFLOWS_DIR = WORKFLOWS_ROOT


class Reference(Implementation):
    name = "reference"
    requires = ("cpu",)

    def run(self, *, value: float, factor: float) -> dict:
        return {"scaled": value * factor}


class Scale(Block):
    """Public logical block with one public implementation."""

    type = "probe/scale@v1"
    aliases = ("probe/scale-old",)
    outputs = {"scaled": Output(FLOAT_KIND, description="value times factor")}
    implementations = (Reference,)

    class Params(BlockParams):
        value: Ref(FLOAT_KIND) = Field(description="Number to scale.")
        factor: float = Field(default=2.0, ge=0.0, description="Multiplier.")


class Offset(Block):
    """Public ordinary block: the class is its own ``default`` implementation."""

    type = "probe/offset@v1"
    outputs = {"shifted": Output(FLOAT_KIND)}

    def __init__(self, *, delta: float = 1.0):
        self.delta = delta

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value: float) -> dict:
        return {"shifted": value + self.delta}


def fast(name: str = "fast", requires: tuple = ("cpu", "fast-kernel")) -> type:
    class Fast(Implementation):
        def run(self, *, value: float, factor: float) -> dict:
            return {"scaled": value * factor}

    Fast.name = name
    Fast.requires = requires

    return Fast


class FastOffset(Implementation):
    name = "fast-offset"
    requires = ("fast-kernel",)

    def run(self, *, value: float) -> dict:
        return {"shifted": value + 1.0}


FAST = fast()


def contribution(
    *implementations: type,
    block: type = Scale,
    block_type: Any = None,
    contract: Any = None,
    provider: str = "example-provider",
    version: str = "1.0.0",
) -> ImplementationContribution:
    declared = ImplementationContribution(
        block_type=block_type or block.type,
        contract=contract or contract_digest(block),
        implementations=implementations or (FAST,),
        provider=provider,
        version=version,
    )

    return declared


def public() -> Catalogue:
    catalogue = Catalogue([Scale, Offset], namespace="probe")

    return catalogue


def names(catalogue: Catalogue, block_type: str = Scale.type) -> list:
    listed = [item.name for item in catalogue.entry(block_type).spec.implementations]

    return listed


# --- applying contributions --------------------------------------------------


def test_contribution_extends_a_new_catalogue_and_changes_no_class_or_catalogue() -> (
    None
):
    base = public()
    before = spec_of(Scale)
    base_description = base.describe()

    effective = base.with_contributions([contribution()])

    assert spec_of(Scale) is before
    assert Scale.implementations == (Reference,)
    assert base.entry(Scale.type).spec is before
    assert base.describe() == base_description
    assert names(effective) == ["fast", "reference"]
    added = effective.entry(Scale.type).spec.implementations[0]
    assert added.origin == ImplementationOrigin(
        provider="example-provider", version="1.0.0", contract=contract_digest(Scale)
    )
    assert (
        effective.entry(Scale.type).spec.implementations[1] is before.implementations[0]
    )


def test_no_contributions_give_an_equal_description() -> None:
    assert public().with_contributions([]).describe() == public().describe()


def test_contributions_apply_in_host_order_before_the_blocks_own() -> None:
    first = contribution(fast("a"), provider="provider-a")
    second = contribution(fast("b"), provider="provider-b")

    together = public().with_contributions([first, second])
    stepwise = public().with_contributions([second]).with_contributions([first])

    assert names(together) == ["a", "b", "reference"]
    assert names(stepwise) == ["b", "a", "reference"]


def test_ordinary_block_keeps_its_default_after_contributions() -> None:
    effective = public().with_contributions([contribution(FastOffset, block=Offset)])
    spec = effective.entry(Offset.type).spec

    assert names(effective, Offset.type) == ["fast-offset", "default"]
    assert spec.implementations[1].implementation_class is Offset
    assert spec.resources is None
    assert "resources" not in spec.describe()
    assert spec.describe()["implementations"][1]["resources"][0]["name"] == "delta"


def test_describe_shows_origin_only_for_contributed_implementations() -> None:
    described = public().with_contributions([contribution()]).describe()
    scale = next(block for block in described["blocks"] if block["type"] == Scale.type)
    public_described = public().describe()

    assert [item.get("origin") for item in scale["implementations"]] == [
        {
            "provider": "example-provider",
            "version": "1.0.0",
            "contract": contract_digest(Scale),
        },
        None,
    ]
    assert all(
        "origin" not in item
        for block in public_described["blocks"]
        for item in block["implementations"]
    )


def test_with_blocks_keeps_contributed_descriptions() -> None:
    class Extra(Offset):
        type = "probe/extra@v1"

    effective = public().with_contributions([contribution()])

    extended = effective.with_blocks([Extra], namespace="dynamic")

    assert extended.entry(Scale.type).spec is effective.entry(Scale.type).spec


# --- rejected contributions --------------------------------------------------


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: contribution(block_type="probe/missing@v1"), "unknown block type"),
        (lambda: contribution(block_type="probe/scale-old"), "canonical type"),
        (lambda: contribution(fast("reference")), "already listed"),
        (lambda: contribution(FAST, fast("fast")), "repeat name"),
        (lambda: contribution(FastOffset), "does not accept Params"),
    ],
    ids=["unknown", "alias", "public-name", "repeated-name", "narrow-run"],
)
def test_contributions_that_do_not_fit_the_block_are_rejected(
    build: Any, message: str
) -> None:
    declared = build()

    with pytest.raises(CatalogueError, match=message):
        public().with_contributions([declared])
    assert message in public().contribution_problem(declared)


def test_a_name_another_contribution_already_added_is_rejected() -> None:
    first = contribution(fast("shared"), provider="provider-a")
    second = contribution(fast("shared"), provider="provider-b")

    with pytest.raises(CatalogueError, match="already listed"):
        public().with_contributions([first, second])


def test_restated_contract_and_managed_state_are_rejected() -> None:
    class Restating(Implementation):
        name = "restating"
        outputs = {"scaled": Output(FLOAT_KIND)}

        def run(self, *, value: float, factor: float) -> dict:
            return {}

    class Stateful(Implementation):
        name = "stateful"

        def __init__(self, *, managed_state: Any):
            self.state = managed_state

        def run(self, *, value: float, factor: float) -> dict:
            return {}

    assert "restates contract" in public().contribution_problem(contribution(Restating))
    assert "managed_state" in public().contribution_problem(contribution(Stateful))


def test_another_contract_is_rejected_and_reported_without_raising() -> None:
    stale = contribution(contract="v1:sha256:" + "0" * 64)

    problem = public().contribution_problem(stale)

    assert "was built for contract v1:sha256:000" in problem
    assert contract_digest(Scale) in problem
    assert public().contribution_problem(contribution()) is None


def test_another_digest_version_is_unsupported_not_compared() -> None:
    future = contribution(contract="v2:sha256:" + "0" * 64)

    assert "version v2 is not supported" in public().contribution_problem(future)


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"contract": "sha256:" + "0" * 64}, "contract_digest"),
        ({"provider": "bad provider"}, "provider"),
        ({"version": ""}, "version"),
        ({"implementations": FAST}, "tuple of Implementation"),
        ({"implementations": ()}, "no implementations"),
        ({"block_type": ""}, "block_type"),
    ],
)
def test_malformed_contributions_are_rejected_at_construction(
    fields: dict, message: str
) -> None:
    values = {
        "block_type": Scale.type,
        "contract": contract_digest(Scale),
        "implementations": (FAST,),
        "provider": "example-provider",
        "version": "1.0.0",
        **fields,
    }

    with pytest.raises(CatalogueError, match=message):
        ImplementationContribution(**values)


def test_only_contributions_are_accepted() -> None:
    assert "Expected an ImplementationContribution" in public().contribution_problem(
        FAST
    )


# --- contract digest ---------------------------------------------------------


def test_digest_is_versioned_and_stable_in_fresh_interpreters() -> None:
    digest = contract_digest(Scale)
    code = (
        "from tests.v2.unit.engine.authoring.test_contributions import Scale\n"
        "from roboflow_workflows.v2.engine import contract_digest\n"
        "print(contract_digest(Scale))\n"
    )

    found = set()
    for seed in ("1", "2"):
        environment = {
            **os.environ,
            "PYTHONHASHSEED": seed,
            "PYTHONPATH": os.pathsep.join(
                [str(WORKFLOWS_DIR), os.environ.get("PYTHONPATH", "")]
            ),
        }
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            check=True,
            env=environment,
            text=True,
        )
        found.add(completed.stdout.strip())

    assert digest.startswith(f"v{CONTRACT_DIGEST_VERSION}:sha256:")
    assert found == {digest}


def test_digest_ignores_documentation_aliases_metadata_and_implementations() -> None:
    class Documented(Block):
        """Entirely different documentation."""

        type = Scale.type
        aliases = ("another-alias",)
        metadata = {"ui": {"icon": "x"}}
        outputs = {"scaled": Output(FLOAT_KIND, description="Reworded.")}
        implementations = (Reference, fast("extra"))

        class Params(BlockParams):
            """Reworded parameters."""

            value: Ref(FLOAT_KIND) = Field(
                description="Reworded.", title="Value", examples=["$inputs.x"]
            )
            factor: float = Field(
                default=2.0, ge=0.0, description="Reworded.", examples=[1.0]
            )

    assert contract_digest(Documented) == contract_digest(Scale)


def test_digest_ignores_nested_documentation_of_outputs_fields_and_events() -> None:
    original = spec_of(Emitting)
    documented = dataclasses.replace(
        original,
        outputs={"count": Output(INTEGER_KIND, description="Reworded output.")},
        fields={
            name: dataclasses.replace(field, description="Reworded field.")
            for name, field in original.fields.items()
        },
        events={
            "crossed": Event(
                fields={"level": FLOAT_KIND}, description="Reworded event."
            )
        },
    )

    assert contract_digest(documented) == contract_digest(original)


class Emitting(Block):
    type = "probe/emitting@v1"
    outputs = {"count": Output(INTEGER_KIND, description="How many.")}
    events = {
        "crossed": Event(fields={"level": FLOAT_KIND}, description="Level crossed.")
    }

    class Params(BlockParams):
        value: Ref(FLOAT_KIND) = Field(description="Level.")

    def run(self, *, value: float) -> dict:
        return {"count": 1}


def described_block(
    *,
    description_default: str = "keep",
    options_default: Any = None,
    factor_bound: float = 0.0,
    output_kind: Any = FLOAT_KIND,
    batch: str = "never",
    mutates: tuple = (),
) -> type:
    """A block whose real field is named ``description``, with tunable contract."""
    options_value = {"description": "data, not documentation"}
    if options_default is not None:
        options_value = options_default

    declared_mutates = mutates

    class Described(Block):
        type = "probe/described@v1"
        outputs = {"scaled": Output(output_kind)}
        implementations = (AnyFields,)
        mutates = declared_mutates

        class Params(BlockParams):
            value: Ref(FLOAT_KIND, batch=batch)
            factor: float = Field(default=2.0, ge=factor_bound)
            description: str = Field(default=description_default, title="Doc")
            options: dict = Field(default=options_value)

    return Described


class AnyFields(Implementation):
    name = "reference"

    def run(self, *, value: Any, factor: float, **ignored: Any) -> dict:
        return {"scaled": value}


@pytest.mark.parametrize(
    "change",
    [
        {"description_default": "other"},
        {"options_default": {"description": "changed data"}},
        {"factor_bound": 1.0},
        {"output_kind": STRING_KIND},
        {"batch": "always"},
        {"mutates": ("value",)},
    ],
    ids=[
        "field-named-description-default",
        "default-data-with-description-key",
        "constraint",
        "output-kind",
        "selector-batch",
        "mutates",
    ],
)
def test_digest_changes_with_real_contract_changes(change: dict) -> None:
    baseline = contract_digest(described_block())

    assert contract_digest(described_block()) == baseline
    assert contract_digest(described_block(**change)) != baseline


def test_field_named_description_is_part_of_the_contract() -> None:
    class Without(Block):
        type = "probe/described@v1"
        outputs = {"scaled": Output(FLOAT_KIND)}
        implementations = (AnyFields,)

        class Params(BlockParams):
            value: Ref(FLOAT_KIND)
            factor: float = Field(default=2.0, ge=0.0)
            options: dict = Field(default={"description": "data, not documentation"})

    assert contract_digest(Without) != contract_digest(described_block())


# --- merging -----------------------------------------------------------------


def offset_block(
    *,
    params_name: str = "OffsetParams",
    description: str = "Added.",
    example: int = 1,
    output_doc: str = "Result.",
) -> type:
    params = type(
        params_name,
        (BlockParams,),
        {
            "__annotations__": {"offset": int},
            "offset": Field(default=1, description=description, examples=[example]),
        },
    )

    class Offsetting(Block):
        type = "probe/offsetting@v1"
        Params = params
        outputs = {"value": Output(INTEGER_KIND, description=output_doc)}

        def run(self, *, offset: int = 1) -> dict:
            return {"value": offset}

    return Offsetting


@pytest.mark.parametrize(
    "edit",
    [
        {"description": "Added to x."},
        {"example": 2},
        {"params_name": "OffsetParameters"},
        {"output_doc": "The result."},
    ],
    ids=["field-description", "field-examples", "params-renamed", "output-doc"],
)
def test_digest_ignores_documentation_and_params_class_name(edit: dict) -> None:
    assert contract_digest(offset_block(**edit)) == contract_digest(offset_block())


def nested_block(*, model_name: str = "Region", minimum: int = 0) -> type:
    region_model = type(
        model_name,
        (BaseModel,),
        {
            "__annotations__": {"width": int, "mode": Literal["fit", "fill"]},
            "width": Field(default=8, ge=minimum, description="Pixels."),
            "mode": "fit",
        },
    )

    class Nested(Block):
        type = "probe/nested@v1"
        outputs = {"value": Output(INTEGER_KIND)}

        class Params(BlockParams):
            value: Ref(FLOAT_KIND)
            region: region_model = Field(default_factory=region_model)
            regions: List[Union[region_model, int]] = Field(default=[])

        def run(self, *, value: float, region: Any, regions: Any) -> dict:
            return {"value": 1}

    return Nested


def test_digest_ignores_nested_model_names_but_keeps_their_constraints() -> None:
    baseline = contract_digest(nested_block())
    schema = spec_of(nested_block()).params_schema()

    assert "$defs" in schema and "Region" in schema["$defs"]
    assert contract_digest(nested_block(model_name="Area")) == baseline
    assert contract_digest(nested_block(minimum=1)) != baseline


def test_real_fields_named_title_and_description_stay_in_the_contract() -> None:
    def block(field_name: str) -> type:
        params = type(
            "TitledParams",
            (BlockParams,),
            {"__annotations__": {field_name: str}, field_name: "x"},
        )

        def run(self: Any, **fields: Any) -> dict:
            return {"value": 1}

        titled = type(
            "Titled",
            (Block,),
            {
                "type": "probe/titled@v1",
                "outputs": {"value": Output(INTEGER_KIND)},
                "Params": params,
                "run": run,
            },
        )

        return titled

    digests = {
        contract_digest(block(name)) for name in ("title", "description", "label")
    }

    assert len(digests) == 3


def test_independently_built_equal_catalogues_merge_both_ways() -> None:
    first = public().with_contributions([contribution()])
    second = public().with_contributions([contribution()])

    forward = Catalogue.merge(first, second)
    backward = Catalogue.merge(second, first)

    assert names(forward) == names(backward) == ["fast", "reference"]
    assert forward.entry(Scale.type).spec is first.entry(Scale.type).spec
    assert backward.entry(Scale.type).spec is second.entry(Scale.type).spec


@pytest.mark.parametrize(
    "other",
    [
        lambda: public(),
        lambda: public().with_contributions([contribution(version="1.1.0")]),
        lambda: public().with_contributions(
            [contribution(provider="another-provider")]
        ),
        lambda: public().with_contributions([contribution(fast())]),
        lambda: public().with_contributions(
            [contribution(fast("a")), contribution(fast("b"))]
        ),
    ],
    ids=["public-only", "version", "provider", "another-class", "extra"],
)
def test_different_implementation_lists_of_one_block_do_not_merge(other: Any) -> None:
    effective = public().with_contributions([contribution()])

    with pytest.raises(CatalogueError, match="apply contributions once"):
        Catalogue.merge(effective, other())
    with pytest.raises(CatalogueError, match="apply contributions once"):
        Catalogue.merge(other(), effective)


def test_order_of_two_contributions_matters_when_merging() -> None:
    a = contribution(fast("a"), provider="provider-a")
    b = contribution(fast("b"), provider="provider-b")

    with pytest.raises(CatalogueError, match="apply contributions once"):
        Catalogue.merge(
            public().with_contributions([a, b]), public().with_contributions([b, a])
        )


def test_equal_contributions_still_need_identical_providers() -> None:
    first = Catalogue.merge(
        public().with_contributions([contribution()]),
        Catalogue(namespace="probe", providers={"kernel_cache": object()}),
    )
    second = Catalogue.merge(
        public().with_contributions([contribution()]),
        Catalogue(namespace="probe", providers={"kernel_cache": object()}),
    )

    with pytest.raises(CatalogueError, match="two different providers"):
        Catalogue.merge(first, second)
